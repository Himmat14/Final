"""
Several control laws on one satellite (step 14): a real Starlink satellite does not only keep a 500 m
band. Over 400 days this mean-element run contains

    orbit raising          continuous thrust from 6890 km up to the operational band (days 0-~1)
    station keeping        the 500 m deadband (the law of steps 5-9)
    band change            at day 150 the band is moved 3 km up (re-phasing / altitude change)
    collision avoidance    six unscheduled short burns (3 min) at random times, wherever the SMA is

Every burn carries its TRUE label. The learner sees only the noisy measured SMA (every 10 min) and must
    1. detect the burns (mean_element.detect_burns),
    2. label each burn's regime from three observable features: duration, SMA gain, and where it starts
       relative to the lower edge learned from the recent station-keeping burns:
           long / large gain                              -> orbit raising or band change ("transfer")
           short, not starting at the learned lower edge  -> collision avoidance
           otherwise                                      -> station keeping
       and compare with an unsupervised 3-component GMM on [log duration, log gain] (no rules),
    3. detect the band change as a CHANGEPOINT: station-keeping burns whose start SMA jumps away from
       the running median edge,
    4. learn each regime's law (SINDy-style constant rates per regime).
"""
from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture

from .mean_element import DAY_S, DT_S, calibrate, detect_burns, measure, simulate, space_weather

DAYS = 400
RAISE_FROM_KM = 6890.0
BAND_SHIFT_DAY = 150.0
BAND_SHIFT_KM = 3.0
N_CAM = 6
CAM_S = 180.0
REGIMES = ("transfer", "station keeping", "collision avoidance")
SIGMA_KM = 0.005                      # a good multi-pass orbit determination
SPAN = 3                              # compare samples 30 min apart (short CAM burns)


@dataclass
class RegimeRun:
    run: object
    labels: np.ndarray            # true label per burn (index into REGIMES)
    cam_times: np.ndarray


def simulate_regimes(seed=0):
    c = calibrate()
    rng = np.random.default_rng(seed)
    weather = space_weather(DAYS, seed=seed + 10)
    cam_times = np.sort(rng.uniform(20, DAYS - 5, N_CAM)) * DAY_S
    lower0, upper0 = c["lower"], c["upper"]

    def schedule(i, t, a, s):
        shift = BAND_SHIFT_KM if t >= BAND_SHIFT_DAY * DAY_S else 0.0
        lo, up = lower0 + shift, upper0 + shift
        for cam in cam_times:                             # collision avoidance: thrust for CAM_S, whatever a is
            if cam <= t < cam + CAM_S:
                return lo, up, 1
            if cam + CAM_S <= t < cam + CAM_S + DT_S:     # then hand back to the normal law, coasting
                return lo, up, 0
        return lo, up, None

    run = simulate(weather, a0=RAISE_FROM_KM, schedule=schedule)
    labels = []
    for on, off in zip(run.onsets, run.ends):
        if any(abs(on - cam) < 2 * DT_S for cam in cam_times):
            labels.append(2)
        elif off - on > 3 * c["burn_s"]:
            labels.append(0)
        else:
            labels.append(1)
    return RegimeRun(run=run, labels=np.array(labels), cam_times=cam_times)


@dataclass
class RegimeAnalysis:
    t: np.ndarray
    a: np.ndarray
    onsets: np.ndarray
    ends: np.ndarray
    a_on: np.ndarray
    a_off: np.ndarray
    rule_labels: np.ndarray
    gmm_labels: np.ndarray
    true_labels: np.ndarray       # matched to the detected burns (-1 = false detection)
    running_edge: np.ndarray      # learned lower edge at each detected burn
    changepoints: list            # days at which the band was found to move
    laws: dict                    # regime -> dict(rate_km_per_day, n)


def _match(detected, true_onsets, true_labels, tolerance_s=1800.0):
    out = np.full(detected.size, -1)
    for k, t in enumerate(detected):
        d = np.abs(true_onsets - t)
        if d.size and d.min() < tolerance_s:
            out[k] = true_labels[np.argmin(d)]
    return out


def analyse(rr, seed=0, window=7):
    c = calibrate()
    t, a = measure(rr.run, 600.0, SIGMA_KM, seed=seed)
    onsets, ends, a_on, a_off = detect_burns(t, a, SIGMA_KM, span=SPAN)
    duration, gain = ends - onsets, a_off - a_on
    typical_s = np.median(duration)
    # running lower edge from the previous `window` station-keeping-like burns (median: robust to CAMs)
    edge, history, labels, changepoints = np.full(onsets.size, np.nan), [], np.empty(onsets.size, dtype=int), []
    for k in range(onsets.size):
        if duration[k] > 3 * typical_s or gain[k] > 3 * (c["upper"] - c["lower"]):
            labels[k] = 0                                            # transfer: long burn, large gain
            history = []                                             # the band may have moved: start again
            if onsets[k] > 5 * DAY_S:
                changepoints.append(onsets[k] / DAY_S)
            continue
        edge[k] = np.median(history[-window:]) if len(history) >= 2 else np.nan
        short = duration[k] < 0.5 * typical_s
        off_edge = np.isfinite(edge[k]) and abs(a_on[k] - edge[k]) > 0.05
        labels[k] = 2 if (short or off_edge) else 1
        if labels[k] == 1:
            history.append(a_on[k])
    X = np.column_stack([np.log(duration), np.log(np.maximum(gain, 1e-3))])
    gmm = GaussianMixture(3, n_init=5, random_state=seed).fit(X)
    raw = gmm.predict(X)
    order = np.argsort(-gmm.means_[:, 0])                             # longest mean duration = transfer
    remap = {order[0]: 0, order[1]: 1, order[2]: 2}
    gmm_labels = np.array([remap[r] for r in raw])
    truth = _match(onsets, rr.run.onsets, rr.labels)
    laws = {}
    for k, name in enumerate(REGIMES):
        mine = labels == k
        if mine.any():
            laws[name] = dict(rate_km_per_day=float(np.median(gain[mine] / duration[mine]) * DAY_S), n=int(mine.sum()))
    coast = []
    for end, nxt in zip(ends[:-1], onsets[1:]):
        keep = (t > end + 1800) & (t < nxt - 1800)
        if keep.sum() > 20:
            coast.append(np.polyfit(t[keep], a[keep], 1)[0])
    laws["coast"] = dict(rate_km_per_day=float(np.median(coast) * DAY_S), n=len(coast))
    return RegimeAnalysis(t=t, a=a, onsets=onsets, ends=ends, a_on=a_on, a_off=a_off, rule_labels=labels,
                          gmm_labels=gmm_labels, true_labels=truth, running_edge=edge,
                          changepoints=changepoints, laws=laws)


def confusion(predicted, truth, n=len(REGIMES)):
    """(n x n) counts: rows = true regime, columns = predicted; false detections (truth -1) are left out."""
    m = np.zeros((n, n), dtype=int)
    for p, q in zip(predicted, truth):
        if q >= 0:
            m[q, p] += 1
    return m

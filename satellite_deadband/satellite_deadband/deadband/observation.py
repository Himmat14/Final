"""
A realistic observation model (step 11): instead of a full state every 6 s, the satellite is seen in
short PASSES (a radar or laser station: 8 minutes of noisy positions every 12 s) a few times a day, with
gaps of hours in between. Burns happen in the gaps.

Pipeline
--------
1. Orbit determination per pass: the 6 initial-state parameters of a gravity + J2 orbit are fitted to
   the pass's noisy positions by batched Gauss-Newton (all passes at once, RK4 propagation, finite-
   difference Jacobian). Started from a polynomial fit of the positions.
2. One mean SMA per pass, from the fitted state (energy including the J2 potential).
3. Gap test: between consecutive passes the mean SMA should fall by the drag decay k x gap. A residual
   da + k x gap far above the noise means a burn happened in the gap (thrust only RAISES the SMA).
   k and one pass's SMA noise are learned on TRAIN (days 0-40) by robust statistics; threshold =
   N sigma of the tested difference (pass-to-pass, or the 1-day multi-pass windows).
4. Score per gap: detected burns vs the true burn onsets falling in that gap (precision / recall / F1),
   and the burn size estimated from the residual.

Also: a reader for CCSDS OEM ephemeris files (the format Starlink publishes its ephemerides in) so the
same gap test can be run on real data placed in data/ephemeris/.
"""
from pathlib import Path

import numpy as np

from .constants import DU, SECONDS, TU
from .derivative_detection import known_accel
from .long_run import TRAIN_DAYS, controlled_run
from .smooth_controller import mean_sma_km_series

PASS_MINUTES = 8.0
PASS_STEP_S = 12.0                     # two samples of the 6-s controlled run
N_MAD = 6.0                            # pass-to-pass threshold [sigma]
N_SIGMA_WINDOW = 4.0                   # multi-pass threshold [sigma of the window difference]
TESTS = {"pass-to-pass": 0.0, "1-day multi-pass": 1.0}
GN_ITERATIONS = 6
PASSES_PER_DAY = (1, 2, 4, 8)
NOISES_KM = (0.01, 0.1, 0.3, 1.0)
STUDY_DAYS = 120


def _propagate(x, n_steps, h):
    """(P, 3, n_steps) positions from (6, P) initial states, RK4 under gravity + J2, step h (non-dim)."""
    r, v = x[0:3].copy(), x[3:6].copy()
    out = np.empty((x.shape[1], 3, n_steps))
    out[:, :, 0] = r.T
    for k in range(1, n_steps):
        k1r, k1v = v, known_accel(r)
        k2r, k2v = v + 0.5 * h * k1v, known_accel(r + 0.5 * h * k1r)
        k3r, k3v = v + 0.5 * h * k2v, known_accel(r + 0.5 * h * k2r)
        k4r, k4v = v + h * k3v, known_accel(r + h * k3r)
        r = r + h / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        out[:, :, k] = r.T
    return out, np.vstack([r, v])


def fit_passes(obs, h):
    """
    Batched Gauss-Newton orbit determination. obs: (P, 3, M) noisy positions (non-dim), equally spaced by h.
    Returns (6, P) states at the LAST sample of each pass.
    """
    P, _, M = obs.shape
    tau = np.arange(M) * h
    # starting guess: quadratic fit of each coordinate gives position and velocity at sample 0
    coef = np.polynomial.polynomial.polyfit(tau, obs.transpose(2, 0, 1).reshape(M, -1), 4)
    x = np.vstack([coef[0].reshape(P, 3).T, coef[1].reshape(P, 3).T])
    scale = np.array([1e-6] * 3 + [1e-6] * 3)[:, None]
    for _ in range(GN_ITERATIONS):
        model, _ = _propagate(x, M, h)
        residual = (obs - model).reshape(P, -1)                        # (P, 3M)
        J = np.empty((P, 3 * M, 6))
        for j in range(6):
            dx = np.zeros_like(x)
            dx[j] = scale[j]
            J[:, :, j] = ((_propagate(x + dx, M, h)[0] - model).reshape(P, -1)) / scale[j]
        JT = J.transpose(0, 2, 1)
        step = np.linalg.solve(JT @ J, (JT @ residual[:, :, None]))[:, :, 0]
        x = x + step.T
    _, last = _propagate(x, M, h)
    return last


def pass_sma(noise_km, passes_per_day, days=STUDY_DAYS, seed=0):
    """(pass end times [days], mean SMA per pass [km], true SMA at the pass [km]) for one configuration."""
    run = controlled_run()
    rng = np.random.default_rng(seed)
    stride = int(round(PASS_STEP_S / 6.0))
    m = int(PASS_MINUTES * 60 / PASS_STEP_S)
    samples_per_day = int(86400 / 6.0)
    starts = []
    for d in range(days):
        times = np.sort(rng.uniform(0, samples_per_day - m * stride - 1, passes_per_day)).astype(int)
        starts.extend(d * samples_per_day + times)
    starts = np.array(starts)
    idx = starts[:, None] + np.arange(m)[None, :] * stride                   # (P, M) sample indices
    obs = run.r[:, idx].transpose(1, 0, 2) + noise_km / DU * rng.standard_normal((len(starts), 3, m))
    state = fit_passes(obs, PASS_STEP_S * SECONDS)
    end = idx[:, -1]
    return (run.t[end] * TU / 86400, mean_sma_km_series(state),
            run.mean_sma_km[end])


def gap_test(t_days, sma, true_onsets_days, train_days=TRAIN_DAYS, window_days=0.0):
    """
    Burn detection in the gaps between passes. Returns (scores dict, residual km, flags, burns per gap).

    window_days = 0: compare each pass with the previous one (pass-to-pass).
    window_days > 0: multi-pass test: the drag-corrected mean SMA of ALL passes within window_days AFTER the
        gap minus that of all passes within window_days BEFORE it, so the noise falls as 1/sqrt(passes);
        a burn then also raises the neighbouring gaps' residuals, so only local maxima are kept.
    """
    gaps = np.diff(t_days)
    dsma = np.diff(sma)
    in_gap = np.array([np.sum((true_onsets_days > a) & (true_onsets_days <= b))
                       for a, b in zip(t_days[:-1], t_days[1:])])
    train = t_days[1:] < train_days
    # drag decay [km/day] from pass pairs ~1 day apart (a long baseline beats the pass noise; pairs that
    # straddle a burn are outliers and the median ignores them)
    j = np.searchsorted(t_days, t_days + 1.0)
    pairs = (j < t_days.size) & (t_days < train_days)
    i_idx, j_idx = np.flatnonzero(pairs), j[pairs]
    decay = -np.median((sma[j_idx] - sma[i_idx]) / (t_days[j_idx] - t_days[i_idx]))
    pass_to_pass = dsma + decay * gaps
    centre = np.median(pass_to_pass[train])
    sigma_pass = 1.4826 * np.median(np.abs(pass_to_pass[train] - centre)) / np.sqrt(2)   # one pass's SMA noise
    if window_days <= 0:
        residual = pass_to_pass - centre
        sigma = np.full(gaps.size, sigma_pass * np.sqrt(2))
        n_sigma = N_MAD
    else:
        level = sma + decay * t_days                                      # drag removed: flat between burns
        residual, sigma = np.empty(gaps.size), np.empty(gaps.size)
        for g in range(gaps.size):
            before = (t_days <= t_days[g]) & (t_days > t_days[g] - window_days)
            after = (t_days >= t_days[g + 1]) & (t_days < t_days[g + 1] + window_days)
            residual[g] = level[after].mean() - level[before].mean()      # means: a graded peak at the burn gap
            sigma[g] = sigma_pass * np.sqrt(1 / before.sum() + 1 / after.sum())
        n_sigma = N_SIGMA_WINDOW
    flags = residual > n_sigma * sigma
    if window_days > 0:                                                   # keep only the local maximum
        n = max(1, int(np.ceil(window_days / np.median(gaps))))
        peak = np.array([g == max(0, g - n) + int(np.argmax(residual[max(0, g - n):g + n + 1]))
                         for g in range(gaps.size)])
        flags &= peak
    mad = float(np.median(sigma))
    test = ~train
    tp = int(np.sum(flags & (in_gap > 0) & test))
    fp = int(np.sum(flags & (in_gap == 0) & test))
    fn = int(np.sum(~flags & (in_gap > 0) & test))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    sizes = residual[flags & (in_gap > 0) & test]
    return (dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                 f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                 mean_gap_h=float(np.mean(gaps) * 24), noise_floor_m=float(mad * 1000),
                 burn_size_m=float(np.median(sizes) * 1000) if sizes.size else float("nan")),
            residual, flags, in_gap)


def stress(days=STUDY_DAYS):
    """Gap-test scores for every (passes per day, noise) and the per-pass SMA error."""
    from .mean_element import DAY_S
    run = controlled_run()
    from .derivative_detection import runs_of_ones
    onsets = np.array([s for s, _ in runs_of_ones(run.t * TU, run.true_on)]) / DAY_S
    rows, examples = [], {}
    for per_day in PASSES_PER_DAY:
        for noise in NOISES_KM:
            t_days, sma, truth = pass_sma(noise, per_day, days)
            for test, window in TESTS.items():
                scores, residual, flags, in_gap = gap_test(t_days, sma, onsets, window_days=window)
                rows.append(dict(test=test, passes_per_day=per_day, noise_km=noise,
                                 sma_error_m=float(np.sqrt(np.mean((sma - truth) ** 2)) * 1000), **scores))
                examples[(test, per_day, noise)] = (t_days, sma, truth, residual, flags, in_gap)
    return rows, examples, onsets


# ---------------------------------------------------------------------------
# CCSDS OEM ephemeris files (e.g. the Starlink ephemerides on space-track.org)
# ---------------------------------------------------------------------------
def read_oem(path):
    """(t [s from the first epoch], r (3, N) [km], v (3, N) [km/s]) from a CCSDS OEM text file."""
    from datetime import datetime
    times, states, in_data, in_cov = [], [], False, False
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("COMMENT"):
            continue
        if text.startswith("META_START"):
            in_data = False
            continue
        if text.startswith("META_STOP"):
            in_data = True
            continue
        if text.startswith("COVARIANCE_START"):
            in_cov = True
            continue
        if text.startswith("COVARIANCE_STOP"):
            in_cov = False
            continue
        if in_data and not in_cov:
            parts = text.split()
            if len(parts) >= 7 and "T" in parts[0]:
                stamp = parts[0].rstrip("Z")
                fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in stamp else "%Y-%m-%dT%H:%M:%S"
                times.append(datetime.strptime(stamp, fmt))
                states.append([float(p) for p in parts[1:7]])
    states = np.array(states).T
    t = np.array([(x - times[0]).total_seconds() for x in times])
    return t, states[0:3], states[3:6]


def oem_mean_sma(path):
    """(t [days], mean SMA [km]) of an OEM ephemeris (J2 energy, as everywhere in this project)."""
    t, r, v = read_oem(path)
    state = np.vstack([r / DU, v / (DU / TU)])
    return t / 86400, mean_sma_km_series(state)


def find_ephemerides(folder=None):
    folder = Path(folder) if folder else Path(__file__).resolve().parents[1] / "data" / "ephemeris"
    return sorted(folder.glob("*.oem")) + sorted(folder.glob("*.txt")) if folder.exists() else []

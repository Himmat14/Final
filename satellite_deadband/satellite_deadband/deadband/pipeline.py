"""
The whole analysis chain, end to end (step 20): measured r, v  ->  burn detection (GMM)  ->  mean SMA and
its rate  ->  drag c_d, thrust T and switching edges (SINDy / BINDy / joint regression)  ->  free-running
burn forecast (learned law, Gaussian-process coast curve, periodic baseline).

Three studies
-------------
1. Stage accuracy and error budget ("oracle swaps"). Every stage is run with its LEARNED output, then the
   true value is swapped in one stage at a time (true burn labels, true c_d, true T, true edges, all true).
   The drop in forecast error when a stage is made perfect is that stage's share of the final error.
2. Sampling robustness. The whole chain is re-run on the same 400-day run sampled every 6 s, 30 s, 1, 2, 5
   and 10 min. Below 2 min the GMM uses the step 6 Savitzky-Golay derivative; from 2 min the one-step
   propagation feature of the step 6 stress test (deadband/stress.py). The SMA smoothing window and the
   edge buffers keep their length in TIME.
3. Thrust-law (Gaussian process) robustness to sampling, and the space-weather forecaster (step 10)
   robustness to the measurement cadence.

The law forecast uses the learned rates directly. With constant drag the hysteresis law gives a periodic
burn sequence: coast time = band width / coast rate, burn time = band width / net burn rate. A relative
period error e then makes the n-th burn wrong by n e P, so the mean |onset error| over a horizon H is
close to (1/2) H |e| (checked in step 20, "calculation checks").
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.mixture import GaussianMixture

from . import stress
from .burn_folding import fit_thrust_law, stack_bursts
from .classifiers import REFERENCE_NOISE, add_noise, build_features, detection_dataset, mixture_burn_components
from .constants import DU, MU, SECONDS, SMOOTH_CD, THRUST_ACCEL, TU
from .derivative_detection import event_scores, group_into_events, runs_of_ones
from .long_run import HORIZONS_DAYS, TRAIN_DAYS
from .settings import tuned
from .sindy import drag_shape, learn_switching_law, sma_data, thrust_shape
from .sindy_split import _ard

DAY = 86400.0
STRIDES = (1, 5, 10, 20, 50, 100)                     # x 6 s: 6 s, 30 s, 1, 2, 5, 10 min
GP_STRIDES = (1, 2, 5, 10, 20)                        # thrust-law study: up to 2 min (a burn is 23 min)
CADENCES_S = (600, 1800, 3600, 7200, 14400)           # space-weather study: 10 min ... 4 h
DERIVATIVE_LIMIT_S = 120.0                            # above this step the propagation feature is used
FIT_SAMPLES = 20_000
MODELS = ("SINDy split (D)", "SINDy joint (A)", "BINDy split (ARD)", "GP coast curve", "periodic baseline")


@dataclass
class Prepared:
    dataset: object
    events: np.ndarray
    data: object
    f1: float
    feature: str
    step_s: float
    smoothing_s: float
    source: str = "j2_drag"


# ---------------------------------------------------------------------------
# Stage 1-2: detection and SMA data at any sampling step
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2)
def true_onsets_days(source="j2_drag"):
    ds = detection_dataset(source)
    return np.array([s for s, _ in runs_of_ones(ds.t, ds.true_on)]) * TU / DAY


def catalogue_accel_ms2(t, r):
    """(3, N) Moon + Sun + SRP acceleration [m/s^2] at times t (TU) and positions r: the catalogue force model."""
    from .constants import ACCEL_UNIT_MS2
    from .perturbations import MOON_DISTANCE, MOON_RATE, MU_MOON, MU_SUN, SRP_ACCEL, SUN_DISTANCE, SUN_RATE
    out = np.zeros_like(r)
    for rate, distance, mu in ((MOON_RATE, MOON_DISTANCE, MU_MOON), (SUN_RATE, SUN_DISTANCE, MU_SUN)):
        body = distance * np.array([np.cos(rate * t), np.sin(rate * t), np.zeros_like(t)])
        to_body = body - r
        out += mu * (to_body / np.linalg.norm(to_body, axis=0) ** 3 - body / distance ** 3)
        if mu == MU_SUN:
            out += SRP_ACCEL * (-to_body) / np.linalg.norm(to_body, axis=0)
    return out * ACCEL_UNIT_MS2


def _matrix(accel_ms2, v):
    """[log |a|, along-track a / 1e-4] from a (3, N) unmodelled acceleration (as classifiers.Features.matrix)."""
    from .classifiers import FLOOR_MS2
    along = np.sum(accel_ms2 * v, axis=0) / np.linalg.norm(v, axis=0)
    return np.column_stack([np.log(np.linalg.norm(accel_ms2, axis=0) + FLOOR_MS2), along / 1e-4])


def _features(ds, r, v, t_catalogue=None):
    """Classifier features; with t_catalogue the Moon / Sun / SRP model is subtracted as well."""
    if ds.step_s >= DERIVATIVE_LIMIT_S:
        # the propagation feature's noise (> 1e-6 m/s^2 at these steps) is far above the third bodies
        return stress.propagation_features(r, v, ds.step_s), "one-step propagation", ds.step_s
    window, order = stress.derivative_settings(ds.step_s)
    accel = build_features(r, v, ds.step_s, window=window, order=order, method="Savitzky-Golay").accel_from_v
    if t_catalogue is not None:
        accel = accel - catalogue_accel_ms2(t_catalogue, r)
    return _matrix(accel, v), "Savitzky-Golay derivative", window * ds.step_s


def prepare(stride=1, labels="gmm", noise_scale=1.0, seed=1, source="j2_drag", catalogue=False):
    """
    source: "j2_drag" or "full" (Moon, Sun, SRP in the truth). catalogue=True: the analyst also models the
    Moon, Sun and SRP, so they are removed from the classifier features and from the measured SMA rate.
    """
    base = detection_dataset(source)
    ds = base if stride == 1 else base.every(stride)
    r, v = add_noise(ds, REFERENCE_NOISE[0] * noise_scale, REFERENCE_NOISE[1] * noise_scale, seed)
    step = ds.step_s
    X, feature, smear_s = _features(ds, r, v, ds.t if catalogue else None)
    if labels == "truth":
        events = ds.true_on.astype(int)
    else:
        rows = np.flatnonzero(ds.train)
        pick = np.random.default_rng(0).choice(rows, size=min(FIT_SAMPLES, rows.size), replace=False)
        model = GaussianMixture(n_components=tuned("gmm_components"), n_init=3, random_state=0).fit(X[pick])
        is_burn = mixture_burn_components(model)
        events = group_into_events(is_burn[model.predict(X)].astype(int), min_samples=max(1, int(np.ceil(60 / step))))
    test = ds.test
    f1 = event_scores(ds.t[test], events[test], ds.true_on[test], step, max(60.0, step))["f1"]
    smoothing_s = tuned("sma_window") * 6.0                       # the tuned SMA smoothing, kept in time
    window = max(5, int(round(smoothing_s / step)) | 1)
    data = sma_data(ds.t, r, v, events, step, window=window, edge_buffer_s=120.0 + max(smear_s, window * step) / 2)
    if catalogue:
        # rate of the mean SMA caused by the modelled Moon / Sun / SRP: da/dt = (2 a^2 / mu) v . a_cat,
        # smoothed with the same filter as the measured rate, then removed
        from scipy.signal import savgol_filter
        from .constants import ACCEL_UNIT_MS2
        a_cat = catalogue_accel_ms2(ds.t, r) / ACCEL_UNIT_MS2
        rate = 2 * data.a ** 2 / MU * np.sum(v * a_cat, axis=0)
        data.a_dot = data.a_dot - savgol_filter(rate, window, 2)
    return Prepared(ds, events, data, float(f1), feature, step, window * step, source)


# ---------------------------------------------------------------------------
# Stage 3: drag, thrust and edges
# ---------------------------------------------------------------------------
@dataclass
class Law:
    name: str
    cd: float
    T: float
    lower_km: float
    upper_km: float
    cd_std: float = 0.0
    T_std: float = 0.0


def _shapes(p):
    d = p.data
    tr = p.dataset.train
    D = drag_shape(d.a, d.speed)
    B = thrust_shape(d.a, d.speed)
    coast, burn = d.coast & tr, d.burn & tr
    D_bar = float(np.mean(D[coast])) if coast.any() else float(np.mean(D[tr]))
    B_bar = float(np.mean(B[burn])) if burn.sum() >= 3 else float(np.mean(B[tr]))
    return D, B, coast, burn, D_bar, B_bar


def _gain_thrust(p, cd, D_bar, B_bar):
    """
    T from the SMA gained across each detected TRAIN burn (used when too few clean burn samples are left).
    The gain is read outside the smoothing window on both sides; across that window of length W the satellite
    also coasts, so gain = T B_bar t_b - c_d D_bar W, with t_b the detected burn duration.
    """
    d, ds = p.data, p.dataset
    settle = (120.0 + p.smoothing_s / 2 + p.step_s) / TU
    # each burn edge is only known to within one sample: the centred duration estimate adds half a step
    extra = 0.5 * p.step_s / TU
    gains, durations = [], []
    for start, end in runs_of_ones(ds.t[ds.train], p.events[ds.train]):
        i0 = max(np.searchsorted(ds.t, start - settle), 0)
        i1 = min(np.searchsorted(ds.t, end + settle), len(ds.t) - 1)
        gains.append(d.a[i1] - d.a[i0] + cd * D_bar * (ds.t[i1] - ds.t[i0]))
        durations.append(end - start + extra)
    # ratio of sums: the sampled durations are unbiased on average, their reciprocals are not
    return float(np.sum(gains) / np.sum(durations)) / B_bar if gains else np.nan


def learn_laws(p):
    """{name: Law} for the three regression variants; edges from the detected burns."""
    D, B, coast, burn, D_bar, B_bar = _shapes(p)
    d, tr = p.data, p.dataset.train
    switching = learn_switching_law(p.dataset.t, d.a, p.events, tr, settle_s=120.0 + p.step_s)
    lower, upper = switching.lower * DU, switching.upper * DU
    laws = {}
    # split: c_d on clean coast samples, then T on clean burn samples given c_d
    cd = float(-np.sum(d.a_dot[coast] * D[coast]) / np.sum(D[coast] ** 2))
    if burn.sum() >= 5:
        z = d.a_dot[burn] + cd * D[burn]
        T = float(np.sum(z * B[burn]) / np.sum(B[burn] ** 2))
    else:
        T = _gain_thrust(p, cd, D_bar, B_bar)
    laws["SINDy split (D)"] = Law("SINDy split (D)", cd, T, lower, upper)
    # BINDy (ARD) on the same split, with posterior standard deviations
    (cd_b,), (cd_sd,) = _ard((-D[coast])[:, None], d.a_dot[coast])
    if burn.sum() >= 5:
        (T_b,), (T_sd,) = _ard(B[burn][:, None], d.a_dot[burn] + cd_b * D[burn])
    else:
        T_b, T_sd = _gain_thrust(p, cd_b, D_bar, B_bar), 0.0
    laws["BINDy split (ARD)"] = Law("BINDy split (ARD)", float(cd_b), float(T_b), lower, upper, float(cd_sd), float(T_sd))
    # joint: one regression over every TRAIN sample with the detected thruster state
    s = p.events.astype(float)
    X = np.column_stack([-D, s * B])[tr]
    cd_j, T_j = np.linalg.lstsq(X / np.linalg.norm(X, axis=0), d.a_dot[tr], rcond=None)[0] / np.linalg.norm(X, axis=0)
    laws["SINDy joint (A)"] = Law("SINDy joint (A)", float(cd_j), float(T_j), lower, upper)
    return laws, (D_bar, B_bar)


# ---------------------------------------------------------------------------
# Stage 4: forecasts
# ---------------------------------------------------------------------------
def _last_train_end_days(p):
    ends = [e for _, e in runs_of_ones(p.dataset.t[p.dataset.train], p.events[p.dataset.train])]
    return ends[-1] * TU / DAY


def law_forecast(law, shapes, start_day):
    """Periodic onsets [days] implied by a law: coast time and burn time between the edges."""
    D_bar, B_bar = shapes
    width = law.upper_km - law.lower_km
    coast_rate = law.cd * D_bar * DU / TU * DAY                       # km/day (decay)
    burn_rate = (law.T * B_bar - law.cd * D_bar) * DU / TU * DAY
    if coast_rate <= 0 or burn_rate <= 0 or width <= 0:
        return None, np.nan
    period = width / coast_rate + width / burn_rate
    first = start_day + width / coast_rate
    return first + period * np.arange(0, int(400 / period) + 2), period


def score_onsets(predicted, source="j2_drag"):
    actual = true_onsets_days(source)
    actual = actual[actual > TRAIN_DAYS]
    if predicted is None or len(predicted) == 0:
        return {f"{h}d": np.nan for h in HORIZONS_DAYS}
    predicted = np.asarray(predicted)
    errors = np.array([predicted[np.argmin(np.abs(predicted - x))] - x for x in actual]) * 1440
    lead = actual - TRAIN_DAYS
    return {f"{h}d": float(np.mean(np.abs(errors[lead <= h]))) for h in HORIZONS_DAYS}


def all_forecasts(p, laws, shapes):
    """{model: (errors by horizon, period days)} for the five forecasters."""
    from stages.report_step9 import gp_coast_forecast
    start = _last_train_end_days(p)
    out = {}
    for name in ("SINDy split (D)", "SINDy joint (A)", "BINDy split (ARD)"):
        onsets, period = law_forecast(laws[name], shapes, start)
        out[name] = (score_onsets(onsets, p.source), period)
    ds, d = p.dataset, p.data
    switching = learn_switching_law(ds.t, d.a, p.events, ds.train, settle_s=120.0 + p.step_s)
    try:
        gp = gp_coast_forecast(ds, p.events, d, switching, np.array([TRAIN_DAYS * DAY, 400 * DAY]))
        out["GP coast curve"] = (score_onsets(gp["onsets"] / DAY, p.source), np.nan)
    except Exception:
        out["GP coast curve"] = (score_onsets(None, p.source), np.nan)
    train_onsets = np.array([s for s, _ in runs_of_ones(ds.t[ds.train], p.events[ds.train])]) * TU / DAY
    period = float(np.mean(np.diff(train_onsets)))
    out["periodic baseline"] = (score_onsets(train_onsets[-1] + period * np.arange(1, int(400 / period) + 2), p.source),
                                period)
    return out


# ---------------------------------------------------------------------------
# Study 1: stage accuracy and error budget
# ---------------------------------------------------------------------------
def true_law(source="j2_drag"):
    from .mean_element import calibrate
    c = calibrate(source)
    return Law("truth", SMOOTH_CD, THRUST_ACCEL, c["lower"], c["upper"])


def effective_law(shapes, source="j2_drag"):
    """
    The c_d and T that make the constant-rate law reproduce the full run EXACTLY: the coast rate equals the
    measured coast slope k and the burn rate the measured gain rate (mean_element.calibrate). They differ
    from the true c_d and T by the model-form error of replacing D(a, v) and B(a, v) by their TRAIN means
    (about 0.04 % for c_d), which is the floor of any constant-rate law.
    """
    from .mean_element import calibrate
    c = calibrate(source)
    D_bar, B_bar = shapes
    scale = DU / TU
    cd = c["k"] / (D_bar * scale)
    T = (c["T"] - c["k"] + cd * D_bar * scale) / (B_bar * scale)
    return Law("effective truth", cd, T, c["lower"], c["upper"])


def stage_metrics(p, law):
    t = true_law(p.source)
    return dict(f1=p.f1, cd_error_pct=100 * (law.cd / SMOOTH_CD - 1), T_error_pct=100 * (law.T / THRUST_ACCEL - 1),
                lower_error_m=1000 * (law.lower_km - t.lower_km), upper_error_m=1000 * (law.upper_km - t.upper_km))


def error_budget(source="j2_drag", catalogue=False):
    """Forecast error of the split law with each stage swapped for the truth (6-s data, reference noise)."""
    learned = prepare(1, "gmm", source=source, catalogue=catalogue)
    laws, shapes = learn_laws(learned)
    oracle_labels = prepare(1, "truth", source=source, catalogue=catalogue)
    laws_true_labels, shapes_true = learn_laws(oracle_labels)
    t = effective_law(shapes, source)
    phys = true_law(source)
    base = laws["SINDy split (D)"]
    start = _last_train_end_days(learned)
    variants = {
        "pipeline as learned": (base, shapes),
        "+ true burn labels": (laws_true_labels["SINDy split (D)"], shapes_true),
        "+ true c_d": (Law("", t.cd, base.T, base.lower_km, base.upper_km), shapes),
        "+ true thrust T": (Law("", base.cd, t.T, base.lower_km, base.upper_km), shapes),
        "+ true edges": (Law("", base.cd, base.T, t.lower_km, t.upper_km), shapes),
        "all stages true": (Law("", t.cd, t.T, t.lower_km, t.upper_km), shapes),
        "physical c_d, T (model-form floor)": (Law("", phys.cd, phys.T, t.lower_km, t.upper_km), shapes),
    }
    rows = []
    for name, (law, sh) in variants.items():
        onsets, period = law_forecast(law, sh, start)
        rows.append(dict(variant=name, period_days=period, errors=score_onsets(onsets, source),
                         **stage_metrics(learned, law)))
    forecasts = {"GMM labels": all_forecasts(learned, laws, shapes),
                 "true labels": all_forecasts(oracle_labels, laws_true_labels, shapes_true)}
    metrics = {name: stage_metrics(learned, law) | dict(cd_std_pct=100 * law.cd_std / SMOOTH_CD,
                                                       T_std_pct=100 * law.T_std / THRUST_ACCEL)
               for name, law in laws.items()}
    effective = dict(cd_ratio=t.cd / SMOOTH_CD, T_ratio=t.T / THRUST_ACCEL)
    return rows, forecasts, metrics, learned, effective


# ---------------------------------------------------------------------------
# Study 2: the whole chain vs sampling step
# ---------------------------------------------------------------------------
def sampling_study(strides=STRIDES):
    rows = []
    for stride in strides:
        p = prepare(stride, "gmm")
        laws, shapes = learn_laws(p)
        fc = all_forecasts(p, laws, shapes)
        split = laws["SINDy split (D)"]
        rows.append(dict(step_s=p.step_s, feature=p.feature, f1=p.f1, smoothing_s=p.smoothing_s,
                         **{k: v for k, v in stage_metrics(p, split).items() if k != "f1"},
                         cd_joint_error_pct=100 * (laws["SINDy joint (A)"].cd / SMOOTH_CD - 1),
                         T_joint_error_pct=100 * (laws["SINDy joint (A)"].T / THRUST_ACCEL - 1),
                         cd_bindy_std_pct=100 * laws["BINDy split (ARD)"].cd_std / SMOOTH_CD,
                         n_coast=int((p.data.coast & p.dataset.train).sum()), n_burn=int((p.data.burn & p.dataset.train).sum()),
                         forecast={m: fc[m][0] for m in MODELS}))
        del p
    return rows


# ---------------------------------------------------------------------------
# Study 3a: Gaussian-process thrust law vs sampling step
# ---------------------------------------------------------------------------
def thrust_law_sampling(strides=GP_STRIDES):
    from stages.report_step8 import true_law_numbers
    rows = []
    for stride in strides:
        base = detection_dataset()
        ds = base if stride == 1 else base.every(stride)
        r, v = add_noise(ds, 2 * REFERENCE_NOISE[0], 2 * REFERENCE_NOISE[1], seed=0)       # step 8 noise level
        window = max(5, int(round(tuned("sg_window_stacked") * 6.0 / ds.step_s)) | 1)
        order = min(tuned("sg_order_stacked"), window - 2)
        features = build_features(r, v, ds.step_s, window=window, order=order, method="Savitzky-Golay")
        signal = features.along_track
        events = ds.true_on                                                                  # labels: step 6 truth
        stack = stack_bursts(ds.t, signal, events, ds.step_s)
        truth = true_law_numbers(stack_bursts(ds.t, ds.thrust_ms2, events, ds.step_s))
        truth_full = true_law_numbers(stack_bursts(base.t, base.thrust_ms2, base.true_on, base.step_s))
        law = fit_thrust_law(stack)
        rows.append(dict(step_s=ds.step_s, window_s=window * ds.step_s, bursts=len(stack.windows),
                         plateau_ratio=law.plateau / truth_full["plateau"],
                         duration_ratio=law.duration_s / truth_full["duration_s"],
                         rise_time_s=law.rise_time_s, true_rise_time_s=truth_full["rise_time_s"],
                         offsets_s=stack.offsets_s.tolist(), mean=law.mean.tolist()))
        del base
    return rows


# ---------------------------------------------------------------------------
# Study 3b: space-weather forecasting vs measurement cadence
# ---------------------------------------------------------------------------
def cadence_study(cadences_s=CADENCES_S):
    from .forecasting import SCENARIO_SIGMA_KM, horizon_table, learn, rolling_errors, scenario
    from .mean_element import measure
    sc = scenario()
    forecasters = ("periodic baseline", "law, latest coast", "law + space weather", "law + perfect forecast")
    rows = []
    for every in cadences_s:
        t, a = measure(sc.run, every, SCENARIO_SIGMA_KM, seed=3)
        L = learn(t, a, SCENARIO_SIGMA_KM, sc.weather)
        table = horizon_table(rolling_errors(L, sc.weather, sc.run.onsets, forecasters=forecasters), forecasters)
        rows.append(dict(cadence_s=every, detected=int(L.onsets.size), true=int(sc.run.onsets.size),
                         lower_error_m=1000 * (L.lower - sc.learned.lower), k_ratio=L.k_proxy / sc.learned.k_proxy,
                         errors={n: dict(zip([f"{h}d" for h in HORIZONS_DAYS], v)) for n, v in table.items()}))
    return rows


# ---------------------------------------------------------------------------
# The GMM step in detail
# ---------------------------------------------------------------------------
def gmm_anatomy(max_k=7, seed=1):
    """
    What the Gaussian mixture of step 6 actually fits (6-s data, reference noise): the features, the fitted
    components and which are called burn, the information criteria against K, and P(burn) through one burn.
    """
    ds = detection_dataset()
    r, v = add_noise(ds, REFERENCE_NOISE[0], REFERENCE_NOISE[1], seed)
    X, _, _ = _features(ds, r, v)
    rows = np.flatnonzero(ds.train)
    pick = np.random.default_rng(0).choice(rows, size=min(FIT_SAMPLES, rows.size), replace=False)
    ic = []
    for k in range(1, max_k + 1):
        m = GaussianMixture(n_components=k, n_init=3, random_state=0).fit(X[pick])
        ic.append(dict(k=k, bic=float(m.bic(X[pick])), aic=float(m.aic(X[pick])),
                       log_likelihood=float(m.score(X[pick]) * len(pick))))
    model = GaussianMixture(n_components=tuned("gmm_components"), n_init=3, random_state=0).fit(X[pick])
    is_burn = mixture_burn_components(model)
    coast = int(np.argmax(model.weights_))
    threshold = float(model.means_[coast, 0] + 3 * np.sqrt(model.covariances_[coast][0, 0]))
    burn_prob = model.predict_proba(X)[:, is_burn].sum(axis=1)
    labels = is_burn[model.predict(X)].astype(int)
    test = ds.test
    truth = ds.true_on.astype(bool)
    confusion = dict(tp=int((labels[test] & truth[test]).sum()), fp=int((labels[test] & ~truth[test]).sum()),
                     fn=int((~labels[test].astype(bool) & truth[test]).sum()),
                     tn=int((~labels[test].astype(bool) & ~truth[test]).sum()))
    # one TEST burn for the time series
    starts = [s for s, _ in runs_of_ones(ds.t[test], ds.true_on[test])]
    centre = np.searchsorted(ds.t, starts[len(starts) // 2])
    span = slice(max(centre - 400, 0), centre + 600)
    components = [dict(weight=float(w), mean=m.tolist(), cov=np.asarray(c).tolist(), burn=bool(b))
                  for w, m, c, b in zip(model.weights_, model.means_, model.covariances_, is_burn)]
    return dict(X=X[pick], truth=truth[pick], components=components, threshold=threshold, ic=ic,
                confusion=confusion, t_min=(ds.t[span] - ds.t[centre]) * TU / 60, prob=burn_prob[span],
                feature=X[span], on=ds.true_on[span], thrust=ds.thrust_ms2[span],
                clean_feature=_features(ds, ds.r[:, span], ds.v[:, span])[0])

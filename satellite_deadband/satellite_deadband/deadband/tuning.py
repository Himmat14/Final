"""
Bias-variance tuning: choose every adjustable setting of the pipeline from data, and carry the
choice forward (deadband/settings.py) into every later step.

The bias-variance decomposition
-------------------------------
Repeat an estimate many times with independent noise (Monte Carlo seeds, or bootstrap re-fits).
For an estimate y_hat of a true value y:
    bias      = mean(y_hat) - y                   how far the AVERAGE estimate is off (systematic)
    variance  = mean((y_hat - mean(y_hat))^2)      how much it scatters from one data set to the next
    MSE       = mean((y_hat - y)^2) = bias^2 + variance
A setting that makes the method more flexible / less smoothed usually lowers the bias and raises the
variance; the best setting minimises the MSE. For time series the decomposition is done sample by
sample and averaged (bias^2 = mean over samples of the squared mean error).

The six studies (all on the report's cached 400-day data)
---------------------------------------------------------
1. fd_order          FD stencil order (2, 4, 6, 8) for cd from noisy positions (steps 2-4).
                     Low order: truncation BIAS. High order: noise amplification, VARIANCE.
2. sg_window, sg_order   Savitzky-Golay window x order for the classifier's v' (steps 6-9).
                     Short window: noise VARIANCE. Long window: smears the burn edges, BIAS.
                     Scored on coast and burn samples with equal weight (burns are only 0.4% of samples).
3. derivative_method WALK-FORWARD and WALK-BACKWARD testing of the burn GMM for each way of
                     differentiating the velocity (SG with the study-2 settings, central 2nd-8th order):
                     forward  = train on days 0-40,    test on the next 15, 30, ... 360 days
                     backward = train on days 360-400, test on the 15, 30, ... 360 days BEFORE day 360
                     Run at 0.25, 0.5 and 1 x the reference noise. The method with the best mean
                     event F1 over the 12 out-of-sample periods AT THE REFERENCE NOISE wins
                     (ties: fewer false events, then smaller burn-start error).
4. gmm_components    K = 1 ... 8 Gaussians. Bootstrap re-fits on TRAIN; the predicted burn probability
                     P(burn | x) on TEST is split into bias^2 and variance (the Brier score, coast and
                     burn weighted equally). Small K: BIAS (cannot represent the clusters). Large K:
                     VARIANCE (components move from one re-fit to the next). Chosen: the smallest K
                     within one standard error of the lowest Brier score (the "one-standard-error rule").
5. sma_window        Savitzky-Golay window of the mean-SMA derivative used by SINDy (step 9).
6. stlsq_threshold   SINDy sparsity threshold: too high drops real terms (BIAS), too low keeps
                     noise-fitted distractors (VARIANCE). Scored on the predicted da/dt on TEST.
"""
import numpy as np
from scipy.signal import savgol_filter
from sklearn.mixture import GaussianMixture

from . import settings
from .classifiers import (EVENT_TOLERANCE_S, REFERENCE_NOISE, add_noise, build_features, detection_dataset,
                          mixture_burn_components)
from .constants import ACCEL_UNIT_MS2, DU, J2, SAMPLE_STEP_S, SECONDS, SMOOTH_CD
from .derivative_detection import DERIVATIVE_METHODS, event_scores, group_into_events
from .long_run import HORIZONS_DAYS, LONG_DAYS, TRAIN_DAYS, natural_run
from .regression import add_position_noise, fit_cd
from .sindy import learn_cluster_laws, library, sma_data, stlsq
from .smooth_controller import mean_sma_km_series

# --- study grids -------------------------------------------------------------------------------------
FD_ORDERS = (2, 4, 6, 8)
FD_STEPS_MIN = (1, 2, 5)
FD_DAYS, FD_SEEDS = 15, 20
FD_NOISE_KM = 0.01                                # the reference noise of steps 3-4: the setting is chosen here
FD_NOISES_KM = (0.0, 0.001, 0.01)                 # also shown: how the optimum moves with the noise

SG_WINDOWS = (9, 15, 21, 31, 41, 61, 81, 121)
SG_ORDERS = (3, 5, 7)
SG_DAYS, SG_SEEDS = 20, 4

BACKTEST_TRAIN_DAYS = TRAIN_DAYS                  # 40-day training windows at both ends of the run
BACKTEST_NOISE_SCALES = (0.25, 0.5, 1.0)          # x the reference noise; the choice is made at 1.0
FIT_SAMPLES = 20_000

GMM_KS = (1, 2, 3, 4, 5, 6, 7, 8)
GMM_BOOTSTRAPS = 6
GMM_TEST_DAYS = 60                                # TEST = days 40-100 (the step 7 window)
GMM_TEST_SAMPLES = 60_000

SMA_WINDOWS = (11, 21, 41, 81, 161, 321)
SINDY_SEEDS = 5
STLSQ_THRESHOLDS = (0.0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.6, 0.9)
TO_M_PER_HOUR = DU * 1e3 * 3600 * SECONDS            # non-dimensional da/dt -> metres per hour


def _decompose(estimates, truth):
    """bias^2, variance, MSE of (seeds, ...) estimates against a truth of shape (...): averaged over samples."""
    errors = np.asarray(estimates) - truth
    bias_sq = float(np.mean(np.mean(errors, axis=0) ** 2))
    variance = float(np.mean(np.var(errors, axis=0)))
    return bias_sq, variance, bias_sq + variance


# ---------------------------------------------------------------------------
# 1. FD stencil order
# ---------------------------------------------------------------------------
def study_fd_order():
    """cd error [%] bias^2 / variance / MSE for every (noise, sampling step, order)."""
    t, states = natural_run()
    rows = []
    for noise in FD_NOISES_KM:
        seeds = range(FD_SEEDS) if noise > 0 else range(1)            # clean data: no variance, one run is enough
        for step in FD_STEPS_MIN:
            keep = slice(0, int(FD_DAYS * 1440), step)
            positions, h = states[0:3, keep], (t[1] - t[0]) * step
            for order in FD_ORDERS:
                estimates = [100 * (fit_cd(add_position_noise(positions, noise, seed), h, order=order, j2=J2)
                                    / SMOOTH_CD - 1) for seed in seeds]
                bias_sq, variance, mse = _decompose(estimates, 0.0)
                rows.append(dict(noise_km=noise, step_min=step, order=order, bias_sq=bias_sq, variance=variance,
                                 mse=mse))
    reference = [r for r in rows if r["step_min"] == FD_STEPS_MIN[0] and r["noise_km"] == FD_NOISE_KM]
    return rows, int(min(reference, key=lambda r: r["mse"])["order"])


# ---------------------------------------------------------------------------
# 2. Savitzky-Golay window and order
# ---------------------------------------------------------------------------
def _true_unmodelled_ms2(dataset):
    """The real unmodelled acceleration [m/s^2]: thrust along the velocity plus drag."""
    v = dataset.v
    speed = np.linalg.norm(v, axis=0)
    drag = -SMOOTH_CD * speed * v * ACCEL_UNIT_MS2
    return dataset.thrust_ms2 * v / speed + drag


def study_sg():
    dataset = detection_dataset().until(SG_DAYS)
    truth = _true_unmodelled_ms2(dataset) * 1e4               # in units of 1e-4 m/s^2 (thrust = 2)
    on = dataset.true_on == 1
    noisy = [add_noise(dataset, *REFERENCE_NOISE, seed=seed) for seed in range(SG_SEEDS)]
    rows = []
    for order in SG_ORDERS:
        for window in SG_WINDOWS:
            if window <= order + 1:
                continue
            estimates = np.array([build_features(r, v, dataset.step_s, window=window, order=order,
                                                 method="Savitzky-Golay").accel_from_v * 1e4 for r, v in noisy])
            parts = {}
            for name, mask in (("coast", ~on), ("burn", on)):
                parts[name] = _decompose(estimates[:, :, mask], truth[:, mask])
            # equal weight to coast and burn samples (otherwise the 0.4% burns would not count)
            bias_sq = 0.5 * (parts["coast"][0] + parts["burn"][0])
            variance = 0.5 * (parts["coast"][1] + parts["burn"][1])
            rows.append(dict(window=window, order=order, window_s=window * dataset.step_s, bias_sq=bias_sq,
                             variance=variance, mse=bias_sq + variance,
                             coast_mse=parts["coast"][2], burn_mse=parts["burn"][2]))
    best = min(rows, key=lambda r: r["mse"])
    return rows, int(best["window"]), int(best["order"])


# ---------------------------------------------------------------------------
# 3. Walk-forward / walk-backward choice of the derivative method
# ---------------------------------------------------------------------------
def _fit_gmm(X, rows, n_components, seed=0):
    pick = np.random.default_rng(seed).choice(rows, size=min(FIT_SAMPLES, rows.size), replace=False)
    model = GaussianMixture(n_components=n_components, n_init=3, random_state=seed).fit(X[pick])
    return model, mixture_burn_components(model)


def _window_scores(dataset, events, start_day, end_day):
    keep = (dataset.days >= start_day) & (dataset.days < end_day)
    scores = event_scores(dataset.t[keep], events[keep], dataset.true_on[keep], dataset.step_s, EVENT_TOLERANCE_S)
    return {k: scores[k] for k in ("f1", "precision", "recall", "n_false_events", "n_found", "n_burns",
                                   "mean_abs_start_error_s")}


def study_walk_forward(sg_window, sg_order, n_components):
    dataset = detection_dataset()
    days = dataset.days
    test_end = LONG_DAYS - BACKTEST_TRAIN_DAYS                           # day 360
    train_forward = np.flatnonzero(days < BACKTEST_TRAIN_DAYS)
    train_backward = np.flatnonzero(days >= test_end)
    methods = ("Savitzky-Golay",) + tuple(m for m in DERIVATIVE_METHODS if m != "Savitzky-Golay")
    rows = []
    for scale in BACKTEST_NOISE_SCALES:
        r, v = add_noise(dataset, REFERENCE_NOISE[0] * scale, REFERENCE_NOISE[1] * scale, seed=1)
        for method in methods:
            X = build_features(r, v, dataset.step_s, window=sg_window, order=sg_order, method=method).matrix
            for direction, train_rows in (("forward", train_forward), ("backward", train_backward)):
                model, is_burn = _fit_gmm(X, train_rows, n_components)
                events = group_into_events(is_burn[model.predict(X)].astype(int))
                for horizon in HORIZONS_DAYS:
                    if direction == "forward":
                        window = (BACKTEST_TRAIN_DAYS, BACKTEST_TRAIN_DAYS + horizon)
                    else:
                        window = (test_end - horizon, test_end)
                    rows.append(dict(noise_scale=scale, method=method, direction=direction, horizon_days=horizon,
                                     **_window_scores(dataset, events, *window)))
            del X
        del r, v
    summary = {}
    for method in methods:
        mine = [row for row in rows if row["method"] == method and row["noise_scale"] == 1.0]
        summary[method] = dict(mean_f1=float(np.mean([r["f1"] for r in mine])),
                               std_f1=float(np.std([r["f1"] for r in mine])),
                               false_events=int(np.sum([r["n_false_events"] for r in mine])),
                               start_error_s=float(np.nanmean([r["mean_abs_start_error_s"] for r in mine])))
    best = max(summary, key=lambda m: (round(summary[m]["mean_f1"], 3), -summary[m]["false_events"],
                                       -np.nan_to_num(summary[m]["start_error_s"], nan=1e9)))
    return rows, summary, best


# ---------------------------------------------------------------------------
# 4. Number of GMM components (bias-variance of the burn probability)
# ---------------------------------------------------------------------------
def study_gmm_components(sg_window, sg_order, method):
    dataset = detection_dataset().until(TRAIN_DAYS + GMM_TEST_DAYS)
    r, v = add_noise(dataset, *REFERENCE_NOISE, seed=1)
    X = build_features(r, v, dataset.step_s, window=sg_window, order=sg_order, method=method).matrix
    train_rows = np.flatnonzero(dataset.train)
    test = dataset.test
    rng = np.random.default_rng(0)
    test_rows = np.flatnonzero(test)
    # balanced probe set: every TEST burn sample plus as many random TEST coast samples
    burn_rows = test_rows[dataset.true_on[test_rows] == 1]
    coast_rows = rng.choice(test_rows[dataset.true_on[test_rows] == 0], size=burn_rows.size, replace=False)
    probe = np.concatenate([coast_rows, burn_rows])
    label = dataset.true_on[probe].astype(float)
    rows = []
    for k in GMM_KS:
        probabilities, f1, test_nll, train_nll = [], [], [], []
        for b in range(GMM_BOOTSTRAPS):
            model, is_burn = _fit_gmm(X, train_rows, k, seed=b)
            probabilities.append(model.predict_proba(X[probe])[:, is_burn].sum(axis=1))
            flags = np.zeros(len(X), dtype=int)
            flags[test] = is_burn[model.predict(X[test])]
            events = group_into_events(flags)
            f1.append(event_scores(dataset.t[test], events[test], dataset.true_on[test], dataset.step_s,
                                   EVENT_TOLERANCE_S)["f1"])
            train_nll.append(-model.score(X[rng.choice(train_rows, 20_000, replace=False)]))
            test_nll.append(-model.score(X[rng.choice(test_rows, GMM_TEST_SAMPLES, replace=False)]))
        bias_sq, variance, brier = _decompose(np.array(probabilities), label)
        per_fit = np.mean((np.array(probabilities) - label) ** 2, axis=1)
        rows.append(dict(k=k, bias_sq=bias_sq, variance=variance, brier=brier,
                         brier_se=float(np.std(per_fit) / np.sqrt(GMM_BOOTSTRAPS)),
                         f1_mean=float(np.mean(f1)), f1_std=float(np.std(f1)),
                         train_nll=float(np.mean(train_nll)), test_nll=float(np.mean(test_nll)),
                         test_nll_std=float(np.std(test_nll))))
    best = min(rows, key=lambda r: r["brier"])
    chosen = min(r["k"] for r in rows if r["brier"] <= best["brier"] + best["brier_se"])   # one-SE rule
    return rows, int(chosen)


# ---------------------------------------------------------------------------
# 5-6. SINDy: SMA smoothing window and STLSQ threshold
# ---------------------------------------------------------------------------
def _sindy_data(window, seeds):
    dataset = detection_dataset().until(TRAIN_DAYS + GMM_TEST_DAYS)
    h = dataset.step_s * SECONDS
    clean_a = mean_sma_km_series(np.vstack([dataset.r, dataset.v])) / DU
    truth = savgol_filter(clean_a, 11, 2, deriv=1, delta=h)              # clean data: a short window is exact
    noisy = []
    for seed in range(seeds):
        r, v = add_noise(dataset, *REFERENCE_NOISE, seed=seed)
        noisy.append(sma_data(dataset.t, r, v, dataset.true_on, dataset.step_s, window=window))
    return dataset, truth, noisy


def study_sma_window():
    rows = []
    for window in SMA_WINDOWS:
        dataset, truth, noisy = _sindy_data(window, SINDY_SEEDS)
        on = dataset.true_on == 1
        estimates = np.array([d.a_dot for d in noisy]) * TO_M_PER_HOUR
        truth_m = truth * TO_M_PER_HOUR
        coast = _decompose(estimates[:, ~on], truth_m[~on])
        burn = _decompose(estimates[:, on], truth_m[on])
        rows.append(dict(window=window, window_s=window * SAMPLE_STEP_S, bias_sq=0.5 * (coast[0] + burn[0]),
                         variance=0.5 * (coast[1] + burn[1]), mse=0.5 * (coast[2] + burn[2])))
    return rows, int(min(rows, key=lambda r: r["mse"])["window"])


def study_stlsq(sma_window):
    dataset, truth, noisy = _sindy_data(sma_window, SINDY_SEEDS)
    test = dataset.test & (noisy[0].coast | noisy[0].burn)
    rows = []
    for threshold in STLSQ_THRESHOLDS:
        predictions, kept = [], []
        for data in noisy:
            rows_fit = (data.coast | data.burn) & dataset.train
            a_ref = float(np.median(data.a[rows_fit]))
            theta = library(data.a[rows_fit], data.burn[rows_fit].astype(float), a_ref)
            xi = stlsq(theta, data.a_dot[rows_fit], threshold=threshold)
            kept.append(int(np.sum(xi != 0)))
            predictions.append(library(data.a[test], data.burn[test].astype(float), a_ref) @ xi * TO_M_PER_HOUR)
        on = dataset.true_on[test] == 1
        predictions = np.array(predictions)
        truth_m = truth[test] * TO_M_PER_HOUR
        coast = _decompose(predictions[:, ~on], truth_m[~on])
        burn = _decompose(predictions[:, on], truth_m[on])
        rows.append(dict(threshold=threshold, bias_sq=0.5 * (coast[0] + burn[0]), variance=0.5 * (coast[1] + burn[1]),
                         mse=0.5 * (coast[2] + burn[2]), terms_kept=float(np.mean(kept))))
    best = min(rows, key=lambda r: r["mse"])
    # among (near-)equal MSEs prefer the sparser law: the largest threshold within 1% of the best MSE
    chosen = max(r["threshold"] for r in rows if r["mse"] <= 1.01 * best["mse"])
    return rows, float(chosen)


# ---------------------------------------------------------------------------
# Run everything, in order (each study uses the settings chosen before it)
# ---------------------------------------------------------------------------
def run_all(log=print):
    studies, chosen = {}, {}
    log("    tuning 1/6: FD stencil order ...")
    studies["fd_order"], chosen["fd_order"] = study_fd_order()
    log("    tuning 2/6: Savitzky-Golay window and order ...")
    studies["sg"], chosen["sg_window"], chosen["sg_order"] = study_sg()
    log("    tuning 3/6: walk-forward / walk-backward test of the derivative methods ...")
    rows, summary, chosen["derivative_method"] = study_walk_forward(chosen["sg_window"], chosen["sg_order"],
                                                                   settings.DEFAULTS["gmm_components"])
    studies["walk_forward"], studies["walk_forward_summary"] = rows, summary
    log("    tuning 4/6: number of GMM components ...")
    studies["gmm_components"], chosen["gmm_components"] = study_gmm_components(
        chosen["sg_window"], chosen["sg_order"], chosen["derivative_method"])
    log("    tuning 5/6: SMA smoothing window ...")
    studies["sma_window"], chosen["sma_window"] = study_sma_window()
    log("    tuning 6/6: SINDy sparsity threshold ...")
    studies["stlsq"], chosen["stlsq_threshold"] = study_stlsq(chosen["sma_window"])
    settings.save(chosen, studies)
    return chosen, studies

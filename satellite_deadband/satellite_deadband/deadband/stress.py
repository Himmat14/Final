"""
Stress test of the step 6 burn classifier: how coarse can the sampling get, and how much noise can
it take, before the GMM stops finding the burns?

Data and model are exactly those of step 6 (days 0-100 of the 400-day controlled run, TRAIN days 0-40,
TEST days 40-100, the tuned derivative and GMM), except that only every `stride`-th sample is kept,
so the sampling step is stride x 6 s (6 s ... 20 min), and the measurement noise is a multiple of the
step 6 reference noise (0.1 m, 0.5 mm/s).

Two things are scaled with the step so that the test measures the CLASSIFIER, not bookkeeping:
    * the Savitzky-Golay derivative keeps its tuned length in TIME (tuned window x 6 s), but never
      fewer than MIN_WINDOW samples (the polynomial order is lowered to fit a short window);
    * an event must last MIN_EVENT_S seconds (not 10 samples): a 23-minute burn sampled every 3 min
      has only ~8 samples, and a 10-sample rule would delete it whatever the classifier did.
The timing tolerance of the event score is max(60 s, one step).

"Still works" = event F1 >= WORKS_F1 on TEST. For each feature and noise level the WORKING RANGE of
steps is reported (finest and coarsest step with F1 >= WORKS_F1), and the "switched" classifier uses
whichever feature is better at that step (derivative for short steps, propagation for long ones).

Two ways of getting the unmodelled acceleration are stressed:
    "derivative (step 6)"     Savitzky-Golay derivative of the measured velocity minus gravity + J2.
                              Needs short steps: a polynomial through 5 samples spanning a big piece of
                              the curved orbit leaves a truncation error larger than the thrust.
    "one-step propagation"    propagate each measured state ONE step ahead with the known gravity + J2
                              (vectorised RK4), and divide the velocity mismatch at the next measurement
                              by the step: the mean unmodelled acceleration over that interval. No
                              truncation error from the orbit's curvature, and the noise (~ sqrt(2)
                              x velocity noise / step) SHRINKS as the step grows, until a step is as long
                              as a burn and the burn is diluted into one interval.
"""
import numpy as np
from sklearn.mixture import GaussianMixture

from .classifiers import (FLOOR_MS2, REFERENCE_NOISE, add_noise, build_features, detection_dataset,
                          mixture_burn_components)
from .constants import ACCEL_UNIT_MS2, SECONDS
from .derivative_detection import known_accel
from .derivative_detection import event_scores, group_into_events
from .long_run import TRAIN_DAYS
from .settings import tuned

STRIDES = (1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 450, 600)   # x 6 s: 6 s ... 60 min
NOISE_SCALES = (0.0, 0.5, 1.0, 2.0, 4.0)
WINDOW_DAYS = TRAIN_DAYS + 60
MIN_WINDOW = 5
MIN_EVENT_S = 60.0
WORKS_F1 = 0.9
FIT_SAMPLES = 20_000
PROPAGATION_SUBSTEP_S = 10.0                                 # RK4 sub-step of the one-step propagation
FEATURES = ("derivative (step 6)", "one-step propagation")
BURN_MINUTES = 22.8                                          # every burn lasts ~22.8 min (step 5)


def derivative_settings(step_s):
    """(window samples, polynomial order) that keep the tuned derivative length in time."""
    window_s = tuned("sg_window") * 6.0
    window = max(MIN_WINDOW, int(round(window_s / step_s)) | 1)
    order = min(tuned("sg_order"), window - 2)
    return window, order


def propagate_rk4(r, v, h, substep):
    """Every column of (r, v) propagated by h under gravity + J2 only (vectorised RK4, non-dimensional)."""
    n = max(1, int(np.ceil(h / substep)))
    dt = h / n
    for _ in range(n):
        k1r, k1v = v, known_accel(r)
        k2r, k2v = v + 0.5 * dt * k1v, known_accel(r + 0.5 * dt * k1r)
        k3r, k3v = v + 0.5 * dt * k2v, known_accel(r + 0.5 * dt * k2r)
        k4r, k4v = v + dt * k3v, known_accel(r + dt * k3r)
        r = r + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r, v


def propagation_features(r, v, step_s):
    """
    (N, 2) features from the one-step propagation residual: [log |a|, along-track a / 1e-4 m/s^2], where
    a = (v_measured[k+1] - v_propagated[k -> k+1]) / h is the mean unmodelled acceleration over the
    interval ending at sample k+1 (sample 0 copies sample 1).
    """
    h = step_s * SECONDS
    _, v_predicted = propagate_rk4(r[:, :-1], v[:, :-1], h, PROPAGATION_SUBSTEP_S * SECONDS)
    accel = (v[:, 1:] - v_predicted) / h * ACCEL_UNIT_MS2
    accel = np.concatenate([accel[:, :1], accel], axis=1)
    along = np.sum(accel * v, axis=0) / np.linalg.norm(v, axis=0)
    return np.column_stack([np.log(np.linalg.norm(accel, axis=0) + FLOOR_MS2), along / 1e-4])


def classify(dataset, noise_scale, feature=FEATURES[0], seed=1):
    """Fit the tuned GMM on TRAIN of this (possibly coarse) dataset and return TEST event scores."""
    r, v = add_noise(dataset, REFERENCE_NOISE[0] * noise_scale, REFERENCE_NOISE[1] * noise_scale, seed)
    window, order = derivative_settings(dataset.step_s)
    if feature == FEATURES[0]:
        X = build_features(r, v, dataset.step_s, window=window, order=order, method="Savitzky-Golay").matrix
    else:
        X, window, order = propagation_features(r, v, dataset.step_s), 2, 0
    rows = np.flatnonzero(dataset.train)
    pick = np.random.default_rng(0).choice(rows, size=min(FIT_SAMPLES, rows.size), replace=False)
    model = GaussianMixture(n_components=tuned("gmm_components"), n_init=3, random_state=0).fit(X[pick])
    is_burn = mixture_burn_components(model)
    test = dataset.test
    flags = np.zeros(len(X), dtype=int)
    flags[test] = is_burn[model.predict(X[test])]
    min_samples = max(1, int(np.ceil(MIN_EVENT_S / dataset.step_s)))
    events = group_into_events(flags, min_samples=min_samples)
    scores = event_scores(dataset.t[test], events[test], dataset.true_on[test], dataset.step_s,
                          max(60.0, dataset.step_s))
    return dict(feature=feature, step_s=dataset.step_s, noise_scale=noise_scale, window=window, order=order,
                window_s=window * dataset.step_s, samples_per_burn=BURN_MINUTES * 60 / dataset.step_s,
                **{k: scores[k] for k in ("f1", "precision", "recall", "n_found", "n_burns", "n_false_events",
                                          "mean_abs_start_error_s", "mean_coverage")})


def sampling_stress(strides=STRIDES, noise_scales=NOISE_SCALES):
    """Rows for every (feature, step, noise), and {feature: {noise: largest working step [s]}}."""
    base = detection_dataset().until(WINDOW_DAYS)
    rows = []
    for stride in strides:
        dataset = base if stride == 1 else base.every(stride)
        for feature in FEATURES:
            for scale in noise_scales:
                rows.append(classify(dataset, scale, feature))
    for row in rows:
        row["works"] = bool(row["f1"] >= WORKS_F1)
    ranges = {}
    for feature in FEATURES + ("switched",):
        ranges[feature] = {}
        for scale in noise_scales:
            if feature == "switched":                   # the better of the two features at every step
                steps = sorted({r["step_s"] for r in rows})
                works = {s: any(r["works"] for r in rows if r["step_s"] == s and r["noise_scale"] == scale)
                         for s in steps}
            else:
                works = {r["step_s"]: r["works"] for r in rows
                         if r["feature"] == feature and r["noise_scale"] == scale}
            good = [s for s, ok in sorted(works.items()) if ok]
            ranges[feature][scale] = dict(min_step_s=good[0] if good else None, max_step_s=good[-1] if good else None,
                                          all_steps_work=bool(good) and len(good) == len(works))
    return rows, ranges

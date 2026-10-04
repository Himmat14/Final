"""
Five basic burn detectors (Workstream C/G), on the SAME data and with the SAME scoring as report
steps 6-7.

Data: the 400-day controlled run (classifiers.detection_dataset), days 0-100 (the step 7 window):
TRAIN = days 0-40, TEST = days 40-100. The evidence signal is the 1-D feature
    x = log |unmodelled acceleration from v'|      (column 0 of classifiers.Features.matrix)
from Savitzky-Golay derivatives of measured r and v, at the step 6 reference noise
(0.1 m, 0.5 mm/s) unless stated otherwise. Every detector's flags are grouped into burn EVENTS and
scored per event on TEST (found burns, false events, precision / recall / F1), as in step 6.

Two kinds of detector:
    fitted on TRAIN, then applied to TEST          fixed threshold (MAD), k-means, GMM
    online (look only at the past, no fitting)     rolling z-score, CUSUM

The Week 4 version used per-sample jumps of the osculating SMA from an impulsive-burn run, noise
as a fraction of the signal's standard deviation, and per-sample scores.
"""
from functools import lru_cache

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .classifiers import EVENT_TOLERANCE_S, detection_dataset, mixture_burn_components, noisy_features
from .derivative_detection import event_scores, group_into_events
from .long_run import TRAIN_DAYS
from .settings import tuned

WINDOW_DAYS = TRAIN_DAYS + 60       # days 0-100, the same window as the step 7 sweeps
FIT_SAMPLES = 20_000                # TRAIN samples the fitted detectors see (random subsample)
ROLLING_WINDOW = 25                 # samples of history for the rolling z-score (2.5 min at 6 s)
ROLLING_THRESHOLD = 3.5
CUSUM_DRIFT, CUSUM_THRESHOLD = 0.5, 5.0
MAD_FACTOR = 5.0

ROLLING_ZSCORE = "Rolling z-score"
FIXED_THRESHOLD = "Fixed threshold (MAD)"
KMEANS = "K-means (2 clusters)"
GMM = "Gaussian mixture (tuned K)"
CUSUM = "CUSUM"


# ---------------------------------------------------------------------------
# Data (shared with gmm_stage and heldout_stage)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def window_dataset():
    """Days 0-100 of the controlled run (TRAIN 0-40, TEST 40-100)."""
    return detection_dataset().until(WINDOW_DAYS)


@lru_cache(maxsize=2)                # each entry holds ~1.4 M samples of features: keep only two
def window_features(noise_scale=1.0, seed=0, stride=1):
    """(dataset, Features) at `noise_scale` x the reference noise, keeping every `stride`-th sample."""
    dataset = window_dataset() if stride == 1 else window_dataset().every(stride)
    features, _ = noisy_features(dataset, noise_scale, seed)
    return dataset, features


def train_subsample(x, dataset, n=FIT_SAMPLES, seed=0):
    rows = np.flatnonzero(dataset.train)
    return x[np.random.default_rng(seed).choice(rows, size=min(n, rows.size), replace=False)]


# ---------------------------------------------------------------------------
# The detectors: each takes (x_train, x_test) and returns 0/1 flags for x_test
# ---------------------------------------------------------------------------
def rolling_zscore_detector(x_train, x):
    """Flag x[i] more than ROLLING_THRESHOLD standard deviations above the previous ROLLING_WINDOW samples."""
    w = ROLLING_WINDOW
    cumulative = np.concatenate([[0.0], np.cumsum(x)])
    cumulative_sq = np.concatenate([[0.0], np.cumsum(x**2)])
    index = np.arange(len(x))
    start = np.maximum(index - w, 0)
    count = np.maximum(index - start, 1)
    mean = (cumulative[index] - cumulative[start]) / count
    variance = np.maximum((cumulative_sq[index] - cumulative_sq[start]) / count - mean**2, 1e-12)
    return ((x - mean) / np.sqrt(variance) > ROLLING_THRESHOLD).astype(int)


def fixed_threshold_detector(x_train, x):
    """Flag x above the TRAIN median by more than MAD_FACTOR median absolute deviations."""
    median = np.median(x_train)
    return (x > median + MAD_FACTOR * np.median(np.abs(x_train - median))).astype(int)


def kmeans_detector(x_train, x):
    """Two k-means clusters on TRAIN; the cluster with the larger centre is burn."""
    model = KMeans(n_clusters=2, n_init=5, random_state=0).fit(x_train[:, None])
    burn = int(np.argmax(model.cluster_centers_.ravel()))
    return (model.predict(x[:, None]) == burn).astype(int)


def gmm_detector(x_train, x, n_components=None):
    """1-D Gaussian mixture on TRAIN (tuned K); burn components = means more than 3 coast sd above the coast mean."""
    n_components = n_components or tuned("gmm_components")
    model = GaussianMixture(n_components=n_components, n_init=3, random_state=0).fit(x_train[:, None])
    burn = mixture_burn_components(model)
    return burn[model.predict(x[:, None])].astype(int)


def cusum_detector(x_train, x):
    """One-sided CUSUM around the TRAIN median and spread; the accumulator resets each time it fires."""
    centre, spread = np.median(x_train), np.std(x_train) + 1e-12
    accumulator = 0.0
    flags = np.zeros(len(x), dtype=int)
    for i, value in enumerate((x - centre) / spread - CUSUM_DRIFT):
        accumulator = max(0.0, accumulator + value)
        if accumulator > CUSUM_THRESHOLD:
            flags[i] = 1
            accumulator = 0.0
    return flags


DETECTORS = {
    ROLLING_ZSCORE: rolling_zscore_detector,
    FIXED_THRESHOLD: fixed_threshold_detector,
    KMEANS: kmeans_detector,
    GMM: gmm_detector,
    CUSUM: cusum_detector,
}


# ---------------------------------------------------------------------------
# Running and scoring
# ---------------------------------------------------------------------------
def run_detectors(dataset, features, detectors=DETECTORS):
    """{name: (events on every sample (0 on TRAIN), event scores on TEST)}."""
    x = features.matrix[:, 0]
    x_train = train_subsample(x, dataset)
    test = dataset.test
    out = {}
    for name, detector in detectors.items():
        flags = np.zeros(len(x), dtype=int)
        flags[test] = detector(x_train, x[test])
        events = group_into_events(flags)
        out[name] = (events, event_scores(dataset.t[test], events[test], dataset.true_on[test], dataset.step_s,
                                          EVENT_TOLERANCE_S))
    return out


def noise_sweep(noise_scales):
    """[{noise_scale, method, f1, precision, recall, false events}] for every detector and noise level."""
    rows = []
    for scale in noise_scales:
        dataset, features = window_features(scale, seed=1)
        for name, (_, scores) in run_detectors(dataset, features).items():
            rows.append(dict(noise_scale=scale, method=name, **{k: scores[k] for k in
                             ("f1", "precision", "recall", "n_false_events", "n_found", "n_burns")}))
    return rows


def cadence_sweep(strides):
    """The same at sampling steps of stride x 6 s (reference noise)."""
    rows = []
    for stride in strides:
        dataset, features = window_features(1.0, seed=1, stride=stride)
        for name, (_, scores) in run_detectors(dataset, features).items():
            rows.append(dict(step_s=dataset.step_s, method=name, **{k: scores[k] for k in
                             ("f1", "precision", "recall", "n_false_events")}))
    return rows

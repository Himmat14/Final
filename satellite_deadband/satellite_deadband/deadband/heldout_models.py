"""
Held-out evaluation of more burn detectors (Workstream C/G, further extended), on the SAME data,
split and scoring as report step 6:

    data      days 0-100 of the 400-day controlled run, step 6 reference noise (0.1 m, 0.5 mm/s)
    split     chronological: TRAIN = days 0-40, TEST = days 40-100 (detectors.window_features)
    features  [log |unmodelled accel from v'|, along-track part / 1e-4 m/s^2] (classifiers.Features)
    scoring   flags -> burn EVENTS -> found burns, false events, precision / recall / F1 on TEST

Models
------
    GMM (tuned K)       fitted on TRAIN; burn components by the step 6 rule (gmm_2d.fit_feature_gmm)
    K-means (2D)        fitted on TRAIN; the cluster with the larger log|v'| centre is burn
    Bayesian GMM        up to 6 components, its prior switches off the unused ones
    Agglomerative       Ward clustering of a TRAIN subsample; new points go to the nearest centroid
    Isolation Forest    anomaly detector (1% contamination), flags only points above the coast median
    DBSCAN              has no predict(): run IN-SAMPLE on TEST (every 10th sample, 60 s), flagged as such
    GP anomaly          a Gaussian process learns how the mean SMA decays while coasting, as a function
                        of the time since the SMA last touched the UPPER band edge (observable, no labels:
                        every coast starts there). A sample is a burn when the measured SMA departs from
                        the GP's coast curve UPWARDS (thrust only raises it) by more than Z_THRESHOLD
                        times the scatter of the TRAIN coast samples around the curve.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.mixture import BayesianGaussianMixture

from .classifiers import EVENT_TOLERANCE_S, add_noise, mixture_burn_components
from .constants import SMOOTH_A_UPPER_KM, TU
from .derivative_detection import event_scores, group_into_events
from .detectors import train_subsample, window_features
from .gmm_2d import fit_feature_gmm
from .smooth_controller import mean_sma_km_series

AGGLOMERATIVE_SAMPLES = 4000          # Ward clustering needs memory ~ n^2
DBSCAN_STRIDE = 10                    # DBSCAN on every 10th TEST sample (60 s)
GP_POINTS = 600
GP_CHUNK = 100_000                    # predict in chunks: one call on 1.4 M samples would need ~7 GB
Z_THRESHOLD = 5.0
EDGE_TOLERANCE_KM = 0.005             # "touched the upper edge" = within 5 m of it
REARM_KM = 0.05                       # the clock can only reset again once the SMA has fallen 50 m below the edge

GMM_NAME = "GMM (tuned K)"
KMEANS_NAME = "K-means (2D)"
BAYESIAN_GMM_NAME = "Bayesian GMM"
AGGLOMERATIVE_NAME = "Agglomerative (Ward)"
ISOLATION_FOREST_NAME = "Isolation Forest"
DBSCAN_NAME = "DBSCAN (in-sample on TEST, 60 s)"
GP_NAME = "GP coast-curve anomaly"


# ---------------------------------------------------------------------------
# Inductive models: fit(X_train) -> predict(X) -> 0/1
# ---------------------------------------------------------------------------
def fit_gmm(X_train):
    gmm = fit_feature_gmm(X_train)
    return gmm.predict, dict(model=gmm.model, is_burn=gmm.is_burn)


def fit_kmeans(X_train):
    model = KMeans(n_clusters=2, n_init=8, random_state=0).fit(X_train)
    burn = int(np.argmax(model.cluster_centers_[:, 0]))
    return (lambda X: (model.predict(X) == burn).astype(int)), dict(model=model)


def fit_bayesian_gmm(X_train):
    model = BayesianGaussianMixture(n_components=6, covariance_type="full", random_state=0,
                                    weight_concentration_prior=0.01, max_iter=1000).fit(X_train)
    burn = mixture_burn_components(model)
    return (lambda X: burn[model.predict(X)].astype(int)), dict(model=model, n_active=int(np.sum(model.weights_ > 0.01)))


def fit_agglomerative(X_train):
    """Ward clustering has no predict(): new points go to the nearest cluster centroid."""
    X = X_train[:AGGLOMERATIVE_SAMPLES]
    labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(X).labels_
    centroids = np.array([X[labels == k].mean(axis=0) for k in (0, 1)])
    burn = int(np.argmax(centroids[:, 0]))

    def predict(Xn):
        distances = np.linalg.norm(Xn[:, None, :] - centroids[None, :, :], axis=2)
        return (np.argmin(distances, axis=1) == burn).astype(int)

    return predict, dict(centroids=centroids)


def fit_isolation_forest(X_train):
    model = IsolationForest(n_estimators=200, contamination=0.01, random_state=0).fit(X_train)
    median = np.median(X_train[:, 0])
    return (lambda X: ((model.predict(X) == -1) & (X[:, 0] > median)).astype(int)), dict(model=model)


INDUCTIVE = {GMM_NAME: fit_gmm, KMEANS_NAME: fit_kmeans, BAYESIAN_GMM_NAME: fit_bayesian_gmm,
             AGGLOMERATIVE_NAME: fit_agglomerative, ISOLATION_FOREST_NAME: fit_isolation_forest}


def dbscan_flags(X):
    """DBSCAN on X; the smallest cluster plus all noise points above the median log|v'| are burn."""
    labels = DBSCAN(eps=0.15, min_samples=10).fit(X).labels_
    clusters = [c for c in set(labels) if c != -1]
    if not clusters:
        return np.zeros(len(X), dtype=int)
    biggest = max(clusters, key=lambda c: np.sum(labels == c))
    return ((labels != biggest) & (X[:, 0] > np.median(X[:, 0]))).astype(int)


# ---------------------------------------------------------------------------
# GP coast-curve anomaly detector
# ---------------------------------------------------------------------------
@dataclass
class GpDetection:
    elapsed_hours: np.ndarray   # time since the measured SMA last touched the upper edge
    sma_km: np.ndarray          # measured mean SMA
    mean: np.ndarray            # GP coast curve at each sample
    z_score: np.ndarray
    flags: np.ndarray
    fit_rows: np.ndarray
    model: GaussianProcessRegressor


def time_since_upper_edge(t, sma_km):
    """
    Hours since each coast began: the FIRST time the measured SMA came within EDGE_TOLERANCE_KM of the
    upper band edge. With hysteresis: after a reset the clock re-arms only once the SMA has fallen
    REARM_KM below the edge, so noise flickering around the edge in the first hour of a coast (the
    SMA falls only ~5 m per hour) cannot keep resetting it. NaN before the first coast starts.
    """
    near_edge = sma_km >= SMOOTH_A_UPPER_KM - EDGE_TOLERANCE_KM
    well_below = sma_km < SMOOTH_A_UPPER_KM - REARM_KM
    elapsed = np.full(len(t), np.nan)
    start, armed = None, True
    for i in range(len(t)):
        if armed and near_edge[i]:
            start, armed = t[i], False          # a new coast starts here
        elif well_below[i]:
            armed = True
        if start is not None:
            elapsed[i] = (t[i] - start) * TU / 3600
    return elapsed


def fit_gp_detector(dataset, sma_km, coast_labels, seed=0):
    """coast_labels: 1 where the GMM says coast (NOT the truth: the GP must not see true labels)."""
    elapsed = time_since_upper_edge(dataset.t, sma_km)
    rows = np.flatnonzero(dataset.train & (coast_labels == 1) & (elapsed > 0))   # NaN compares False
    fit_rows = np.random.default_rng(seed).choice(rows, size=min(GP_POINTS, rows.size), replace=False)
    offset = sma_km[fit_rows].mean()
    kernel = ConstantKernel(1.0) * RBF(length_scale=20.0, length_scale_bounds=(1.0, 500.0)) + WhiteKernel(1e-6)
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)
    model.fit(elapsed[fit_rows, None], sma_km[fit_rows] - offset)
    known = np.flatnonzero(np.isfinite(elapsed))
    mean = np.full(len(elapsed), np.nan)
    for start in range(0, len(known), GP_CHUNK):
        rows_k = known[start:start + GP_CHUNK]
        mean[rows_k] = model.predict(elapsed[rows_k, None]) + offset
    residual = sma_km - mean
    coast_train = np.flatnonzero(dataset.train & (coast_labels == 1) & np.isfinite(mean))
    scatter = 1.4826 * np.median(np.abs(residual[coast_train] - np.median(residual[coast_train])))   # robust sd
    z = np.where(np.isfinite(mean), residual / scatter, 0.0)                                         # one-sided
    return GpDetection(elapsed_hours=elapsed, sma_km=sma_km, mean=mean, z_score=z,
                       flags=(z > Z_THRESHOLD).astype(int), fit_rows=fit_rows, model=model)


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------
@dataclass
class HeldOutExperiment:
    dataset: object
    X: np.ndarray
    predictors: dict             # name -> predict function (inductive models)
    info: dict                   # name -> fitted-model details
    events: dict                 # name -> event flags on every sample (TEST only)
    rows: list                   # one score dict per method, on TEST
    gp: GpDetection
    dbscan_rows: np.ndarray      # TEST samples DBSCAN ran on
    dbscan_flags: np.ndarray


def _score(dataset, events, rows, stride=1):
    """Event scores on the samples `rows` (every `stride`-th sample, so the sampling step is stride x 6 s)."""
    scores = event_scores(dataset.t[rows], events[rows], dataset.true_on[rows], dataset.step_s * stride,
                          EVENT_TOLERANCE_S)
    return {k: v for k, v in scores.items() if k != "per_burn"}


def run_heldout_experiment():
    dataset, features = window_features(1.0, seed=1)
    X = features.matrix
    X_train = train_subsample(X, dataset)
    test = np.flatnonzero(dataset.test)

    predictors, info, events, rows = {}, {}, {}, []
    for name, fit in INDUCTIVE.items():
        predict, details = fit(X_train)
        predictors[name], info[name] = predict, details
        flags = np.zeros(len(X), dtype=int)
        flags[test] = predict(X[test])
        events[name] = group_into_events(flags)
        rows.append(dict(method=name, **_score(dataset, events[name], test)))

    dbscan_rows = test[::DBSCAN_STRIDE]
    flags = dbscan_flags(X[dbscan_rows])
    coarse = group_into_events(flags)
    events[DBSCAN_NAME] = np.zeros(len(X), dtype=int)
    events[DBSCAN_NAME][dbscan_rows] = coarse
    rows.append(dict(method=DBSCAN_NAME, **_score(dataset, events[DBSCAN_NAME], dbscan_rows, DBSCAN_STRIDE)))

    r, v = add_noise(dataset, 0.1, 0.5, seed=1)                    # same measured r, v as the features
    sma = mean_sma_km_series(np.vstack([r, v]))
    gmm_train = np.zeros(len(X), dtype=int)                        # GMM labels on TRAIN pick the GP's coast samples
    gmm_train[dataset.train] = predictors[GMM_NAME](X[dataset.train])
    gp = fit_gp_detector(dataset, sma, coast_labels=1 - group_into_events(gmm_train))
    flags = np.zeros(len(X), dtype=int)
    flags[test] = gp.flags[test]
    events[GP_NAME] = group_into_events(flags)
    rows.append(dict(method=GP_NAME, **_score(dataset, events[GP_NAME], test)))

    return HeldOutExperiment(dataset=dataset, X=X, predictors=predictors, info=info, events=events, rows=rows, gp=gp,
                             dbscan_rows=dbscan_rows, dbscan_flags=coarse)

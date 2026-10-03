"""
Held-out evaluation of several burn detectors on a long run (Workstream C/G, further extended).

* A 60-orbit simulation is split CHRONOLOGICALLY: the first 70% of time trains every
  model, the last 30% (never seen while fitting) scores it. A random split would leak
  future information backwards and overstate every score.
* Models: GMM, k-means, Bayesian GMM, Agglomerative (Ward), Isolation Forest, DBSCAN, and
  a Gaussian Process that models undisturbed coasting and flags wherever reality departs
  from it.
* DBSCAN has no predict() for new points (it is transductive), so it is scored in-sample
  only and flagged as such.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from .constants import PERIOD
from .controller import DeadbandSimulation, simulate_deadband
from .evidence import BurnEvidence, build_evidence
from .metrics import score_predictions
from .physics import osculating_sma_km

N_ORBITS_LONG = 60
TRAIN_FRACTION = 0.7

# Display names, shared with the plotting code
GMM_NAME = "GMM (2D)"
KMEANS_NAME = "K-means (2D)"
BAYESIAN_GMM_NAME = "Bayesian GMM"
AGGLOMERATIVE_NAME = "Agglomerative (ward)"
ISOLATION_FOREST_NAME = "Isolation Forest"
DBSCAN_TRAIN_NAME = "DBSCAN (in-sample, train)"
DBSCAN_TEST_NAME = "DBSCAN (in-sample, test)"
GP_NAME = "Gaussian Process (dynamics anomaly)"


@dataclass
class FittedDetector:
    """A model fitted on training data. `predict` maps (n, 2) features to 0/1 burn flags."""
    name: str
    predict: Callable[[np.ndarray], np.ndarray]
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Models that can be fitted on one set and applied to another
# ---------------------------------------------------------------------------
def _burn_cluster(centres):
    """The cluster whose centre has the largest summed coordinates is "burn"."""
    return int(np.argmax(centres.sum(axis=1)))


def fit_gmm(train_features):
    model = GaussianMixture(n_components=2, n_init=8, random_state=0, covariance_type="full").fit(train_features)
    burn = _burn_cluster(model.means_)
    return FittedDetector(GMM_NAME, lambda X: (model.predict(X) == burn).astype(int),
                          info=dict(model=model, burn=burn))


def fit_kmeans(train_features):
    model = KMeans(n_clusters=2, n_init=8, random_state=0).fit(train_features)
    burn = _burn_cluster(model.cluster_centers_)
    return FittedDetector(KMEANS_NAME, lambda X: (model.predict(X) == burn).astype(int),
                          info=dict(model=model, burn=burn))


def fit_bayesian_gmm(train_features):
    """Up to 6 components; a sparsity-inducing prior prunes the unused ones."""
    model = BayesianGaussianMixture(n_components=6, covariance_type="full", random_state=0,
                                    weight_concentration_prior=0.01, max_iter=500).fit(train_features)
    burn = _burn_cluster(model.means_)
    n_active = int(np.sum(model.weights_ > 0.01))
    return FittedDetector(BAYESIAN_GMM_NAME, lambda X: (model.predict(X) == burn).astype(int),
                          info=dict(model=model, burn=burn, n_active=n_active))


def fit_agglomerative(train_features):
    """
    Ward clustering has no predict(), so new points go to the nearest training-cluster
    centroid (the standard workaround).
    """
    clustering = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(train_features)
    centroids = np.array([train_features[clustering.labels_ == k].mean(axis=0) for k in (0, 1)])
    burn = _burn_cluster(centroids)

    def predict(X):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        return (np.argmin(distances, axis=1) == burn).astype(int)

    return FittedDetector(AGGLOMERATIVE_NAME, predict, info=dict(centroids=centroids, burn=burn))


def fit_isolation_forest(train_features, train_labels):
    """Contamination is set from the TRAINING burn fraction (no peeking at test labels)."""
    contamination = max(float(np.mean(train_labels)), 1e-3)
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=0).fit(train_features)
    return FittedDetector(ISOLATION_FOREST_NAME, lambda X: (model.predict(X) == -1).astype(int),
                          info=dict(model=model))


def dbscan_in_sample(features):
    """Run DBSCAN on `features` and call the smallest cluster plus all noise points "burn"."""
    eps = 0.6 * np.std(features, axis=0).mean()
    labels = DBSCAN(eps=eps, min_samples=5).fit(features).labels_
    clusters = [c for c in set(labels) if c != -1]
    if not clusters:
        return (labels == -1).astype(int)
    sizes = {c: np.sum(labels == c) for c in clusters}
    smallest = min(sizes, key=sizes.get)
    return ((labels == smallest) | (labels == -1)).astype(int)


# ---------------------------------------------------------------------------
# Gaussian Process dynamics-anomaly detector
# ---------------------------------------------------------------------------
@dataclass
class GpDetection:
    elapsed: np.ndarray      # time since the most recent burn, per sample
    sma_km: np.ndarray
    mean: np.ndarray         # GP prediction of the coast SMA
    std: np.ndarray
    z_score: np.ndarray      # |SMA - mean| / std
    predicted: np.ndarray    # 0/1 burn flags (z above the threshold)
    threshold: float
    fit_indices: np.ndarray  # training samples the GP was fitted on
    model: GaussianProcessRegressor


def elapsed_since_reset(t, burn_times):
    """Time since the most recent burn (or since the start of the run before the first burn)."""
    burn_times = np.asarray(burn_times)
    if len(burn_times) == 0:
        return t - t[0]
    last = np.searchsorted(burn_times, t, side="right") - 1
    last_reset = np.where(last >= 0, burn_times[np.clip(last, 0, len(burn_times) - 1)], t[0])
    return t - last_reset


def fit_gp_detector(sim: DeadbandSimulation, labels, train_mask, n_fit_points=400, seed=0):
    """
    Fit a GP to coast-only TRAINING samples, regressing SMA on time-since-last-burn.

    Using elapsed time (not absolute time) "phase-folds" ~180 near-identical coast
    segments onto one curve, which a single smooth GP can fit. A sample is flagged as a
    burn when its SMA departs from the GP by more than `threshold` standard deviations;
    the threshold is chosen on TRAIN data only and then frozen.
    """
    sma_km = osculating_sma_km(sim.states)
    elapsed = elapsed_since_reset(sim.t, sim.burn_times)

    coast_train = np.where(train_mask & (labels == 0))[0]
    fit_indices = np.random.default_rng(seed).choice(coast_train, size=min(n_fit_points, len(coast_train)),
                                                     replace=False)

    # The length scale is bounded well above the sample spacing (~0.002) so the GP cannot
    # collapse into a near-noise-free interpolator that snaps through every training point.
    kernel = (ConstantKernel(1.0, (1e-3, 1e4)) * RBF(length_scale=0.05, length_scale_bounds=(5e-3, 2.0))
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1.0)))
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=8,
                                     random_state=seed).fit(elapsed[fit_indices].reshape(-1, 1), sma_km[fit_indices])

    mean, std = model.predict(elapsed.reshape(-1, 1), return_std=True)
    z_score = np.abs(sma_km - mean) / np.maximum(std, 1e-6)

    best_threshold, best_score = None, -1
    for threshold in np.arange(1.0, 8.0, 0.25):
        train_score = score_predictions((z_score[train_mask] > threshold).astype(int),
                                        labels[train_mask])["balanced_accuracy"]
        if train_score > best_score:
            best_score, best_threshold = train_score, threshold

    return GpDetection(elapsed=elapsed, sma_km=sma_km, mean=mean, std=std, z_score=z_score,
                       predicted=(z_score > best_threshold).astype(int), threshold=float(best_threshold),
                       fit_indices=fit_indices, model=model)


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------
@dataclass
class HeldOutExperiment:
    sim: DeadbandSimulation
    evidence: BurnEvidence
    features: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    t_split: float
    detectors: dict          # name -> FittedDetector (inductive models only)
    gp: GpDetection
    rows: list               # one score dict per method, all on the held-out test set

    @property
    def train_labels(self):
        return self.evidence.true_label[self.train_mask]

    @property
    def test_labels(self):
        return self.evidence.true_label[self.test_mask]

    @property
    def test_features(self):
        return self.features[self.test_mask]

    def row(self, method_name):
        return next(r for r in self.rows if r["method"] == method_name)


def chronological_split(t, train_fraction=TRAIN_FRACTION):
    t_split = t[0] + train_fraction * (t[-1] - t[0])
    return t < t_split, t >= t_split, t_split


def run_heldout_experiment(sim: DeadbandSimulation = None):
    sim = sim or simulate_deadband(PERIOD * N_ORBITS_LONG)
    evidence = build_evidence(sim)
    features, labels = evidence.features, evidence.true_label
    train_mask, test_mask, t_split = chronological_split(evidence.t)
    train_X, test_X = features[train_mask], features[test_mask]
    train_y, test_y = labels[train_mask], labels[test_mask]

    detectors = {d.name: d for d in (
        fit_gmm(train_X), fit_kmeans(train_X), fit_bayesian_gmm(train_X),
        fit_agglomerative(train_X), fit_isolation_forest(train_X, train_y),
    )}
    gp = fit_gp_detector(sim, labels, train_mask)

    rows = [dict(method=name, **score_predictions(det.predict(test_X), test_y)) for name, det in detectors.items()]
    rows.append(dict(method=DBSCAN_TRAIN_NAME, **score_predictions(dbscan_in_sample(train_X), train_y)))
    rows.append(dict(method=DBSCAN_TEST_NAME, **score_predictions(dbscan_in_sample(test_X), test_y)))
    rows.append(dict(method=GP_NAME, **score_predictions(gp.predicted[test_mask], test_y)))

    return HeldOutExperiment(sim=sim, evidence=evidence, features=features, train_mask=train_mask,
                             test_mask=test_mask, t_split=t_split, detectors=detectors, gp=gp, rows=rows)

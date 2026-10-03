"""
GMM versus k-means in more depth (Workstream C/G, extended): decision regions, how they
move under noise, and how detection quality depends on sampling cadence.
"""
import numpy as np
from sklearn.cluster import KMeans

from .constants import from_minutes
from .detectors import DETECTORS, GMM, KMEANS, noisy_log_signal
from .evidence import log_transform
from .gmm_2d import burn_component, fit_gmm_2d
from .metrics import balanced_accuracy, score_predictions
from .physics import osculating_sma_series
from .controller import DeadbandSimulation
from .evidence import BurnEvidence


# ---------------------------------------------------------------------------
# Decision regions in feature space
# ---------------------------------------------------------------------------
def fit_kmeans_2d(features):
    """Two k-means clusters; returns (model, index of the burn cluster)."""
    model = KMeans(n_clusters=2, n_init=8, random_state=0).fit(features)
    return model, int(np.argmax(model.cluster_centers_.sum(axis=1)))


def region_labels(model, burn_cluster, x_grid, y_grid):
    """1 where the fitted model calls "burn" on a feature-space grid, else 0."""
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    return (model.predict(points) == burn_cluster).astype(int).reshape(x_grid.shape)


def feature_grid(features, pad=0.5, n=150):
    """A grid covering the data: returns (X, Y, x_limits, y_limits)."""
    x_limits = (features[:, 0].min() - pad, features[:, 0].max() + pad)
    y_limits = (features[:, 1].min() - pad, features[:, 1].max() + pad)
    x_grid, y_grid = np.meshgrid(np.linspace(*x_limits, n), np.linspace(*y_limits, n))
    return x_grid, y_grid, x_limits, y_limits


def fit_regions_under_noise(evidence: BurnEvidence, noise_fracs=(0.0, 3.0), seed=0):
    """Fit GMM and k-means on clean / noisy evidence. Returns {noise_frac: dict of fits and predictions}."""
    results = {}
    for noise_frac in noise_fracs:
        noisy = evidence.with_noise(noise_frac, seed=seed + int(noise_frac * 10) + 1)
        features = noisy.features
        gmm = fit_gmm_2d(features)
        gmm_burn = burn_component(gmm)
        kmeans, kmeans_burn = fit_kmeans_2d(features)
        results[noise_frac] = dict(
            evidence=noisy, features=features,
            gmm=gmm, gmm_burn=gmm_burn, gmm_predicted=(gmm.predict(features) == gmm_burn).astype(int),
            kmeans=kmeans, kmeans_burn=kmeans_burn, kmeans_predicted=(kmeans.predict(features) == kmeans_burn).astype(int),
        )
    return results


# ---------------------------------------------------------------------------
# Precision / recall versus noise
# ---------------------------------------------------------------------------
def precision_recall_noise_sweep(evidence: BurnEvidence, noise_fracs):
    rows = []
    for noise_frac in noise_fracs:
        signal = noisy_log_signal(evidence, noise_frac)
        for name in (GMM, KMEANS):
            scores = score_predictions(DETECTORS[name](signal.copy()), evidence.true_label)
            rows.append(dict(noise_frac=noise_frac, method=name,
                             precision=float(scores["precision"]), recall=float(scores["sensitivity"])))
    return rows


# ---------------------------------------------------------------------------
# Sampling cadence
# ---------------------------------------------------------------------------
def resample_at_cadence(sim: DeadbandSimulation, spacing):
    """
    Keep the nearest native sample to each point on a uniform grid of `spacing`
    (non-dimensional). A burn is labelled on the first kept sample at or after its true
    time, as a fixed-cadence tracker would see it. Returns (sma, labels).
    """
    sma = osculating_sma_series(sim.states)
    grid = np.arange(0, sim.t[-1], spacing)
    kept = np.unique(np.clip(np.searchsorted(sim.t, grid), 0, len(sim.t) - 1))
    kept_times = sim.t[kept]

    labels = np.zeros(len(kept), dtype=int)
    for burn_time in sim.burn_times:
        labels[min(np.searchsorted(kept_times, burn_time), len(kept) - 1)] = 1
    return sma[kept], labels


def cadence_detection_sweep(sim: DeadbandSimulation, cadences_min):
    """Every basic detector's balanced accuracy and precision at each sampling cadence."""
    rows = []
    for cadence_min in cadences_min:
        sma, labels = resample_at_cadence(sim, from_minutes(cadence_min))
        jump = np.abs(np.diff(sma))
        signal = log_transform(np.concatenate([[jump[0]], jump]))
        for name, detector in DETECTORS.items():
            predicted = detector(signal.copy())
            rows.append(dict(dt_min=cadence_min, method=name,
                             balanced_accuracy=float(balanced_accuracy(predicted, labels)),
                             precision=float(score_predictions(predicted, labels)["precision"]),
                             n_samples=int(len(labels))))
    return rows

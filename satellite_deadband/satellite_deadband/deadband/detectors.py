"""
Five basic burn detectors (Workstream C/G), all operating on a 1-D log evidence signal.

Each detector takes the signal and returns a 0/1 array (1 = burn). `DETECTORS` maps the
display name used in plots to the function.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .evidence import BurnEvidence, NOISE_FLOOR, add_noise, log_transform
from .metrics import accuracy, balanced_accuracy

ROLLING_ZSCORE = "Rolling z-score"
FIXED_THRESHOLD = "Fixed threshold"
KMEANS = "K-means (2 clusters)"
GMM = "Gaussian mixture model"
CUSUM = "CUSUM"


def rolling_zscore_detector(x, window=25, threshold=3.5):
    """Flag points more than `threshold` standard deviations from the previous `window` points."""
    flags = np.zeros(len(x), dtype=int)
    for i in range(len(x)):
        history = x[max(0, i - window):i] if i > 0 else x[0:1]
        z = (x[i] - np.mean(history)) / (np.std(history) + 1e-12)
        flags[i] = int(abs(z) > threshold)
    return flags


def fixed_threshold_detector(x, k=6.0):
    """Flag points above k times the median absolute deviation (measured from the median)."""
    cutoff = k * np.median(np.abs(x - np.median(x)))
    return (x > cutoff).astype(int)


def kmeans_detector(x):
    """Two k-means clusters; the cluster with the larger centre is "burn"."""
    model = KMeans(n_clusters=2, n_init=5, random_state=0).fit(x.reshape(-1, 1))
    burn_cluster = np.argmax(model.cluster_centers_.ravel())
    return (model.labels_ == burn_cluster).astype(int)


def gmm_detector(x):
    """Two-component Gaussian mixture; the component with the larger mean is "burn"."""
    model = GaussianMixture(n_components=2, n_init=5, random_state=0).fit(x.reshape(-1, 1))
    burn_component = np.argmax(model.means_.ravel())
    return (model.predict(x.reshape(-1, 1)) == burn_component).astype(int)


def cusum_detector(x, drift=0.5, threshold=5.0):
    """One-sided CUSUM; the accumulator resets each time it fires."""
    centre, spread = np.median(x), np.std(x) + 1e-12
    accumulator = 0.0
    flags = np.zeros(len(x), dtype=int)
    for i in range(len(x)):
        accumulator = max(0, accumulator + (x[i] - centre) / spread - drift)
        if accumulator > threshold:
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


def noisy_log_signal(evidence: BurnEvidence, noise_frac):
    """The SMA-jump channel with noise added to the raw jumps, then log-transformed."""
    noisy = np.clip(add_noise(evidence.jump_sma, noise_frac, seed=int(noise_frac * 1000) + 1), NOISE_FLOOR, None)
    return log_transform(noisy)


def noise_robustness_sweep(evidence: BurnEvidence, noise_fracs):
    """Accuracy and balanced accuracy of every detector at each noise level."""
    rows = []
    for noise_frac in noise_fracs:
        signal = noisy_log_signal(evidence, noise_frac)
        for name, detector in DETECTORS.items():
            predicted = detector(signal.copy())
            rows.append(dict(noise_frac=noise_frac, method=name,
                             accuracy=accuracy(predicted, evidence.true_label),
                             balanced_accuracy=balanced_accuracy(predicted, evidence.true_label)))
    return rows

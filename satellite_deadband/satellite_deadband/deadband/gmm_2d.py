"""
Two-feature Gaussian mixture burn detector (Workstream C/G, extended) plus the geometry
helpers needed to draw it: density grids and covariance ellipses.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture

from .detectors import gmm_detector
from .evidence import BurnEvidence
from .metrics import balanced_accuracy, score_predictions


def burn_component(model):
    """Index of the component with the larger mean (the burn regime)."""
    return int(np.argmax(model.means_.sum(axis=1)))


def fit_gmm_2d(features):
    """Two-component, full-covariance GMM on the (N, 2) feature matrix."""
    return GaussianMixture(n_components=2, n_init=8, random_state=0, covariance_type="full").fit(features)


@dataclass
class Gmm2dSummary:
    model: GaussianMixture
    features: np.ndarray
    predicted: np.ndarray
    burn_component: int
    balanced_accuracy_1d: float     # SMA-jump channel only
    balanced_accuracy_2d: float     # SMA-jump + speed-jump channels
    scores: dict                    # confusion counts, sensitivity, specificity, precision


def summarize_gmm_2d(evidence: BurnEvidence):
    features = evidence.features
    model = fit_gmm_2d(features)
    component = burn_component(model)
    predicted = (model.predict(features) == component).astype(int)

    predicted_1d = gmm_detector(evidence.log_sma.copy())
    return Gmm2dSummary(model=model, features=features, predicted=predicted, burn_component=component,
                        balanced_accuracy_1d=float(balanced_accuracy(predicted_1d, evidence.true_label)),
                        balanced_accuracy_2d=float(balanced_accuracy(predicted, evidence.true_label)),
                        scores=score_predictions(predicted, evidence.true_label))


# ---------------------------------------------------------------------------
# Geometry for plotting
# ---------------------------------------------------------------------------
def gaussian_2d_pdf(x_grid, y_grid, mean, covariance):
    position = np.dstack((x_grid, y_grid))
    inverse = np.linalg.inv(covariance)
    offset = position - mean
    exponent = -0.5 * np.einsum("...k,kl,...l->...", offset, inverse, offset)
    return np.exp(exponent) / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))


def mixture_density_grid(model, x_limits, y_limits, n=120):
    """Mixture density evaluated on an n x n grid: returns (X, Y, Z)."""
    x_grid, y_grid = np.meshgrid(np.linspace(*x_limits, n), np.linspace(*y_limits, n))
    density = np.zeros_like(x_grid)
    for k in range(model.n_components):
        density += model.weights_[k] * gaussian_2d_pdf(x_grid, y_grid, model.means_[k], model.covariances_[k])
    return x_grid, y_grid, density


def ellipse_points(mean, covariance, n_std=2.0, n=100):
    """Points on the n_std covariance ellipse of a 2-D Gaussian: returns (x, y)."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    scaled = (eigenvectors @ np.diag(n_std * np.sqrt(np.maximum(eigenvalues, 1e-12)))) @ circle
    return mean[0] + scaled[0], mean[1] + scaled[1]

"""
Two-feature Gaussian mixture burn detector and the geometry needed to draw it (density grids,
burn-probability grids, covariance ellipses).

Features: the step 6 classifier features (classifiers.Features.matrix), from VELOCITY data:
    column 0: log |unmodelled acceleration from v'|
    column 1: along-track part / 1e-4 m/s^2      (thrust pushes forwards, noise does not)
The mixture is fitted on TRAIN samples only; its number of components is the tuned one
(settings.tuned("gmm_components")): one or two for coast, one for the burn plateau and spares for the
thruster ramps / noise tail. A component is BURN when its mean
log|v'| is more than 3 coast standard deviations above the coast mean (the step 6 rule).

The "weighting" of a sample is its posterior probability of belonging to a burn component,
    P(burn | x) = sum over burn components k of  w_k N(x | mu_k, S_k) / sum over all j of w_j N(x | mu_j, S_j)
which the 3D figures use as a colour.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture

from .classifiers import mixture_burn_components
from .settings import tuned


@dataclass
class FeatureGmm:
    model: GaussianMixture
    is_burn: np.ndarray            # (K,) True for burn components

    def predict(self, X):
        """0/1 burn flags."""
        return self.is_burn[self.model.predict(X)].astype(int)

    def burn_probability(self, X):
        """P(burn | x): the posterior weight of the burn components."""
        return self.model.predict_proba(X)[:, self.is_burn].sum(axis=1)


def fit_feature_gmm(X_train, n_components=None):
    n_components = n_components or tuned("gmm_components")
    model = GaussianMixture(n_components=n_components, n_init=3, random_state=0, covariance_type="full").fit(X_train)
    return FeatureGmm(model=model, is_burn=mixture_burn_components(model))


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


def refined_grid(model, x_limits, y_limits, n=120, n_local=40):
    """
    A (non-uniform) grid that is fine around every component: a uniform grid plus extra lines within
    +-4 sd of each mean. The burn component can be far narrower than a uniform grid's spacing; without
    this its peak would be drawn as aliasing spikes. Returns (X, Y) from np.meshgrid.
    """
    axes = []
    for dim, limits in enumerate((x_limits, y_limits)):
        values = [np.linspace(*limits, n)]
        for k in range(model.n_components):
            sd = np.sqrt(model.covariances_[k][dim, dim])
            values.append(model.means_[k, dim] + sd * np.linspace(-4, 4, n_local))
        values = np.unique(np.concatenate(values))
        axes.append(values[(values >= limits[0]) & (values <= limits[1])])
    return np.meshgrid(*axes)


def mixture_density_on(model, x_grid, y_grid):
    """Mixture density on any (X, Y) grid."""
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    return np.exp(model.score_samples(points)).reshape(x_grid.shape)


def probability_grid(gmm: FeatureGmm, x_grid, y_grid):
    """P(burn | x) on a grid."""
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    return gmm.burn_probability(points).reshape(x_grid.shape)


def ellipse_points(mean, covariance, n_std=2.0, n=100):
    """Points on the n_std covariance ellipse of a 2-D Gaussian: returns (x, y)."""
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    scaled = (eigenvectors @ np.diag(n_std * np.sqrt(np.maximum(eigenvalues, 1e-12)))) @ circle
    return mean[0] + scaled[0], mean[1] + scaled[1]

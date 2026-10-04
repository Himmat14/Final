"""
GMM versus k-means in the 2-feature plane (Workstream C/G, extended): decision regions and how they
move from clean to noisy data. Same data and features as report step 6 (days 0-100 of the controlled
run, TRAIN days 0-40, TEST days 40-100; see detectors.window_features).
"""
import numpy as np
from sklearn.cluster import KMeans

from .detectors import train_subsample, window_features
from .gmm_2d import fit_feature_gmm


def fit_kmeans_2d(X_train):
    """Two k-means clusters; returns a predict function (1 = the cluster with the larger log|v'| centre)."""
    model = KMeans(n_clusters=2, n_init=8, random_state=0).fit(X_train)
    burn = int(np.argmax(model.cluster_centers_[:, 0]))
    return lambda X: (model.predict(X) == burn).astype(int)


def feature_grid(features, pad=0.5, n=150, percentile=(0.01, 99.99)):
    """A grid covering (almost all of) the data: returns (X, Y, x_limits, y_limits)."""
    lo, hi = np.percentile(features, percentile, axis=0)
    x_limits, y_limits = (lo[0] - pad, hi[0] + pad), (lo[1] - pad, hi[1] + pad)
    x_grid, y_grid = np.meshgrid(np.linspace(*x_limits, n), np.linspace(*y_limits, n))
    return x_grid, y_grid, x_limits, y_limits


def region_labels(predict, x_grid, y_grid):
    """1 where `predict` calls "burn" on a feature-space grid, else 0."""
    return predict(np.column_stack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)


def fit_regions_under_noise(noise_scales=(0.0, 1.0)):
    """{noise_scale: dict(dataset, X, gmm, kmeans predict, TEST predictions)} fitted on TRAIN."""
    results = {}
    for scale in noise_scales:
        dataset, features = window_features(scale, seed=1)
        X = features.matrix
        X_train = train_subsample(X, dataset)
        gmm = fit_feature_gmm(X_train)
        kmeans = fit_kmeans_2d(X_train)
        test = dataset.test
        results[scale] = dict(dataset=dataset, X=X, gmm=gmm, kmeans=kmeans,
                              gmm_test=gmm.predict(X[test]), kmeans_test=kmeans(X[test]))
    return results

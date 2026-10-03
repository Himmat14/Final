"""
Steps 6-7 of the report: classify every sample as COAST (natural physics) or BURN (thruster on),
train on one part of the data and test on another, and score whole burn EVENTS.

Pipeline
--------
1. DATASET   The 400-day smooth-deadband run (long_run.controlled_run: 500 m band, J2 + drag,
             ~100 burns), sampled every 0.1 min. The first 40 days are TRAIN, days 40-400 are TEST
             (chronological split: in real operations you only ever have the past to learn from).
             Every TEST score is also reported over the first 15, 30, 60, 120, 240 and 360 days.
2. NOISE     Optional Gaussian noise on the measured position (metres) and velocity (mm/s).
3. FEATURES  Unmodelled acceleration = (derivative of the data) - (known gravity + J2), in two
             "spaces": r'' from positions and v' from velocities. Derivatives use a
             Savitzky-Golay filter: a local polynomial fit over a short window, i.e. a finite
             difference with built-in averaging, which keeps noise from exploding.
4. MODELS    Fitted on TRAIN only (on a random subsample, for speed), then applied to everything.
             Classifier features use the velocity data: [log |v' residual|, along-track v' residual].
5. EVENTS    Flagged samples -> burn events (short gaps filled, single points dropped), then
             scored per event on TEST: burns found, false events, precision / recall / F1.
             To save time the models only predict on the samples being scored.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from .constants import ACCEL_UNIT_MS2, DU, SAMPLE_STEP_S, SECONDS, VU, from_days
from .derivative_detection import event_scores, group_into_events, known_accel
from .long_run import HORIZONS_DAYS, TRAIN_DAYS, controlled_run

# --- Features ----------------------------------------------------------------
SG_WINDOW = 21                          # samples in each local polynomial fit (21 x 6 s = 2 min)
SG_ORDER = 5                            # polynomial order (high enough not to bias the orbit curvature)
REFERENCE_NOISE = (0.1, 0.5)            # (position m, velocity mm/s) used for the "noisy" case
FIT_SUBSAMPLE = 20_000                  # training samples each model is fitted on
EVENT_TOLERANCE_S = 60.0
FLOOR_MS2 = 1e-12


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------
@dataclass
class DetectionDataset:
    t: np.ndarray            # (N,) sample times (non-dimensional)
    r: np.ndarray            # (3, N) true positions
    v: np.ndarray            # (3, N) true velocities
    true_on: np.ndarray      # (N,) 1 where the thruster is more than half open
    thrust_ms2: np.ndarray   # (N,) true thrust acceleration
    mean_sma_km: np.ndarray  # (N,) true mean SMA
    step_s: float

    @property
    def days(self):
        return self.t / from_days(1)

    @property
    def train(self):
        """Boolean mask of the training samples (the first TRAIN_DAYS)."""
        return self.days < TRAIN_DAYS

    @property
    def test(self):
        return ~self.train

    def test_horizon(self, horizon_days):
        """The first `horizon_days` of the TEST period."""
        return self.test & (self.days < TRAIN_DAYS + horizon_days)

    def until(self, days):
        """A shorter copy ending at `days` (keeps the same TRAIN period)."""
        keep = self.days < days
        return DetectionDataset(self.t[keep], self.r[:, keep], self.v[:, keep], self.true_on[keep],
                                self.thrust_ms2[keep], self.mean_sma_km[keep], self.step_s)

    def every(self, stride):
        """A coarser copy keeping every `stride`-th sample (for the sampling-step robustness test)."""
        return DetectionDataset(self.t[::stride], self.r[:, ::stride], self.v[:, ::stride], self.true_on[::stride],
                                self.thrust_ms2[::stride], self.mean_sma_km[::stride], self.step_s * stride)


@lru_cache(maxsize=1)
def detection_dataset():
    """The 400-day controlled run (see long_run.py), as a DetectionDataset."""
    run = controlled_run()
    return DetectionDataset(t=run.t, r=run.r, v=run.v, true_on=run.true_on, thrust_ms2=run.thrust_ms2,
                            mean_sma_km=run.mean_sma_km, step_s=SAMPLE_STEP_S)


# ---------------------------------------------------------------------------
# 2. Noise
# ---------------------------------------------------------------------------
def add_noise(dataset: DetectionDataset, position_m, velocity_mms, seed=0):
    """Measured (r, v): the truth plus Gaussian noise (position in metres, velocity in mm/s)."""
    rng = np.random.default_rng(seed)
    r = dataset.r + position_m / 1000 / DU * rng.standard_normal(dataset.r.shape)
    v = dataset.v + velocity_mms / 1e6 / VU * rng.standard_normal(dataset.v.shape)
    return r, v


# ---------------------------------------------------------------------------
# 3. Features
# ---------------------------------------------------------------------------
@dataclass
class Features:
    accel_from_r: np.ndarray   # (3, N) unmodelled acceleration from positions [m/s^2]
    accel_from_v: np.ndarray   # (3, N) unmodelled acceleration from velocities [m/s^2]
    along_track: np.ndarray    # (N,) component of accel_from_v along the velocity [m/s^2]

    @property
    def matrix(self):
        """
        (N, 2) classifier features, from the VELOCITY data only:
            column 0: log |v' residual|                      (how much unmodelled acceleration)
            column 1: along-track v' residual / 1e-4 m/s^2   (thrust pushes forwards, noise does not)
        Position-based r'' is kept for the r'' vs v' comparison but not used here: with realistic
        position noise its second derivative is pure noise (see step 6 / 7 figures).
        """
        return np.column_stack([np.log(np.linalg.norm(self.accel_from_v, axis=0) + FLOOR_MS2),
                                self.along_track / 1e-4])

    @property
    def log_r_and_v(self):
        """(N, 2) [log |r'' residual|, log |v' residual|] for comparing the two derivative spaces."""
        return np.column_stack([np.log(np.linalg.norm(self.accel_from_r, axis=0) + FLOOR_MS2),
                                np.log(np.linalg.norm(self.accel_from_v, axis=0) + FLOOR_MS2)])


def build_features(r, v, step_s, window=SG_WINDOW, order=SG_ORDER):
    """Savitzky-Golay derivatives of the measured r and v, minus the known gravity + J2."""
    h = step_s * SECONDS
    window = max(window, order + 2 + (order % 2 == 0))            # must be odd and longer than the order
    r_second_derivative = savgol_filter(r, window, order, deriv=2, delta=h, axis=1)
    v_first_derivative = savgol_filter(v, window, order, deriv=1, delta=h, axis=1)
    modelled = known_accel(r)
    from_r = (r_second_derivative - modelled) * ACCEL_UNIT_MS2
    from_v = (v_first_derivative - modelled) * ACCEL_UNIT_MS2
    along = np.sum(from_v * v, axis=0) / np.linalg.norm(v, axis=0)
    return Features(accel_from_r=from_r, accel_from_v=from_v, along_track=along)


# ---------------------------------------------------------------------------
# 4. Models (each returns (predict function: features -> 0/1 burn flags, fitted model))
# ---------------------------------------------------------------------------
def mixture_burn_components(model, n_sigma=3.0):
    """
    Which mixture components are BURN? The heaviest component is coast. A component is burn if
    its mean log|v'| sits more than n_sigma coast standard deviations above the coast mean
    (thrust can only ADD acceleration). Components in between are extra pieces of coast.
    """
    coast = int(np.argmax(model.weights_))
    coast_sd = np.sqrt(np.atleast_2d(model.covariances_[coast])[0, 0])
    return model.means_[:, 0] > model.means_[coast, 0] + n_sigma * coast_sd


def fit_gmm(X_train, n_components=2):
    """Gaussian mixture with a fixed number of components."""
    model = GaussianMixture(n_components=n_components, n_init=3, random_state=0).fit(X_train)
    burn = mixture_burn_components(model)
    return lambda X: burn[model.predict(X)].astype(int), model


def fit_bayesian_gmm(X_train, max_components=6):
    """Bayesian mixture with up to 6 components; its prior switches off the ones it does not need."""
    model = BayesianGaussianMixture(n_components=max_components, weight_concentration_prior=0.01,
                                    random_state=0, max_iter=1000).fit(X_train)
    burn = mixture_burn_components(model)
    return lambda X: burn[model.predict(X)].astype(int), model


def fit_kmeans(X_train):
    """Two k-means clusters; the one with the larger log|v'| centre is burn."""
    model = KMeans(n_clusters=2, n_init=5, random_state=0).fit(X_train)
    burn = int(np.argmax(model.cluster_centers_[:, 0]))
    return lambda X: (model.predict(X) == burn).astype(int), model


def fit_isolation_forest(X_train, contamination=0.01):
    """Anomaly detector: flags points that are easy to isolate (and lie above the coast median)."""
    model = IsolationForest(contamination=contamination, random_state=0).fit(X_train)
    median = np.median(X_train, axis=0)
    return lambda X: ((model.predict(X) == -1) & (X[:, 0] > median[0])).astype(int), model


def fit_threshold(X_train, n_mad=5.0):
    """Baseline: flag log|v'| more than n_mad median-absolute-deviations above the training median."""
    median = np.median(X_train[:, 0])
    mad = np.median(np.abs(X_train[:, 0] - median))
    return lambda X: (X[:, 0] > median + n_mad * mad).astype(int), None


MODELS = {
    "GMM": fit_gmm,
    "Bayesian GMM": fit_bayesian_gmm,
    "k-means": fit_kmeans,
    "Isolation Forest": fit_isolation_forest,
    "Threshold (MAD)": fit_threshold,
}


def training_subsample(X, train_mask, n=FIT_SUBSAMPLE, seed=0):
    rows = np.flatnonzero(train_mask)
    rng = np.random.default_rng(seed)
    return X[rng.choice(rows, size=min(n, rows.size), replace=False)]


# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
@dataclass
class MethodResult:
    name: str
    flagged: np.ndarray        # raw per-sample output (0 outside the scored samples)
    events: np.ndarray         # after grouping into events
    test_scores: dict          # event scores on the scored TEST samples
    horizon_scores: dict       # horizon [days] -> event scores over the first `horizon` TEST days
    model: object


def score_segment(dataset, events, mask):
    return event_scores(dataset.t[mask], events[mask], dataset.true_on[mask], dataset.step_s, EVENT_TOLERANCE_S)


def evaluate(dataset: DetectionDataset, features: Features, methods=MODELS, n_fit=FIT_SUBSAMPLE, seed=0,
             horizons=(), predict_all=False, **fit_kwargs):
    """
    Fit every method on (a random subsample of) TRAIN, predict on TEST, group into events and score.
    `horizons` (days): also score the first 15 / 30 / ... days of TEST separately.
    `predict_all`: also label the TRAIN samples (needed when later steps learn from TRAIN clusters).
    """
    X = features.matrix
    X_train = training_subsample(X, dataset.train, n_fit, seed)
    test_rows = np.arange(len(X)) if predict_all else np.flatnonzero(dataset.test)
    results = {}
    for name, fit in methods.items():
        predict, model = fit(X_train, **fit_kwargs.get(name, {}))
        flagged = np.zeros(len(X), dtype=int)
        flagged[test_rows] = predict(X[test_rows])
        events = group_into_events(flagged)
        results[name] = MethodResult(
            name=name, flagged=flagged, events=events, model=model,
            test_scores=score_segment(dataset, events, dataset.test),
            horizon_scores={h: score_segment(dataset, events, dataset.test_horizon(h)) for h in horizons})
    return results


def noisy_features(dataset, noise_scale=1.0, seed=0, window=SG_WINDOW):
    """Features at `noise_scale` x the reference noise (0 = clean)."""
    position_m, velocity_mms = REFERENCE_NOISE
    r, v = add_noise(dataset, position_m * noise_scale, velocity_mms * noise_scale, seed)
    return build_features(r, v, dataset.step_s, window=window), (r, v)

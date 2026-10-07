"""
Step 9 of the report: learn the CONTROL LAW from data, then use it to PREDICT future burns.

What is learned
---------------
The mean SMA a(t) obeys (energy derivation, Week 2 deck Step 10):
        da/dt = (2 a^2 / mu) * (v . a_pert)
    coast: a_pert = drag = -cd |v| v      ->  da/dt = -cd * D(a),   D = (2 a^2/mu) |v|^3
    burn:  a_pert = drag + T v/|v|        ->  da/dt = -cd * D(a) + T * B(a),   B = (2 a^2/mu) |v|
Inside a 100 m band a and |v| hardly change, so D and B are CONSTANTS to 1 part in 10^5: a
library containing D, B and a constant column would be singular (see the library-correlation
figure). The honest library is therefore built from the CLUSTER labels (coast / burn, from the
step 6 classifier) plus distractor terms that a good method must reject:
        da/dt = k_coast * (1 - s) + k_burn * s + c1 (a - a_ref) + c2 s (a - a_ref) + c3 (a - a_ref)^2
and the physics is read off afterwards:  cd = -k_coast / D,   T = (k_burn + cd * D) / B.

    SINDy  (Brunton et al. 2016): sequentially thresholded least squares (STLSQ):
           solve, zero every small coefficient, re-solve with what is left, repeat.
    BINDy-style: sparse BAYESIAN regression (sklearn ARDRegression): every coefficient gets a
           posterior mean AND standard deviation, i.e. "how confident am I". (The BINDy paper uses
           its own prior and inference; ARD is a simple stand-in with the same idea.)

The SWITCHING law (when the thruster turns on/off) is learned from the detected burn events:
    lower edge = SMA when burns start,  upper edge = SMA when they stop  (a hysteresis loop).
A memory-less sigmoid P(on | a) is fitted too, to show why the memory is needed (Week 3 result).

Finally every learned model is run FORWARD from the start of the test period, with no further
data, to forecast when the test burns happen.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ARDRegression, LogisticRegression

from .constants import DU, MU, SECONDS, TU
from .derivative_detection import runs_of_ones
from .settings import tuned
from .smooth_controller import mean_sma_km_series

# the SMA smoothing window and the STLSQ threshold are tuned (settings.tuned("sma_window" / "stlsq_threshold"))
EDGE_BUFFER_S = 120.0       # samples this close to a burn edge are left out of both clusters
LIBRARY_NAMES = ["coast (1-s)", "burn s", "a - a_ref", "s (a - a_ref)", "(a - a_ref)^2"]
PHYSICS_LIBRARY_NAMES = ["drag shape D", "thrust shape B", "constant"]


# ---------------------------------------------------------------------------
# Data: mean SMA, its rate, and the two clusters
# ---------------------------------------------------------------------------
@dataclass
class SmaData:
    t_s: np.ndarray        # time [s]
    a: np.ndarray          # mean SMA (non-dimensional), smoothed
    a_dot: np.ndarray      # d a / dt (non-dimensional)
    speed: np.ndarray      # |v| (non-dimensional)
    coast: np.ndarray      # bool: clearly coasting
    burn: np.ndarray       # bool: clearly burning


def sma_data(t, r, v, events, step_s, window=None, edge_buffer_s=None):
    """
    Mean SMA from measured r, v; its derivative; and coast / burn masks from the detected events.
    edge_buffer_s: samples this close to a detected burn edge belong to neither cluster (default: the
    step 9 rule, which grows with the classifier's derivative window).
    """
    window = window or tuned("sma_window")
    a = mean_sma_km_series(np.vstack([r, v])) / DU
    h = step_s * SECONDS
    a_smooth = savgol_filter(a, window, 2)
    a_dot = savgol_filter(a, window, 2, deriv=1, delta=h)
    near_edge = np.zeros(len(t), dtype=bool)
    # the detected burn edges are smeared by half the classifier's derivative window (tuned "sg_window"),
    # so the buffer grows with it: otherwise thruster-ramp samples leak into the coast and burn clusters
    if edge_buffer_s is None:
        edge_buffer_s = EDGE_BUFFER_S + tuned("sg_window") * step_s / 2
    buffer = int(edge_buffer_s / step_s)
    starts = np.flatnonzero(np.diff(events) != 0)
    for i in starts:
        near_edge[max(0, i - buffer):i + buffer + 1] = True
    return SmaData(t_s=t * TU, a=a_smooth, a_dot=a_dot, speed=np.linalg.norm(v, axis=0),
                   coast=(events == 0) & ~near_edge, burn=(events == 1) & ~near_edge)


def drag_shape(a, speed):
    """D = (2 a^2 / mu) |v|^3 : da/dt per unit cd."""
    return 2 * a**2 / MU * speed**3


def thrust_shape(a, speed):
    """B = (2 a^2 / mu) |v| : da/dt per unit thrust acceleration."""
    return 2 * a**2 / MU * speed


def physics_library(a, speed):
    """The 'obvious' library [D, B, 1]: kept only to show it is singular over a narrow band."""
    return np.column_stack([drag_shape(a, speed), thrust_shape(a, speed), np.ones_like(a)])


def library(a, s, a_ref):
    """Candidate terms for da/dt: cluster indicators plus distractors (offsets scaled to ~metres)."""
    offset = (a - a_ref) * DU * 1e3          # metres from the reference SMA
    return np.column_stack([1 - s, s, offset, s * offset, offset**2])


# ---------------------------------------------------------------------------
# SINDy and the Bayesian version
# ---------------------------------------------------------------------------
def stlsq(theta, target, threshold=None, iterations=10):
    """Sequentially thresholded least squares on unit-scaled columns. Returns unscaled coefficients."""
    threshold = threshold if threshold is not None else tuned("stlsq_threshold")
    scale = np.linalg.norm(theta, axis=0)
    empty = scale == 0                                     # a column that is zero in the data cannot be fitted
    scale = np.where(empty, 1.0, scale)
    X = theta / scale
    active = ~empty
    xi = np.zeros(X.shape[1])
    for _ in range(iterations):
        xi[:] = 0
        xi[active] = np.linalg.lstsq(X[:, active], target, rcond=None)[0]
        new_active = np.abs(xi) >= threshold * np.max(np.abs(xi))
        if np.array_equal(new_active, active):
            break
        active = new_active
    return xi / scale


def stlsq_significance(theta, target, t_min=3.0, iterations=10):
    """
    Sequentially thresholded least squares with a STATISTICAL threshold: a term is dropped when its
    coefficient is smaller than t_min standard errors (|xi| / se < t_min), not when it is small compared
    with the LARGEST coefficient. Classic STLSQ wrongly prunes a small but well-measured term (the coast
    drag, ~5 m/h) next to a huge rare one (the burn, ~1300 m/h); this version keeps it.
    """
    n, k = theta.shape
    active = np.linalg.norm(theta, axis=0) > 0
    xi = np.zeros(k)
    for _ in range(iterations):
        xi[:] = 0
        X = theta[:, active]
        coef, *_ = np.linalg.lstsq(X, target, rcond=None)
        resid = target - X @ coef
        dof = max(n - active.sum(), 1)
        cov = np.linalg.pinv(X.T @ X) * (resid @ resid) / dof
        se = np.sqrt(np.maximum(np.diag(cov), 1e-300))
        xi[active] = coef
        keep = np.abs(coef) / se >= t_min
        if keep.all():
            break
        idx = np.flatnonzero(active)
        active[idx[~keep]] = False
        if not active.any():
            break
    return xi


def bayesian_sparse_fit(theta, target):
    """ARD regression: posterior mean and standard deviation of every coefficient (unscaled)."""
    scale = np.linalg.norm(theta, axis=0)
    target_scale = np.std(target) or 1.0
    model = ARDRegression(fit_intercept=False).fit(theta / scale, target / target_scale)
    mean = model.coef_ * target_scale / scale
    std = np.sqrt(np.diag(model.sigma_)) if model.sigma_.shape[0] == theta.shape[1] else np.zeros(theta.shape[1])
    # sklearn keeps sigma_ only for the coefficients it did not prune; expand to the full set
    full_std = np.zeros(theta.shape[1])
    kept = np.abs(model.coef_) > 0
    if model.sigma_.shape[0] == kept.sum():
        full_std[kept] = np.sqrt(np.diag(model.sigma_))
    elif std.any():
        full_std = std
    return mean, full_std * target_scale / scale


@dataclass
class ClusterLaws:
    a_ref: float
    sindy: np.ndarray        # coefficients for LIBRARY_NAMES
    bindy_mean: np.ndarray
    bindy_std: np.ndarray
    drag_shape: float        # D at the reference SMA
    thrust_shape: float      # B at the reference SMA

    def physics(self, coefficients):
        """(cd, T) implied by the two cluster rates."""
        cd = -coefficients[0] / self.drag_shape
        thrust = (coefficients[1] + cd * self.drag_shape) / self.thrust_shape
        return cd, thrust


def learn_cluster_laws(data: SmaData, train_mask):
    """One hybrid regression over both clusters (TRAIN data only), with SINDy and the Bayesian version."""
    rows = (data.coast | data.burn) & train_mask
    a_ref = float(np.median(data.a[rows]))
    theta = library(data.a[rows], data.burn[rows].astype(float), a_ref)
    mean, std = bayesian_sparse_fit(theta, data.a_dot[rows])
    speed_ref = float(np.median(data.speed[rows]))
    return ClusterLaws(a_ref=a_ref, sindy=stlsq(theta, data.a_dot[rows]), bindy_mean=mean, bindy_std=std,
                       drag_shape=float(drag_shape(a_ref, speed_ref)), thrust_shape=float(thrust_shape(a_ref, speed_ref)))


# ---------------------------------------------------------------------------
# Switching law
# ---------------------------------------------------------------------------
@dataclass
class SwitchingLaw:
    start_smas: np.ndarray     # SMA [non-dim] at every detected burn start (train)
    end_smas: np.ndarray       # SMA at every detected burn end (train)

    @property
    def lower(self):
        return float(np.median(self.start_smas))

    @property
    def upper(self):
        return float(np.median(self.end_smas))


def learn_switching_law(t, a, events, train_mask, settle_s=EDGE_BUFFER_S):
    """
    Read the SMA `settle_s` away from each burn so the smoothing window (which is centred) does
    not mix in the burn itself:
        lower edge: SMA `settle_s` BEFORE each detected burn start (coasting changes it by < 0.2 m),
        upper edge: SMA `settle_s` AFTER each burn end, where coasting really begins.
    """
    settle = settle_s / TU
    starts, ends = [], []
    for start, end in runs_of_ones(t[train_mask], events[train_mask]):
        starts.append(a[max(np.searchsorted(t, start - settle), 0)])
        ends.append(a[min(np.searchsorted(t, end + settle), len(a) - 1)])
    return SwitchingLaw(np.array(starts), np.array(ends))


def hysteresis_states(a, lower, upper, s0=0):
    """The deadband WITH memory: on below `lower`, off above `upper`, otherwise keep the last state."""
    s = np.zeros(len(a), dtype=int)
    state = s0
    for i, value in enumerate(a):
        if value <= lower:
            state = 1
        elif value >= upper:
            state = 0
        s[i] = state
    return s


def memoryless_sigmoid(a_train, on_train):
    """Logistic regression P(on | a) using the current SMA only (no memory)."""
    scale = 1e3   # work in metres-ish units so the fit is well conditioned
    model = LogisticRegression(class_weight="balanced").fit(((a_train - a_train.mean()) * scale)[:, None], on_train)
    return lambda a: model.predict_proba(((a - a_train.mean()) * scale)[:, None])[:, 1]


# ---------------------------------------------------------------------------
# Forecasting the test period with a learned model
# ---------------------------------------------------------------------------
def circular_speed(a):
    return np.sqrt(MU / a)


def run_hybrid(a0, s0, n_steps, h, rate, lower, upper):
    """
    Free-run a learned hybrid model: da/dt = rate(a, s), with the hysteresis switching law.
    Simple explicit Euler steps of length h. A step that would carry the SMA past an edge stops
    exactly ON the edge (the switch happens there), so the step size does not bias the timing.
    """
    a = np.empty(n_steps)
    s = np.empty(n_steps, dtype=int)
    a[0], s[0] = a0, s0
    for k in range(1, n_steps):
        state = s[k - 1]
        if a[k - 1] <= lower:
            state = 1
        elif a[k - 1] >= upper:
            state = 0
        s[k] = state
        a[k] = a[k - 1] + h * rate(a[k - 1], state)
        if state == 1 and a[k] > upper:
            a[k] = upper
        elif state == 0 and a[k] < lower:
            a[k] = lower
    return a, s


def sindy_rate(coefficients, a_ref):
    """da/dt = library(a, s) . coefficients."""
    def rate(a, s):
        return float((library(np.array([a]), np.array([float(s)]), a_ref) @ coefficients)[0])
    return rate


def random_forest_rate(data: SmaData, train_mask, grid):
    """
    Black-box comparison: one random forest PER CLUSTER learns da/dt from a alone (no physics library),
    tabulated on a grid of a for speed. Each leaf must average about 1% of its cluster's samples, so
    the forest averages away the derivative noise (~15x the coast rate) instead of fitting it.
    """
    tables = {}
    for state, mask in ((0, data.coast), (1, data.burn)):
        rows = mask & train_mask
        X = ((data.a[rows] - grid.mean()) * 1e3)[:, None]
        leaf = max(5, rows.sum() // 100)
        forest = RandomForestRegressor(n_estimators=100, min_samples_leaf=leaf, random_state=0).fit(X, data.a_dot[rows])
        tables[state] = forest.predict(((grid - grid.mean()) * 1e3)[:, None])
    return lambda a, s: float(np.interp(a, grid, tables[s]))


def onset_times(t_s, s):
    """Times [s] where a 0/1 sequence switches on."""
    return t_s[1:][np.diff(s) == 1]


def match_onsets(predicted, actual):
    """Timing error [s] of each actual onset against the nearest predicted onset (NaN if none)."""
    if len(predicted) == 0:
        return np.full(len(actual), np.nan)
    return np.array([predicted[np.argmin(np.abs(predicted - t))] - t for t in actual])

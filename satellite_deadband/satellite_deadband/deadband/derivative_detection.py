"""
Week 5: burn detection from finite-difference derivatives of the raw r, v data.

The idea
--------
Tracking data gives positions r (and maybe velocities v) at a fixed cadence h. Differencing
them gives the acceleration the satellite actually felt:

    from positions  ("rdot space"):  r'' ~ (r[k+1] - 2 r[k] + r[k-1]) / h^2
    from velocities ("vdot space"):  v'  ~ (v[k+1] - v[k-1]) / (2 h)

Gravity and J2 are known, so subtracting them leaves the UNMODELLED acceleration:
drag (~1e-6 m/s^2) while coasting, drag + thrust (~2e-4 m/s^2) during a burn. That is a
jump of two orders of magnitude, so a 2-component Gaussian mixture on log |residual| splits
the samples into coast and burn. Unlike the Week 4 SMA-jump features this is not fooled by
J2, which makes the osculating SMA swing by ~12 km per orbit.

The finite differences have a truncation error that grows like h^2 (h^2/12 for r'', h^2/6
for v'), so the cadence matters: `cadence_sweep` measures where each space stops working.

From flagged samples to burn EVENTS
-----------------------------------
A burn lasts ~23 minutes, i.e. hundreds of samples, so the GMM's per-sample flags are turned
into events before anything is scored:
    1. gaps of up to MAX_GAP_SAMPLES unflagged samples inside a run are filled in,
    2. runs shorter than MIN_EVENT_SAMPLES are thrown away (an isolated point is not a burn).

Scoring is EVENT-level only: how many true burns were flagged (recall), how many detected
events were real burns (precision), plus coverage and start/end timing for the found ones.
A detected event counts for a burn if it overlaps it, widened by a timing `tolerance`.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture

from .constants import ACCEL_UNIT_MS2, DU, EARTH_EQUATORIAL_RADIUS_KM, J2, MU, SECONDS, THRUST_ACCEL_MS2, TU
from .smooth_controller import SmoothDeadbandSimulation, throttle

FEATURE_FLOOR_MS2 = 1e-12   # added before taking logs so log(0) never happens
MAX_GAP_SAMPLES = 2         # unflagged samples inside a run that are filled in
MIN_EVENT_SAMPLES = 10      # a run must be at least this long (10 x 6 s = 1 min) to count as a burn


# ---------------------------------------------------------------------------
# Known (modelled) acceleration, vectorised over many samples
# ---------------------------------------------------------------------------
def known_accel(r, j2=J2):
    """Gravity + J2 acceleration for a (3, N) array of positions (non-dimensional)."""
    radius = np.linalg.norm(r, axis=0)
    earth_radius = EARTH_EQUATORIAL_RADIUS_KM / DU   # J2 reference radius
    z_ratio_sq = (r[2] / radius) ** 2
    factor = 1.5 * j2 * MU * earth_radius**2 / radius**4
    j2_part = factor * np.array([r[0] / radius * (5 * z_ratio_sq - 1),
                                 r[1] / radius * (5 * z_ratio_sq - 1),
                                 r[2] / radius * (5 * z_ratio_sq - 3)])
    return -MU * r / radius**3 + j2_part


# ---------------------------------------------------------------------------
# Measurements and residual accelerations
# ---------------------------------------------------------------------------
@dataclass
class DerivativeEvidence:
    """Everything a detector sees, at one sampling cadence. Accelerations are in m/s^2."""
    step_s: float                 # sampling cadence h [s]
    t: np.ndarray                 # (N,) sample times (non-dimensional)
    true_on: np.ndarray           # (N,) 1 where the thruster is more than half open
    true_thrust_ms2: np.ndarray   # (N,) true thrust acceleration magnitude
    accel_from_r: np.ndarray      # (3, N) unmodelled acceleration from the position 2nd difference
    accel_from_v: np.ndarray      # (3, N) unmodelled acceleration from the velocity central difference
    along_track: np.ndarray       # (3, N) unit velocity vectors (thrust direction)

    @property
    def log_accel_from_r(self):
        return np.log(np.linalg.norm(self.accel_from_r, axis=0) + FEATURE_FLOOR_MS2)

    @property
    def log_accel_from_v(self):
        return np.log(np.linalg.norm(self.accel_from_v, axis=0) + FEATURE_FLOOR_MS2)

    def features(self, space):
        """(N, k) feature matrix for 'rdot', 'vdot' or 'both'."""
        columns = {"rdot": [self.log_accel_from_r], "vdot": [self.log_accel_from_v],
                   "both": [self.log_accel_from_r, self.log_accel_from_v]}[space]
        return np.column_stack(columns)


def build_derivative_evidence(sim: SmoothDeadbandSimulation, step_s, t_start=None, t_end=None):
    """Sample the simulation every `step_s` seconds and finite-difference r and v."""
    h = step_s * SECONDS
    t_start = sim.t[0] if t_start is None else t_start
    t_end = sim.t[-1] if t_end is None else t_end
    times = np.arange(t_start, t_end, h)
    samples = sim.sample(times)
    r, v, s = samples[0:3], samples[3:6], samples[6]

    # interior samples only: each difference needs a neighbour on both sides
    r_second_difference = (r[:, 2:] - 2 * r[:, 1:-1] + r[:, :-2]) / h**2
    v_central_difference = (v[:, 2:] - v[:, :-2]) / (2 * h)
    modelled = known_accel(r[:, 1:-1])

    thrust = throttle(s[1:-1])
    return DerivativeEvidence(
        step_s=step_s, t=times[1:-1], true_on=(thrust > 0.5).astype(int),
        true_thrust_ms2=thrust * THRUST_ACCEL_MS2,
        accel_from_r=(r_second_difference - modelled) * ACCEL_UNIT_MS2,
        accel_from_v=(v_central_difference - modelled) * ACCEL_UNIT_MS2,
        along_track=v[:, 1:-1] / np.linalg.norm(v[:, 1:-1], axis=0))


# ---------------------------------------------------------------------------
# Gaussian mixture detector
# ---------------------------------------------------------------------------
@dataclass
class DerivativeGmmFit:
    space: str                # 'rdot', 'vdot' or 'both'
    model: GaussianMixture
    burn_component: int
    flagged: np.ndarray       # (N,) raw per-sample GMM output, 1 = burn-like sample
    predicted: np.ndarray     # (N,) after grouping into events: 1 = inside a detected burn event
    event_scores: dict        # events found, false events, precision / recall / F1, coverage, timing


def fit_derivative_gmm(evidence: DerivativeEvidence, space, tolerance_s=60.0):
    """2-component GMM on log |unmodelled acceleration| in the chosen space, grouped into events, then scored."""
    features = evidence.features(space)
    model = GaussianMixture(n_components=2, n_init=3, random_state=0, covariance_type="full").fit(features)
    burn = int(np.argmax(model.means_.sum(axis=1)))     # burns are the high-acceleration component
    coast = 1 - burn

    # A wide burn Gaussian also "wins" far out in the LOW tail of the coast cluster. Thrust can
    # only make the unmodelled acceleration bigger, so a burn must also lie above the coast mean.
    above_coast_mean = features.sum(axis=1) > model.means_[coast].sum()
    flagged = ((model.predict(features) == burn) & above_coast_mean).astype(int)
    predicted = group_into_events(flagged)
    return DerivativeGmmFit(space=space, model=model, burn_component=burn, flagged=flagged, predicted=predicted,
                            event_scores=event_scores(evidence.t, predicted, evidence.true_on,
                                                      evidence.step_s, tolerance_s))


# ---------------------------------------------------------------------------
# From flagged samples to events
# ---------------------------------------------------------------------------
def _runs(flags):
    """(first index, last index) of each contiguous run of 1s in `flags`."""
    padded = np.concatenate([[0], np.asarray(flags, dtype=int), [0]])
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return list(zip(starts, ends))


def group_into_events(flags, max_gap=MAX_GAP_SAMPLES, min_samples=MIN_EVENT_SAMPLES):
    """Fill short gaps inside runs, then drop runs shorter than `min_samples` (no single-point detections)."""
    events = np.asarray(flags, dtype=int).copy()
    for first, last in _runs(1 - events):                 # runs of 0s
        is_inside = first > 0 and last < len(events) - 1
        if is_inside and last - first + 1 <= max_gap:
            events[first:last + 1] = 1
    for first, last in _runs(events):
        if last - first + 1 < min_samples:
            events[first:last + 1] = 0
    return events


def runs_of_ones(t, flags):
    """(start, end) times of each contiguous run of 1s in `flags`."""
    return [(t[first], t[last]) for first, last in _runs(flags)]


# ---------------------------------------------------------------------------
# Event-level scoring
# ---------------------------------------------------------------------------
def event_scores(t, predicted, true_on, step_s, tolerance_s=60.0):
    """
    Score detected EVENTS against true burns (never individual samples).

    found            a true burn is found if a detected event overlaps it (+- tolerance)
    false events     detected events that overlap no true burn
    recall           found burns / true burns   ("how many true events were flagged")
    precision        detected events that are real burns / all detected events
    coverage         fraction of a found burn's duration inside a detected event
    start/end error  first/last detected sample minus the true switch-on/off [s]
    """
    tolerance = tolerance_s * SECONDS
    true_events = runs_of_ones(t, true_on)
    detected_events = runs_of_ones(t, predicted)

    def overlaps(detected, burn):
        return detected[0] <= burn[1] + tolerance and detected[1] >= burn[0] - tolerance

    per_burn = []
    for burn in true_events:
        matches = [d for d in detected_events if overlaps(d, burn)]
        inside = (t >= burn[0]) & (t <= burn[1])
        if matches:
            per_burn.append(dict(found=True, coverage=float(predicted[inside].mean()),
                                 start_error_s=float((matches[0][0] - burn[0]) * TU),
                                 end_error_s=float((matches[-1][1] - burn[1]) * TU)))
        else:
            per_burn.append(dict(found=False, coverage=0.0, start_error_s=np.nan, end_error_s=np.nan))

    n_found = sum(b["found"] for b in per_burn)
    n_real = sum(1 for d in detected_events if any(overlaps(d, burn) for burn in true_events))
    recall = n_found / len(true_events) if true_events else 0.0
    precision = n_real / len(detected_events) if detected_events else 0.0
    found = [b for b in per_burn if b["found"]]
    return dict(
        n_burns=len(true_events), n_found=n_found, n_detected_events=len(detected_events),
        n_false_events=len(detected_events) - n_real, recall=recall, precision=precision,
        f1=2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0,
        mean_coverage=float(np.mean([b["coverage"] for b in found])) if found else 0.0,
        mean_abs_start_error_s=float(np.mean([abs(b["start_error_s"]) for b in found])) if found else np.nan,
        mean_abs_end_error_s=float(np.mean([abs(b["end_error_s"]) for b in found])) if found else np.nan,
        per_burn=per_burn)


# ---------------------------------------------------------------------------
# Several ways to differentiate the data (comparison of derivative methods)
# ---------------------------------------------------------------------------
# Why this matters: with the 2nd-order central difference at h = 6 s the COASTING residual is
# not drag (~1e-6 m/s^2) but the method's own truncation error:
#     from v:  (h^2 / 6)  |v'''|  ~ (h^2 / 6)  n^3 |v|  ~ 6e-5 m/s^2
#     from r:  (h^2 / 12) |r''''| ~ (h^2 / 12) n^4 |r|  ~ 3e-5 m/s^2
# That error is only ~4x below the thrust (2e-4 m/s^2), so coast and burn sit close together
# in the GMM's feature. A higher-order stencil shrinks the error like h^4, h^6, h^8, which pushes
# the coast cluster down towards the real drag and separates the two clusters by far more.
# With measurement NOISE the story flips: higher orders amplify noise slightly, while the
# Savitzky-Golay filter (a least-squares polynomial fitted through a window) averages it away.
#
# Central-difference weights for points k-m ... k+m (divide by divisor * h for a 1st derivative,
# by divisor * h^2 for a 2nd derivative).
CENTRAL_FIRST = {2: ([-1, 0, 1], 2), 4: ([1, -8, 0, 8, -1], 12),
                 6: ([-1, 9, -45, 0, 45, -9, 1], 60), 8: ([3, -32, 168, -672, 0, 672, -168, 32, -3], 840)}
CENTRAL_SECOND = {2: ([1, -2, 1], 1), 4: ([-1, 16, -30, 16, -1], 12),
                  6: ([2, -27, 270, -490, 270, -27, 2], 180),
                  8: ([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9], 5040)}
SAVGOL_WINDOW = 21          # samples in the Savitzky-Golay window (2 minutes at 6 s)
SAVGOL_ORDER = 6            # degree of the polynomial fitted through the window
DERIVATIVE_METHODS = ("central 2nd order", "central 4th order", "central 6th order", "central 8th order",
                      "Savitzky-Golay")
TRIM = SAVGOL_WINDOW // 2   # every method is evaluated on the same interior samples (widest stencil)


def _central(signal, h, order, derivative):
    """Central difference of a (3, N) signal; returns (3, N) with NaN where the stencil does not fit."""
    weights, divisor = (CENTRAL_FIRST if derivative == 1 else CENTRAL_SECOND)[order]
    m = len(weights) // 2
    n = signal.shape[1]
    out = np.full(signal.shape, np.nan)
    out[:, m:n - m] = sum(w * signal[:, k:n - 2 * m + k] for k, w in enumerate(weights)) / (divisor * h**derivative)
    return out


def differentiate(signal, h, method, derivative=1):
    """1st (velocity -> acceleration) or 2nd (position -> acceleration) derivative of a (3, N) signal."""
    if method == "Savitzky-Golay":
        from scipy.signal import savgol_filter
        return savgol_filter(signal, SAVGOL_WINDOW, SAVGOL_ORDER, deriv=derivative, delta=h, axis=1)
    order = int(method.split()[1][0])           # "central 4th order" -> 4
    return _central(signal, h, order, derivative)


def method_evidence(sim: SmoothDeadbandSimulation, step_s, method, position_noise_m=0.0, velocity_noise_mms=0.0,
                    seed=0):
    """
    Same as build_derivative_evidence, but with any of DERIVATIVE_METHODS and optional measurement
    noise (Gaussian, position in metres, velocity in mm/s). r'' and v' use the same method family.
    """
    h = step_s * SECONDS
    times = np.arange(sim.t[0], sim.t[-1], h)
    samples = sim.sample(times)
    rng = np.random.default_rng(seed)
    r = samples[0:3] + position_noise_m / (DU * 1000) * rng.standard_normal((3, len(times)))
    v = samples[3:6] + velocity_noise_mms / (ACCEL_UNIT_MS2 * TU * 1000) * rng.standard_normal((3, len(times)))
    keep = slice(TRIM, len(times) - TRIM)
    modelled = known_accel(r[:, keep])
    thrust = throttle(samples[6, keep])
    v_true = samples[3:6, keep]
    return DerivativeEvidence(
        step_s=step_s, t=times[keep], true_on=(thrust > 0.5).astype(int), true_thrust_ms2=thrust * THRUST_ACCEL_MS2,
        accel_from_r=(differentiate(r, h, method, 2)[:, keep] - modelled) * ACCEL_UNIT_MS2,
        accel_from_v=(differentiate(v, h, method, 1)[:, keep] - modelled) * ACCEL_UNIT_MS2,
        along_track=v_true / np.linalg.norm(v_true, axis=0))


ALONG_TRACK_UNIT_MS2 = 1e-4     # along-track feature is reported in units of 1e-4 m/s^2 (thrust = 2)
# the number of components of the method GMM is the tuned one (settings.tuned("gmm_components"))


def velocity_features(evidence: DerivativeEvidence):
    """
    (N, 2) features from VELOCITY data only:
        column 0: log |unmodelled acceleration from v'|   (how big)
        column 1: along-track part / 1e-4 m/s^2           (which way: thrust pushes FORWARDS,
                                                            truncation error and noise do not)
    """
    along = np.sum(evidence.accel_from_v * evidence.along_track, axis=0) / ALONG_TRACK_UNIT_MS2
    return np.column_stack([evidence.log_accel_from_v, along])


def burn_components(model, n_sigma=3.0):
    """
    Which mixture components are BURN? The heaviest component is coast. A component is burn if its
    mean log|v'| sits more than n_sigma coast standard deviations above the coast mean (thrust only
    ADDS acceleration). Components in between are extra pieces of coast (ramps, noise tail).
    """
    coast = int(np.argmax(model.weights_))
    coast_sd = np.sqrt(model.covariances_[coast][0, 0])
    return model.means_[:, 0] > model.means_[coast, 0] + n_sigma * coast_sd


def fit_method_gmm(evidence: DerivativeEvidence, n_components=None, tolerance_s=60.0):
    """
    GMM on the two velocity features, with `n_components` Gaussians. Why 4 and not 2: the
    thruster's tanh ramp passes through every level between drag and full thrust, and with a
    2-component mixture the burn Gaussian must stretch to cover those ramp samples, so it is no
    longer centred on the burn. With noise the coast cloud is also skewed (log of a noise vector),
    which uses up a second coast Gaussian. Two spare components absorb the ramps and the skew,
    and the burn Gaussian then sits exactly on the burn (clean: 3 is enough; noisy: 4 is needed).
    """
    from .settings import tuned
    n_components = n_components or tuned("gmm_components")
    features = velocity_features(evidence)
    model = GaussianMixture(n_components=n_components, n_init=3, random_state=0, covariance_type="full").fit(features)
    is_burn = burn_components(model)
    flagged = is_burn[model.predict(features)].astype(int)
    predicted = group_into_events(flagged)
    top = int(np.argmax(np.where(is_burn, model.means_[:, 0], -np.inf))) if is_burn.any() else int(np.argmax(model.means_[:, 0]))
    return DerivativeGmmFit(space="v' + along-track", model=model, burn_component=top, flagged=flagged,
                            predicted=predicted,
                            event_scores=event_scores(evidence.t, predicted, evidence.true_on, evidence.step_s, tolerance_s))


def cluster_separation(evidence: DerivativeEvidence, space="vdot"):
    """How far apart coast and burn are, in coast standard deviations: (burn median - coast median) / coast sd."""
    feature = evidence.features(space)[:, 0]
    coast, burn = feature[evidence.true_on == 0], feature[evidence.true_on == 1]
    return float((np.median(burn) - np.median(coast)) / np.std(coast))


# ---------------------------------------------------------------------------
# Cadence sweep: how fine must the time step be?  (not plotted in the Week 5 talk)
# ---------------------------------------------------------------------------
def cadence_sweep(sim, steps_s=(1, 2, 5, 10, 20, 30, 60), spaces=("rdot", "vdot"), tolerance_s=60.0):
    """Event-level GMM scores for each sampling step and each derivative space: {space: [row per step]}."""
    table = {space: [] for space in spaces}
    for step_s in steps_s:
        evidence = build_derivative_evidence(sim, step_s)
        for space in spaces:
            scores = fit_derivative_gmm(evidence, space, tolerance_s).event_scores
            table[space].append(dict(
                step_s=step_s, **{k: scores[k] for k in ("n_found", "n_burns", "n_false_events", "recall",
                                                         "precision", "f1", "mean_coverage")},
                coast_median_ms2=float(np.median(np.exp(evidence.features(space)[evidence.true_on == 0, 0]))),
            ))
    return table

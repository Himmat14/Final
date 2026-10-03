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

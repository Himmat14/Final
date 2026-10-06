"""
A mean-element model of the station-keeping satellite: only the mean SMA a(t) and the thruster
state s(t) (0 = coast, 1 = burn), the two quantities steps 8-9 showed the controller actually uses.

    da/dt = -k * w(t) * exp(-(a - a_ref) / H)  +  T * s          [km/s]
    s switches ON when a <= lower edge, OFF when a >= upper edge   (hysteresis: the controller's memory)

    k      coast decay rate at the reference SMA with average space weather   (drag)
    w(t)   space-weather factor: atmospheric density relative to its average (1 = constant drag)
    H      density scale height (60 km at 550 km)
    T      thrust contribution to da/dt (SMA gain per second while the thruster is on)

Why: the full 6-s orbit integration of the 400-day controlled run takes ~20 minutes for ONE scenario.
The extension steps (10-16) need many scenarios (variable drag, regimes, a 40-satellite fleet,
thousands of forecasts), so they use this model, CALIBRATED on the full controlled run (`calibrate`)
and checked against it in the tests (same burn period to <1%, same burn length).

Space weather (`space_weather`): w(t) = solar-rotation wave (27 days) + slow solar-cycle trend +
geomagnetic storms (fast rise, ~1.5-day decay, density x2-x4). The observable proxy (an F10.7 / Ap-like
daily index) is w averaged per day with 5% measurement noise.
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .constants import SMOOTH_A_LOWER_KM, SMOOTH_A_UPPER_KM

DAY_S = 86400.0
DT_S = 60.0                          # integration / sampling step of the mean-element model
SCALE_HEIGHT_KM = 60.0
SOLAR_ROTATION_DAYS = 27.0


# ---------------------------------------------------------------------------
# Calibration on the full 400-day controlled run
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def calibrate():
    """
    k [km/s], T [km/s], a_ref [km], band edges and burn period measured on the full controlled run:
    k = -median coast slope of the mean SMA, T = (SMA gained per burn) / (burn duration) + k.
    """
    from .derivative_detection import runs_of_ones
    from .long_run import controlled_run
    from .constants import TU
    run = controlled_run()
    t_s = run.t * TU
    sma, on = run.mean_sma_km, run.true_on
    burns = runs_of_ones(t_s, on)
    slopes = []
    for (_, end), (start, _) in zip(burns[:-1], burns[1:]):
        arc = (t_s > end + 600) & (t_s < start - 600)
        idx = np.flatnonzero(arc)[::50]
        slopes.append(np.polyfit(t_s[idx], sma[idx], 1)[0])
    k = -float(np.median(slopes))
    gains, durations, a_on, a_off = [], [], [], []
    for start, end in burns:
        i0, i1 = np.searchsorted(t_s, [start, end])
        i1 = min(i1, len(sma) - 1)
        gains.append(sma[i1] - sma[i0])
        durations.append(end - start)
        a_on.append(sma[i0]); a_off.append(sma[i1])
    starts = np.array([s for s, _ in burns])
    ends = np.array([e for _, e in burns])
    # EFFECTIVE band: the coast lasts from one burn's end to the next burn's start, so the SMA span it
    # covers is (coast rate) x (coast duration). Taking the edges as the SMA where the throttle crosses 1/2
    # would miss the thrust ramps (~3 m): the coast and burn durations of the model then match the full run.
    coast_s = float(np.median(starts[1:] - ends[:-1]))
    span = k * coast_s
    lower = float(np.median(a_on))
    T = span / float(np.median(durations)) + k
    return dict(k=k, T=T, a_ref=float(np.median(sma)), lower=lower, upper=lower + span,
                nominal_lower=SMOOTH_A_LOWER_KM, nominal_upper=SMOOTH_A_UPPER_KM,
                period_days=float(np.mean(np.diff(starts)) / DAY_S), burn_s=float(np.median(durations)))


# ---------------------------------------------------------------------------
# Space weather
# ---------------------------------------------------------------------------
@dataclass
class SpaceWeather:
    t: np.ndarray                    # (N,) time [s] on the DT_S grid
    w: np.ndarray                    # (N,) true density factor
    proxy: np.ndarray                # (N,) observable daily index (noisy daily mean of w), held per day
    storms: list = field(default_factory=list)    # storm onset days


def space_weather(days, seed=0, constant=False, n_storms=8, trend=0.4, rotation=0.25):
    t = np.arange(0, days * DAY_S, DT_S)
    if constant:
        w = np.ones_like(t)
        return SpaceWeather(t, w, w.copy(), [])
    rng = np.random.default_rng(seed)
    d = t / DAY_S
    w = 1 + rotation * np.sin(2 * np.pi * d / SOLAR_ROTATION_DAYS + rng.uniform(0, 2 * np.pi))
    w *= 1 + trend * (d / days - 0.5)                                 # slow solar-cycle change
    onsets = np.sort(rng.uniform(5, days - 5, n_storms))
    for onset in onsets:                                              # storms: fast rise, slow decay
        amplitude = rng.uniform(1.0, 3.0)
        lag = d - onset
        lag = np.maximum(lag, 0.0)
        w *= 1 + amplitude * (1 - np.exp(-lag / 0.1)) * np.exp(-lag / 1.5)
    w /= np.mean(w)                                                   # average density = calibration value
    per_day = (t // DAY_S).astype(int)
    daily = np.bincount(per_day, weights=w) / np.bincount(per_day)
    daily *= 1 + 0.05 * rng.standard_normal(daily.size)
    return SpaceWeather(t, w, daily[per_day], list(onsets))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
@dataclass
class MeanRun:
    t: np.ndarray                    # (N,) time [s]
    a: np.ndarray                    # (N,) true mean SMA [km]
    s: np.ndarray                    # (N,) thruster state 0 / 1
    weather: SpaceWeather
    onsets: np.ndarray               # (B,) burn start times [s]
    ends: np.ndarray                 # (B,) burn end times [s]
    labels: list = field(default_factory=list)    # per burn: regime label (multi-regime runs only)

    @property
    def days(self):
        return self.t / DAY_S


def simulate(weather, k=None, T=None, lower=None, upper=None, a0=None, schedule=None):
    """
    Integrate the mean-element model on the weather grid (explicit Euler, 60 s, exact enough: the
    largest change per step is 22 m during a burn, versus a 500 m band).
    `schedule(i, t, a, s) -> (lower, upper, forced)`: optional per-step override of the band and a
    forced thruster state (None = normal hysteresis), used by the multi-regime runs.
    """
    c = calibrate()
    k, T = k or c["k"], T or c["T"]
    lower, upper = lower or c["lower"], upper or c["upper"]
    a_ref = c["a_ref"]
    n = len(weather.t)
    a = np.empty(n)
    s = np.zeros(n, dtype=np.int8)
    a[0] = a0 if a0 is not None else upper - 0.05
    state = 0
    for i in range(1, n):
        lo, up, forced = (lower, upper, None) if schedule is None else schedule(i, weather.t[i], a[i - 1], state)
        if forced is not None:
            state = forced
        elif state == 0 and a[i - 1] <= lo:
            state = 1
        elif state == 1 and a[i - 1] >= up:
            state = 0
        s[i] = state
        rate = -k * weather.w[i] * np.exp(-(a[i - 1] - a_ref) / SCALE_HEIGHT_KM) + T * state
        a[i] = a[i - 1] + rate * DT_S
        if forced is None:                     # a step that would cross an edge stops ON the edge
            if state == 1 and a[i] > up:
                a[i] = up
            elif state == 0 and a[i] < lo:
                a[i] = lo
    edges = np.diff(s.astype(int))
    onsets = weather.t[1:][edges == 1]
    ends = weather.t[1:][edges == -1]
    ends = ends[ends > onsets[0]] if onsets.size else ends
    return MeanRun(t=weather.t, a=a, s=s, weather=weather, onsets=onsets, ends=ends[:len(onsets)])


def measure(run, every_s=600.0, sigma_km=0.02, seed=0):
    """(times [s], measured SMA [km]): the mean SMA every `every_s` seconds with Gaussian noise."""
    stride = max(1, int(round(every_s / DT_S)))
    rng = np.random.default_rng(seed)
    t = run.t[::stride]
    return t, run.a[::stride] + sigma_km * rng.standard_normal(t.size)


def detect_burns(t, a_meas, sigma_km, span=1):
    """
    Burns from measured SMA: a sample-to-sample rise much larger than the noise (drag only ever
    LOWERS the SMA). Returns (onsets, ends, onset SMAs, end SMAs) of the grouped rises.
    The first and last sampling intervals of a burn are only PARTLY thrusting, so they rise less and are
    often below the threshold. The start and end are therefore placed where the fitted coast line
    (6 h before / after) meets the fitted burn line, which removes that bias (tens of metres otherwise).
    span > 1 compares samples `span` apart, so a short burn split across two intervals is still seen.
    """
    rise = a_meas[span:] - a_meas[:-span]
    flags = rise > 5 * sigma_km * np.sqrt(2)
    groups, i = [], 0
    while i < flags.size:
        if flags[i]:
            j = i
            while j + 1 < flags.size and flags[j + 1]:
                j += 1
            groups.append((i, j + span))                    # samples i .. j+span bracket the clearly-burning part
            i = j + 1
        else:
            i += 1
    if not groups:
        empty = np.array([])
        return empty, empty, empty, empty
    step = t[1] - t[0]
    def burn_line(i, j):
        """Slope and anchor of the burn: from the steepest single-step rises in the bracket (fully thrusting)."""
        steps = np.diff(a_meas[i:j + 1])
        k = int(np.argmax(steps))
        steep = steps[steps >= 0.5 * steps[k]]
        return np.mean(steep) / step, 0.5 * (t[i + k] + t[i + k + 1]), 0.5 * (a_meas[i + k] + a_meas[i + k + 1])

    onsets, ends, a_on, a_off = [], [], [], []
    for i, j in groups:
        burn_slope, burn_mid_t, burn_mid_a = burn_line(i, j)
        for side in ("before", "after"):
            if side == "before":
                keep = (t < t[i] - step) & (t > t[i] - 6 * 3600)
            else:
                keep = (t > t[j] + step) & (t < t[j] + 6 * 3600)
            if keep.sum() >= 5:
                slope, intercept = np.polyfit(t[keep] - burn_mid_t, a_meas[keep], 1)
                # coast line a = intercept + slope x; burn line a = burn_mid_a + burn_slope x
                x = (intercept - burn_mid_a) / (burn_slope - slope)
                edge_t, edge_a = burn_mid_t + x, intercept + slope * x
            else:
                edge_t, edge_a = (t[i], a_meas[i]) if side == "before" else (t[j], a_meas[j])
            if side == "before":
                onsets.append(edge_t); a_on.append(edge_a)
            else:
                ends.append(edge_t); a_off.append(edge_a)
    return np.array(onsets), np.array(ends), np.array(a_on), np.array(a_off)

"""
The long (400-day) simulations every report step draws its data from.

Why 400 days: long enough to compare methods at horizons of 15, 30, 60, 120, 240 and 360 days.
Why the realistic drag (SMOOTH_CD, ~0.13 km/day decay) everywhere: the Week 2-4 demo drag
(CD_TRUE, ~45 km/day) would bring the satellite down in about 12 days.

Four runs, each computed once and cached on disk in .cache/ (the first run takes several minutes):

    natural_run()      no control, every force (gravity, drag, J2, Moon, Sun, SRP), sampled every 1 min
                       -> steps 1-4 (regression, noise studies, spectra, Kalman filter)
    divergence_runs()  the same orbit with gravity + drag only, and with each extra force added
                       on its own, sampled every 1 min  -> step 1 (what each force does over a year)
    controlled_run()   the smooth tanh deadband (500 m band, J2 + drag + thrust), sampled every
                       0.1 min (6 s)  -> steps 5-9 (deadband, classifiers, stacking, SINDy)
    drag_only_run()    the controlled orbit's twin with no thruster, sampled every 1 min -> step 5

Steps 6-9 learn from the first TRAIN_DAYS of the controlled run and are scored on the
following 360 days, so every metric can be reported at each horizon in HORIZONS_DAYS.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .constants import (DU, SAMPLE_STEP_S, SECONDS, SMOOTH_CD, SMOOTH_DAYS, THRUST_ACCEL_MS2, from_days,
                        from_minutes)
from .perturbations import FORCE_NAMES, propagate_full
from .physics import propagate
from .smooth_controller import mean_sma_km_series, simulate_smooth_deadband, smooth_initial_state, throttle

LONG_DAYS = 400
HORIZONS_DAYS = (15, 30, 60, 120, 240, 360)
TRAIN_DAYS = 40                      # steps 6-9: learn from days 0-40, evaluate on days 40-400
NATURAL_STEP_MIN = 1.0
DIVERGENCE_STEP_MIN = 1.0           # 1 min so the runs can also draw ground tracks
CHUNK_DAYS = SMOOTH_DAYS             # the controlled run is integrated in 10-day pieces (keeps memory low);
                                     # the first piece IS the Week 5 run (smooth_controller.default_runs)
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
CACHE_VERSION = "v2"                 # bump when the physics changes (v2: J2 uses the equatorial radius)


def _cached(name, build):
    """Load `name` from the disk cache, or build it with build() -> dict of arrays and save it."""
    path = CACHE_DIR / f"{name}_{CACHE_VERSION}.npz"
    if path.exists():
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    arrays = build()
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(path, **arrays)
    return arrays


# ---------------------------------------------------------------------------
# Uncontrolled runs (steps 1-4)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def natural_run():
    """(t, states): every force, realistic drag, 400 days, one sample per minute."""
    def build():
        t = np.arange(0, from_days(LONG_DAYS), from_minutes(NATURAL_STEP_MIN))
        return dict(t=t, states=propagate_full(t, cd=SMOOTH_CD, forces=FORCE_NAMES))
    data = _cached("natural_run", build)
    return data["t"], data["states"]


@lru_cache(maxsize=1)
def divergence_runs():
    """(t, {name: positions}) for 'baseline' (gravity + drag), each extra force alone, and 'All combined'."""
    def build():
        t = np.arange(0, from_days(LONG_DAYS), from_minutes(DIVERGENCE_STEP_MIN))
        runs = {"baseline": propagate_full(t, cd=SMOOTH_CD, forces=())[0:3]}
        for name in FORCE_NAMES:
            runs[name] = propagate_full(t, cd=SMOOTH_CD, forces=(name,))[0:3]
        runs["All combined"] = propagate_full(t, cd=SMOOTH_CD, forces=FORCE_NAMES)[0:3]
        return dict(t=t, **runs)
    data = _cached(f"divergence_runs_{DIVERGENCE_STEP_MIN:g}min", build)
    t = data.pop("t")
    return t, data


# ---------------------------------------------------------------------------
# Controlled runs (steps 5-9)
# ---------------------------------------------------------------------------
@dataclass
class ControlledRun:
    t: np.ndarray            # (N,) sample times, every SAMPLE_STEP_S (non-dimensional)
    r: np.ndarray            # (3, N) positions
    v: np.ndarray            # (3, N) velocities
    s: np.ndarray            # (N,) thruster on/off state
    mean_sma_km: np.ndarray  # (N,)

    @property
    def thrust_ms2(self):
        return throttle(self.s) * THRUST_ACCEL_MS2

    @property
    def true_on(self):
        return (throttle(self.s) > 0.5).astype(int)


@lru_cache(maxsize=1)
def controlled_run():
    """The 400-day smooth-deadband run, integrated in 10-day chunks, sampled every 0.1 min."""
    def build():
        step = SAMPLE_STEP_S * SECONDS
        chunk = from_days(CHUNK_DAYS)
        state = smooth_initial_state()
        times, samples = [], []
        for k in range(int(np.ceil(LONG_DAYS / CHUNK_DAYS))):
            # The dynamics do not depend on absolute time (no Moon/Sun here), so each chunk can
            # start its clock at zero; the samples are shifted to the right absolute time below.
            sim = simulate_smooth_deadband(chunk, state0=state)
            local = np.arange(0, chunk, step)
            times.append(local + k * chunk)
            samples.append(sim.sample(local))
            state = sim.states[:, -1]
        return dict(t=np.concatenate(times), states=np.concatenate(samples, axis=1))
    data = _cached("controlled_run", build)
    states = data["states"]
    return ControlledRun(t=data["t"], r=states[0:3], v=states[3:6], s=states[6],
                         mean_sma_km=mean_sma_km_series(states))


@lru_cache(maxsize=1)
def drag_only_run():
    """(t, states): the controlled orbit's starting state with no thruster, one sample per minute."""
    def build():
        t = np.arange(0, from_days(LONG_DAYS), from_minutes(NATURAL_STEP_MIN))
        from .constants import J2
        return dict(t=t, states=propagate(SMOOTH_CD, t, j2=J2, state0=smooth_initial_state()[0:6]))
    data = _cached("drag_only_run", build)
    return data["t"], data["states"]


def horizon_mask(t, start_days, horizon_days):
    """Samples with start_days <= time < start_days + horizon_days."""
    days = t / from_days(1)
    return (days >= start_days) & (days < start_days + horizon_days)


__all__ = ["LONG_DAYS", "HORIZONS_DAYS", "TRAIN_DAYS", "natural_run", "divergence_runs", "controlled_run",
           "drag_only_run", "horizon_mask", "DU"]

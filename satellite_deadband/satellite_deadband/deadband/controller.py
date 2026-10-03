"""
State-dependent deadband station-keeping controller (the Week 4 "Workstream B" fix).

How it works
------------
Drag only ever lowers the orbit, so the controller has ONE trigger: the osculating
semi-major axis decaying to the lower edge of the band. When that happens, two
tangential impulses are applied back-to-back at the same instant:

    1. raise : aim slightly ABOVE the upper edge (a deliberate overshoot)
    2. trim  : a small opposing burn that settles the orbit exactly on the upper edge

There is no burn-duration parameter and no timer anywhere: the only thing that ever
changes the controller's behaviour is the current osculating SMA.

Why not trigger the trim separately at the upper edge? The raise burn ends exactly on
the quantity that would trigger that second event, so a separate trim trigger would
produce dv = 0 by construction (a structural redundancy, not a solver bug).
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.integrate import solve_ivp

from .constants import (
    A_LOWER, A_UPPER, CD_TRUE, COAST_MAX_STEP, COAST_SAMPLE_SPACING,
    CONTROLLER_SOLVER_TOLERANCE, RAISE_OVERSHOOT_FRACTION,
)
from .physics import equations_of_motion, initial_state, osculating_sma, osculating_sma_km, speed_for_sma

MAX_BURN_EVENTS = 2000  # safety cap on the number of burn pairs in one run


@dataclass
class DeadbandSimulation:
    """Result of `simulate_deadband` (everything is in non-dimensional units)."""
    t: np.ndarray                # (N,) sample times
    states: np.ndarray           # (6, N) position and velocity
    mode: np.ndarray             # (N,) thruster history state: 1 at the raise sample, else 0
    burn_times: np.ndarray       # (n_burns,) instant of each raise + trim pair
    raise_dv: np.ndarray         # (n_burns,) raise impulse
    trim_dv: np.ndarray          # (n_burns,) trim impulse (opposing, so negative)

    @cached_property
    def sma_km(self):
        """Osculating SMA at every sample [km]."""
        return osculating_sma_km(self.states)

    @cached_property
    def burn_labels(self):
        """Ground truth: 1 for every sample stored at a burn instant, 0 for coasting."""
        labels = np.zeros(len(self.t), dtype=int)
        for burn_time in self.burn_times:
            labels[np.isclose(self.t, burn_time)] = 1
        return labels

    @property
    def n_burns(self):
        return len(self.burn_times)


def _lower_edge_event():
    """Solver event: osculating SMA crossing the lower band edge while decaying."""
    def event(t, state):
        return osculating_sma(state[0:3], state[3:6]) - A_LOWER
    event.terminal = True
    event.direction = -1
    return event


def _rescale_speed(state, new_speed):
    """Same position and flight direction, velocity magnitude replaced by `new_speed`."""
    velocity = state[3:6]
    return np.concatenate([state[0:3], velocity / np.linalg.norm(velocity) * new_speed])


def _apply_raise_and_trim(state):
    """Return (state after raise, state after trim, raise dv, trim dv) for a burn at `state`."""
    radius = np.linalg.norm(state[0:3])
    raise_target = A_UPPER + RAISE_OVERSHOOT_FRACTION * (A_UPPER - A_LOWER)
    speed_after_raise = speed_for_sma(radius, raise_target)
    speed_after_trim = speed_for_sma(radius, A_UPPER)

    raised = _rescale_speed(state, speed_after_raise)
    trimmed = _rescale_speed(raised, speed_after_trim)
    raise_dv = speed_after_raise - np.linalg.norm(state[3:6])
    trim_dv = speed_after_trim - speed_after_raise
    return raised, trimmed, raise_dv, trim_dv


def simulate_deadband(t_end, cd=CD_TRUE, state0=None, tolerance=CONTROLLER_SOLVER_TOLERANCE):
    """
    Simulate gravity + drag under the deadband controller until `t_end` (non-dimensional).

    Stored samples are evenly spaced within each coast segment. At every burn, three
    samples share the burn instant: the last coast state, the post-raise state, and the
    post-trim state; all three count as "burn" in `burn_labels`.
    """
    t, state = 0.0, (initial_state() if state0 is None else state0.copy())
    times, states, modes = [t], [state.copy()], [0]
    burn_times, raise_dvs, trim_dvs = [], [], []

    for _ in range(MAX_BURN_EVENTS):
        if t >= t_end:
            break

        # Coast (no thrust) until the lower edge is hit or time runs out.
        coast = solve_ivp(equations_of_motion(cd), [t, t_end], state,
                          method="DOP853", rtol=tolerance, atol=tolerance, events=[_lower_edge_event()],
                          dense_output=True, max_step=COAST_MAX_STEP)
        n_samples = max(2, int((coast.t[-1] - coast.t[0]) / COAST_SAMPLE_SPACING))
        for sample_time in np.linspace(coast.t[0], coast.t[-1], n_samples)[1:]:
            times.append(sample_time)
            states.append(coast.sol(sample_time).copy())
            modes.append(0)

        t, state = coast.t[-1], coast.y[:, -1]
        if t >= t_end or coast.t_events[0].size == 0:
            break

        # Burn: raise then trim, both at the same instant.
        raised, trimmed, raise_dv, trim_dv = _apply_raise_and_trim(state)
        times.extend([t, t])
        states.extend([raised, trimmed.copy()])
        modes.extend([1, 0])
        burn_times.append(t)
        raise_dvs.append(raise_dv)
        trim_dvs.append(trim_dv)
        state = trimmed

    return DeadbandSimulation(t=np.array(times), states=np.array(states).T, mode=np.array(modes),
                              burn_times=np.array(burn_times), raise_dv=np.array(raise_dvs),
                              trim_dv=np.array(trim_dvs))

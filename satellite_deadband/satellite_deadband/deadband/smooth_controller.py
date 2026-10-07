"""
Week 5: a SMOOTH deadband controller, with J2, integrated as one ordinary ODE.

Why change the Week 4 controller?
---------------------------------
Week 4 used an `if` statement (a solver event): when the SMA hits the lower edge, stop the
integrator, jump the velocity, restart. An `if` is a step ("jump") function, so the
right-hand side is discontinuous and the result depends on event handling rather than on
the dynamics. Here every switch is a tanh, so the right-hand side is smooth and the solver
shrinks its own time step automatically when the thruster is switching.

The state has 7 entries: [x, y, z, vx, vy, vz, s]
---------------------------------------------------
s is the thruster on/off state. It sits at 0 almost all the time and moves smoothly to 1
when the mean SMA reaches the lower edge, then stays at 1 until the mean SMA reaches the
upper edge, then goes back to 0. That memory is what makes it a DEADBAND (two edges,
hysteresis) rather than a thermostat with a single set point.

    ds/dt = ( on(a) * (1 - s)          push s up    while a is below the lower edge
            - off(a) * s               push s down  while a is above the upper edge
            + LATCH_GAIN * s(1-s)(2s-1)  hold s at whichever of 0 or 1 it is nearer
            ) / SWITCH_TIME

    on(a)  = rising_switch(A_LOWER - a)       0 inside the band, 1 below it
    off(a) = rising_switch(a - A_UPPER)       0 inside the band, 1 above it
    rising_switch(x) = (1 + tanh(x / width)) / 2

The slope of rising_switch is sech^2(x / width) / (2 * width): `width` (metres of SMA)
and SWITCH_TIME (seconds) set how fast the thruster switches. `stages/smooth_deadband_stage`
sweeps both.

The thrust itself is a_thrust = THRUST_ACCEL * throttle(s) along the velocity, where
throttle(s) = rising_switch(s - 0.9) is a steep tanh. The thruster only opens once s is past
0.9, i.e. after the latch has already committed to "on" (anything past 1/2 is pulled to 1).
If it opened earlier, the first bit of thrust would lift the SMA back out of the "on" zone
before s had committed, s would fall back to 0, and the burn would stop part-way.

Why a MEAN SMA?
---------------
With J2 the osculating SMA swings by ~12 km every orbit, far more than the 500 m band, so a
controller watching it would fire every orbit (J2 makes it unstable). The controller instead
watches the SMA computed from the total orbital energy INCLUDING the J2 potential energy.
J2 is a conservative force, so that energy -- and this "mean" SMA -- stays constant under J2
(to ~1 cm) and changes only through drag and thrust, which are exactly what the deadband is
meant to respond to.
"""
from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np
from scipy.integrate import solve_ivp

from .constants import (
    ACCEL_UNIT_MS2, DU, EARTH_EQUATORIAL_RADIUS_KM, J2, LATCH_GAIN, MU, SAMPLE_STEP_MIN, SECONDS, SMOOTH_A_LOWER,
    SMOOTH_A_START,
    SMOOTH_A_UPPER, SMOOTH_CD, SMOOTH_DAYS, SMOOTH_SOLVER, SMOOTH_SOLVER_TOLERANCE, SWITCH_SMA_WIDTH,
    SWITCH_TIME, THROTTLE_OPENS_AT, THROTTLE_WIDTH, THRUST_ACCEL, from_days,
)
from .physics import drag_accel, gravity_accel, initial_state, j2_accel, osculating_sma


# ---------------------------------------------------------------------------
# Mean SMA (energy including the J2 potential)
# ---------------------------------------------------------------------------
def j2_potential(r, j2=J2):
    """Potential energy per unit mass from J2 (its gradient is `physics.j2_accel`)."""
    radius = np.linalg.norm(r)
    earth_radius = EARTH_EQUATORIAL_RADIUS_KM / DU   # J2 reference radius
    return MU * j2 * earth_radius**2 / (2 * radius**3) * (3 * (r[2] / radius) ** 2 - 1)


def mean_sma(r, v, j2=J2):
    """SMA from the total energy (kinetic + point-mass + J2 potential). Constant under J2."""
    energy = np.dot(v, v) / 2 - MU / np.linalg.norm(r) + j2_potential(r, j2)
    return -MU / (2 * energy)


def mean_sma_km_series(states, j2=J2):
    """Mean SMA [km] for every column of a (>=6, N) state array (vectorised version of `mean_sma`)."""
    r, v = states[0:3], states[3:6]
    radius = np.linalg.norm(r, axis=0)
    earth_radius = EARTH_EQUATORIAL_RADIUS_KM / DU   # J2 reference radius
    j2_energy = MU * j2 * earth_radius**2 / (2 * radius**3) * (3 * (r[2] / radius) ** 2 - 1)
    energy = np.sum(v * v, axis=0) / 2 - MU / radius + j2_energy
    return -MU / (2 * energy) * DU


def osculating_sma_km_series(states):
    """Osculating (two-body) SMA [km] for every column of a (>=6, N) state array."""
    r, v = states[0:3], states[3:6]
    energy = np.sum(v * v, axis=0) / 2 - MU / np.linalg.norm(r, axis=0)
    return -MU / (2 * energy) * DU


# ---------------------------------------------------------------------------
# The tanh switches
# ---------------------------------------------------------------------------
def rising_switch(x, width):
    """Smooth step: ~0 for x << -width, 1/2 at x = 0, ~1 for x >> width."""
    return 0.5 * (1 + np.tanh(x / width))


def rising_switch_slope(x, width):
    """d/dx of `rising_switch`: sech^2(x / width) / (2 * width). Large slope = fast switch."""
    return 0.5 / width / np.cosh(x / width) ** 2


def throttle(s):
    """How open the thruster is (0..1) for thruster state s: opens steeply as s passes THROTTLE_OPENS_AT."""
    return rising_switch(s - THROTTLE_OPENS_AT, THROTTLE_WIDTH)


def a_thrust(v, s, thrust_accel=THRUST_ACCEL):
    """Thrust acceleration: along the velocity, magnitude THRUST_ACCEL * throttle(s)."""
    return thrust_accel * throttle(s) * v / np.linalg.norm(v)


def thruster_state_rate(sma, s, sma_width=SWITCH_SMA_WIDTH, switch_time=SWITCH_TIME,
                        a_lower=SMOOTH_A_LOWER, a_upper=SMOOTH_A_UPPER):
    """ds/dt for the thruster on/off state s (see the module docstring)."""
    turn_on = rising_switch(a_lower - sma, sma_width)    # 1 once the SMA is below the lower edge
    turn_off = rising_switch(sma - a_upper, sma_width)   # 1 once the SMA is above the upper edge
    latch = LATCH_GAIN * s * (1 - s) * (2 * s - 1)              # holds s at 0 or 1 inside the band
    return (turn_on * (1 - s) - turn_off * s + latch) / switch_time


# ---------------------------------------------------------------------------
# Equations of motion: [r, v, s]
# ---------------------------------------------------------------------------
def smooth_deadband_rhs(cd=SMOOTH_CD, j2=J2, controlled=True, use_mean_sma=True,
                        sma_width=SWITCH_SMA_WIDTH, switch_time=SWITCH_TIME,
                        a_lower=SMOOTH_A_LOWER, a_upper=SMOOTH_A_UPPER, extra_forces=(), t_offset=0.0):
    """
    Right-hand side f(t, [r, v, s]) for scipy's solve_ivp.

    controlled=False gives the drag-only reference orbit (s stays 0, no thrust).
    use_mean_sma=False makes the controller watch the osculating SMA instead (the J2 failure case).
    extra_forces: any of "Moon", "Sun", "SRP" (perturbations.py), evaluated at the absolute time
    t + t_offset (so a run integrated in chunks keeps the right Moon and Sun positions).
    The controller still watches the J2 mean SMA: an operator does not need the third bodies to keep a band.
    """
    from .perturbations import MU_MOON, MU_SUN, moon_position, srp_accel, sun_position, third_body_accel

    def rhs(t, state):
        r, v, s = state[0:3], state[3:6], state[6]
        accel = gravity_accel(r) + drag_accel(v, cd) + j2_accel(r, j2)
        if extra_forces:
            clock = t + t_offset
            if "Moon" in extra_forces:
                accel = accel + third_body_accel(r, moon_position(clock), MU_MOON)
            if "Sun" in extra_forces or "SRP" in extra_forces:
                r_sun = sun_position(clock)
                if "Sun" in extra_forces:
                    accel = accel + third_body_accel(r, r_sun, MU_SUN)
                if "SRP" in extra_forces:
                    accel = accel + srp_accel(r, r_sun)
        ds_dt = 0.0
        if controlled:   # a fixed choice of model, not a switch inside the dynamics
            accel = accel + a_thrust(v, s)
            sma = mean_sma(r, v, j2) if use_mean_sma else osculating_sma(r, v)
            ds_dt = thruster_state_rate(sma, s, sma_width, switch_time, a_lower, a_upper)
        return np.concatenate([v, accel, [ds_dt]])
    return rhs


def smooth_initial_state(target_mean_sma=SMOOTH_A_START, j2=J2):
    """Week 4 initial position/direction, speed set so the MEAN SMA starts at `target_mean_sma`; s = 0."""
    state = initial_state()
    r, v = state[0:3], state[3:6]
    target_energy = -MU / (2 * target_mean_sma)
    speed = np.sqrt(2 * (target_energy + MU / np.linalg.norm(r) - j2_potential(r, j2)))
    return np.concatenate([r, v / np.linalg.norm(v) * speed, [0.0]])


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------
@dataclass
class SmoothDeadbandSimulation:
    """Result of `simulate_smooth_deadband` (non-dimensional units unless the name says otherwise)."""
    t: np.ndarray          # (N,) the solver's own steps
    states: np.ndarray     # (7, N) [r, v, s] at those steps
    solution: object       # scipy OdeSolution: continuous solution, call it with any times
    n_rhs_evaluations: int
    solver: str

    @cached_property
    def mean_sma_km(self):
        return np.array([mean_sma(self.states[0:3, i], self.states[3:6, i]) for i in range(len(self.t))]) * DU

    @cached_property
    def osculating_sma_km(self):
        return np.array([osculating_sma(self.states[0:3, i], self.states[3:6, i]) for i in range(len(self.t))]) * DU

    @property
    def thruster_state(self):
        return self.states[6]

    @property
    def throttle(self):
        return throttle(self.states[6])

    @property
    def thrust_accel_ms2(self):
        return THRUST_ACCEL * self.throttle * ACCEL_UNIT_MS2

    @property
    def step_sizes(self):
        """Time step the solver chose at every step (non-dimensional)."""
        return np.diff(self.t)

    def sample(self, times):
        """(7, len(times)) states at arbitrary times, from the continuous solution."""
        return self.solution(times)

    def sample_times(self, step=SAMPLE_STEP_MIN * 60 * SECONDS):
        """Evenly spaced times over the whole run, every SAMPLE_STEP_MIN minutes by default."""
        return np.arange(self.t[0], self.t[-1], step)

    def burn_intervals(self, resolution=None):
        """List of (start, end) times when the throttle is more than half open."""
        times = self.t if resolution is None else np.arange(self.t[0], self.t[-1], resolution)
        on = throttle(self.sample(times)[6]) > 0.5
        edges = np.flatnonzero(np.diff(on.astype(int)))
        starts = list(times[edges[on[edges + 1]] + 1])
        ends = list(times[edges[~on[edges + 1]] + 1])
        if on[0]:
            starts.insert(0, times[0])
        if on[-1]:
            ends.append(times[-1])
        return list(zip(starts, ends))


def simulate_smooth_deadband(t_end, cd=SMOOTH_CD, j2=J2, controlled=True, use_mean_sma=True,
                             sma_width=SWITCH_SMA_WIDTH, switch_time=SWITCH_TIME, state0=None,
                             method=SMOOTH_SOLVER, tolerance=SMOOTH_SOLVER_TOLERANCE,
                             a_lower=SMOOTH_A_LOWER, a_upper=SMOOTH_A_UPPER, extra_forces=(), t_offset=0.0):
    """Integrate [r, v, s] from 0 to `t_end` (non-dimensional) in a single solve_ivp call. No events."""
    state0 = smooth_initial_state(j2=j2) if state0 is None else state0
    rhs = smooth_deadband_rhs(cd, j2, controlled, use_mean_sma, sma_width, switch_time, a_lower, a_upper,
                              extra_forces, t_offset)
    result = solve_ivp(rhs, [0.0, t_end], state0, method=method, rtol=tolerance, atol=tolerance,
                       dense_output=True)
    if not result.success:
        raise RuntimeError(f"smooth deadband integration failed: {result.message}")
    return SmoothDeadbandSimulation(t=result.t, states=result.y, solution=result.sol,
                                    n_rhs_evaluations=result.nfev, solver=method)


@lru_cache(maxsize=1)
def default_runs():
    """The 8-day controlled run and its drag-only twin. Cached: several stages share them."""
    t_end = from_days(SMOOTH_DAYS)
    return simulate_smooth_deadband(t_end), simulate_smooth_deadband(t_end, controlled=False)

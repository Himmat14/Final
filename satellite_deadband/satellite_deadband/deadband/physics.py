"""Orbit dynamics: accelerations, orbital elements, and free (uncontrolled) propagation."""
import numpy as np
from scipy.integrate import solve_ivp

from .constants import (
    ARGUMENT_OF_PERIGEE, DU, ECCENTRICITY, EARTH_MU_KM3_S2, EARTH_EQUATORIAL_RADIUS_KM, INCLINATION,
    INITIAL_TRUE_ANOMALY, J2, MU, RAAN, SEMI_MAJOR_AXIS_KM, SOLVER_TOLERANCE, VU,
)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------
def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def _rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def initial_state():
    """Non-dimensional [x, y, z, vx, vy, vz] at the start of the reference orbit."""
    a, e, nu = SEMI_MAJOR_AXIS_KM, ECCENTRICITY, INITIAL_TRUE_ANOMALY
    semi_latus_rectum = a * (1 - e**2)

    # position and velocity in the perifocal frame
    r_perifocal = semi_latus_rectum / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0])
    v_perifocal = np.sqrt(EARTH_MU_KM3_S2 / semi_latus_rectum) * np.array([-np.sin(nu), e + np.cos(nu), 0])

    # rotate into the inertial frame
    to_inertial = _rotation_z(-RAAN) @ _rotation_x(-INCLINATION) @ _rotation_z(-ARGUMENT_OF_PERIGEE)
    return np.concatenate([to_inertial @ r_perifocal / DU, to_inertial @ v_perifocal / VU])


# ---------------------------------------------------------------------------
# Accelerations (single position / velocity vectors, non-dimensional)
# ---------------------------------------------------------------------------
def gravity_accel(r):
    """Two-body gravity."""
    return -MU * r / np.linalg.norm(r) ** 3


def drag_accel(v, cd):
    """Quadratic drag opposing the velocity: -cd * |v| * v."""
    return -cd * np.linalg.norm(v) * v


def j2_accel(r, j2=J2):
    """Acceleration from Earth's oblateness (J2)."""
    radius = np.linalg.norm(r)
    x, y, z = r
    earth_radius = EARTH_EQUATORIAL_RADIUS_KM / DU   # J2 reference radius
    factor = 1.5 * j2 * MU * earth_radius**2 / radius**4
    z_ratio_sq = (z / radius) ** 2
    return factor * np.array([
        x / radius * (5 * z_ratio_sq - 1),
        y / radius * (5 * z_ratio_sq - 1),
        z / radius * (5 * z_ratio_sq - 3),
    ])


# ---------------------------------------------------------------------------
# Orbital elements
# ---------------------------------------------------------------------------
def osculating_sma(r, v):
    """Osculating semi-major axis (energy method): a = -mu / (2 * specific energy)."""
    energy = np.dot(v, v) / 2 - MU / np.linalg.norm(r)
    return -MU / (2 * energy)


def osculating_sma_series(states):
    """Osculating SMA for every column of a (6, N) state history (non-dimensional)."""
    return np.array([osculating_sma(states[0:3, i], states[3:6, i]) for i in range(states.shape[1])])


def osculating_sma_km(states):
    """Osculating SMA in km for a (6, N) state history."""
    return osculating_sma_series(states) * DU


def speed_for_sma(radius, target_sma):
    """Vis-viva: the speed that puts the orbit at `target_sma` when at `radius`."""
    return np.sqrt(max(2 * (-MU / (2 * target_sma) + MU / radius), 0))


# ---------------------------------------------------------------------------
# Equations of motion and propagation
# ---------------------------------------------------------------------------
def state_derivative(state, cd, j2=0.0):
    """d/dt of [r, v]: gravity + drag, plus J2 when `j2` is non-zero."""
    r, v = state[0:3], state[3:6]
    accel = gravity_accel(r) + drag_accel(v, cd)
    if j2:
        accel = accel + j2_accel(r, j2)
    return np.concatenate([v, accel])


def equations_of_motion(cd, j2=0.0):
    """Right-hand side f(t, state) in the form scipy's ODE solvers expect."""
    return lambda t, state: state_derivative(state, cd, j2)


def propagate(cd, t_eval, j2=0.0, state0=None, tolerance=SOLVER_TOLERANCE):
    """Integrate gravity + drag (+ J2) and return the (6, N) states at `t_eval`."""
    if state0 is None:
        state0 = initial_state()
    solution = solve_ivp(equations_of_motion(cd, j2), [t_eval[0], t_eval[-1]], state0,
                         t_eval=t_eval, method="DOP853", rtol=tolerance, atol=tolerance)
    return solution.y

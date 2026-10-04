"""
Steps 2-3 of the report: learn the coefficients of the equations of motion from POSITION data.

The idea (Week 2 deck, Steps 6-10)
-----------------------------------
1. Measure positions r(t_i) every h (plus noise).
2. Estimate velocity and acceleration by finite differences (FD): a weighted sum of
   neighbouring positions divided by h (or h^2).
3. The model is LINEAR in each unknown coefficient, so one least-squares solve finds it:
       mu:   a = mu * (-r / |r|^3)                  ->  mu_hat = (T'T)^-1 T'a,  T = -r/|r|^3
       cd:   a - gravity = cd * (-|v| v)            ->  same formula with T = -|v| v
       joint: a - gravity = [cd, J2, mu_moon, mu_sun, P*CR] . [shape columns]
4. The ENERGY method (Step 10) avoids the second derivative altogether:
       a_sma = -mu / (2 * energy),   d(a_sma)/dt = -(2 a^2 / mu) cd |v|^3
   so cd comes from the SLOPE of a straight line fitted through the SMA history.
"""
import numpy as np

from .constants import DU, EARTH_EQUATORIAL_RADIUS_KM, J2, MU
from .perturbations import (MOON_DISTANCE, MU_MOON, MU_SUN, SRP_ACCEL, moon_position, srp_accel, sun_position,
                            third_body_accel)
from .physics import gravity_accel, j2_accel
from .settings import tuned

# ---------------------------------------------------------------------------
# Finite-difference stencils (central differences, weights for points i-k ... i+k)
# ---------------------------------------------------------------------------
FIRST_DERIVATIVE = {      # order: (weights, divisor)  ->  v_i = sum(w * r) / (divisor * h)
    2: (np.array([-1, 0, 1]), 2),
    4: (np.array([1, -8, 0, 8, -1]), 12),
    6: (np.array([-1, 9, -45, 0, 45, -9, 1]), 60),
    8: (np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]), 840),
}
SECOND_DERIVATIVE = {     # order: (weights, divisor)  ->  a_i = sum(w * r) / (divisor * h^2)
    2: (np.array([1, -2, 1]), 1),
    4: (np.array([-1, 16, -30, 16, -1]), 12),
    6: (np.array([2, -27, 270, -490, 270, -27, 2]), 180),
    8: (np.array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]), 5040),
}


def apply_stencil(signal, weights, divisor, h, power):
    """
    Slide a stencil along the LAST axis of `signal` (shape (3, N)).
    Returns the derivative at the interior points (N - len(weights) + 1 of them).
    """
    n = signal.shape[-1]
    width = len(weights)
    out = np.zeros(signal.shape[:-1] + (n - width + 1,))
    for k, w in enumerate(weights):
        out += w * signal[..., k:n - width + 1 + k]
    return out / (divisor * h**power)


def fd_velocity_and_acceleration(positions, h, order=None):
    """
    FD velocity and acceleration from (3, N) positions; both trimmed to the same interior points.
    `order` defaults to the stencil order chosen by the bias-variance tuning (settings.tuned("fd_order")).
    """
    order = order or tuned("fd_order")
    weights_v, div_v = FIRST_DERIVATIVE[order]
    weights_a, div_a = SECOND_DERIVATIVE[order]
    velocity = apply_stencil(positions, weights_v, div_v, h, 1)
    acceleration = apply_stencil(positions, weights_a, div_a, h, 2)
    half = len(weights_a) // 2
    return positions[:, half:-half], velocity, acceleration


# ---------------------------------------------------------------------------
# Least-squares fits (each unknown multiplies a known "shape" column)
# ---------------------------------------------------------------------------
def least_squares(columns, target):
    """Solve target ~ sum_k c_k * columns[k] for the coefficients c (all 3-vector components stacked)."""
    theta = np.column_stack([np.ravel(c) for c in columns])   # one column per unknown
    # Columns can differ in size by 10+ orders of magnitude (the Sun term is tiny), so solve with
    # every column scaled to unit length, then undo the scaling. Same answer, no round-off trouble.
    scale = np.linalg.norm(theta, axis=0)
    scaled_coefficients, *_ = np.linalg.lstsq(theta / scale, np.ravel(target), rcond=None)
    return scaled_coefficients / scale, theta


def gravity_shape(r):
    """-r / |r|^3 for a (3, N) array of positions."""
    return -r / np.linalg.norm(r, axis=0) ** 3


def known_j2(r, j2):
    """J2 acceleration for (3, N) positions (zero when j2 = 0): known physics, subtracted before fitting."""
    if not j2:
        return np.zeros_like(r)
    from .derivative_detection import known_accel          # vectorised gravity + J2
    return known_accel(r, j2) - known_accel(r, 0.0)


def fit_mu(positions, h, order=None, j2=0.0):
    """Recover mu from positions alone (gravity model; a known J2 term is subtracted first if j2 != 0)."""
    r, _, accel = fd_velocity_and_acceleration(positions, h, order)
    (mu_hat,), _ = least_squares([gravity_shape(r)], accel - known_j2(r, j2))
    return mu_hat


def fit_cd(positions, h, order=None, mu=MU, j2=0.0):
    """Recover cd with gravity (and J2, if j2 != 0) known: (a - gravity - J2) = cd * (-|v| v)."""
    r, v, accel = fd_velocity_and_acceleration(positions, h, order)
    residual = accel - mu * gravity_shape(r) - known_j2(r, j2)
    drag_shape = -np.linalg.norm(v, axis=0) * v
    (cd_hat,), _ = least_squares([drag_shape], residual)
    return cd_hat


def _third_body_shape(r, r_body):
    """Third-body acceleration per unit mu_body, for (3, N) positions and (3, N) body positions."""
    to_body = r_body - r
    return to_body / np.linalg.norm(to_body, axis=0) ** 3 - r_body / np.linalg.norm(r_body, axis=0) ** 3


def _body_positions(t, distance, rate):
    """(3, N) positions of a body on a circular equatorial orbit (same model as perturbations.py)."""
    return distance * np.array([np.cos(rate * t), np.sin(rate * t), np.zeros_like(t)])


def perturbation_shapes(t, r, v):
    """The five 'shape' columns of the joint fit: each force with its unknown strength set to 1 (vectorised)."""
    from .derivative_detection import known_accel
    from .perturbations import MOON_RATE, SUN_DISTANCE, SUN_RATE
    drag = -np.linalg.norm(v, axis=0) * v
    j2_shape = known_accel(r, 1.0) - known_accel(r, 0.0)          # J2 part only, with J2 = 1
    r_sun = _body_positions(t, SUN_DISTANCE, SUN_RATE)
    moon = _third_body_shape(r, _body_positions(t, MOON_DISTANCE, MOON_RATE))
    sun = _third_body_shape(r, r_sun)
    away_from_sun = r - r_sun
    srp = away_from_sun / np.linalg.norm(away_from_sun, axis=0)
    return {"cd": drag, "J2": j2_shape, "mu_moon": moon, "mu_sun": sun, "SRP (P*CR*A/m)": srp}


def fit_all_perturbations(t, positions, h, order=None, mu=MU):
    """Joint fit of cd, J2, mu_moon, mu_sun and the SRP acceleration. Returns (estimates dict, column correlation)."""
    r, v, accel = fd_velocity_and_acceleration(positions, h, order)
    half = (positions.shape[1] - r.shape[1]) // 2
    shapes = perturbation_shapes(t[half:len(t) - half], r, v)
    residual = accel - mu * gravity_shape(r)
    coefficients, theta = least_squares(list(shapes.values()), residual)
    correlation = np.corrcoef(theta, rowvar=False)
    return dict(zip(shapes, coefficients)), correlation


TRUE_PERTURBATION_VALUES = {"J2": J2, "mu_moon": MU_MOON, "mu_sun": MU_SUN, "SRP (P*CR*A/m)": SRP_ACCEL}


# ---------------------------------------------------------------------------
# Energy method for cd (Step 10)
# ---------------------------------------------------------------------------
def energy_sma(r, v, mu=MU, j2=0.0):
    """
    SMA from energy, for (3, N) arrays. With j2 != 0 the J2 potential energy is included, which
    removes J2's ~12 km per-orbit swing (the 'mean SMA' of smooth_controller.py).
    """
    radius = np.linalg.norm(r, axis=0)
    energy = np.sum(v * v, axis=0) / 2 - mu / radius
    if j2:
        earth_radius = EARTH_EQUATORIAL_RADIUS_KM / DU
        energy = energy + mu * j2 * earth_radius**2 / (2 * radius**3) * (3 * (r[2] / radius) ** 2 - 1)
    return -mu / (2 * energy)


def fit_cd_energy(t, positions, h, order=None, mu=MU, j2=0.0):
    """
    cd from the secular SMA decay: fit a straight line a(t) = a0 + a_dot * t through every sample,
    then cd = -a_dot * mu / (2 a^2 |v|^3) using the mean a and mean |v|^3.
    Only ONE derivative (velocity) is needed, which is why it is far less noise-sensitive.
    """
    r, v, _ = fd_velocity_and_acceleration(positions, h, order)
    half = (positions.shape[1] - r.shape[1]) // 2
    times = t[half:len(t) - half]
    sma = energy_sma(r, v, mu, j2)
    a_dot, _ = np.polyfit(times, sma, 1)
    speed_cubed = np.mean(np.linalg.norm(v, axis=0) ** 3)
    return -a_dot * mu / (2 * np.mean(sma) ** 2 * speed_cubed), times, sma


def add_position_noise(positions, noise_km, seed=0):
    """Gaussian position noise with standard deviation `noise_km` on every component."""
    if noise_km <= 0:
        return positions
    return positions + (noise_km / DU) * np.random.default_rng(seed).standard_normal(positions.shape)


def fd_acceleration_error(positions, h, true_accel, order=None):
    """RMS error of the FD acceleration against the exact acceleration at the same interior points."""
    order = order or tuned("fd_order")
    weights_a, div_a = SECOND_DERIVATIVE[order]
    accel = apply_stencil(positions, weights_a, div_a, h, 2)
    half = len(weights_a) // 2
    return float(np.sqrt(np.mean((accel - true_accel[:, half:-half]) ** 2)))


__all__ = ["fd_velocity_and_acceleration", "fit_mu", "fit_cd", "fit_all_perturbations", "fit_cd_energy",
           "add_position_noise", "fd_acceleration_error", "TRUE_PERTURBATION_VALUES", "MOON_DISTANCE",
           "gravity_accel"]

"""
The extra forces from the Week 2 deck (Step 3): Moon, Sun and solar radiation pressure (SRP),
plus a "full model" right-hand side that can switch each force on or off.

Simplifications (same as the Week 2 deck, slide "What I simplified"):
    * Moon and Sun move on circular orbits in Earth's equatorial plane.
    * SRP ignores eclipses (the satellite is always in sunlight) and uses a flat plate facing the Sun.

Everything is non-dimensional (DU = Earth radius, TU = 1 / Earth spin rate), like the rest of the code.
"""
import numpy as np
from scipy.integrate import solve_ivp

from .constants import ACCEL_UNIT_MS2, CD_TRUE, DU, J2, SOLVER_TOLERANCE, TU
from .physics import drag_accel, gravity_accel, initial_state, j2_accel

# --- Third bodies (Week 2 deck, "Reference: every number used") -------------
MU_MOON_KM3_S2 = 4902.8
MU_SUN_KM3_S2 = 1.327e11
MOON_DISTANCE_KM = 384_400.0
SUN_DISTANCE_KM = 1.496e8
MOON_PERIOD_DAYS = 27.32
SUN_PERIOD_DAYS = 365.25

# non-dimensional versions
MU_MOON = MU_MOON_KM3_S2 * TU**2 / DU**3
MU_SUN = MU_SUN_KM3_S2 * TU**2 / DU**3
MOON_DISTANCE = MOON_DISTANCE_KM / DU
SUN_DISTANCE = SUN_DISTANCE_KM / DU
MOON_RATE = 2 * np.pi / (MOON_PERIOD_DAYS * 86400 / TU)   # rad per TU
SUN_RATE = 2 * np.pi / (SUN_PERIOD_DAYS * 86400 / TU)

# --- Solar radiation pressure -----------------------------------------------
SOLAR_PRESSURE_N_M2 = 4.56e-6        # P, at 1 AU
REFLECTIVITY_CR = 1.3                # C_R
AREA_TO_MASS_M2_KG = 5.0 / 260.0     # A/m for a Starlink-like satellite (5 m^2, 260 kg)
SRP_ACCEL_MS2 = SOLAR_PRESSURE_N_M2 * REFLECTIVITY_CR * AREA_TO_MASS_M2_KG
SRP_ACCEL = SRP_ACCEL_MS2 / ACCEL_UNIT_MS2

# names used everywhere (plots, tables)
FORCE_NAMES = ("J2", "Moon", "Sun", "SRP")


# ---------------------------------------------------------------------------
# Where the Moon and Sun are
# ---------------------------------------------------------------------------
def moon_position(t):
    """Moon position at time t: circle in the equatorial plane."""
    angle = MOON_RATE * t
    return MOON_DISTANCE * np.array([np.cos(angle), np.sin(angle), 0.0])


def sun_position(t):
    """Sun position at time t: circle in the equatorial plane."""
    angle = SUN_RATE * t
    return SUN_DISTANCE * np.array([np.cos(angle), np.sin(angle), 0.0])


# ---------------------------------------------------------------------------
# Accelerations
# ---------------------------------------------------------------------------
def third_body_accel(r, r_body, mu_body):
    """Pull of a third body on the satellite MINUS its pull on Earth (Earth is our origin)."""
    to_body = r_body - r
    return mu_body * (to_body / np.linalg.norm(to_body) ** 3 - r_body / np.linalg.norm(r_body) ** 3)


def srp_accel(r, r_sun):
    """Constant-size push directly away from the Sun."""
    from_sun = r - r_sun
    return SRP_ACCEL * from_sun / np.linalg.norm(from_sun)


def force_accelerations(t, r, v, cd=CD_TRUE, j2=J2):
    """Every force at one instant, as a dict name -> 3-vector (non-dimensional)."""
    r_moon, r_sun = moon_position(t), sun_position(t)
    return {
        "Gravity": gravity_accel(r),
        "Drag": drag_accel(v, cd),
        "J2": j2_accel(r, j2),
        "Moon": third_body_accel(r, r_moon, MU_MOON),
        "Sun": third_body_accel(r, r_sun, MU_SUN),
        "SRP": srp_accel(r, r_sun),
    }


# ---------------------------------------------------------------------------
# Full model: gravity + drag, plus any of the extra forces
# ---------------------------------------------------------------------------
def full_rhs(cd=CD_TRUE, forces=FORCE_NAMES):
    """f(t, [r, v]) with gravity + drag + the listed extra forces."""
    def rhs(t, state):
        r, v = state[0:3], state[3:6]
        accel = gravity_accel(r) + drag_accel(v, cd)
        if "J2" in forces:
            accel = accel + j2_accel(r)
        if "Moon" in forces:
            accel = accel + third_body_accel(r, moon_position(t), MU_MOON)
        if "Sun" in forces or "SRP" in forces:
            r_sun = sun_position(t)
            if "Sun" in forces:
                accel = accel + third_body_accel(r, r_sun, MU_SUN)
            if "SRP" in forces:
                accel = accel + srp_accel(r, r_sun)
        return np.concatenate([v, accel])
    return rhs


def propagate_full(t_eval, cd=CD_TRUE, forces=FORCE_NAMES, method="DOP853", tolerance=SOLVER_TOLERANCE,
                   state0=None):
    """Integrate the full model and return (6, N) states at `t_eval`."""
    state0 = initial_state() if state0 is None else state0
    solution = solve_ivp(full_rhs(cd, forces), [t_eval[0], t_eval[-1]], state0, t_eval=t_eval,
                         method=method, rtol=tolerance, atol=tolerance)
    return solution.y

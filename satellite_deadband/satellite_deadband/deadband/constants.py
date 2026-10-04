"""
Physical constants, the non-dimensional unit system, and the orbit / controller settings.

Unit system (same as the Week 2-4 slides)
-----------------------------------------
    DU = Earth radius                  = 6371 km
    TU = 1 / Earth spin rate           = 13713.441 s  (3.8093 h)
    VU = DU / TU                       [km/s per non-dimensional speed]

All simulation state is stored in these units. Names ending in `_km`, `_s`, `_hours`
etc. are dimensional. Use the helpers at the bottom to convert times.
"""
import numpy as np

# --- Earth ------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
EARTH_MU_KM3_S2 = 3.98600441e5
EARTH_SPIN_RATE_RAD_S = 7.2921159e-5
J2 = 0.00108263                       # oblateness coefficient
# J2 is defined relative to the EQUATORIAL radius, so every J2 term must use this radius
# (not the 6371 km mean radius used as the length unit DU). Using 6371 km makes J2 0.22% too weak.
EARTH_EQUATORIAL_RADIUS_KM = 6378.137

# --- Non-dimensional units --------------------------------------------------
DU = EARTH_RADIUS_KM
TU = 1.0 / EARTH_SPIN_RATE_RAD_S
VU = DU / TU
MU = EARTH_MU_KM3_S2 * TU**2 / DU**3  # gravitational parameter in these units (289.873)

# --- Reference orbit (Starlink-style) ---------------------------------------
ALTITUDE_KM = 550.0
SEMI_MAJOR_AXIS_KM = EARTH_RADIUS_KM + ALTITUDE_KM
ECCENTRICITY = 1e-4
INCLINATION = np.deg2rad(53.0)
RAAN = np.deg2rad(30.0)
ARGUMENT_OF_PERIGEE = 0.0
INITIAL_TRUE_ANOMALY = 0.0

ORBITAL_PERIOD_S = 2 * np.pi * np.sqrt(SEMI_MAJOR_AXIS_KM**3 / EARTH_MU_KM3_S2)
PERIOD = ORBITAL_PERIOD_S / TU        # one orbit, non-dimensional time

# --- Drag -------------------------------------------------------------------
# a_drag = -cd * |v| * v. The demo value is ~1200x the physical one so the decay is
# visible within a week of simulated data (Week 2 slides).
CD_PER_KM = 5e-9
CD_TRUE = CD_PER_KM * DU              # 3.1855e-5, non-dimensional

# --- Deadband controller ----------------------------------------------------
A_LOWER_KM = 6920.500                 # raise burn fires when the osculating SMA decays to here
A_UPPER_KM = 6921.500                 # the trim burn settles the orbit back at this edge
A_LOWER = A_LOWER_KM / DU
A_UPPER = A_UPPER_KM / DU
RAISE_OVERSHOOT_FRACTION = 0.15       # raise aims this fraction of the band above A_UPPER

# --- Simulation sampling ----------------------------------------------------
COAST_SAMPLE_SPACING = 0.002          # target spacing of stored samples while coasting (TU)
COAST_MAX_STEP = 0.01                 # largest solver step while coasting (TU)
SOLVER_TOLERANCE = 1e-10              # rtol = atol for free propagation
CONTROLLER_SOLVER_TOLERANCE = 1e-11   # rtol = atol for the controlled simulation


# --- Time conversions (non-dimensional <-> dimensional) ---------------------
def to_hours(t_star):
    return t_star * TU / 3600


def to_minutes(t_star):
    return t_star * TU / 60.0


def from_hours(hours):
    return hours * 3600 / TU


def from_minutes(minutes):
    return minutes * 60.0 / TU


# =============================================================================
# Week 5: smooth (tanh) deadband controller with J2
# -----------------------------------------------------------------------------
# Everything below is used only by `smooth_controller.py` and the Week 5 stages.
# The Week 4 settings above are unchanged.
# =============================================================================
SECONDS = 1.0 / TU                    # one second, non-dimensional
METRES = 1e-3 / DU                    # one metre, non-dimensional
ACCEL_UNIT_MS2 = DU * 1e3 / TU**2     # one non-dimensional acceleration [m/s^2] (0.0339)

# --- Deadband (Figure 1 of the proposal: 500 m band, roughly one burn every 4 days) ---
SMOOTH_A_LOWER_KM = 6921.900          # thruster switches ON when the mean SMA decays to here
SMOOTH_A_UPPER_KM = 6922.400          # thruster switches OFF when the mean SMA climbs to here
SMOOTH_A_START_KM = 6922.250          # mean SMA at t = 0
SMOOTH_A_LOWER = SMOOTH_A_LOWER_KM / DU
SMOOTH_A_UPPER = SMOOTH_A_UPPER_KM / DU
SMOOTH_A_START = SMOOTH_A_START_KM / DU

# --- Drag: tuned so the mean SMA decays ~0.13 km/day, matching Figure 1 -----
# (about 3.4x the physical value at 550 km; the Week 4 CD_TRUE decays 45 km/day)
SMOOTH_CD = CD_TRUE * 2.82e-3

# --- Thruster (Hall-effect class: ~50 mN on a ~250 kg satellite) -----------
THRUST_ACCEL_MS2 = 2e-4               # full-throttle acceleration [m/s^2]
THRUST_ACCEL = THRUST_ACCEL_MS2 / ACCEL_UNIT_MS2

# --- tanh switch shape (the "slopes" to vary) -------------------------------
SWITCH_SMA_WIDTH_M = 5.0              # how many metres of SMA the on/off switch is smeared over
SWITCH_SMA_WIDTH = SWITCH_SMA_WIDTH_M * METRES
SWITCH_TIME_S = 60.0                  # time constant of the thruster on/off state
SWITCH_TIME = SWITCH_TIME_S * SECONDS
LATCH_GAIN = 1.0                      # strength of the pull of the on/off state towards 0 or 1
THROTTLE_OPENS_AT = 0.9               # the throttle opens only once the on/off state is past this
THROTTLE_WIDTH = 0.02                 # sharpness of the throttle opening

# --- Integration ------------------------------------------------------------
SMOOTH_DAYS = 10.0                    # length of the Week 5 run = the first 10-day chunk of the 400-day
                                      # controlled run (long_run.CHUNK_DAYS), so both are the same data
SMOOTH_SOLVER = "LSODA"               # switches to a stiff (BDF) method automatically when needed
SMOOTH_SOLVER_TOLERANCE = 1e-11       # rtol = atol


def to_days(t_star):
    return t_star * TU / 86400


def from_days(days):
    return days * 86400 / TU

# --- Sampling used for every Week 5 plot and for the GMM detector ------------
SAMPLE_STEP_MIN = 0.1                 # one sample every 0.1 minutes (6 s)
SAMPLE_STEP_S = SAMPLE_STEP_MIN * 60.0

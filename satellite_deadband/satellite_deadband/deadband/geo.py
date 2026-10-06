"""
GEO station keeping (step 17): the brief's second crowded regime.

Dynamics (mean longitude lambda [deg], time in days)
    lambda'' = -A sin(2 (lambda - lambda_s))                Earth's ellipticity (triaxiality, J22)
    A = 0.0017 deg/day^2, lambda_s = 75.07 deg E            (one of the two stable points)
    inclination grows by ~0.85 deg/year (Sun / Moon), here as one signed component i(t)

Control laws
    east-west (impulsive chemical): box lambda0 +- 0.05 deg. When the drift carries the satellite to the
        box edge it is accelerating towards, an impulse reverses the drift rate so the free-drift parabola
        just reaches the other edge:  lambda' -> -sign(lambda'') sqrt(2 |lambda''| x box width).
    north-south (impulsive): when i > +0.05 deg, a burn resets it to -0.05 deg.
    east-west (electric propulsion, continuous): a small thrust against the natural acceleration, on
        when lambda leaves an inner band on that side, off at the centre (the LEO hysteresis law again).

Phase 1 (days 0-120) is a FREE DRIFT from 60 deg E (as after a relocation): the satellite librates about
75 deg E over ~25 deg, which is what makes A and lambda_s identifiable. Phase 2 is station keeping at
70 deg E (inside the box the longitude changes by only 0.1 deg, so sin 2 lambda, cos 2 lambda and a constant
are indistinguishable there: only the local acceleration can be learned).

Tracking: longitude and inclination every 6 h with 0.0005 deg noise (~370 m at GEO).
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

A_TRUE = 0.0017
LAMBDA_S = 75.07
LAMBDA_FREE0 = 60.0
LAMBDA_BOX = 70.0
BOX_HALF = 0.05
INC_RATE = 0.85 / 365.25
INC_HALF = 0.05
DAYS = 400
FREE_DAYS = 120
DT = 0.01
MEAS_STEP = 0.25
SIGMA_DEG = 0.0005
EP_ACCEL_FACTOR = 3.0
EP_RETURN_RATE = 0.002                     # deg/day: stop thrusting once drifting back in this fast
EP_BAND = 0.02


def natural_accel(lam):
    return -A_TRUE * np.sin(2 * np.radians(lam - LAMBDA_S))


@dataclass
class GeoRun:
    t: np.ndarray
    lam: np.ndarray
    inc: np.ndarray
    ew_burns: np.ndarray          # times of east-west impulses (or EP switch-on times)
    ns_burns: np.ndarray
    thrusting: np.ndarray         # EP: 1 while the thruster is on


def simulate(propulsion="chemical", days=DAYS):
    n = int(days / DT)
    t = np.arange(n) * DT
    lam, rate, inc = np.empty(n), 0.0, np.empty(n)
    lam[0], inc[0] = LAMBDA_FREE0, 0.0
    ew, ns, on = [], [], np.zeros(n, dtype=np.int8)
    state = 0
    for k in range(1, n):
        x = lam[k - 1]
        acc = natural_accel(x)
        if t[k] >= FREE_DAYS:
            if t[k - 1] < FREE_DAYS:                         # relocation manoeuvre into the box (labelled)
                x, rate = LAMBDA_BOX, 0.0
            if propulsion == "chemical":
                edge = LAMBDA_BOX + np.sign(acc) * BOX_HALF  # the edge the acceleration pushes towards
                if np.sign(acc) * (x - edge) >= 0 and np.sign(rate) == np.sign(acc):
                    rate = -np.sign(acc) * np.sqrt(2 * abs(acc) * 2 * BOX_HALF)
                    ew.append(t[k])
            else:
                outward = np.sign(acc)                       # the direction the natural drift pushes
                if state == 0 and outward * (x - LAMBDA_BOX) > EP_BAND and outward * rate > 0:
                    state = 1                                # outside the inner band and drifting out: thrust
                    ew.append(t[k])
                elif state == 1 and outward * rate < -EP_RETURN_RATE:
                    state = 0                                # drift reversed (heading back in): stop
                acc = acc - state * EP_ACCEL_FACTOR * abs(natural_accel(LAMBDA_BOX)) * np.sign(acc)
        on[k] = state
        rate += acc * DT
        lam[k] = x + rate * DT
        i_new = inc[k - 1] + INC_RATE * DT
        if t[k] >= FREE_DAYS and i_new > INC_HALF:
            i_new = -INC_HALF
            ns.append(t[k])
        inc[k] = i_new
    return GeoRun(t=t, lam=lam, inc=inc, ew_burns=np.array(ew), ns_burns=np.array(ns), thrusting=on)


def measure(run, seed=0):
    rng = np.random.default_rng(seed)
    stride = int(round(MEAS_STEP / DT))
    t = run.t[::stride]
    return t, run.lam[::stride] + SIGMA_DEG * rng.standard_normal(t.size), \
        run.inc[::stride] + SIGMA_DEG * rng.standard_normal(t.size)


# ---------------------------------------------------------------------------
# Learning the longitude law with SINDy (candidate libraries)
# ---------------------------------------------------------------------------
def second_derivative(t, lam, window=41, order=3):
    h = t[1] - t[0]
    return savgol_filter(lam, window, order, deriv=2, delta=h), savgol_filter(lam, window, order)


def recover(coefficients, names):
    """(A, lambda_s) from the sin 2 lambda / cos 2 lambda coefficients (None if they are not in the law)."""
    if "sin 2l" not in names or "cos 2l" not in names:
        return None, None
    cs, cc = coefficients[names.index("sin 2l")], coefficients[names.index("cos 2l")]
    # -A sin(2l - 2ls) = (-A cos 2ls) sin 2l + (A sin 2ls) cos 2l
    amplitude = float(np.hypot(cs, cc))
    lam_s = float(np.degrees(0.5 * np.arctan2(cc, -cs))) % 180
    return amplitude, lam_s


def drift_rate_jumps(t, lam, window_days=8.0):
    """
    Drift-rate jump at every sample: quadratic fit of the `window_days` AFTER minus the one BEFORE, both
    evaluated at the sample. An impulse changes the rate by ~0.016 deg/day, ~400 x the noise of a
    32-sample fit, but only ~3 x the noise of a single second difference.
    """
    n = int(round(window_days / (t[1] - t[0])))
    jumps = np.zeros(t.size)
    for k in range(n, t.size - n):
        before = np.polyfit(t[k - n:k + 1] - t[k], lam[k - n:k + 1], 2)[1]
        after = np.polyfit(t[k:k + n + 1] - t[k], lam[k:k + n + 1], 2)[1]
        jumps[k] = after - before
    return jumps


def detect_ew_burns(t, lam, threshold_sigma=10.0, window_days=8.0):
    """Impulses: local extrema of the drift-rate jump far above its robust noise level."""
    jumps = drift_rate_jumps(t, lam, window_days)
    noise = 1.4826 * np.median(np.abs(jumps - np.median(jumps)))
    n = int(round(window_days / (t[1] - t[0])))
    events = []
    for k in np.flatnonzero(np.abs(jumps) > threshold_sigma * noise):
        lo, hi = max(0, k - n), min(t.size, k + n + 1)
        if np.abs(jumps[k]) >= np.abs(jumps[lo:hi]).max():
            events.append(t[k])
    return np.array(events), jumps


def forecast_ew(t0, lam0, rate0, accel, edge_low, edge_high, until):
    """Impulse times after t0 predicted with a constant local acceleration (learned) and the learned box edges."""
    out, t, x, v = [], t0, lam0, rate0
    while t < until:
        edge = edge_high if accel > 0 else edge_low
        # time to reach the edge under constant acceleration: x + v s + 0.5 a s^2 = edge
        a, b, c = 0.5 * accel, v, x - edge
        disc = b * b - 4 * a * c
        if disc < 0:
            break
        s = max((-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a))
        if s <= 0:
            break
        t += s
        out.append(t)
        x, v = edge, -np.sign(accel) * np.sqrt(2 * abs(accel) * (edge_high - edge_low))
    return np.array(out)

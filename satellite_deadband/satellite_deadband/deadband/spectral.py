"""
Spectral (FFT) method for spotting the J2 signature, and its robustness to noise and
sampling cadence (Workstream A).

J2 shows up at TWICE the orbital frequency in the radial residual acceleration
(finite-difference acceleration minus the known gravity + drag baseline). Two things can
hide it: position noise, and sampling slower than the Nyquist limit for that 2x tone.
"""
from dataclasses import dataclass

import numpy as np

from .constants import CD_TRUE, DU, J2, PERIOD, TU, from_minutes
from .physics import drag_accel, gravity_accel, propagate

# 6th-order central-difference stencil for the second derivative.
_SECOND_DERIVATIVE_STENCIL = np.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90])
_STENCIL_HALF_WIDTH = 3

DC_BINS_TO_SKIP = 5           # ignore the strong near-zero-frequency (secular drift) bins
PEAK_HALF_WIDTH_BINS = 2      # tolerance around the expected peak for spectral leakage
GUARD_HALF_WIDTH_BINS = 4     # bins excluded around the peak when estimating the noise floor


@dataclass
class Periodogram:
    frequencies: np.ndarray   # cycles per non-dimensional time unit
    power: np.ndarray
    orbital_frequency: float
    j2_frequency: float
    peak_power: float
    background_power: float
    snr_db: float


def j2_nyquist_limit_minutes():
    """Slowest cadence that still resolves a tone at 2x the orbital frequency (= T_orbit / 4)."""
    return PERIOD * TU / 60.0 / 4.0


# ---------------------------------------------------------------------------
# Finite differences
# ---------------------------------------------------------------------------
def second_derivative(y, h):
    """6th-order central second derivative; the output is 6 samples shorter than the input."""
    y = np.asarray(y)
    n = y.shape[0]
    out = np.zeros((n - 6,) + y.shape[1:])
    for k, coefficient in enumerate(_SECOND_DERIVATIVE_STENCIL):
        out += coefficient * y[k:n - 6 + k]
    return out / h**2


def first_derivative(y, h):
    """2nd-order central first derivative (only used for the small drag term)."""
    y = np.asarray(y)
    return (y[2:] - y[:-2]) / (2 * h)


# ---------------------------------------------------------------------------
# Residual signal and periodogram
# ---------------------------------------------------------------------------
def true_j2_trajectory(days=3.0, n_points=20000):
    """Dense reference trajectory including J2: returns (times, states)."""
    t_eval = np.linspace(0, days * 86400.0 / TU, n_points)
    return t_eval, propagate(CD_TRUE, t_eval, j2=J2)


def radial_residual_series(t_eval, states, cadence_min, noise_km, seed=0):
    """
    Sample the trajectory at `cadence_min`, add position noise, rebuild the acceleration by
    finite differences, subtract the gravity + drag baseline, and keep the radial component.
    Returns (times, radial_residual), or (None, None) if there are too few samples.
    """
    rng = np.random.default_rng(seed)
    dt = from_minutes(cadence_min)
    t_samples = np.arange(t_eval[0], t_eval[-1], dt)
    positions = np.array([np.interp(t_samples, t_eval, states[i]) for i in range(3)]).T
    positions = positions + (noise_km / DU) * rng.standard_normal(positions.shape)

    if len(t_samples) < 8:
        return None, None

    half = _STENCIL_HALF_WIDTH
    acceleration = second_derivative(positions, dt)
    velocity = first_derivative(positions, dt)[2:-2]   # aligned with the acceleration window
    inner_positions = positions[half:-half]

    baseline = np.array([gravity_accel(inner_positions[i]) + drag_accel(velocity[i], CD_TRUE)
                         for i in range(len(inner_positions))])
    residual = acceleration - baseline

    radial_direction = inner_positions / np.linalg.norm(inner_positions, axis=1, keepdims=True)
    return t_samples[half:-half], np.sum(residual * radial_direction, axis=1)


def periodogram_snr(t, signal, background_max_cycles_per_orbit=None, peak_search_fraction=None):
    """
    Hann-windowed FFT; SNR = peak power near 2x orbital frequency over the median noise floor.

    background_max_cycles_per_orbit (Week 5 addition): if given, the noise floor is taken only
    from frequencies up to this many cycles per orbit. Without it, a fine cadence (e.g. 0.1 min)
    puts thousands of bins of high-frequency finite-difference noise into the median and the
    SNR collapses even though the J2 peak stands far above its neighbours.

    peak_search_fraction (Week 5 addition): if given, look for the peak within +- this fraction of
    the J2 frequency instead of +- 2 bins. The real J2 line sits ~0.1% off exactly twice the
    Keplerian orbital frequency; on a long record (fine bins) a +-2-bin window misses it.
    """
    n = len(signal)
    dt = t[1] - t[0]
    spectrum = np.fft.rfft((signal - np.mean(signal)) * np.hanning(n))
    frequencies = np.fft.rfftfreq(n, d=dt)
    power = np.abs(spectrum) ** 2

    orbital_frequency = 1.0 / PERIOD
    j2_frequency = 2 * orbital_frequency
    j2_bin = int(np.argmin(np.abs(frequencies - j2_frequency)))

    if peak_search_fraction is None:
        peak_window = slice(max(1, j2_bin - PEAK_HALF_WIDTH_BINS), min(len(frequencies), j2_bin + PEAK_HALF_WIDTH_BINS + 1))
        guard = slice(max(0, j2_bin - GUARD_HALF_WIDTH_BINS), j2_bin + GUARD_HALF_WIDTH_BINS + 1)
    else:
        near = np.flatnonzero(np.abs(frequencies - j2_frequency) <= peak_search_fraction * j2_frequency)
        if near.size == 0:                       # short record: bins wider than the window, use the nearest bin
            near = np.array([j2_bin])
        peak_window = slice(near[0], near[-1] + 1)
        guard = slice(max(0, near[0] - GUARD_HALF_WIDTH_BINS), near[-1] + GUARD_HALF_WIDTH_BINS + 1)
    peak_power = float(np.max(power[peak_window]))

    is_background = np.ones(len(frequencies), dtype=bool)
    is_background[:DC_BINS_TO_SKIP] = False
    is_background[guard] = False
    if background_max_cycles_per_orbit is not None:
        is_background &= frequencies <= background_max_cycles_per_orbit * orbital_frequency
    background = float(np.median(power[is_background])) if np.any(is_background) else 1e-30

    snr_db = 10 * np.log10(max(peak_power, 1e-300) / max(background, 1e-300))
    return Periodogram(frequencies, power, orbital_frequency, j2_frequency, peak_power, background, snr_db)


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
def noise_sweep(t_eval, states, noise_list_km, cadence_min=5.0, seed0=1, background_max_cycles_per_orbit=None):
    """J2 peak SNR versus position noise at a fixed (Nyquist-safe) cadence."""
    rows = []
    for i, noise_km in enumerate(noise_list_km):
        t, residual = radial_residual_series(t_eval, states, cadence_min, noise_km, seed=seed0 + i)
        snr_db = periodogram_snr(t, residual, background_max_cycles_per_orbit).snr_db
        rows.append(dict(noise_km=noise_km, cadence_min=cadence_min, snr_db=snr_db))
    return rows


def cadence_sweep(t_eval, states, cadence_list_min, noise_km=0.3, seed=7, background_max_cycles_per_orbit=None):
    """J2 peak SNR versus sampling cadence at fixed noise."""
    nyquist_limit = j2_nyquist_limit_minutes()
    rows = []
    for cadence_min in cadence_list_min:
        t, residual = radial_residual_series(t_eval, states, cadence_min, noise_km, seed=seed)
        rows.append(dict(cadence_min=cadence_min, noise_km=noise_km,
                         snr_db=periodogram_snr(t, residual, background_max_cycles_per_orbit).snr_db,
                         violates_j2_nyquist=bool(cadence_min > nyquist_limit)))
    return rows

"""
Realistic measurement noise: a MIXTURE of noises at many frequencies, instead of pure white noise.

Why
---
Real tracking errors are not white. A position fix carries fast receiver jitter, but also errors
that change slowly: the orbit-determination solution is re-fitted every few minutes or hours
(an error that is constant for a while, then jumps), clocks and atmosphere drift (1/f, "pink"),
small biases wander (random walk, "brown"), and some errors repeat once per orbit or once per day
(thermal flexing, station visibility). Each of these has a different spectrum, and a derivative
treats each frequency differently:
    * the 2nd derivative (finite-difference acceleration) multiplies the noise spectrum by f^4,
      so FAST noise hurts it and slow noise barely matters;
    * the energy method fits a slow trend in the SMA, so SLOW noise (drifts, random walk) can look
      like drag decay and hurts it most;
    * noise that repeats once per orbit has the SAME frequency as some force shapes (SRP, J2's
      cross-track term) and can be mistaken for them.

The mixture (`realistic_noise`)
-------------------------------
Every component is made with zero mean and unit RMS, then weighted by the share of the total noise
VARIANCE it carries (MIXTURE_SHARES), and the sum is scaled to the requested RMS per axis. So
"white" and "realistic" noise of the same RMS have exactly the same size; only the spectrum differs.

    white, every sample                          receiver jitter
    white held for 10 / 100 / 1000 samples,      re-fitted solutions: white noise at slower rates,
      joined by straight lines                   i.e. "various frequencies of white noise"
    pink (power ~ 1/f)                           drifting clocks / atmosphere
    brown (power ~ 1/f^2, a random walk)         wandering biases
    once-per-orbit and once-per-day sinusoids    repeating errors (random phase and size per axis)
"""
import numpy as np

from .constants import DU, PERIOD

MIXTURE_SHARES = {            # share of the noise VARIANCE carried by each component (sums to 1)
    "white (every sample)": 0.35,
    "white held 10 samples": 0.15,
    "white held 100 samples": 0.12,
    "white held 1000 samples": 0.10,
    "pink 1/f": 0.12,
    "brown 1/f^2 (random walk)": 0.08,
    "once per orbit + once per day": 0.08,
}
DAY = 86400.0


def _unit(x):
    """Zero mean, unit RMS (along the last axis)."""
    x = x - x.mean(axis=-1, keepdims=True)
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True))


def held_white(n, hold, rng):
    """White noise drawn every `hold` samples, joined by straight lines: white noise at a slower rate."""
    knots = rng.standard_normal((3, n // hold + 2))
    x = np.arange(n) / hold
    return np.array([np.interp(x, np.arange(knots.shape[1]), k) for k in knots])


def power_law(n, alpha, rng):
    """Noise with power spectrum ~ 1 / f^alpha (alpha = 1 pink, 2 brown), made by shaping white noise."""
    spectrum = np.fft.rfft(rng.standard_normal((3, n)), axis=1)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]                                  # avoid dividing by zero at DC
    return np.fft.irfft(spectrum / f ** (alpha / 2), n=n, axis=1)


def noise_components(t, step_s, rng):
    """{name: (3, N) unit-RMS noise} for every component of the mixture. `t` is non-dimensional time."""
    n = len(t)
    orbit_phase = 2 * np.pi * t / PERIOD
    day_phase = 2 * np.pi * np.arange(n) * step_s / DAY
    phases = rng.uniform(0, 2 * np.pi, (2, 3, 1))
    sizes = rng.uniform(0.5, 1.5, (2, 3, 1))
    return {
        "white (every sample)": _unit(rng.standard_normal((3, n))),
        "white held 10 samples": _unit(held_white(n, 10, rng)),
        "white held 100 samples": _unit(held_white(n, 100, rng)),
        "white held 1000 samples": _unit(held_white(n, 1000, rng)),
        "pink 1/f": _unit(power_law(n, 1, rng)),
        "brown 1/f^2 (random walk)": _unit(power_law(n, 2, rng)),
        "once per orbit + once per day": _unit(sizes[0] * np.sin(orbit_phase + phases[0])
                                               + sizes[1] * np.sin(day_phase + phases[1])),
    }


def realistic_noise(t, step_s, rms_m, seed=0):
    """(3, N) realistic noise in DU with `rms_m` metres RMS per axis: the weighted mixture of every component."""
    rng = np.random.default_rng(seed)
    parts = noise_components(t, step_s, rng)
    mixture = sum(np.sqrt(MIXTURE_SHARES[name]) * part for name, part in parts.items())
    return _unit(mixture) * rms_m / (DU * 1000)


def white_noise(t, rms_m, seed=0):
    """(3, N) white Gaussian noise in DU with `rms_m` metres RMS per axis (the usual assumption)."""
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal((3, len(t)))) * rms_m / (DU * 1000)

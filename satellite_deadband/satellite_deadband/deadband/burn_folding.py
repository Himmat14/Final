"""
Step 8 of the report: use the REPEATING structure of the deadband to beat the noise.

The deadband fires the same burn over and over. A single burn is buried in measurement noise,
but the noise is different every time while the burn is the same, so:

1. AUTOCORRELATION  of the along-track unmodelled acceleration shows peaks at the burn period:
                    proof the signal repeats, and an estimate of how often.
2. STACKING         (epoch folding): cut a window around every detected burn, line them up on
                    their start time and average. Noise falls like 1 / sqrt(number of burns).
3. GAUSSIAN PROCESS on the stacked data gives a smooth average thrust law with an uncertainty
                    band: thrust level, burn duration and ramp-up time, learned from data.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .constants import TU
from .derivative_detection import runs_of_ones

STACK_BEFORE_S = 180.0     # seconds of coasting kept before each burn start
STACK_AFTER_S = 1800.0     # seconds kept after each burn start (covers the ~23-minute burns)
GP_MAX_POINTS = 1500       # a GP costs ~ (number of points)^3, so cap the training set


# ---------------------------------------------------------------------------
# 1. Autocorrelation
# ---------------------------------------------------------------------------
def autocorrelation(signal, max_lag):
    """Normalised autocorrelation for lags 0..max_lag samples (computed with an FFT)."""
    x = signal - signal.mean()
    n = len(x)
    spectrum = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum))[:max_lag + 1]
    return acf / acf[0]


def dominant_period(acf, step_s, min_lag_s):
    """Lag (seconds) of the highest autocorrelation peak beyond `min_lag_s`."""
    start = int(min_lag_s / step_s)
    return (start + int(np.argmax(acf[start:]))) * step_s


# ---------------------------------------------------------------------------
# 2. Stacking
# ---------------------------------------------------------------------------
@dataclass
class Stack:
    offsets_s: np.ndarray      # (W,) seconds relative to each burn start
    windows: np.ndarray        # (n_burns, W) signal around each burn
    mean: np.ndarray           # (W,) average over burns
    std_error: np.ndarray      # (W,) standard error of that average


def stack_bursts(t, signal, events, step_s, before_s=STACK_BEFORE_S, after_s=STACK_AFTER_S):
    """Cut a window around the start of every event (0/1 array) and line them up."""
    before, after = int(before_s / step_s), int(after_s / step_s)
    starts = [np.searchsorted(t, start) for start, _ in runs_of_ones(t, events)]
    windows = np.array([signal[i - before:i + after] for i in starts if i - before >= 0 and i + after <= len(t)])
    offsets = np.arange(-before, after) * step_s
    return Stack(offsets_s=offsets, windows=windows, mean=windows.mean(axis=0),
                 std_error=windows.std(axis=0) / np.sqrt(len(windows)))


def noise_vs_bursts(stack: Stack, truth_profile, counts=(1, 2, 4, 8, 16)):
    """RMS error of the stacked mean (against the true profile) using only the first n bursts."""
    rows = []
    for n in counts:
        if n <= len(stack.windows):
            rows.append((n, float(np.sqrt(np.mean((stack.windows[:n].mean(axis=0) - truth_profile) ** 2)))))
    return rows


# ---------------------------------------------------------------------------
# 3. Gaussian-process thrust law
# ---------------------------------------------------------------------------
@dataclass
class ThrustLaw:
    offsets_s: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    plateau: float             # learned full-thrust level
    duration_s: float          # time the learned law spends above half the plateau
    rise_time_s: float         # 10% -> 90% of the plateau
    kernel: str


def fit_thrust_law(stack: Stack, max_points=GP_MAX_POINTS):
    """Gaussian process through the stacked bursts (at most `max_points` random samples, for speed)."""
    offsets = np.tile(stack.offsets_s, len(stack.windows))
    values = stack.windows.ravel()
    pick = np.random.default_rng(0).choice(offsets.size, size=min(max_points, offsets.size), replace=False)
    scale = np.std(values)
    kernel = ConstantKernel(1.0) * RBF(length_scale=60.0) + WhiteKernel(noise_level=0.5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gp.fit(offsets[pick, None], values[pick] / scale)
    mean, std = gp.predict(stack.offsets_s[:, None], return_std=True)
    mean, std = mean * scale, std * scale

    # full-thrust level = median of the curve while the thruster is clearly on (above half its peak);
    # a high percentile would pick up the residual noise ripple and come out biased high
    plateau = float(np.median(mean[mean > 0.5 * mean.max()]))
    above_half = stack.offsets_s[mean > 0.5 * plateau]
    rising = (stack.offsets_s > -120) & (stack.offsets_s < 240)
    t10 = stack.offsets_s[rising][np.argmax(mean[rising] > 0.1 * plateau)]
    t90 = stack.offsets_s[rising][np.argmax(mean[rising] > 0.9 * plateau)]
    return ThrustLaw(offsets_s=stack.offsets_s, mean=mean, std=std, plateau=plateau,
                     duration_s=float(above_half.max() - above_half.min()) if above_half.size else 0.0,
                     rise_time_s=float(t90 - t10), kernel=str(gp.kernel_))


def seconds(t_nondim):
    return t_nondim * TU

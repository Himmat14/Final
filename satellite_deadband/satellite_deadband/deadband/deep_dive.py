"""
Supporting analyses for step 21: the Savitzky-Golay derivative in the frequency domain, a parametric
(sigmoid) thrust law against the Gaussian-process one, the controller as a two-state ODE (hysteresis with its
direction of travel), and the SMA sawtooth.

Savitzky-Golay (Savitzky & Golay 1964; Schafer 2011)
---------------------------------------------------
Fit a polynomial of order p by least squares to the 2m + 1 samples around each point and take its
derivative at the centre. Because the fit is linear in the data, this is a fixed convolution:
    y'_k = sum_{j=-m..m} c_j y_{k+j},     c = row of (A^T A)^-1 A^T,  A_{ji} = (j h)^i,
so its frequency response is H(w) = sum_j c_j exp(i w j h). An ideal differentiator has H = i w. The filter
is exact for polynomials up to order p (no bias at low frequency) and rolls off above a cut-off frequency
that falls as the window grows: it differentiates the slow orbit and suppresses the fast noise. The white
noise passed is sigma^2 sum_j c_j^2 (the "noise gain").
"""
import numpy as np
from scipy import optimize
from scipy.signal import savgol_coeffs

from .constants import SAMPLE_STEP_S, TU
from .derivative_detection import runs_of_ones
from .settings import tuned

DAY = 86400.0


# ---------------------------------------------------------------------------
# Savitzky-Golay in the frequency domain
# ---------------------------------------------------------------------------
FD_STENCILS = {   # first-derivative central differences (coefficients for offsets -m..m, divided by h)
    "central, 2nd order": np.array([-1 / 2, 0, 1 / 2]),
    "central, 4th order": np.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]),
    "central, 8th order": np.array([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280]),
}


def response(coeffs, step_s, freqs_hz):
    """|H(f)| of a derivative filter with coefficients for offsets -m..m (already divided by h)."""
    m = (len(coeffs) - 1) // 2
    offsets = np.arange(-m, m + 1) * step_s
    return np.abs(np.exp(2j * np.pi * np.outer(freqs_hz, offsets)) @ coeffs)


def sg_study(step_s=SAMPLE_STEP_S, windows=(11, 21, 41, 61, 101, 201), orders=(2, 3, 5, 7)):
    nyquist = 0.5 / step_s
    freqs = np.logspace(-5, np.log10(nyquist), 400)
    tuned_window, tuned_order = tuned("sg_window"), tuned("sg_order")
    curves = {"ideal differentiator": 2 * np.pi * freqs}
    for name, c in FD_STENCILS.items():
        curves[name] = response(c / step_s, step_s, freqs)
    for w in sorted({21, tuned_window, 201}):
        c = savgol_coeffs(w, tuned_order, deriv=1, delta=step_s, use="dot")
        curves[f"Savitzky-Golay, {w} samples, order {tuned_order}"] = response(c, step_s, freqs)
    gains = {}
    for p in orders:
        gains[p] = [float(np.sqrt(np.sum(savgol_coeffs(w, p, deriv=1, delta=step_s) ** 2))) if w > p + 1 else np.nan
                    for w in windows]
    # cut-off: frequency where the tuned filter's gain falls to half of the ideal one
    c = savgol_coeffs(tuned_window, tuned_order, deriv=1, delta=step_s, use="dot")
    ratio = response(c, step_s, freqs) / (2 * np.pi * freqs)
    cutoff = float(freqs[np.argmax(ratio < 0.5)])
    return dict(freqs=freqs, curves=curves, windows=list(windows), gains=gains, cutoff_hz=cutoff,
                tuned=(tuned_window, tuned_order), step_s=step_s)


# ---------------------------------------------------------------------------
# Thrust law: parametric sigmoid model vs Gaussian process
# ---------------------------------------------------------------------------
def logistic(x):
    return 0.5 * (1 + np.tanh(x / 2))


def sigmoid_burn(t, plateau, t_on, t_off, rise, fall, offset):
    """Piecewise-smooth burn: plateau x sigma((t - t_on)/rise) x sigma((t_off - t)/fall) + offset."""
    return plateau * logistic((t - t_on) / rise) * logistic((t_off - t) / fall) + offset


def fit_sigmoid_burn(offsets_s, values):
    p0 = [values.max(), 0.0, 1368.0, 15.0, 15.0, 0.0]
    params, cov = optimize.curve_fit(sigmoid_burn, offsets_s, values, p0=p0, maxfev=20000)
    names = ("plateau", "t_on", "t_off", "rise", "fall", "offset")
    out = dict(zip(names, map(float, params)))
    out["std"] = dict(zip(names, map(float, np.sqrt(np.diag(cov)))))
    out["duration_s"] = out["t_off"] - out["t_on"]
    out["rise_10_90_s"] = 2 * np.log(9) * abs(out["rise"])           # logistic: 10% -> 90% in 2 ln 9 rise units
    return out


def thrust_models(noise_scale=2.0, seed=0):
    """Stack the burns (step 8 setting) and fit the GP and the sigmoid model; also fit the TRUE profile."""
    from .burn_folding import fit_thrust_law, stack_bursts
    from .classifiers import REFERENCE_NOISE, add_noise, build_features, detection_dataset
    ds = detection_dataset()
    r, v = add_noise(ds, noise_scale * REFERENCE_NOISE[0], noise_scale * REFERENCE_NOISE[1], seed)
    features = build_features(r, v, ds.step_s, window=tuned("sg_window_stacked"), order=tuned("sg_order_stacked"),
                              method="Savitzky-Golay")
    stack = stack_bursts(ds.t, features.along_track, ds.true_on, ds.step_s)
    truth = stack_bursts(ds.t, ds.thrust_ms2, ds.true_on, ds.step_s)
    gp = fit_thrust_law(stack)
    offsets = np.tile(stack.offsets_s, len(stack.windows))
    sig = fit_sigmoid_burn(offsets, stack.windows.ravel())
    sig_true = fit_sigmoid_burn(truth.offsets_s, truth.mean)
    curve = sigmoid_burn(stack.offsets_s, *[sig[k] for k in ("plateau", "t_on", "t_off", "rise", "fall", "offset")])
    rms = lambda y: float(np.sqrt(np.mean((y - truth.mean) ** 2)))
    return dict(offsets_s=stack.offsets_s, stack_mean=stack.mean, truth=truth.mean, gp_mean=gp.mean, gp_std=gp.std,
                sigmoid=curve, sigmoid_params=sig, sigmoid_true_params=sig_true,
                rms_gp=rms(gp.mean), rms_sigmoid=rms(curve), rms_stack=rms(stack.mean),
                gp=dict(plateau=gp.plateau, duration_s=gp.duration_s, rise_time_s=gp.rise_time_s, kernel=gp.kernel),
                bursts=len(stack.windows))


# ---------------------------------------------------------------------------
# The controller as a two-state ODE: x = (a, s)
# ---------------------------------------------------------------------------
def reduced_field(a_km, s, k_m_per_h=5.335, climb_m_per_h=1316.0):
    """
    (da/dt [m/h], ds/dt [1/h]) of the reduced controller:
        da/dt = -k + (climb) theta(s)
        ds/dt = [on(a)(1 - s) - off(a) s + latch(s)] / tau
    with the switches of smooth_controller (5 m wide tanh, tau = 60 s, latch gain 1).
    """
    from .constants import LATCH_GAIN, SMOOTH_A_LOWER_KM, SMOOTH_A_UPPER_KM, SWITCH_SMA_WIDTH_M, SWITCH_TIME_S
    from .smooth_controller import throttle
    sw = lambda x: 0.5 * (1 + np.tanh(x / (SWITCH_SMA_WIDTH_M / 1000)))
    on, off = sw(SMOOTH_A_LOWER_KM - a_km), sw(a_km - SMOOTH_A_UPPER_KM)
    latch = LATCH_GAIN * s * (1 - s) * (2 * s - 1)
    s_dot = (on * (1 - s) - off * s + latch) / SWITCH_TIME_S * 3600
    a_dot = -k_m_per_h + climb_m_per_h * throttle(s)
    return a_dot, s_dot


def hysteresis_data(cycles=2, start_day=40.0):
    """Two TEST cycles of the true run: mean SMA, thruster state, throttle and the SMA rate."""
    from .long_run import controlled_run
    run = controlled_run()
    days = run.t * TU / DAY
    starts = [s * TU / DAY for s, _ in runs_of_ones(run.t, run.true_on)]
    first = next(s for s in starts if s > start_day)
    end = next(s for s in starts if s > first + (cycles - 0.5) * 3.9)
    keep = (days > first - 1.0) & (days < end + 0.3)
    a = run.mean_sma_km[keep]
    rate = np.gradient(a, SAMPLE_STEP_S / 3600) * 1000
    return dict(days=days[keep], a=a, s=run.s[keep], on=run.true_on[keep], rate=rate)


def sawtooth_data(days=20.0, measured_every_s=600, seed=0):
    """Mean SMA over the first `days` (true), its noisy measurement every 10 min, and the burns."""
    from .classifiers import REFERENCE_NOISE, add_noise, detection_dataset
    from .smooth_controller import mean_sma_km_series
    ds = detection_dataset()
    keep = ds.days < days
    stride = int(measured_every_s / ds.step_s)
    sub = ds.until(days).every(stride)
    r, v = add_noise(sub, 10 * REFERENCE_NOISE[0], 10 * REFERENCE_NOISE[1], seed)
    measured = mean_sma_km_series(np.vstack([r, v]))
    burns = [(s * TU / DAY, e * TU / DAY) for s, e in runs_of_ones(ds.t[keep], ds.true_on[keep])]
    return dict(days=ds.days[keep][::10], a=ds.mean_sma_km[keep][::10], measured_days=sub.days, measured=measured,
                burns=burns)

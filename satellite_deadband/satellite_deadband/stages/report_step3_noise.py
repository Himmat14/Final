"""
Report step 3 (part 2): realistic noise. Every estimator so far was tested with WHITE noise. Real
tracking noise is a mixture of noises at many frequencies (deadband/noise_models.py). Here every
estimator is run on the same orbit with white noise and with realistic noise of the SAME RMS, so
any difference comes from the noise SPECTRUM alone.

Estimators (all from noisy positions of the 400-day natural run, 1-min samples):
    joint FD regression   cd, J2, mu_moon, mu_sun and SRP together (step 2)
    energy method         cd from the slope of the mean SMA (step 3)

Figures (outputs/report/step3_drag_noise/)
    step3_noise_models        what the realistic noise looks like: time series, spectrum of every
                              component, and what it does to the FD acceleration and the mean SMA
    step3_noise_estimators    error of every perturbation parameter vs noise RMS, white vs realistic,
                              for 30-day and 120-day records
    step3_noise_ratio         realistic / white error at 1 m RMS: which estimators the spectrum helps or hurts
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from deadband.constants import ACCEL_UNIT_MS2, DU, J2, ORBITAL_PERIOD_S, SMOOTH_CD
from deadband.long_run import natural_run
from deadband.noise_models import MIXTURE_SHARES, noise_components, realistic_noise, white_noise
from deadband.regression import (TRUE_PERTURBATION_VALUES, energy_sma, fd_velocity_and_acceleration,
                                 fit_all_perturbations, fit_cd_energy)
from .common import AMBER, GREEN, GREY, NAVY, PURPLE, RUST, SLATE
from .report_common import panel_label, report_style, save, step_dir

NOISE_RMS_M = (0.01, 0.1, 1.0, 10.0)        # noise RMS per axis [m]
HORIZONS = (30, 120)                        # record lengths [days]
SEEDS = range(3)
RATIO_NOISE_M = 1.0                         # noise level of the ratio figure
PARAMETERS = ("cd", "J2", "mu_moon", "mu_sun", "SRP (P*CR*A/m)", "cd (energy)")
TRUTH = {"cd": SMOOTH_CD, **TRUE_PERTURBATION_VALUES, "cd (energy)": SMOOTH_CD}
NOISE_STYLES = {"white": dict(ls="-", marker="o"), "realistic": dict(ls="--", marker="s")}
HORIZON_COLORS = {30: RUST, 120: NAVY}
COMPONENT_COLORS = dict(zip(MIXTURE_SHARES, (GREY, "#7FA7D9", "#3F7FBF", NAVY, GREEN, AMBER, PURPLE)))


def _window(days):
    t, states = natural_run()
    n = int(days * 1440)
    return t[:n], states[0:3, :n], t[1] - t[0]


def _noise(kind, t, rms_m, seed):
    return realistic_noise(t, 60.0, rms_m, seed) if kind == "realistic" else white_noise(t, rms_m, seed)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def estimator_errors():
    """errors[kind][days][parameter] = (len(NOISE_RMS_M), len(SEEDS)) array of |error| in %."""
    errors = {kind: {days: {p: np.zeros((len(NOISE_RMS_M), len(SEEDS))) for p in PARAMETERS} for days in HORIZONS}
              for kind in NOISE_STYLES}
    for days in HORIZONS:
        t, positions, h = _window(days)
        for kind in NOISE_STYLES:
            for i, rms in enumerate(NOISE_RMS_M):
                for j, seed in enumerate(SEEDS):
                    measured = positions + _noise(kind, t, rms, seed)
                    estimates, _ = fit_all_perturbations(t, measured, h)
                    estimates["cd (energy)"] = fit_cd_energy(t, measured, h, j2=J2)[0]
                    for p in PARAMETERS:
                        errors[kind][days][p][i, j] = 100 * abs(estimates[p] / TRUTH[p] - 1)
    return errors


def noise_examples(days=2, rms_m=1.0):
    """Noise samples, spectra and their effect on the FD acceleration and the mean SMA (2 days)."""
    t, positions, h = _window(days)
    step_s = 60.0
    parts = noise_components(t, step_s, np.random.default_rng(0))
    white, real = white_noise(t, rms_m, 0), realistic_noise(t, step_s, rms_m, 0)
    spectra = {}
    in_metres = [("white noise", white * DU * 1000), ("realistic mixture", real * DU * 1000)]
    in_metres += [(name, rms_m * np.sqrt(MIXTURE_SHARES[name]) * part) for name, part in parts.items()]
    for name, x in in_metres:                    # every curve in m^2 / Hz; the components add up to the mixture
        f, p = welch(x[0], fs=1 / step_s, nperseg=1024)
        spectra[name] = (f * ORBITAL_PERIOD_S, p)          # Hz -> cycles per orbit
    effects = {}
    for name, noise in (("white", white), ("realistic", real)):
        r_clean, v_clean, a_clean = fd_velocity_and_acceleration(positions, h)
        r_noisy, v_noisy, a_noisy = fd_velocity_and_acceleration(positions + noise, h)
        effects[name] = dict(accel=np.linalg.norm(a_noisy - a_clean, axis=0) * ACCEL_UNIT_MS2,
                             sma=(energy_sma(r_noisy, v_noisy, j2=J2) - energy_sma(r_clean, v_clean, j2=J2)) * DU * 1000)
    hours = np.arange(len(t)) / 60.0             # samples are 1 minute apart
    return dict(hours=hours, white=white[0] * DU * 1000, real=real[0] * DU * 1000, spectra=spectra,
                effects=effects, effect_hours=hours[_trim(len(hours), effects["white"]["accel"])])


def _trim(n_samples, trimmed):
    """The slice of the original samples that an FD result of len(trimmed) covers (equal trim at both ends)."""
    half = (n_samples - len(trimmed)) // 2
    return slice(half, n_samples - half)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_series(ax, ex):
    ax.plot(ex["hours"], ex["white"], color=GREY, lw=0.5, label="white noise")
    ax.plot(ex["hours"], ex["real"], color=RUST, lw=0.8, label="realistic mixture")
    ax.set(xlabel="time [hours]", ylabel="x-position noise [m]", title="Same RMS (1 m), different spectrum")
    ax.legend(fontsize=8)


def _draw_spectra(ax, ex):
    for name, (f, p) in ex["spectra"].items():
        if name in ("white noise", "realistic mixture"):
            ax.loglog(f[1:], p[1:], color="black" if name == "realistic mixture" else GREY, lw=2, label=name)
        else:
            ax.loglog(f[1:], p[1:], color=COMPONENT_COLORS[name], lw=0.9,
                      label=f"{name} ({100 * MIXTURE_SHARES[name]:.0f}%)")
    ax.axvline(1, color=SLATE, ls=":", lw=0.8)
    ax.set(xlabel="frequency [cycles per orbit]", ylabel="PSD of x-position noise [m$^2$/Hz]",
           title="The mixture: white noise at several rates + 1/f + random walk + periodic")
    ax.legend(fontsize=6.5, ncol=2, loc="lower left")


def _draw_effects(ax_accel, ax_sma, ex):
    for name, color in (("white", GREY), ("realistic", RUST)):
        effect = ex["effects"][name]
        ax_accel.semilogy(ex["effect_hours"], effect["accel"], color=color, lw=0.6,
                          label=f"{name}: median {np.median(effect['accel']):.1e} m/s$^2$")
        ax_sma.plot(ex["effect_hours"], effect["sma"], color=color, lw=0.6,
                    label=f"{name}: RMS {np.sqrt(np.mean(effect['sma'] ** 2)):.1f} m")
    ax_accel.set(xlabel="time [hours]", ylabel="FD acceleration error [m/s$^2$]",
                 title="2nd derivative: fast noise hurts, slow noise is filtered out")
    ax_sma.set(xlabel="time [hours]", ylabel="mean-SMA error [m]",
               title="Mean-SMA error: smaller per sample, but its slow part does not average away")
    ax_accel.legend(fontsize=8)
    ax_sma.legend(fontsize=8)


def _draw_estimator(ax, errors, parameter):
    for days in HORIZONS:
        for kind, style in NOISE_STYLES.items():
            values = errors[kind][days][parameter]
            median = np.median(values, axis=1)
            ax.loglog(NOISE_RMS_M, np.maximum(median, 1e-6), color=HORIZON_COLORS[days], **style,
                      label=f"{kind}, {days} d")
            ax.fill_between(NOISE_RMS_M, np.maximum(values.min(axis=1), 1e-6), np.maximum(values.max(axis=1), 1e-6),
                            color=HORIZON_COLORS[days], alpha=0.08)
    ax.axhline(100, color=GREY, ls=":", lw=0.8)
    ax.set(xlabel="noise RMS per axis [m]", ylabel="|error| [%]", title=parameter)


def _draw_ratio(ax, errors):
    i = NOISE_RMS_M.index(RATIO_NOISE_M)
    x = np.arange(len(PARAMETERS))
    for k, days in enumerate(HORIZONS):
        ratios = [np.median(errors["realistic"][days][p][i]) / np.median(errors["white"][days][p][i]) for p in PARAMETERS]
        ax.bar(x + (k - 0.5) * 0.38, ratios, 0.38, color=HORIZON_COLORS[days], label=f"{days} days")
        for xi, value in zip(x + (k - 0.5) * 0.38, ratios):
            ax.text(xi, value * 1.15, f"{value:.2g}", ha="center", fontsize=7)
    ax.axhline(1, color="black", lw=1)
    ax.set(yscale="log", xticks=x, xticklabels=PARAMETERS, xlabel="estimated parameter",
           ylabel="realistic error / white error",
           title=f"Same RMS ({RATIO_NOISE_M:g} m), different spectrum: >1 = realistic noise is worse")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend()


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    folder = step_dir(out_dir, "step3_drag_noise")
    ex = noise_examples()
    errors = estimator_errors()

    with report_style():
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        _draw_series(axes[0, 0], ex); _draw_spectra(axes[0, 1], ex); _draw_effects(axes[1, 0], axes[1, 1], ex)
        for ax, letter in zip(axes.ravel(), "abcd"):
            panel_label(ax, letter)
        save(fig, folder, "step3_noise_models.png", suptitle="Realistic noise: a mixture of frequencies (1 m RMS)")

        fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
        for ax, parameter in zip(axes.ravel(), PARAMETERS):
            _draw_estimator(ax, errors, parameter)
        axes[0, 0].legend(fontsize=8)
        save(fig, folder, "step3_noise_estimators.png",
             suptitle="Every estimator: white (solid) vs realistic (dashed) noise of the same RMS, median of 3 seeds")

        fig, ax = plt.subplots(figsize=(10, 4.8)); _draw_ratio(ax, errors); save(fig, folder, "step3_noise_ratio.png")

    i = NOISE_RMS_M.index(RATIO_NOISE_M)
    return dict(report_step3_noise=dict(
        mixture_variance_shares=MIXTURE_SHARES,
        median_error_percent={kind: {f"{days}d": {p: dict(zip([f"{n:g}m" for n in NOISE_RMS_M],
                                                               np.median(v, axis=1).tolist()))
                                                  for p, v in per_days.items()} for days, per_days in per_kind.items()}
                              for kind, per_kind in errors.items()},
        realistic_over_white_at_1m={f"{days}d": {p: float(np.median(errors["realistic"][days][p][i]) /
                                                          np.median(errors["white"][days][p][i])) for p in PARAMETERS}
                                    for days in HORIZONS}))

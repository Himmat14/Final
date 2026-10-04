"""
Workstream A: robustness of the FFT method for spotting the J2 signature, on the SAME data and with
the SAME settings as report step 4: the 400-day natural run, the radial unmodelled acceleration from
6th-order finite differences of noisy positions (report_step4.radial_residual), and the J2 peak SNR
with the noise floor taken from 0-6 cycles/orbit and the peak searched within +-1% of 2 cycles/orbit.

The Week 4 version used its own 3-day J2-only trajectory and km-level noise at one cadence.

Figures (outputs/report/step4_spectral_kalman/)
    spectral_snr_vs_noise           J2 SNR vs position noise, for 3-day and 15-day records
    spectral_snr_vs_cadence         J2 SNR vs sampling step (red = past the J2 Nyquist limit)
    spectral_periodogram_examples   J2 peak resolved at 1 min, lost at 90 min (15 days)
    spectral_snr_surface_3d         3D surface: SNR over (sampling step x noise), 15 days
    spectral_waterfall_3d           3D waterfall: the spectrum for record lengths 15 ... 360 days;
                                    the J2 line sharpens and rises out of the noise as the record grows
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband.long_run import HORIZONS_DAYS
from deadband.spectral import j2_nyquist_limit_minutes, periodogram_snr
from .common import AMBER, NAVY, RUST, SLATE
from .report_common import report_style, save, step_dir
from .report_step4 import BACKGROUND_MAX_CYCLES_PER_ORBIT, PEAK_SEARCH_FRACTION, radial_residual

NOISE_KM = (0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
RECORD_DAYS = (3, 15)
NOISE_STEP_MIN = 5
CADENCES_MIN = (1, 2, 5, 10, 15, 20, 23, 26, 30, 45, 60, 90)
CADENCE_NOISE_KM = 0.1
EXAMPLE_DAYS = 15
GOOD_CADENCE_MIN, BAD_CADENCE_MIN = 1, 90
SURFACE_CADENCES_MIN = (1, 2, 5, 10, 20, 30, 45, 60)
SURFACE_NOISE_KM = (0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
WATERFALL_STEP_MIN = 10
WATERFALL_NOISE_KM = 1.0
WATERFALL_BAND = (1.8, 2.2)        # zoom on the J2 line (cycles per orbit)


def snr(days, step_min, noise_km, seed=0):
    """J2 peak SNR [dB] with the step 4 settings."""
    return periodogram_snr(*radial_residual(days, step_min, noise_km, seed), BACKGROUND_MAX_CYCLES_PER_ORBIT,
                           PEAK_SEARCH_FRACTION).snr_db


def periodogram(days, step_min, noise_km, seed=0):
    return periodogram_snr(*radial_residual(days, step_min, noise_km, seed), BACKGROUND_MAX_CYCLES_PER_ORBIT,
                           PEAK_SEARCH_FRACTION)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_noise(ax, rows):
    for days, color in zip(RECORD_DAYS, (AMBER, NAVY)):
        ax.semilogx([max(n, 1e-3) for n in NOISE_KM], rows[days], "o-", color=color, label=f"{days}-day record")
    ax.set(xlabel="position noise [km] (0 drawn at 0.001)", ylabel="J2 peak SNR [dB]",
           title=f"J2 SNR vs noise ({NOISE_STEP_MIN}-min samples): a longer record buys back the SNR")
    ax.legend()


def _draw_cadence(ax, values):
    limit = j2_nyquist_limit_minutes()
    colors = [RUST if c > limit else NAVY for c in CADENCES_MIN]
    ax.plot(CADENCES_MIN, values, "-", color="#AAAAAA", lw=1, zorder=1)
    ax.scatter(CADENCES_MIN, values, c=colors, s=36, zorder=5)
    ax.axvline(limit, color=SLATE, ls="--", lw=1, label=f"J2 Nyquist limit ({limit:.1f} min)")
    ax.set(xlabel="sampling step [min]", ylabel="J2 peak SNR [dB]",
           title=f"J2 SNR vs sampling step ({EXAMPLE_DAYS} days, {CADENCE_NOISE_KM * 1000:g} m noise)")
    ax.legend()


def _draw_examples(axes, good, bad):
    for ax, p, step in ((axes[0], good, GOOD_CADENCE_MIN), (axes[1], bad, BAD_CADENCE_MIN)):
        ax.semilogy(p.frequencies / p.orbital_frequency, p.power + 1e-30, color=NAVY, lw=0.8)
        ax.axvline(2.0, color=RUST, ls="--", lw=1, label="2 cycles/orbit (J2)")
        verdict = "J2 peak resolved" if p.snr_db > 10 else "J2 peak lost (aliased)"
        ax.set(xlim=(0, 6), xlabel="frequency [cycles per orbit]", ylabel="power",
               title=f"{step} min, {EXAMPLE_DAYS} days: {verdict} (SNR {p.snr_db:.0f} dB)")
        ax.legend()


def _draw_surface(ax, surface):
    x, y = np.meshgrid(np.log10(SURFACE_CADENCES_MIN), np.log10(SURFACE_NOISE_KM))
    ax.plot_surface(x, y, surface.T, cmap="viridis", edgecolor="k", lw=0.2, alpha=0.95)
    ax.contour(x, y, surface.T, levels=[10], zdir="z", offset=np.nanmin(surface), colors=RUST)
    ax.set_xticks(np.log10(SURFACE_CADENCES_MIN))
    ax.set_xticklabels([str(c) for c in SURFACE_CADENCES_MIN], fontsize=7)
    ax.set_yticks(np.log10(SURFACE_NOISE_KM))
    ax.set_yticklabels([f"{n:g}" for n in SURFACE_NOISE_KM], fontsize=7)
    ax.set_xlabel("sampling step [min]", labelpad=6)
    ax.set_ylabel("noise [km]", labelpad=6)
    ax.set_zlabel("J2 SNR [dB]")
    ax.set_title(f"J2 SNR surface ({EXAMPLE_DAYS} days): noise and slow sampling both bury the peak\n"
                 "(red contour on the floor: 10 dB, the 'resolved' line)", fontsize=10)
    ax.view_init(elev=25, azim=-50)


def _draw_waterfall(ax, spectra):
    for k, (days, (f, power)) in enumerate(spectra.items()):
        shown = (f >= WATERFALL_BAND[0]) & (f <= WATERFALL_BAND[1])
        level = 10 * np.log10(power[shown] / np.median(power[shown]) + 1e-30)
        ax.plot(f[shown], np.full(shown.sum(), k), level, color=plt.cm.viridis(k / len(spectra)), lw=0.8)
    ax.set_yticks(range(len(spectra)))
    ax.set_yticklabels([f"{d} d" for d in spectra], fontsize=8)
    ax.set_xlabel("frequency [cycles per orbit]", labelpad=6)
    ax.set_ylabel("record length", labelpad=6)
    ax.set_zlabel("power above the median [dB]")
    ax.set_title(f"Spectrum near 2 cycles/orbit vs record length ({WATERFALL_STEP_MIN}-min data, "
                 f"{WATERFALL_NOISE_KM:g} km noise):\nthe J2 line sharpens (finer bins) and rises out of the noise",
                 fontsize=10)
    ax.view_init(elev=25, azim=-60)


def run(sim, out_dir):
    """`sim` is not used: the data are the report's natural run (see report_step4.radial_residual)."""
    folder = step_dir(out_dir, "step4_spectral_kalman")
    noise_rows = {days: [snr(days, NOISE_STEP_MIN, n) for n in NOISE_KM] for days in RECORD_DAYS}
    cadence_rows = [snr(EXAMPLE_DAYS, c, CADENCE_NOISE_KM) for c in CADENCES_MIN]
    good = periodogram(EXAMPLE_DAYS, GOOD_CADENCE_MIN, CADENCE_NOISE_KM, seed=1)
    bad = periodogram(EXAMPLE_DAYS, BAD_CADENCE_MIN, CADENCE_NOISE_KM, seed=1)
    surface = np.array([[snr(EXAMPLE_DAYS, c, n) for n in SURFACE_NOISE_KM] for c in SURFACE_CADENCES_MIN])
    spectra = {}
    for days in HORIZONS_DAYS:
        p = periodogram(days, WATERFALL_STEP_MIN, WATERFALL_NOISE_KM)
        spectra[days] = (p.frequencies / p.orbital_frequency, p.power)

    with report_style():
        fig, ax = plt.subplots(figsize=(8, 4.5)); _draw_noise(ax, noise_rows); save(fig, folder, "spectral_snr_vs_noise.png")
        fig, ax = plt.subplots(figsize=(8, 4.5)); _draw_cadence(ax, cadence_rows); save(fig, folder, "spectral_snr_vs_cadence.png")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.3)); _draw_examples(axes, good, bad)
        save(fig, folder, "spectral_periodogram_examples.png")
        fig = plt.figure(figsize=(9, 7)); _draw_surface(fig.add_subplot(111, projection="3d"), surface)
        save(fig, folder, "spectral_snr_surface_3d.png")
        fig = plt.figure(figsize=(10, 7)); _draw_waterfall(fig.add_subplot(111, projection="3d"), spectra)
        save(fig, folder, "spectral_waterfall_3d.png")

    return dict(
        spectral_noise_sweep={f"{days}d": dict(zip([f"{n:g}km" for n in NOISE_KM], values))
                              for days, values in noise_rows.items()},
        spectral_cadence_sweep=dict(zip([f"{c}min" for c in CADENCES_MIN], cadence_rows)),
        spectral_nyquist_limit_min=float(j2_nyquist_limit_minutes()),
        spectral_example=dict(good_cadence_min=GOOD_CADENCE_MIN, good_snr_db=good.snr_db,
                              bad_cadence_min=BAD_CADENCE_MIN, bad_snr_db=bad.snr_db),
        spectral_snr_surface_db=surface.tolist())

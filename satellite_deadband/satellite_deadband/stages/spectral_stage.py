"""Workstream A (continued): robustness of the FFT method for spotting the J2 signature."""
import matplotlib.pyplot as plt

from deadband import spectral
from deadband.constants import SAMPLE_STEP_MIN
from .common import NAVY, REFERENCE_LINE, RUST, save_figure

NOISE_LIST_KM = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
# Week 5: SAMPLE_STEP_MIN (0.1 min) added as the first point of the cadence sweep
CADENCE_LIST_MIN = [SAMPLE_STEP_MIN, 2, 5, 8, 10, 12, 15, 18, 20, 22, 23.875, 26, 30, 35, 40, 47.75, 60, 75, 90, 120]
# Week 5: the fixed-cadence plots use SAMPLE_STEP_MIN (was 5.0 min)
GOOD_CADENCE_MIN, BAD_CADENCE_MIN = SAMPLE_STEP_MIN, 90.0
EXAMPLE_NOISE_KM = 0.3
TRAJECTORY_DAYS = 3.0
RESOLVED_SNR_DB = 10.0   # an example periodogram is called "resolved" above this J2 peak SNR
# Week 5: noise floor measured only up to 6 cycles/orbit (the plotted band), so a fine cadence
# is not penalised for finite-difference noise far above the frequencies of interest
BACKGROUND_MAX_CYCLES_PER_ORBIT = 6.0


def _periodogram(t_eval, states, cadence_min):
    t, residual = spectral.radial_residual_series(t_eval, states, cadence_min, EXAMPLE_NOISE_KM, seed=1)
    return spectral.periodogram_snr(t, residual, BACKGROUND_MAX_CYCLES_PER_ORBIT)


def _example_title(cadence_min, periodogram):
    verdict = "J2 peak resolved" if periodogram.snr_db > RESOLVED_SNR_DB else "J2 peak lost in noise"
    return f"{cadence_min:g} min cadence: {verdict} (SNR {periodogram.snr_db:.0f} dB)"


def run(sim, out_dir):
    # reference trajectory stored every SAMPLE_STEP_MIN so the 0.1 min samples need no interpolation
    n_points = int(round(TRAJECTORY_DAYS * 24 * 60 / SAMPLE_STEP_MIN)) + 1
    t_eval, states = spectral.true_j2_trajectory(days=TRAJECTORY_DAYS, n_points=n_points)
    noise_rows = spectral.noise_sweep(t_eval, states, NOISE_LIST_KM, cadence_min=SAMPLE_STEP_MIN,
                                      background_max_cycles_per_orbit=BACKGROUND_MAX_CYCLES_PER_ORBIT)
    cadence_rows = spectral.cadence_sweep(t_eval, states, CADENCE_LIST_MIN, noise_km=EXAMPLE_NOISE_KM,
                                          background_max_cycles_per_orbit=BACKGROUND_MAX_CYCLES_PER_ORBIT)
    nyquist_limit = spectral.j2_nyquist_limit_minutes()

    # SNR versus noise
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot([r["noise_km"] for r in noise_rows], [r["snr_db"] for r in noise_rows], "o-", color=NAVY)
    ax.set(xlabel="Position noise sigma [km]", ylabel="J2 peak SNR [dB] (noise floor: 0-6 cycles/orbit)",
           title=f"Spectral method robustness: J2 peak SNR vs position noise ({SAMPLE_STEP_MIN:g} min cadence)")
    ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "spectral_snr_vs_noise.png")

    # SNR versus cadence (points past the J2 Nyquist limit are drawn in red)
    cadences = [r["cadence_min"] for r in cadence_rows]
    snrs = [r["snr_db"] for r in cadence_rows]
    point_colors = [RUST if r["violates_j2_nyquist"] else NAVY for r in cadence_rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(cadences, snrs, "-", color="#AAAAAA", lw=1, zorder=1)
    ax.scatter(cadences, snrs, c=point_colors, zorder=5, s=36)
    ax.axvline(nyquist_limit, color=REFERENCE_LINE, ls="--", lw=1, label=f"J2-signal Nyquist limit ({nyquist_limit:.1f} min)")
    ax.set(xlabel="Sampling cadence [minutes]", ylabel="J2 peak SNR [dB] (noise floor: 0-6 cycles/orbit)",
           title="Spectral method robustness: J2 peak SNR vs sampling cadence (0.3 km noise)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "spectral_snr_vs_cadence.png")

    # Example periodograms: J2 peak resolved vs lost
    good = _periodogram(t_eval, states, GOOD_CADENCE_MIN)
    bad = _periodogram(t_eval, states, BAD_CADENCE_MIN)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, periodogram, title in [(axes[0], good, _example_title(GOOD_CADENCE_MIN, good)),
                                   (axes[1], bad, _example_title(BAD_CADENCE_MIN, bad) + ", Nyquist violated")]:
        ax.semilogy(periodogram.frequencies / periodogram.orbital_frequency, periodogram.power + 1e-30,
                    color=NAVY, lw=0.8)
        ax.axvline(2.0, color=RUST, ls="--", lw=1, label="2x orbital freq (J2)")
        ax.set(xlim=(0, 6), xlabel="Frequency [cycles per orbit]", ylabel="Power")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "spectral_periodogram_examples.png")

    return dict(
        spectral_noise_sweep=noise_rows,
        spectral_cadence_sweep=cadence_rows,
        spectral_nyquist_limit_min=float(nyquist_limit),
        spectral_example=dict(good_cadence_min=GOOD_CADENCE_MIN, good_snr_db=good.snr_db,
                              bad_cadence_min=BAD_CADENCE_MIN, bad_snr_db=bad.snr_db),
    )

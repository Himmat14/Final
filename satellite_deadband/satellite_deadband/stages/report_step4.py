"""
Report step 4: frequency content, filtering, and Kalman filtering for cd, on the 400-day natural run.

Idea
----
Every force leaves its mark in a different frequency band of the residual acceleration:
    drag   -> near zero frequency (a slow, steady decay)
    J2     -> twice the orbital frequency (latitude^2 terms)
    Moon / Sun / SRP -> tiny; the Moon also modulates over its 27-day orbit
A LONGER record gives finer frequency resolution (resolution = 1 / record length), so lines that
blur together in 15 days separate in 360 days, and the J2 peak rises further above the noise.

Filters make each force VISIBLE on its own. But for ESTIMATING a force whose shape is already
known, filtering does not help: a least-squares fit onto the exact shape is already the optimal
("matched") filter. A Kalman filter fuses each new measurement with the dynamics model and keeps
a running estimate of cd with its own uncertainty.

Figures (outputs/report/step4_spectral_kalman/)
    step4_force_spectra          spectrum of each force's radial acceleration: 15-day vs 360-day record
    step4_j2_snr                 J2 peak SNR vs data length (several noise levels) and vs sampling step
    step4_periodograms           J2 peak resolved at 5 min, lost at 90 min (Nyquist)
    step4_filtering              noisy residual, band-passed (J2) and low-passed (drag), against the truth
    step4_kalman_convergence     EKF cd estimate +- 2 sigma over 30 days
    step4_kalman_vs_methods      cd error at 15 and 30 days: EKF vs energy method vs FD regression
    step4_summary                six-panel summary
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

from deadband.constants import ACCEL_UNIT_MS2, J2, MU, PERIOD, SMOOTH_CD, to_days
from deadband.kalman import run_ekf
from deadband.long_run import HORIZONS_DAYS, natural_run
from deadband.perturbations import force_accelerations
from deadband.physics import drag_accel
from deadband.regression import (add_position_noise, fd_velocity_and_acceleration, fit_cd, fit_cd_energy, gravity_shape,
                                 known_j2)
from deadband.spectral import periodogram_snr
from .common import AMBER, GREEN, NAVY, PURPLE, RUST, SLATE
from .report_common import panel_label, report_style, save, step_dir

FORCE_COLORS = {"Drag": RUST, "J2": NAVY, "Moon": GREEN, "Sun": AMBER, "SRP": PURPLE}
ORBITAL_FREQUENCY = 1 / PERIOD
SPECTRUM_STEP_MIN = 10
SNR_NOISE_KM = (0.1, 1.0, 5.0)
SNR_STEP_MIN = 5
CADENCES_MIN = (1, 2, 5, 10, 15, 20, 23, 26, 30, 45, 60, 90)
BACKGROUND_MAX_CYCLES_PER_ORBIT = 6.0
PEAK_SEARCH_FRACTION = 0.01   # look for the J2 peak within +-1% of twice the orbital frequency
FILTER_DAYS = 3
FILTER_NOISE_KM = 0.01
EDGE_FRACTION = 0.15        # filtered signals ring at both ends (start-up transient); not drawn as truth
KALMAN_DAYS = 30
KALMAN_STEP_MIN = 10
KALMAN_NOISE_KM = (0.01, 0.1)
KALMAN_REPORT_DAYS = (15, 30)


def _window(days, stride_min=1):
    t, states = natural_run()
    n = int(days * 1440)
    return t[:n:stride_min], states[:, :n:stride_min], (t[1] - t[0]) * stride_min


# ---------------------------------------------------------------------------
# Spectra and J2 signal-to-noise
# ---------------------------------------------------------------------------
def force_spectra(days):
    """Power spectrum (vs cycles per orbit) of the radial component of each force over the first `days`."""
    t, states, h = _window(days, SPECTRUM_STEP_MIN)
    radial = {name: np.zeros(len(t)) for name in FORCE_COLORS}
    for i in range(len(t)):
        r_hat = states[0:3, i] / np.linalg.norm(states[0:3, i])
        for name, accel in force_accelerations(t[i], states[0:3, i], states[3:6, i], cd=SMOOTH_CD).items():
            if name in radial:
                radial[name][i] = accel @ r_hat * ACCEL_UNIT_MS2
    frequencies = np.fft.rfftfreq(len(t), d=h) / ORBITAL_FREQUENCY
    window = np.hanning(len(t))
    return frequencies, {name: np.abs(np.fft.rfft((x - x.mean()) * window)) ** 2 for name, x in radial.items()}


def radial_residual(days, step_min, noise_km, seed=0):
    """Radial unmodelled acceleration (FD accel - gravity - drag) from noisy positions; J2 is left IN."""
    t, states, h = _window(days, step_min)
    positions = add_position_noise(states[0:3], noise_km, seed)
    r, v, accel = fd_velocity_and_acceleration(positions, h)
    residual = accel - MU * gravity_shape(r) + SMOOTH_CD * np.linalg.norm(v, axis=0) * v
    radial = np.sum(residual * r / np.linalg.norm(r, axis=0), axis=0)
    half = (len(t) - r.shape[1]) // 2
    return t[half:len(t) - half], radial


def snr_vs_horizon():
    return {noise: [periodogram_snr(*radial_residual(h, SNR_STEP_MIN, noise), BACKGROUND_MAX_CYCLES_PER_ORBIT, PEAK_SEARCH_FRACTION).snr_db
                    for h in HORIZONS_DAYS] for noise in SNR_NOISE_KM}


def snr_vs_cadence(days=15, noise_km=0.1):
    return [periodogram_snr(*radial_residual(days, step, noise_km), BACKGROUND_MAX_CYCLES_PER_ORBIT, PEAK_SEARCH_FRACTION).snr_db
            for step in CADENCES_MIN]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def _filter(signal, h, low_cpo=None, high_cpo=None):
    """Zero-phase Butterworth filter on the last axis; band edges in cycles per orbit."""
    nyquist = 0.5 / h
    if low_cpo is None:
        sos = butter(4, high_cpo * ORBITAL_FREQUENCY / nyquist, btype="low", output="sos")
    else:
        sos = butter(4, [low_cpo * ORBITAL_FREQUENCY / nyquist, high_cpo * ORBITAL_FREQUENCY / nyquist],
                     btype="band", output="sos")
    return sosfiltfilt(sos, signal, axis=-1)


def filtering_data():
    t, states, h = _window(FILTER_DAYS)
    r, v, accel = fd_velocity_and_acceleration(add_position_noise(states[0:3], FILTER_NOISE_KM, 0), h)
    residual = accel - MU * gravity_shape(r)
    return h, r, v, residual, known_j2(r, J2)


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------
def kalman_runs():
    t, states, h = _window(KALMAN_DAYS, KALMAN_STEP_MIN)
    runs, comparison = {}, []
    for noise in KALMAN_NOISE_KM:
        measured = add_position_noise(states[0:3], noise, seed=7)
        run = run_ekf(t, measured, states[:, 0], 1.5 * SMOOTH_CD, noise)
        runs[noise] = run
        for days in KALMAN_REPORT_DAYS:
            n = int(days * 1440 / KALMAN_STEP_MIN)
            comparison.append(dict(noise_km=noise, days=days,
                                   ekf=100 * abs(run.estimates[6, n - 1] / SMOOTH_CD - 1),
                                   energy=100 * abs(fit_cd_energy(t[:n], measured[:, :n], h, j2=J2)[0] / SMOOTH_CD - 1),
                                   fd=100 * abs(fit_cd(measured[:, :n], h, j2=J2) / SMOOTH_CD - 1)))
    return runs, comparison


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_spectra(ax, spectra, names=("J2", "Drag", "Moon", "Sun", "SRP")):
    for (days, (frequencies, power)), style in zip(spectra.items(), ("-", "--")):
        for name in names:
            p = power[name]
            ax.semilogy(frequencies, p / p.max(), color=FORCE_COLORS[name], lw=1 if style == "-" else 0.8, ls=style,
                        label=f"{name}, {days} d" if name in ("J2", "Moon") or style == "-" else None)
    for k in (1, 2, 3):
        ax.axvline(k, color="gray", ls=":", lw=1)
    ax.set(xlim=(0, 3.5), ylim=(1e-10, 2), xlabel="frequency [cycles per orbit]", ylabel="normalised power (radial accel)",
           title="Where each force lives: 360 d (solid) vs 15 d (dashed)")
    ax.legend(ncol=2, fontsize=7)


def _draw_snr_horizon(ax, rows):
    for (noise, snr), color in zip(rows.items(), (NAVY, AMBER, RUST)):
        ax.semilogx(HORIZONS_DAYS, snr, "o-", color=color, label=f"{noise:g} km noise")
    ax.set(xlabel="data length [days]", ylabel="J2 peak SNR [dB]",
           title=f"J2 SNR vs data length ({SNR_STEP_MIN} min; drag slowly shifts the line)",
           xticks=HORIZONS_DAYS)
    ax.set_xticklabels([str(h) for h in HORIZONS_DAYS])
    ax.minorticks_off()
    ax.legend()


def _draw_snr_cadence(ax, snr):
    nyquist = 95.5 / 4
    colors = [RUST if c > nyquist else NAVY for c in CADENCES_MIN]
    ax.plot(CADENCES_MIN, snr, "-", color="#AAAAAA", lw=1)
    ax.scatter(CADENCES_MIN, snr, c=colors, zorder=5)
    ax.axvline(nyquist, color=SLATE, ls="--", label=f"J2-signal Nyquist ({nyquist:.1f} min)")
    ax.set(xlabel="sampling step [min]", ylabel="J2 peak SNR [dB]", title="J2 peak SNR vs sampling step (15 d, 0.1 km)")
    ax.legend()


def _draw_periodograms(axes):
    for ax, step in zip(axes, (5, 90)):
        result = periodogram_snr(*radial_residual(15, step, 0.1), BACKGROUND_MAX_CYCLES_PER_ORBIT, PEAK_SEARCH_FRACTION)
        ax.semilogy(result.frequencies / result.orbital_frequency, result.power + 1e-30, color=NAVY, lw=0.8)
        ax.axvline(2.0, color=RUST, ls="--", label="2x orbital frequency (J2)")
        verdict = "resolved" if result.snr_db > 10 else "lost"
        ax.set(xlim=(0, 6), xlabel="frequency [cycles per orbit]", ylabel="power",
               title=f"{step} min, 15 days: J2 peak {verdict} (SNR {result.snr_db:.0f} dB)")
        ax.legend()


def _draw_filtering(axes):
    h, r, v, residual, j2_true = filtering_data()
    hours = np.arange(r.shape[1]) * h * 13713.441 / 3600
    radial = r / np.linalg.norm(r, axis=0)
    along = v / np.linalg.norm(v, axis=0)
    res_radial = np.sum(residual * radial, 0) * ACCEL_UNIT_MS2
    j2_radial = np.sum(j2_true * radial, 0) * ACCEL_UNIT_MS2
    window = hours < 6
    axes[0].plot(hours[window], res_radial[window], color=SLATE, lw=0.5, alpha=0.6,
                 label=f"raw residual ({FILTER_NOISE_KM * 1000:.0f} m noise)")
    axes[0].plot(hours[window], _filter(res_radial, h, 1.5, 2.5)[window], color=NAVY, lw=1.6, label="band-passed 1.5-2.5 /orbit")
    axes[0].plot(hours[window], _filter(j2_radial, h, 1.5, 2.5)[window], color=AMBER, lw=1.2, ls="--", label="true J2 (same band)")
    axes[0].set(xlabel="time [hours]", ylabel="radial accel [m/s$^2$]", title="Band-pass isolates J2")
    axes[0].legend()
    res_along = np.sum((residual - j2_true) * along, 0) * ACCEL_UNIT_MS2
    drag_along = np.sum(np.column_stack([drag_accel(v[:, i], SMOOTH_CD) for i in range(v.shape[1])]) * along, 0) * ACCEL_UNIT_MS2
    axes[1].plot(hours, _filter(res_along, h, None, 0.3), color=RUST, lw=1.8, label="low-passed < 0.3 /orbit")
    axes[1].plot(hours, drag_along, color=AMBER, lw=1.2, ls="--", label="true drag")
    axes[1].axvspan(0, hours[-1] * EDGE_FRACTION, color="gray", alpha=0.15, label="filter start-up")
    axes[1].axvspan(hours[-1] * (1 - EDGE_FRACTION), hours[-1], color="gray", alpha=0.15)
    level = abs(drag_along.mean())
    axes[1].set(xlabel="time [hours]", ylabel="along-track accel [m/s$^2$]", ylim=(-6 * level, 4 * level),
                title="Low-pass isolates drag (raw noise far off-scale)")
    axes[1].legend()


def _draw_kalman(ax, runs):
    for (noise, run), color in zip(runs.items(), (NAVY, RUST)):
        days = to_days(run.t)
        ratio, sigma = run.estimates[6] / SMOOTH_CD, run.sigmas[6] / SMOOTH_CD
        ax.plot(days, ratio, color=color, lw=1.2, label=f"{noise * 1000:g} m noise")
        ax.fill_between(days, ratio - 2 * sigma, ratio + 2 * sigma, color=color, alpha=0.15)
    ax.axhline(1, color="black", lw=0.8)
    ax.set(xlabel="time [days]", ylabel="cd estimate / true cd", ylim=(0.8, 1.6),
           title="Kalman filter: cd converges as data arrives (+-2 sigma)")
    ax.legend()


def _draw_kalman_comparison(ax, rows):
    labels = [f"{row['noise_km'] * 1000:g} m, {row['days']} d" for row in rows]
    x = np.arange(len(rows))
    for k, (key, color, name) in enumerate((("fd", RUST, "FD regression"), ("energy", NAVY, "energy method"),
                                            ("ekf", GREEN, "Kalman filter"))):
        ax.bar(x + (k - 1) * 0.27, [max(row[key], 1e-6) for row in rows], 0.27, color=color, label=name)
    ax.set(xticks=x, xticklabels=labels, yscale="log", ylabel="cd error [%]", title="cd after 15 and 30 days of data")
    ax.legend()


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    folder = step_dir(out_dir, "step4_spectral_kalman")
    spectra = {360: force_spectra(360), 15: force_spectra(15)}
    snr_horizon = snr_vs_horizon()
    snr_cadence = snr_vs_cadence()
    runs, kalman_rows = kalman_runs()

    with report_style():
        fig, ax = plt.subplots(figsize=(9, 4.8)); _draw_spectra(ax, spectra); save(fig, folder, "step4_force_spectra.png")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        _draw_snr_horizon(axes[0], snr_horizon); _draw_snr_cadence(axes[1], snr_cadence)
        save(fig, folder, "step4_j2_snr.png")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.3)); _draw_periodograms(axes); save(fig, folder, "step4_periodograms.png")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5)); _draw_filtering(axes); save(fig, folder, "step4_filtering.png")
        fig, ax = plt.subplots(figsize=(8, 4.5)); _draw_kalman(ax, runs); save(fig, folder, "step4_kalman_convergence.png")
        fig, ax = plt.subplots(figsize=(8, 4.5)); _draw_kalman_comparison(ax, kalman_rows); save(fig, folder, "step4_kalman_vs_methods.png")

        fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
        _draw_spectra(axes[0, 0], spectra)
        _draw_snr_horizon(axes[0, 1], snr_horizon)
        _draw_snr_cadence(axes[0, 2], snr_cadence)
        _draw_filtering([axes[1, 0], axes[1, 1]])
        _draw_kalman_comparison(axes[1, 2], kalman_rows)
        for ax, letter in zip(axes.ravel(), "abcdef"):
            panel_label(ax, letter)
        save(fig, folder, "step4_summary.png", suptitle="Step 4: spectral analysis, filtering and Kalman filtering (400-day data)")

    return dict(report_step4=dict(
        j2_snr_db_by_horizon={f"{noise:g}km": dict(zip([f"{h}d" for h in HORIZONS_DAYS], snr))
                              for noise, snr in snr_horizon.items()},
        j2_snr_db_by_cadence=dict(zip([f"{c}min" for c in CADENCES_MIN], snr_cadence)),
        kalman_vs_methods=kalman_rows))

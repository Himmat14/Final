"""
Report step 4 (part 2): "force spectroscopy". Can the spectrum of a trajectory's unmodelled
acceleration tell us WHICH perturbations act on it and in WHAT MIXTURE, the way an IR spectrum
tells an organic chemist which functional groups a molecule has? Method: deadband/force_spectroscopy.py.

Data: the 400-day natural run (every force, realistic drag), every SPECTROSCOPY_STEP_MIN minutes.
    "measured" mixture   days 0 .. D             (D = 15, 30, 60, 120, 240, 360)
    reference library    the LAST D days of the run (a different stretch of data, like a reference
                         library recorded on another day); for 240 and 360 days the two overlap.

Questions and answers (figures in outputs/report/step4_spectral_kalman/)
    step4_spectroscopy_spectrum      the "IR spectrum": every force's lines in the radial, along-track and
                                     cross-track spectra, with each force's diagnostic band marked
    step4_spectroscopy_composition   do normalised spectral powers reproduce the step 1 force budget (%)?
                                     power: no (it squares the amplitudes); sqrt(power): yes
    step4_spectroscopy_accuracy      how well each method recovers each force's strength, vs record length
                                     and for random mixtures (each force x 0.1 ... x 10)
    step4_spectroscopy_speed         wall time vs record length: FFT + band reading vs regression
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband.constants import ACCEL_UNIT_MS2, MU
from deadband.force_spectroscopy import (COMPONENTS, FORCES, amplitude_shares, band_reading, budget_shares,
                                         cycles_per_orbit, diagnostic_bands, force_signals, peak_fit, power_shares,
                                         shares_from_strengths, spectrum, time_domain_fit, timed, to_rtn)
from deadband.long_run import HORIZONS_DAYS, LONG_DAYS, natural_run
from deadband.regression import fd_velocity_and_acceleration, gravity_shape
from .common import AMBER, GREEN, GREY, NAVY, PURPLE, RUST, SLATE
from .report_common import report_style, save, step_dir

SPECTROSCOPY_STEP_MIN = 2           # sampling step of the spectroscopy data
EDGE = 4                            # samples left out at each end of the time-domain fit (padded FD ends)
SHOWCASE_DAYS = 120                 # record length of the spectrum / composition figures
N_RANDOM_MIXTURES = 25              # random mixtures in the accuracy test
FORCE_COLORS = {"Drag": RUST, "J2": NAVY, "Moon": GREEN, "Sun": AMBER, "SRP": PURPLE}
METHOD_COLORS = {"band reading (IR-style)": RUST, "phase-aware peak fit": GREEN, "time-domain regression": NAVY}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def window(start_day, days):
    """(t, r, v, step) of the natural run from `start_day` for `days` days, every SPECTROSCOPY_STEP_MIN."""
    t, states = natural_run()
    stride = SPECTROSCOPY_STEP_MIN                      # the natural run is sampled every minute
    first, n = int(start_day * 1440), int(days * 1440)
    keep = slice(first, first + n, stride)
    return t[keep], states[0:3, keep], states[3:6, keep], (t[1] - t[0]) * stride


def library_for(days):
    """Reference spectra from the LAST `days` days of the run (a different stretch of data)."""
    t, r, v, _ = window(LONG_DAYS - days - 1, days)
    signals = force_signals(t, r, v)
    return signals, {name: spectrum(a) for name, a in signals.items()}


def measured_residual(t, r, v, step):
    """
    The unmodelled acceleration as a tracking analyst would get it: finite differences of the
    POSITIONS (the tuned stencil order), minus point-mass gravity, in RTN [m/s^2]. The stencil trims a
    few samples at each end; they are padded back with the end values so the length matches the library.
    """
    r_mid, v_fd, accel = fd_velocity_and_acceleration(r, step)
    residual = to_rtn(accel - MU * gravity_shape(r_mid), r_mid, v_fd) * ACCEL_UNIT_MS2
    half = (r.shape[1] - residual.shape[1]) // 2
    return np.pad(residual, ((0, 0), (half, half)), mode="edge")


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------
def showcase(days=SHOWCASE_DAYS):
    """Everything the spectrum and composition figures need, for one record length."""
    t, r, v, step = window(0, days)
    signals = force_signals(t, r, v)
    spectra = {name: spectrum(a) for name, a in signals.items()}
    lib_signals, library = library_for(days)
    bands = diagnostic_bands(library)
    measured = spectrum(measured_residual(t, r, v, step))
    composition = {
        "force budget (step 1 bar chart)": budget_shares(signals),
        "spectral power share": power_shares(spectra),
        "sqrt(spectral power) share": amplitude_shares(spectra),
        "band reading of the measured spectrum": shares_from_strengths(band_reading(measured, library, bands),
                                                                       lib_signals),
    }
    return dict(frequencies=cycles_per_orbit(len(t), step), spectra=spectra, mixture=spectrum(sum(signals.values())),
                measured=measured, bands=bands, composition=composition, days=days)


def accuracy_vs_horizon():
    """Recovered / true strength of every force for each record length (natural mixture, measured positions)."""
    rows = {method: {name: [] for name in FORCES} for method in METHOD_COLORS}
    for days in HORIZONS_DAYS:
        t, r, v, step = window(0, days)
        signals = force_signals(t, r, v)
        _, library = library_for(days)
        measured_signal = measured_residual(t, r, v, step)
        measured = spectrum(measured_signal)
        same_window = {name: spectrum(a) for name, a in signals.items()}
        results = {"band reading (IR-style)": band_reading(measured, library),
                   "phase-aware peak fit": peak_fit(measured, same_window),
                   "time-domain regression": time_domain_fit(measured_signal[:, EDGE:-EDGE],
                                                             {n: a[:, EDGE:-EDGE] for n, a in signals.items()})}
        for method, strengths in results.items():
            for name in FORCES:
                rows[method][name].append(strengths[name])
    return rows


def random_mixtures(days=SHOWCASE_DAYS, n=N_RANDOM_MIXTURES, seed=0):
    """Scale every force by a random factor 10^U(-1, 1) and recover the factors: {method: {force: ratios}}."""
    t, r, v, _ = window(0, days)
    signals = force_signals(t, r, v)
    same_window = {name: spectrum(a) for name, a in signals.items()}
    _, library = library_for(days)
    rng = np.random.default_rng(seed)
    out = {method: {name: [] for name in FORCES} for method in METHOD_COLORS}
    for _ in range(n):
        scales = dict(zip(FORCES, 10 ** rng.uniform(-1, 1, len(FORCES))))
        mixture_signal = sum(scales[name] * signals[name] for name in FORCES)
        mixture = spectrum(mixture_signal)
        results = {"band reading (IR-style)": band_reading(mixture, library),   # bands fixed by the library
                   "phase-aware peak fit": peak_fit(mixture, same_window),
                   "time-domain regression": time_domain_fit(mixture_signal, signals)}
        for method, strengths in results.items():
            for name in FORCES:
                out[method][name].append(strengths[name] / scales[name])
    return out


def speed_vs_horizon():
    """
    Wall time [ms] of each step for each record length. The band reading reuses a library built
    once, so per new data set it only needs the FFT. Regression and the peak fit need the model
    evaluated along THIS orbit (the force shapes), which is the expensive part.
    """
    rows = []
    for days in HORIZONS_DAYS:
        t, r, v, step = window(0, days)
        _, library = library_for(days)
        bands = diagnostic_bands(library)
        measured_signal = measured_residual(t, r, v, step)
        mixture, fft_s = timed(spectrum, measured_signal)
        _, read_s = timed(band_reading, mixture, library, bands)
        signals, model_s = timed(force_signals, t, r, v)
        same_window = {name: spectrum(a) for name, a in signals.items()}
        _, peak_s = timed(peak_fit, mixture, same_window)
        _, regression_s = timed(time_domain_fit, measured_signal, signals)
        rows.append(dict(days=days, samples=len(t), fft_ms=1e3 * fft_s, band_read_ms=1e3 * read_s,
                         model_ms=1e3 * model_s, peak_fit_ms=1e3 * (fft_s + model_s + peak_s),
                         regression_ms=1e3 * (model_s + regression_s), ir_total_ms=1e3 * (fft_s + read_s)))
    return rows


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_spectrum(axes, d):
    f = d["frequencies"]
    shown = f <= 3.5
    for c, ax in enumerate(axes):
        ax.semilogy(f[shown], np.abs(d["mixture"][c, shown]), color="black", lw=1.4, label="mixture (all forces)")
        for name in FORCES:
            ax.semilogy(f[shown], np.abs(d["spectra"][name][c, shown]), color=FORCE_COLORS[name], lw=0.8, alpha=0.8,
                        label=name)
        labelled, last_centre = 0, None
        for name, info in d["bands"].items():
            if info["component"] == c:
                band = f[info["band"]]
                close = last_centre is not None and abs(band.mean() - last_centre) < 0.1
                labelled = labelled + 1 if close else 0           # stagger only labels that share a band
                last_centre = band.mean()
                ax.axvspan(band[0] - 0.01, band[-1] + 0.01, color=FORCE_COLORS[name], alpha=0.2)
                ax.text(band.mean() + 0.02 + 0.13 * labelled, 0.97 - 0.13 * labelled,
                        f"{name}\n({100 * info['purity']:.0f}% pure)", transform=ax.get_xaxis_transform(),
                        ha="left", va="top", fontsize=7, color=FORCE_COLORS[name], fontweight="bold")
        for k in (1, 2, 3):
            ax.axvline(k, color=GREY, ls=":", lw=0.8)
        ax.set(ylabel=f"{COMPONENTS[c]}\n|FFT| (sum of m/s$^2$ samples)", ylim=(1e-9, None), xlim=(-0.05, 3.5))
    axes[0].set_title(f"Force spectrum of a {d['days']}-day record: each force has its own lines "
                      "(shaded = its diagnostic band, like an IR band assignment)")
    axes[0].legend(ncol=6, fontsize=7, loc="upper right", bbox_to_anchor=(1, 0.8))
    axes[-1].set_xlabel("frequency [cycles per orbit]  (the 'wavenumber' axis)")


def _draw_composition(ax, d):
    names = list(d["composition"])
    colors = (SLATE, GREY, NAVY, RUST)
    x = np.arange(len(FORCES))
    width = 0.8 / len(names)
    for i, (label, color) in enumerate(zip(names, colors)):
        shares = d["composition"][label]
        values = [max(shares[name], 1e-10) for name in FORCES]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, values, width, color=color, label=label,
               hatch="//" if label.startswith("spectral power") else "")
    ax.set(yscale="log", ylim=(1e-9, 300), xticks=x, xticklabels=FORCES, ylabel="share of the perturbing force [%]",
           title=f"Composition from the spectrum ({d['days']} days): sqrt(power) matches the budget, power does not")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
    ax.text(0.99, 0.97, "power ~ amplitude$^2$: J2's 99.98% becomes 99.999999%\n"
                        "Parseval: sum |X|$^2$ = N x mean(a$^2$), so sqrt(power) ~ RMS ~ mean |a|",
            transform=ax.transAxes, fontsize=8, ha="right", va="top", color=SLATE)


def _draw_accuracy_horizon(ax, rows):
    for method, style in zip(METHOD_COLORS, ("o-", "s--", "^:")):
        for name in FORCES:
            ratios = np.abs(np.array(rows[method][name]) - 1) * 100
            ax.loglog(HORIZONS_DAYS, np.maximum(ratios, 1e-6), style, color=FORCE_COLORS[name], ms=4,
                      label=f"{name}" if method == "band reading (IR-style)" else None,
                      alpha=1.0 if method == "band reading (IR-style)" else 0.5)
    ax.set(xticks=HORIZONS_DAYS, xlabel="record length [days]", ylabel="|recovered / true - 1| [%]",
           title="Recovered strength (measured positions): band reading (solid),\npeak fit (dashed), regression (dotted)")
    ax.set_xticklabels([str(h) for h in HORIZONS_DAYS])
    ax.minorticks_off()
    ax.axhline(10, color=GREY, lw=0.8, ls="--")
    ax.legend(fontsize=7, ncol=2)


def _draw_random_mixtures(ax, mixtures):
    positions = np.arange(len(FORCES))
    for i, (method, color) in enumerate(METHOD_COLORS.items()):
        data = [np.clip(mixtures[method][name], 1e-3, 1e3) for name in FORCES]
        parts = ax.boxplot(data, positions=positions + (i - 1) * 0.27, widths=0.22, patch_artist=True,
                           showfliers=True, flierprops=dict(markersize=2))
        for box in parts["boxes"]:
            box.set(facecolor=color, alpha=0.6)
        ax.plot([], [], "s", color=color, label=method)
    ax.axhline(1, color="black", lw=0.8)
    ax.set(yscale="log", ylim=(1e-2, 1e2), xticks=positions, xticklabels=FORCES, ylabel="recovered / true strength",
           title=f"{N_RANDOM_MIXTURES} random mixtures (each force x 0.1 ... x 10), {SHOWCASE_DAYS} days")
    ax.legend(fontsize=7, loc="upper left")


def _draw_speed(ax, rows):
    days = [row["days"] for row in rows]
    for key, label, color, style in (("ir_total_ms", "band reading: FFT + read bands (library reused)", RUST, "o-"),
                                     ("fft_ms", "   of which the FFT", RUST, ":"),
                                     ("model_ms", "evaluating the force model along the orbit", GREY, "--"),
                                     ("peak_fit_ms", "phase-aware peak fit (FFT + model + fit)", GREEN, "s-"),
                                     ("regression_ms", "time-domain regression (model + fit)", NAVY, "^-")):
        ax.loglog(days, [row[key] for row in rows], style, color=color, label=label)
    ax.set(xticks=days, xlabel="record length [days]", ylabel="wall time [ms]",
           title=f"Speed: sampled every {SPECTROSCOPY_STEP_MIN} min "
                 f"({rows[0]['samples']:,} to {rows[-1]['samples']:,} samples)")
    ax.set_xticklabels([str(h) for h in days])
    ax.minorticks_off()
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    folder = step_dir(out_dir, "step4_spectral_kalman")
    d = showcase()
    horizon_rows = accuracy_vs_horizon()
    mixtures = random_mixtures()
    speed_rows = speed_vs_horizon()

    with report_style():
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        _draw_spectrum(axes, d)
        save(fig, folder, "step4_spectroscopy_spectrum.png")
        fig, ax = plt.subplots(figsize=(11, 5)); _draw_composition(ax, d); save(fig, folder, "step4_spectroscopy_composition.png")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        _draw_accuracy_horizon(axes[0], horizon_rows); _draw_random_mixtures(axes[1], mixtures)
        save(fig, folder, "step4_spectroscopy_accuracy.png")
        fig, ax = plt.subplots(figsize=(9, 5)); _draw_speed(ax, speed_rows); save(fig, folder, "step4_spectroscopy_speed.png")

    def median_error(values):
        return float(np.median(np.abs(np.array(values) - 1)) * 100)

    return dict(report_step4_spectroscopy=dict(
        composition_percent={label: {k: float(v) for k, v in shares.items()} for label, shares in d["composition"].items()},
        diagnostic_band_purity={name: info["purity"] for name, info in d["bands"].items()},
        strength_by_horizon={method: {name: dict(zip([f"{h}d" for h in HORIZONS_DAYS], values))
                                      for name, values in per_force.items()} for method, per_force in horizon_rows.items()},
        random_mixture_median_error_percent={method: {name: median_error(values) for name, values in per_force.items()}
                                             for method, per_force in mixtures.items()},
        speed_ms=speed_rows))

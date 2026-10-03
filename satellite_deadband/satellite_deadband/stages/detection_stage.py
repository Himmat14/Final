"""
Workstream C/G: separate thruster (burn) dynamics from orbital (coast) dynamics.

Covers the five basic detectors, the noise and sampling-cadence sweeps, and the
explanatory plots (ground-truth timeline, why the evidence is log-transformed, and what
each method flags).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from deadband import detectors, method_comparison
from deadband.constants import to_hours
from deadband.evidence import build_evidence, log_transform
from .common import GREY, METHOD_COLORS, NAVY, PURPLE, REFERENCE_LINE, RUST, SKY, save_figure

NOISE_FRACS = [0.0, 0.5, 1.0, 2.0, 4.0]
CADENCES_MIN = [0.2, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90]


def _plot_noise_robustness(rows, out_dir):
    noise_fracs = sorted({r["noise_frac"] for r in rows})
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for name, color in zip(detectors.DETECTORS, METHOD_COLORS):
        scores = [next(r["balanced_accuracy"] for r in rows if r["method"] == name and r["noise_frac"] == nf)
                  for nf in noise_fracs]
        ax.plot(noise_fracs, scores, "o-", color=color, label=name)
    ax.axhline(0.5, color=REFERENCE_LINE, ls=":", lw=1, label="chance level")
    ax.set(xlabel="Added noise (fraction of signal std)", ylabel="Balanced accuracy (coast vs. burn)",
           title="Separating thruster dynamics from orbital dynamics: five basic methods vs. noise")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "detect_noise_robustness.png")


def _plot_gmm_split(evidence, log_signal, gmm_flags, out_dir):
    hours = to_hours(evidence.t)
    is_burn = evidence.true_label == 1
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.plot(hours, log_signal, color=GREY, lw=0.7, label="log evidence signal")
    ax.scatter(hours[is_burn], log_signal[is_burn], color=RUST, s=18, zorder=5, label="true burn sample")
    ax.scatter(hours[gmm_flags == 1], log_signal[gmm_flags == 1], facecolors="none", edgecolors=NAVY,
               s=60, zorder=4, label="GMM-flagged sample")
    ax.set(xlabel="Time [hours]", ylabel="log(|delta SMA|)",
           title="Gaussian mixture split on the log evidence signal, zero added noise")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "detect_gmm_split_example.png")


def _plot_activation_timeline(sim, out_dir):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.axhspan(0, 1, color=SKY)
    for burn_hour in to_hours(sim.burn_times):
        ax.axvspan(burn_hour - 0.035, burn_hour + 0.035, color=RUST, lw=0)
    ax.set_yticks([])
    ax.set(xlim=(0, to_hours(sim.t[-1])), xlabel="Time [hours]",
           title="Activation timeline: which dynamics are driving the satellite, instant by instant")
    ax.legend(handles=[Patch(color=SKY, label="coast: orbital dynamics only (gravity + drag)"),
                       Patch(color=RUST, label="burn: thruster dynamics active (raise + trim impulse)")],
              loc="upper right", fontsize=9)
    save_figure(fig, out_dir, "activation_timeline.png")


def _plot_raw_vs_log_histograms(evidence, out_dir):
    raw, log_signal, labels = evidence.jump_sma, evidence.log_sma, evidence.true_label
    raw_bins = np.linspace(raw.min(), raw.max(), 61)
    log_bins = np.linspace(log_signal.min(), log_signal.max(), 61)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, values, bins, xlabel, title in [
        (axes[0], raw, raw_bins, "|delta SMA| (raw, linear scale)",
         "Raw evidence: coast collapses to one hairline spike at 0"),
        (axes[1], log_signal, log_bins, "log(|delta SMA|)",
         "Log-transformed: coast and the 3 burn magnitudes are now comparably spaced"),
    ]:
        ax.hist(values[labels == 0], bins=bins, color=GREY, alpha=0.75, label="coast")
        ax.hist(values[labels == 1], bins=bins, color=RUST, alpha=0.85, label="burn")
        ax.set(xlabel=xlabel, ylabel="count", title=title)
        ax.legend(fontsize=9)
    save_figure(fig, out_dir, "jump_signal_histograms.png")


def _plot_methods_timeline(evidence, log_signal, out_dir):
    hours = to_hours(evidence.t)
    rows = {"Ground truth": evidence.true_label}
    rows.update({name: detector(log_signal.copy()) for name, detector in detectors.DETECTORS.items()})

    fig, ax = plt.subplots(figsize=(10, 4.2))
    for y_position, (name, flags) in enumerate(rows.items()):
        ax.scatter(hours[flags == 1], [y_position] * int(flags.sum()), marker="|", s=220,
                   color=RUST if name == "Ground truth" else NAVY, linewidths=1.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(rows), fontsize=10)
    ax.set(xlim=(0, hours[-1]), xlabel="Time [hours]",
           title="What each method flags as a burn, lined up against ground truth (zero added noise)")
    ax.grid(axis="x", alpha=0.3)
    save_figure(fig, out_dir, "methods_timeline_comparison.png")


def _plot_cadence_sweep(rows, out_dir):
    gmm_rows = [r for r in rows if r["method"] == detectors.GMM]
    cadences = [r["dt_min"] for r in gmm_rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(cadences, [r["balanced_accuracy"] for r in gmm_rows], "o-", color=NAVY,
            label="GMM / k-means balanced accuracy (identical in 1D)")
    ax.plot(cadences, [r["precision"] for r in gmm_rows], "o--", color=RUST, label="GMM / k-means precision")
    ax.axvline(5.0, color=REFERENCE_LINE, ls=":", lw=1, label="Week 3's standard 5-minute cadence")
    ax.set(xlabel="Sampling cadence [minutes]", ylabel="Score",
           title="Detection performance vs sampling cadence: a sweet spot, not monotonic")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "cadence_detection_sweep.png")


def _plot_precision_recall(rows, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, color, line_style in [(detectors.GMM, NAVY, "-"), (detectors.KMEANS, PURPLE, "--")]:
        method_rows = [r for r in rows if r["method"] == name]
        noise = [r["noise_frac"] for r in method_rows]
        ax.plot(noise, [r["precision"] for r in method_rows], line_style, marker="o", color=color,
                label=f"{name} -- precision")
        ax.plot(noise, [r["recall"] for r in method_rows], line_style, marker="s", color=color, alpha=0.45,
                label=f"{name} -- recall")
    ax.set(xlabel="Added noise (fraction of signal std)", ylabel="Score",
           title="Precision and recall vs noise: GMM and k-means")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "precision_recall_noise.png")


def run(sim, out_dir):
    evidence = build_evidence(sim)
    clean_log_signal = log_transform(evidence.jump_sma)
    gmm_flags = detectors.gmm_detector(clean_log_signal.copy())

    sweep_rows = detectors.noise_robustness_sweep(evidence, NOISE_FRACS)
    cadence_rows = method_comparison.cadence_detection_sweep(sim, CADENCES_MIN)
    precision_recall_rows = method_comparison.precision_recall_noise_sweep(evidence, NOISE_FRACS)

    _plot_noise_robustness(sweep_rows, out_dir)
    _plot_gmm_split(evidence, clean_log_signal, gmm_flags, out_dir)
    _plot_activation_timeline(sim, out_dir)
    _plot_raw_vs_log_histograms(evidence, out_dir)
    _plot_methods_timeline(evidence, clean_log_signal, out_dir)
    _plot_cadence_sweep(cadence_rows, out_dir)
    _plot_precision_recall(precision_recall_rows, out_dir)

    return dict(detect_sweep=sweep_rows, cadence_detection_sweep=cadence_rows,
                precision_recall_noise=precision_recall_rows)

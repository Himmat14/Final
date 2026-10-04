"""
Workstream C/G: separate thruster (burn) dynamics from orbital (coast) dynamics with five basic
detectors, on the SAME data, features, split and event scoring as report steps 6-7
(deadband/detectors.py: days 0-100 of the 400-day controlled run, TRAIN days 0-40, TEST days
40-100, evidence = log |unmodelled acceleration from v'|, noise in multiples of the step 6 reference
noise 0.1 m / 0.5 mm/s).

Figures (outputs/report/step7_robustness/)
    detect_noise_robustness        event F1 of the five detectors vs noise
    precision_recall_noise         precision and recall vs noise (GMM and k-means)
    cadence_detection_sweep        event F1 vs sampling step (6 s ... 60 s)
    activation_timeline            which dynamics are acting, first 15 TEST days
    jump_signal_histograms         the evidence, raw (linear) vs log: why detectors work on the log
    detect_gmm_split_example       one TEST burn: the log evidence, the GMM flags and the event
    methods_timeline_comparison    what each detector flags, first 15 TEST days, vs the truth
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from deadband import detectors
from deadband.constants import to_days, to_minutes
from deadband.derivative_detection import runs_of_ones
from deadband.long_run import TRAIN_DAYS
from .common import GREY, METHOD_COLORS, NAVY, PURPLE, RUST, SKY, SLATE
from .report_common import report_style, save, step_dir

NOISE_SCALES = (0.0, 0.5, 1.0, 2.0, 4.0)
STRIDES = (1, 2, 5, 10)                       # sampling steps 6, 12, 30, 60 s
TIMELINE_DAYS = 15


def _draw_noise(ax, rows):
    for name, color in zip(detectors.DETECTORS, METHOD_COLORS):
        ax.plot(NOISE_SCALES, [r["f1"] for r in rows if r["method"] == name], "o-", color=color, label=name)
    ax.set(xlabel="noise (x 0.1 m / 0.5 mm/s)", ylabel="event F1 on TEST", ylim=(-0.05, 1.05),
           title="Five basic detectors vs noise (events, TEST days 40-100)")
    ax.legend(fontsize=8)


def _draw_precision_recall(ax, rows):
    for name, color, style in ((detectors.GMM, NAVY, "-"), (detectors.KMEANS, PURPLE, "--")):
        method_rows = [r for r in rows if r["method"] == name]
        ax.plot(NOISE_SCALES, [r["precision"] for r in method_rows], style, marker="o", color=color,
                label=f"{name}: precision")
        ax.plot(NOISE_SCALES, [r["recall"] for r in method_rows], style, marker="s", color=color, alpha=0.45,
                label=f"{name}: recall")
    ax.set(xlabel="noise (x 0.1 m / 0.5 mm/s)", ylabel="score", ylim=(-0.05, 1.05),
           title="Precision and recall vs noise: k-means keeps finding the burns but drowns them in false events")
    ax.legend(fontsize=8)


def _draw_cadence(ax, rows):
    steps = sorted({r["step_s"] for r in rows})
    for name, color in zip(detectors.DETECTORS, METHOD_COLORS):
        ax.plot(steps, [r["f1"] for r in rows if r["method"] == name], "o-", color=color, label=name)
    ax.set(xscale="log", xticks=steps, xlabel="sampling step [s]", ylabel="event F1 on TEST", ylim=(-0.05, 1.05),
           title="Detection vs sampling step (reference noise)")
    ax.set_xticklabels([f"{s:g}" for s in steps])
    ax.minorticks_off()
    ax.legend(fontsize=8)


def _first_test_days(dataset):
    return dataset.test & (dataset.days < TRAIN_DAYS + TIMELINE_DAYS)


def _draw_activation(ax, dataset):
    ax.axhspan(0, 1, color=SKY)
    window = _first_test_days(dataset)
    for start, end in runs_of_ones(dataset.t[window], dataset.true_on[window]):
        ax.axvspan(to_days(start), to_days(end), color=RUST, lw=0)
    ax.set_yticks([])
    ax.set(xlim=(TRAIN_DAYS, TRAIN_DAYS + TIMELINE_DAYS), xlabel="time [days]",
           title="Activation timeline: which dynamics drive the satellite (each burn lasts ~23 min every ~3.9 days)")
    ax.legend(handles=[Patch(color=SKY, label="coast: gravity + J2 + drag only"),
                       Patch(color=RUST, label="burn: thruster on (throttle > 1/2)")], loc="upper right")


def _draw_histograms(axes, dataset, x):
    raw = np.exp(x)
    on = dataset.true_on == 1
    for ax, values, xlabel, title in (
            (axes[0], raw, "|unmodelled accel from v'| [m/s$^2$] (linear)",
             "Raw evidence: the coast is a narrow spike at 0, the burns a distant tail"),
            (axes[1], x, "log |unmodelled accel from v'|",
             "Log evidence: coast and burn become two comparable bumps")):
        bins = np.linspace(np.percentile(values, 0.01), values.max(), 120)
        ax.hist(values[~on][::10], bins=bins, color=GREY, alpha=0.75, label="coast (every 10th)")
        ax.hist(values[on], bins=bins, color=RUST, alpha=0.85, label="burn")
        ax.set(xlabel=xlabel, ylabel="count", yscale="log", title=title)
        ax.legend()


def _draw_gmm_example(ax, dataset, x, events):
    start, end = next((s, e) for s, e in runs_of_ones(dataset.t, dataset.true_on) if to_days(s) > TRAIN_DAYS)
    window = (dataset.t > start - (end - start)) & (dataset.t < end + (end - start))
    minutes = to_minutes(dataset.t[window] - start)
    ax.plot(minutes, x[window], color=GREY, lw=0.7, label="log evidence")
    on = dataset.true_on[window] == 1
    ax.plot(minutes[on], x[window][on], ".", ms=3, color=RUST, label="true burn samples")
    flagged = events[window] == 1
    ax.plot(minutes[flagged], np.full(flagged.sum(), x[window].max() + 0.3), "|", color=NAVY, ms=8,
            label="GMM burn event")
    ax.set(xlabel="minutes from the burn start", ylabel="log |unmodelled accel from v'|",
           title="One TEST burn with reference noise: the GMM event covers the burn")
    ax.legend(loc="lower right")


def _draw_methods_timeline(ax, dataset, results):
    window = _first_test_days(dataset)
    rows = {"Ground truth": dataset.true_on, **{name: events for name, (events, _) in results.items()}}
    for y, (name, flags) in enumerate(rows.items()):
        for start, end in runs_of_ones(dataset.t[window], flags[window]):
            ax.plot([to_days(start), to_days(end) + 0.02], [y, y], lw=8, solid_capstyle="butt",
                    color=RUST if name == "Ground truth" else NAVY)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(rows))
    ax.set(xlim=(TRAIN_DAYS, TRAIN_DAYS + TIMELINE_DAYS), ylim=(-0.6, len(rows) - 0.4), xlabel="time [days]",
           title="What each detector flags as burn EVENTS, first 15 TEST days (reference noise)")
    for y, (name, (_, scores)) in enumerate(results.items(), start=1):
        ax.text(TRAIN_DAYS + TIMELINE_DAYS, y, f"  F1 {scores['f1']:.2f}, {scores['n_false_events']} false",
                va="center", fontsize=8, color=SLATE)


def run(sim, out_dir):
    """`sim` is not used: the data are the report's controlled run (detectors.window_features)."""
    folder = step_dir(out_dir, "step7_robustness")
    noise_rows = detectors.noise_sweep(NOISE_SCALES)
    cadence_rows = detectors.cadence_sweep(STRIDES)
    dataset, features = detectors.window_features(1.0, seed=1)
    x = features.matrix[:, 0]
    results = detectors.run_detectors(dataset, features)

    with report_style():
        fig, ax = plt.subplots(figsize=(8.5, 4.6)); _draw_noise(ax, noise_rows); save(fig, folder, "detect_noise_robustness.png")
        fig, ax = plt.subplots(figsize=(9, 4.6)); _draw_precision_recall(ax, noise_rows)
        save(fig, folder, "precision_recall_noise.png")
        fig, ax = plt.subplots(figsize=(8.5, 4.4)); _draw_cadence(ax, cadence_rows); save(fig, folder, "cadence_detection_sweep.png")
        fig, ax = plt.subplots(figsize=(11, 2.6)); _draw_activation(ax, dataset); save(fig, folder, "activation_timeline.png")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.3)); _draw_histograms(axes, dataset, x)
        save(fig, folder, "jump_signal_histograms.png")
        fig, ax = plt.subplots(figsize=(9, 4.2)); _draw_gmm_example(ax, dataset, x, results[detectors.GMM][0])
        save(fig, folder, "detect_gmm_split_example.png")
        fig, ax = plt.subplots(figsize=(12, 4.2)); _draw_methods_timeline(ax, dataset, results)
        save(fig, folder, "methods_timeline_comparison.png")

    return dict(detect_sweep=noise_rows, cadence_detection_sweep=cadence_rows,
                detect_reference_noise={name: {k: v for k, v in scores.items() if k != "per_burn"}
                                        for name, (_, scores) in results.items()})

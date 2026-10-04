"""
Report step 7: how robust is each classifier?

Four stress tests, all scored with the event F1 on held-out data. To keep the ~40 evaluations
affordable they use days 0-100 of the 400-day run: TRAIN = days 0-40, TEST = the first 60 days
after training (step 6 reports every method over the full 15 ... 360-day TEST horizons).
    noise          measurement noise from 0 to 8 x the reference (0.1 m position, 0.5 mm/s velocity)
    clusters       number of Gaussian-mixture components (2 ... 6)
    training size  number of training samples the model is fitted on (100 ... 30,000)
    sampling step  0.1 min (6 s) ... 1 min between measurements

Figures (outputs/report/step7_robustness/)
    step7_noise_robustness     F1 and false events vs noise, every method
    step7_noise_heatmap        method x noise F1 table
    step7_n_components         GMM F1 vs number of components, at three noise levels
    step7_training_size        F1 vs number of training samples (mean +- spread over 3 random draws)
    step7_sampling_step        F1 vs sampling step
    step7_summary              six-panel summary
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt

from deadband.classifiers import (MODELS, build_features, detection_dataset, evaluate, fit_gmm,
                                  noisy_features, add_noise, REFERENCE_NOISE)
from deadband.long_run import TRAIN_DAYS
from deadband.settings import tuned
from .common import AMBER, GREEN, NAVY, PURPLE, RUST
from .report_common import panel_label, report_style, save, step_dir

METHOD_COLORS = dict(zip(MODELS, (NAVY, GREEN, RUST, PURPLE, AMBER)))
# distinct marker + line style per method, so methods with identical scores stay visible
METHOD_STYLE = {name: dict(color=METHOD_COLORS[name], marker=marker, ls=ls, ms=ms)
                for name, marker, ls, ms in zip(MODELS, ("o", "s", "^", "D", "v"), ("-", "--", "-", "-.", ":"), (7, 6, 6, 5, 5))}
NOISE_SCALES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
N_COMPONENTS = (2, 3, 4, 5, 6)
COMPONENT_NOISE = (0.0, 1.0, 2.0)
TRAINING_SIZES = (100, 300, 1000, 3000, 10000, 30000)
SIZE_SEEDS = (0, 1, 2)
STRIDES = (1, 2, 5, 10)          # sampling step = stride x 6 s
SWEEP_TEST_DAYS = 60             # TEST window used by every sweep


def _f1(results):
    return {name: r.test_scores["f1"] for name, r in results.items()}


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def noise_sweep(dataset):
    rows = []
    for scale in NOISE_SCALES:
        results = evaluate(dataset, noisy_features(dataset, scale)[0])
        rows.append(dict(scale=scale, f1=_f1(results),
                         false_events={n: r.test_scores["n_false_events"] for n, r in results.items()}))
    return rows


def component_sweep(dataset):
    """GMM with K components (heaviest = coast, clearly-higher components = burn)."""
    table = {}
    for scale in COMPONENT_NOISE:
        features = noisy_features(dataset, scale)[0]
        table[scale] = [evaluate(dataset, features, methods={"GMM": lambda X, k=k: fit_gmm(X, n_components=k)})
                        ["GMM"].test_scores["f1"] for k in N_COMPONENTS]
    return table


def training_size_sweep(dataset, scale=1.0):
    features = noisy_features(dataset, scale)[0]
    table = {name: np.zeros((len(TRAINING_SIZES), len(SIZE_SEEDS))) for name in MODELS}
    for i, n in enumerate(TRAINING_SIZES):
        for j, seed in enumerate(SIZE_SEEDS):
            for name, f1 in _f1(evaluate(dataset, features, n_fit=n, seed=seed)).items():
                table[name][i, j] = f1
    return table


def sampling_step_sweep(dataset, scale=1.0):
    """Coarser sampling: keep every stride-th sample. The derivative window keeps >= 7 samples."""
    rows = []
    for stride in STRIDES:
        coarse = dataset.every(stride)
        r, v = add_noise(coarse, REFERENCE_NOISE[0] * scale, REFERENCE_NOISE[1] * scale)
        window = max(7, tuned("sg_window") // stride | 1)       # the tuned window, in samples at the coarser step
        rows.append(dict(step_s=coarse.step_s, window_s=window * coarse.step_s,
                         f1=_f1(evaluate(coarse, build_features(r, v, coarse.step_s, window=window)))))
    return rows


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_noise(ax, rows):
    scales = [row["scale"] for row in rows]
    x = np.arange(len(scales))
    for name in MODELS:
        ax.plot(x, [row["f1"][name] for row in rows], label=name, **METHOD_STYLE[name])
    ax.set(xticks=x, xticklabels=[f"{s:g}" for s in scales], ylim=(-0.05, 1.05),
           xlabel="noise (x 0.1 m / 0.5 mm/s)", ylabel="event F1 (TEST)", title="Robustness to measurement noise")
    ax.legend()


def _draw_false_events(ax, rows):
    scales = [row["scale"] for row in rows]
    x = np.arange(len(scales))
    for name in MODELS:
        ax.plot(x, [row["false_events"][name] for row in rows], label=name, **METHOD_STYLE[name])
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set(xticks=x, xticklabels=[f"{s:g}" for s in scales],
           xlabel="noise (x 0.1 m / 0.5 mm/s)", ylabel=f"false burn events (TEST, {SWEEP_TEST_DAYS} days)", title="False alarms vs noise")


def _draw_heatmap(ax, rows):
    grid = np.array([[row["f1"][name] for row in rows] for name in MODELS])
    image = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set(xticks=range(len(rows)), xticklabels=[f"{row['scale']:g}" for row in rows], yticks=range(len(MODELS)),
           yticklabels=list(MODELS), xlabel="noise (x reference)", title="Event F1: method x noise")
    ax.grid(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.046)


def _draw_components(ax, table):
    for (scale, f1), color in zip(table.items(), (NAVY, AMBER, RUST)):
        ax.plot(N_COMPONENTS, f1, "o-", color=color, label=f"noise x{scale:g}")
    ax.set(xticks=N_COMPONENTS, ylim=(-0.05, 1.05), xlabel="number of GMM components K", ylabel="event F1 (TEST)",
           title="How many clusters does the GMM need?")
    ax.legend()


def _draw_training_size(ax, table):
    for name, f1 in table.items():
        ax.plot(TRAINING_SIZES, f1.mean(axis=1), label=name, **METHOD_STYLE[name])
        ax.fill_between(TRAINING_SIZES, f1.min(axis=1), f1.max(axis=1), color=METHOD_COLORS[name], alpha=0.15)
    ax.set(xscale="log", ylim=(-0.05, 1.05), xlabel="training samples used to fit", ylabel="event F1 (TEST)",
           title="Robustness to training-set size (noise x1)")


def _draw_sampling(ax, rows):
    steps = [row["step_s"] for row in rows]
    for name in MODELS:
        ax.plot(steps, [row["f1"][name] for row in rows], label=name, **METHOD_STYLE[name])
    ax.set(xticks=steps, ylim=(-0.05, 1.05), xlabel="sampling step [s]", ylabel="event F1 (TEST)",
           title="Robustness to sampling step (noise x1)")


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    warnings.filterwarnings("ignore", category=UserWarning)
    folder = step_dir(out_dir, "step7_robustness")
    dataset = detection_dataset().until(TRAIN_DAYS + SWEEP_TEST_DAYS)
    noise_rows = noise_sweep(dataset)
    components = component_sweep(dataset)
    sizes = training_size_sweep(dataset)
    sampling = sampling_step_sweep(dataset)

    with report_style():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        _draw_noise(axes[0], noise_rows); _draw_false_events(axes[1], noise_rows)
        save(fig, folder, "step7_noise_robustness.png")
        fig, ax = plt.subplots(figsize=(8, 4)); _draw_heatmap(ax, noise_rows); save(fig, folder, "step7_noise_heatmap.png")
        fig, ax = plt.subplots(figsize=(7, 4.5)); _draw_components(ax, components); save(fig, folder, "step7_n_components.png")
        fig, ax = plt.subplots(figsize=(7.5, 4.5)); _draw_training_size(ax, sizes); ax.legend(); save(fig, folder, "step7_training_size.png")
        fig, ax = plt.subplots(figsize=(7.5, 4.5)); _draw_sampling(ax, sampling); ax.legend(); save(fig, folder, "step7_sampling_step.png")

        fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
        _draw_noise(axes[0, 0], noise_rows)
        _draw_false_events(axes[0, 1], noise_rows)
        _draw_heatmap(axes[0, 2], noise_rows)
        _draw_components(axes[1, 0], components)
        _draw_training_size(axes[1, 1], sizes)
        axes[1, 1].legend(fontsize=7)
        _draw_sampling(axes[1, 2], sampling)
        for ax, letter in zip(axes.ravel(), "abcdef"):
            panel_label(ax, letter)
        save(fig, folder, "step7_summary.png",
             suptitle=f"Step 7: classifier robustness (train days 0-{TRAIN_DAYS}, test the next {SWEEP_TEST_DAYS} days)")

    return dict(report_step7=dict(
        noise=noise_rows, gmm_components={f"{k:g}": v for k, v in components.items()},
        training_size={name: dict(sizes=list(TRAINING_SIZES), mean_f1=f1.mean(axis=1).tolist()) for name, f1 in sizes.items()},
        sampling_step=sampling))

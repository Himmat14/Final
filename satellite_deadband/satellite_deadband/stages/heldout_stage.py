"""
Workstream C/G, further extended: a long 60-orbit run, a chronological train/test split,
six detectors plus a Gaussian Process, all scored on data never used for fitting.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from deadband import heldout_models as H
from deadband.constants import from_hours, to_hours
from deadband.method_comparison import feature_grid
from .common import BURN_GREY, NAVY, PURPLE, RUST, save_figure

GRID_RESOLUTION = 160


def _plot_split_timeline(exp, out_dir):
    hours = to_hours(exp.evidence.t)
    sma = exp.gp.sma_km
    split_hour = to_hours(exp.t_split)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(hours[exp.train_mask], sma[exp.train_mask], "-", lw=0.6, color=NAVY, label="train (first 70%)")
    ax.plot(hours[exp.test_mask], sma[exp.test_mask], "-", lw=0.6, color=RUST, label="test (last 30%, held out)")
    ax.axvline(split_hour, color="black", ls="--", lw=1.2)
    ax.annotate("train / test split", xy=(split_hour, sma.max()), xytext=(split_hour + 1.5, sma.max() - 0.05),
                fontsize=9, va="top")
    ax.set(xlabel="Time [hours]", ylabel="Osculating SMA [km]",
           title=f"A much longer run ({H.N_ORBITS_LONG} orbits, {hours[-1]:.1f} hours, {len(hours)} samples, "
                 f"{exp.sim.n_burns} burns) with a chronological split")
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "traintest_split_timeline.png")


def _draw_cluster_panel(ax, grid, test_features, test_labels, predicted, title, row, limits, region=None):
    x_grid, y_grid = grid
    if region is not None:
        ax.contourf(x_grid, y_grid, region, levels=[-0.5, 0.5, 1.5], colors=["#EDEDF5", "#F7D9CF"])
    coast, burn = test_labels == 0, test_labels == 1
    ax.scatter(test_features[coast, 0], test_features[coast, 1], c=BURN_GREY, s=8, alpha=0.6, linewidths=0, zorder=2)
    ax.scatter(test_features[burn, 0], test_features[burn, 1], c=RUST, s=36, alpha=0.95, linewidths=0.4,
               edgecolors="black", zorder=4)
    flagged = predicted == 1
    ax.scatter(test_features[flagged, 0], test_features[flagged, 1], facecolors="none", edgecolors=NAVY,
               s=70, linewidths=0.9, alpha=0.5, zorder=3)
    ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
    ax.set_title(f"{title}\nbal.acc={row['balanced_accuracy'] * 100:.0f}%  prec={row['precision'] * 100:.0f}%", fontsize=10)
    ax.tick_params(labelsize=7)


def _plot_cluster_comparison(exp, out_dir):
    x_grid, y_grid, x_limits, y_limits = feature_grid(exp.features, pad=0.5, n=GRID_RESOLUTION)
    grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    limits = (x_limits, y_limits)
    test_X, test_y = exp.test_features, exp.test_labels

    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    axes = axes.ravel()

    # one panel per model that can predict on new points (shaded region + flagged rings)
    for ax, (name, detector) in zip(axes, exp.detectors.items()):
        region = detector.predict(grid_points).reshape(x_grid.shape)
        title = name
        if name == H.BAYESIAN_GMM_NAME:
            title = f"Bayesian GMM ({detector.info['n_active']} active comp.)"
        elif name == H.GMM_NAME:
            title = "GMM"
        elif name == H.KMEANS_NAME:
            title = "K-means"
        _draw_cluster_panel(ax, (x_grid, y_grid), test_X, test_y, detector.predict(test_X), title,
                            exp.row(name), limits, region)

    # DBSCAN is scored in-sample on the test set and has no region to shade
    _draw_cluster_panel(axes[5], (x_grid, y_grid), test_X, test_y, H.dbscan_in_sample(test_X),
                        "DBSCAN (in-sample on test)", exp.row(H.DBSCAN_TEST_NAME), limits)

    axes[6].axis("off"); axes[7].axis("off")
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BURN_GREY, markersize=7, label="true coast"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RUST, markeredgecolor="black", markersize=9, label="true burn"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=NAVY, markersize=9, label="flagged as burn by method"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#F7D9CF", label='region method calls "burn"'),
        plt.Rectangle((0, 0), 1, 1, facecolor="#EDEDF5", label='region method calls "coast"'),
    ]
    axes[6].legend(handles=legend_items, loc="center", fontsize=11, frameon=False)
    axes[7].text(0.0, 0.85, f"Held-out TEST set only ({len(test_y)} points,\n{int(test_y.sum())} true burns) -- every panel\nscores points never seen during fitting.",
                 fontsize=10.5, transform=axes[7].transAxes, va="top")
    axes[7].text(0.0, 0.45, "log|delta SMA| sits at a near-constant\nfloor for 96%+ of points (coast and most\nburns alike); the real separation signal\nis almost entirely in log|delta speed|\n(vertical axis) -- a 1-feature problem\nin practice, which is why a method that\ncan only draw straight lines (k-means,\nagglomerative) struggles so badly.",
                 fontsize=10.5, transform=axes[7].transAxes, va="top")
    axes[7].text(0.0, 0.03, "DBSCAN has no predict() for new points,\nso it cannot draw a shaded region or be\nfit on train and scored on test.",
                 fontsize=10.5, transform=axes[7].transAxes, va="top")
    fig.suptitle("Clearer cluster comparison on held-out test data: shaded region + true label + ring = flagged", fontsize=13)
    save_figure(fig, out_dir, "clearer_cluster_comparison.png", dpi=180, tight_rect=[0, 0, 1, 0.96])


def _plot_gp_phase_fold(exp, out_dir):
    gp, labels = exp.gp, exp.evidence.true_label
    elapsed_hours = to_hours(gp.elapsed)
    grid_hours = np.linspace(0, elapsed_hours.max(), 300)
    mean, std = gp.model.predict(from_hours(grid_hours).reshape(-1, 1), return_std=True)
    coast, burn = labels == 0, labels == 1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(grid_hours, mean - 2 * std, mean + 2 * std, color=NAVY, alpha=0.15, label="GP +/- 2 std", zorder=1)
    ax.scatter(elapsed_hours[coast], gp.sma_km[coast], c="#2E2E2E", s=5, alpha=0.5, label="true coast", zorder=3)
    ax.plot(grid_hours, mean, color=NAVY, lw=1.6, label="GP mean (fit on 400 train coast points only)", zorder=4)
    ax.scatter(elapsed_hours[burn], gp.sma_km[burn], c=RUST, s=20, alpha=0.95, label="true burn", zorder=5,
               edgecolors="black", linewidths=0.3)
    ax.scatter(elapsed_hours[gp.fit_indices], gp.sma_km[gp.fit_indices], facecolors="none", edgecolors="#FFA500",
               s=26, linewidths=0.8, alpha=0.9, label="the 400 points used to fit the GP", zorder=4.5)
    ax.set(xlabel="Elapsed time since last burn/reset [hours]", ylabel="Osculating SMA [km]",
           title="Phase-folding collapses ~181 coast segments onto one curve a single GP can fit")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "gp_phase_fold.png")


def _plot_gp_zscore_timeline(exp, out_dir):
    gp = exp.gp
    hours = to_hours(exp.evidence.t)
    flagged = gp.predicted == 1
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.scatter(hours[~flagged], gp.z_score[~flagged], s=4, color=BURN_GREY, alpha=0.5, label="below threshold (called coast)")
    ax.scatter(hours[flagged], gp.z_score[flagged], s=16, color=RUST, alpha=0.9, label="above threshold (flagged as burn)")
    ax.axhline(gp.threshold, color=NAVY, ls="--", lw=1.2, label=f"threshold = {gp.threshold:.2f} (chosen on TRAIN only)")
    ax.axvline(to_hours(exp.t_split), color="black", ls=":", lw=1.4, label="train / test split")
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set(xlabel="Time [hours]", ylabel="|z-score| vs GP coast model (symlog)",
           title="GP anomaly score over time: isolated spikes at burns, flat baseline everywhere else")
    ax.legend(fontsize=8.5, loc="center right"); ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "gp_zscore_timeline.png")


def _plot_method_bars(exp, out_dir):
    names = [r["method"] for r in exp.rows]
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(x - width, [r["balanced_accuracy"] * 100 for r in exp.rows], width=width, color=NAVY, label="balanced accuracy")
    ax.bar(x, [r["precision"] * 100 for r in exp.rows], width=width, color=RUST, label="precision")
    ax.bar(x + width, [r["sensitivity"] * 100 for r in exp.rows], width=width, color=PURPLE, label="sensitivity (recall)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
    ax.set(ylabel="Score [%]", ylim=(0, 105),
           title="All methods, scored on the same held-out TEST set (DBSCAN panels are in-sample, flagged)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    save_figure(fig, out_dir, "method_comparison_bars.png")


def run(sim, out_dir):
    """`sim` (the 8-orbit run) is not used: this stage simulates its own, longer run."""
    exp = H.run_heldout_experiment()

    _plot_split_timeline(exp, out_dir)
    _plot_cluster_comparison(exp, out_dir)
    _plot_gp_phase_fold(exp, out_dir)
    _plot_gp_zscore_timeline(exp, out_dir)
    _plot_method_bars(exp, out_dir)

    hours = to_hours(exp.evidence.t)
    return dict(advanced_methods=dict(
        n_orbits=H.N_ORBITS_LONG, train_frac=H.TRAIN_FRACTION,
        n_samples=int(len(hours)), sim_hours=float(hours[-1]), n_burns=int(exp.sim.n_burns),
        n_train=int(exp.train_mask.sum()), n_test=int(exp.test_mask.sum()),
        n_train_burns=int(exp.train_labels.sum()), n_test_burns=int(exp.test_labels.sum()),
        rows=exp.rows,
        bgmm_active_components=exp.detectors[H.BAYESIAN_GMM_NAME].info["n_active"],
        gp_threshold=exp.gp.threshold,
    ))

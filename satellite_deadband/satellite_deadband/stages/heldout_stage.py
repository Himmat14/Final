"""
Workstream C/G, further extended: seven detectors, fitted on TRAIN and scored per burn EVENT on
held-out TEST data, on the SAME data, split and features as report step 6 (days 0-100 of the
400-day controlled run, TRAIN days 0-40, TEST days 40-100, reference noise). Library: heldout_models.py.

The Week 4 version used a 60-orbit impulsive-burn run with a 70 / 30 split and per-sample scores.

Figures (outputs/report/step6_classification/)
    traintest_split_timeline      the mean SMA over days 0-100 with the chronological split
    clearer_cluster_comparison    each model's burn region in the feature plane, with the TEST data
    gp_phase_fold                 every coast arc, as a function of time since the SMA left the upper
                                  edge, collapses onto one curve the GP learns (TRAIN arcs only)
    gp_phase_fold_3d              the same arcs as a 3D surface (hours into the coast x cycle number):
                                  one identical ribbon per cycle, which is why a single GP is enough
    gp_zscore_timeline            the GP anomaly score over the first 20 TEST days
    method_comparison_bars        event F1 / precision / recall of all seven methods on TEST
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from deadband import heldout_models as H
from deadband.long_run import TRAIN_DAYS
from deadband.method_comparison import feature_grid, region_labels
from .common import BURN_GREY, NAVY, PURPLE, RUST, SLATE
from .report_common import report_style, save, step_dir

ZSCORE_DAYS = 20
FOLD_HOURS = 90.0                  # every coast lasts ~93 hours; the fold shows the first 90


def _draw_split(ax, exp):
    d = exp.dataset
    every = slice(None, None, 50)
    ax.plot(d.days[d.train][every], exp.gp.sma_km[d.train][every], color=NAVY, lw=0.6,
            label=f"TRAIN (days 0-{TRAIN_DAYS})")
    ax.plot(d.days[d.test][every], exp.gp.sma_km[d.test][every], color=RUST, lw=0.6,
            label=f"TEST (days {TRAIN_DAYS}-{d.days[-1]:.0f}, held out)")
    ax.axvline(TRAIN_DAYS, color="black", ls="--", lw=1.2)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set(xlabel="time [days]", ylabel="measured mean SMA [km]",
           title=f"Chronological split: {len(_burns(d, d.train))} TRAIN burns, "
                 f"{len(_burns(d, d.test))} TEST burns")
    ax.legend(loc="lower left")


def _burns(dataset, mask):
    from deadband.derivative_detection import runs_of_ones
    return runs_of_ones(dataset.t[mask], dataset.true_on[mask])


def _draw_clusters(axes, exp):
    d, X = exp.dataset, exp.X
    x_grid, y_grid, x_limits, y_limits = feature_grid(X, n=160)
    test = np.flatnonzero(d.test)
    coast, burn = test[d.true_on[test] == 0][::40], test[d.true_on[test] == 1]
    rows = {row["method"]: row for row in exp.rows}
    panels = list(exp.predictors.items()) + [(H.DBSCAN_NAME, None)]
    for ax, (name, predict) in zip(axes, panels):
        if predict is not None:
            ax.contourf(x_grid, y_grid, region_labels(predict, x_grid, y_grid), levels=[-0.5, 0.5, 1.5],
                        colors=["#EDEDF5", "#F7D9CF"])
        else:
            flagged = exp.dbscan_rows[exp.dbscan_flags == 1]
            ax.scatter(X[flagged, 0], X[flagged, 1], facecolors="none", edgecolors=NAVY, s=30, lw=0.6, zorder=3)
        ax.scatter(X[coast, 0], X[coast, 1], c=BURN_GREY, s=3, alpha=0.5, lw=0)
        ax.scatter(X[burn, 0], X[burn, 1], c=RUST, s=4)
        ax.set(xlim=x_limits, ylim=y_limits)
        title = name if name != H.BAYESIAN_GMM_NAME else f"{name} ({exp.info[name]['n_active']} active comp.)"
        row = rows[name]
        ax.set_title(f"{title}\nevent F1 {row['f1']:.2f}, false events {row['n_false_events']}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(panels):]:
        ax.axis("off")
    axes[-1].legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BURN_GREY, markersize=7, label="true coast (TEST)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RUST, markersize=7, label="true burn (TEST)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#F7D9CF", label='region the model calls "burn"'),
        plt.Rectangle((0, 0), 1, 1, facecolor="#EDEDF5", label='region the model calls "coast"'),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor=NAVY, markersize=8,
               label="DBSCAN: flagged (no region: no predict())")], loc="center", fontsize=9)


def _coast_arcs(exp, train_only):
    """[(elapsed hours, SMA - SMA at coast start [m])] for every coast arc."""
    d, gp = exp.dataset, exp.gp
    elapsed = gp.elapsed_hours
    starts = np.flatnonzero(np.diff(np.nan_to_num(elapsed, nan=1e9)) < -1) + 1     # the clock resets
    arcs = []
    for a, b in zip(starts[:-1], starts[1:]):
        if train_only and not d.train[a]:
            continue
        rows = np.arange(a, b)[elapsed[a:b] <= FOLD_HOURS]
        rows = rows[d.true_on[rows] == 0]
        arcs.append((elapsed[rows], (gp.sma_km[rows] - gp.mean[a]) * 1000, rows))
    return arcs


def _draw_fold(axes, exp):
    """Left: every arc and the GP curve. Right: each arc minus the GP curve (TRAIN grey, TEST red)."""
    grid = np.linspace(0, FOLD_HOURS, 300)
    prediction = exp.gp.model.predict(grid[:, None])
    curve = (prediction - prediction[0]) * 1000             # relative to the curve at the start of the coast
    d, gp = exp.dataset, exp.gp
    for hours, metres, rows in _coast_arcs(exp, train_only=False):
        axes[0].plot(hours[::20], metres[::20], color=SLATE, lw=0.4, alpha=0.4)
        residual = (gp.sma_km[rows] - gp.mean[rows]) * 1000
        axes[1].plot(hours[::50], residual[::50], ".", ms=1, color=SLATE if d.train[rows[0]] else RUST, alpha=0.4)
    axes[0].plot(grid, curve, color=RUST, lw=2, label="GP coast curve (fitted on TRAIN arcs)")
    axes[0].set(xlabel="hours since the SMA left the upper edge", ylabel="SMA change since the coast began [m]",
                title="Phase folding: every coast arc (grey) lies on one curve")
    axes[0].legend()
    axes[1].plot([], [], "o", color=SLATE, label="TRAIN arcs")
    axes[1].plot([], [], "o", color=RUST, label="TEST arcs (never seen by the GP)")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(xlabel="hours since the SMA left the upper edge", ylabel="measured SMA - GP curve [m]",
                title="What is left: ~1 m of measurement noise, TEST the same as TRAIN")
    axes[1].legend()


def _draw_fold_3d(ax, exp):
    arcs = _coast_arcs(exp, train_only=False)
    grid = np.linspace(0, FOLD_HOURS, 120)
    surface = np.array([np.interp(grid, hours, metres) for hours, metres, _ in arcs if len(hours) > 10])
    cycle = np.arange(len(surface))
    X, Y = np.meshgrid(grid, cycle)
    ax.plot_surface(X, Y, surface, cmap="viridis", rstride=1, cstride=2, lw=0, alpha=0.95)
    split = next(i for i, (_, _, rows) in enumerate(arcs) if not exp.dataset.train[rows[0]])
    ax.plot(grid, np.full_like(grid, split), surface[split], color=RUST, lw=2, label="first TEST coast")
    ax.set_xlabel("hours into the coast", labelpad=6)
    ax.set_ylabel("coast number", labelpad=6)
    ax.set_zlabel("SMA change [m]")
    ax.set_title("Every coast arc, side by side: one identical ribbon per cycle\n"
                 "(why a single GP fitted on TRAIN predicts TEST)", fontsize=10)
    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=-60)


def _draw_zscore(ax, exp):
    d, gp = exp.dataset, exp.gp
    keep = np.flatnonzero(d.test & (d.days < TRAIN_DAYS + ZSCORE_DAYS))[::5]
    flagged = gp.z_score[keep] > H.Z_THRESHOLD
    ax.scatter(d.days[keep][~flagged], gp.z_score[keep][~flagged], s=1, color=BURN_GREY, label="coast-like")
    ax.scatter(d.days[keep][flagged], gp.z_score[keep][flagged], s=3, color=RUST, label="flagged (burn)")
    ax.axhline(H.Z_THRESHOLD, color=NAVY, ls="--", label=f"threshold z = {H.Z_THRESHOLD:g}")
    ax.set_yscale("symlog", linthresh=5)
    ax.set(xlabel="time [days]", ylabel="(SMA - GP curve) / TRAIN scatter (symlog)",
           title=f"GP anomaly score, first {ZSCORE_DAYS} TEST days: spikes at burns, flat in between")
    ax.legend(loc="upper right")


def _draw_bars(ax, exp):
    names = [row["method"] for row in exp.rows]
    x = np.arange(len(names))
    for offset, key, color, label in ((-0.27, "f1", NAVY, "event F1"), (0, "precision", RUST, "precision"),
                                      (0.27, "recall", PURPLE, "recall (burns found)")):
        ax.bar(x + offset, [row[key] * 100 for row in exp.rows], 0.27, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set(ylabel="score [%]", ylim=(0, 108), title="All methods, event scores on the same held-out TEST days (40-100)")
    ax.legend(loc="upper center", ncol=3)


def run(sim, out_dir):
    """`sim` is not used: the data are the report's controlled run."""
    folder = step_dir(out_dir, "step6_classification")
    exp = H.run_heldout_experiment()

    with report_style():
        fig, ax = plt.subplots(figsize=(11, 4)); _draw_split(ax, exp); save(fig, folder, "traintest_split_timeline.png")
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        _draw_clusters(axes.ravel(), exp)
        save(fig, folder, "clearer_cluster_comparison.png",
             suptitle="Each model's burn region on held-out TEST data (fitted on TRAIN, reference noise)")
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.8)); _draw_fold(axes, exp); save(fig, folder, "gp_phase_fold.png")
        fig = plt.figure(figsize=(9.5, 7)); _draw_fold_3d(fig.add_subplot(111, projection="3d"), exp)
        save(fig, folder, "gp_phase_fold_3d.png")
        fig, ax = plt.subplots(figsize=(11, 4.2)); _draw_zscore(ax, exp); save(fig, folder, "gp_zscore_timeline.png")
        fig, ax = plt.subplots(figsize=(12, 5)); _draw_bars(ax, exp); save(fig, folder, "method_comparison_bars.png")

    d = exp.dataset
    return dict(advanced_methods=dict(
        data="controlled run days 0-100, TRAIN 0-40, TEST 40-100, reference noise, event scores",
        n_train_burns=len(_burns(d, d.train)), n_test_burns=len(_burns(d, d.test)),
        rows=exp.rows, bgmm_active_components=exp.info[H.BAYESIAN_GMM_NAME]["n_active"],
        gp_z_threshold=H.Z_THRESHOLD, gp_kernel=str(exp.gp.model.kernel_)))

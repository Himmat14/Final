"""
Workstream C/G, extended: the two-feature Gaussian mixture, shown in 3D and 2D, and how
GMM and k-means decision regions differ (and move) under noise.
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband import gmm_2d, method_comparison
from deadband.constants import to_hours
from deadband.evidence import build_evidence
from .common import NAVY, RUST, SLATE, GREY, save_figure

NOISE_FRACS_FOR_REGIONS = (0.0, 2.0)
ELLIPSE_COLORS = ["#FFD700", "#00FFFF"]


def _plot_density_surface(summary, density_grid, out_dir):
    x_grid, y_grid, density = density_grid
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_grid, y_grid, density, cmap="viridis", alpha=0.9, linewidth=0, antialiased=True)
    floor = float(density.min()) - 0.02 * abs(float(density.max()))
    ax.contourf(x_grid, y_grid, density, zdir="z", offset=floor, cmap="viridis", alpha=0.35)
    ax.set_xlabel("log |delta SMA|"); ax.set_ylabel("log |delta speed|"); ax.set_zlabel("mixture density")
    ax.set_title("Fitted 2-component Gaussian mixture: density surface")
    ax.view_init(elev=28, azim=-60)
    save_figure(fig, out_dir, "gmm_density_surface_3d.png")


def _plot_scatter_3d(evidence, summary, out_dir):
    hours = to_hours(evidence.t)
    x1, x2 = summary.features[:, 0], summary.features[:, 1]
    is_coast, is_burn = evidence.true_label == 0, evidence.true_label == 1
    is_wrong = summary.predicted != evidence.true_label

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(hours[is_coast], x1[is_coast], x2[is_coast], c=GREY, s=6, alpha=0.5, label="coast (true)")
    ax.scatter(hours[is_burn], x1[is_burn], x2[is_burn], c=RUST, s=40, label="burn (true)")
    ax.scatter(hours[is_wrong], x1[is_wrong], x2[is_wrong], facecolors="none", edgecolors=NAVY, s=90,
               linewidths=1.5, label="GMM misclassified")
    ax.set_xlabel("Time [hours]"); ax.set_ylabel("log |delta SMA|"); ax.set_zlabel("log |delta speed|")
    ax.set_title("Burn evidence in time + 2-feature space, GMM errors circled")
    ax.view_init(elev=22, azim=-50)
    ax.legend(fontsize=8)
    save_figure(fig, out_dir, "gmm_scatter_3d.png")


def _plot_contour_with_ellipses(evidence, summary, density_grid, limits, out_dir):
    x_grid, y_grid, density = density_grid
    features, labels = summary.features, evidence.true_label
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.contourf(x_grid, y_grid, density, levels=25, cmap="viridis", alpha=0.6)
    ax.scatter(features[labels == 0, 0], features[labels == 0, 1], c=SLATE, s=8, alpha=0.5, label="coast")
    ax.scatter(features[labels == 1, 0], features[labels == 1, 1], c=RUST, s=40, label="burn")
    for k in range(summary.model.n_components):
        ex, ey = gmm_2d.ellipse_points(summary.model.means_[k], summary.model.covariances_[k], n_std=2.0)
        ax.plot(ex, ey, color=ELLIPSE_COLORS[k % 2], lw=2, label=f"component {k} (2 sigma)")
    ax.set(xlabel="log |delta SMA| (jump evidence)", ylabel="log |delta speed| (jump evidence)",
           title="2-feature GMM: density, component covariance ellipses, and the data",
           xlim=limits[0], ylim=limits[1])
    ax.legend(fontsize=8, loc="lower right")
    save_figure(fig, out_dir, "gmm_contour_ellipses.png")


def _plot_decision_regions(fits, out_dir):
    """2x2 grid: rows = GMM / k-means, columns = clean / noisy. Returns the region-shift numbers."""
    shift = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for column, noise_frac in enumerate(NOISE_FRACS_FOR_REGIONS):
        fit = fits[noise_frac]
        features, labels = fit["features"], fit["evidence"].true_label
        x_grid, y_grid, x_limits, y_limits = method_comparison.feature_grid(features)
        regions = [
            ("GMM", method_comparison.region_labels(fit["gmm"], fit["gmm_burn"], x_grid, y_grid)),
            ("k-means", method_comparison.region_labels(fit["kmeans"], fit["kmeans_burn"], x_grid, y_grid)),
        ]
        noise_label = "clean (0 noise)" if noise_frac == 0.0 else f"noisy ({noise_frac:.0f}x signal std)"
        for row, (title, region) in enumerate(regions):
            ax = axes[row, column]
            ax.contourf(x_grid, y_grid, region, levels=[-0.5, 0.5, 1.5], colors=["#E8E8F0", "#F5C6BA"])
            ax.scatter(features[labels == 0, 0], features[labels == 0, 1], c=SLATE, s=6, alpha=0.5)
            ax.scatter(features[labels == 1, 0], features[labels == 1, 1], c=RUST, s=30,
                       edgecolors="black", linewidths=0.3)
            ax.set(xlim=x_limits, ylim=y_limits, xlabel="log |delta SMA|", ylabel="log |delta speed|")
            ax.set_title(f"{title} region -- {noise_label}", fontsize=11)
        shift[str(noise_frac)] = dict(
            agreement_frac=float(np.mean(fit["gmm_predicted"] == fit["kmeans_predicted"])),
            gmm_flagged=int(fit["gmm_predicted"].sum()),
            kmeans_flagged=int(fit["kmeans_predicted"].sum()))
    save_figure(fig, out_dir, "decision_regions_grid.png")
    return shift


def _plot_orbit_map(sim, fits, out_dir):
    """Where along the SMA trace each method fires, clean versus noisy."""
    hours = to_hours(sim.t)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, noise_frac in zip(axes, NOISE_FRACS_FOR_REGIONS):
        fit = fits[noise_frac]
        gmm_flagged, kmeans_flagged = fit["gmm_predicted"] == 1, fit["kmeans_predicted"] == 1
        ax.plot(hours, sim.sma_km, "-", color="#CCCCCC", lw=1, zorder=1, label="SMA trace")
        ax.scatter(hours[gmm_flagged], sim.sma_km[gmm_flagged], marker="o", facecolors="none",
                   edgecolors=NAVY, s=70, linewidths=1.3, label="GMM flagged", zorder=3)
        ax.scatter(hours[kmeans_flagged], sim.sma_km[kmeans_flagged], marker="x", color=RUST, s=45,
                   linewidths=1.3, label="k-means flagged", zorder=4)
        ax.set_title("clean measurements" if noise_frac == 0.0 else f"noisy measurements ({noise_frac:.0f}x signal std)",
                     fontsize=11)
        ax.set_xlabel("Time [hours]")
    axes[0].set_ylabel("Osculating SMA [km]")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Where along the orbit trace each method fires: GMM vs k-means, clean vs noisy")
    save_figure(fig, out_dir, "orbit_map_detections.png")


def run(sim, out_dir):
    evidence = build_evidence(sim)
    summary = gmm_2d.summarize_gmm_2d(evidence)

    features = summary.features
    limits = ((features[:, 0].min() - 0.5, features[:, 0].max() + 0.5),
              (features[:, 1].min() - 0.5, features[:, 1].max() + 0.5))
    density_grid = gmm_2d.mixture_density_grid(summary.model, *limits)

    _plot_density_surface(summary, density_grid, out_dir)
    _plot_scatter_3d(evidence, summary, out_dir)
    _plot_contour_with_ellipses(evidence, summary, density_grid, limits, out_dir)

    fits = method_comparison.fit_regions_under_noise(evidence, noise_fracs=NOISE_FRACS_FOR_REGIONS)
    region_shift = _plot_decision_regions(fits, out_dir)
    _plot_orbit_map(sim, fits, out_dir)

    return dict(
        gmm_2d=dict(bal_acc_1d=summary.balanced_accuracy_1d, bal_acc_2d=summary.balanced_accuracy_2d,
                    weights=summary.model.weights_.tolist(),
                    tp=summary.scores["tp"], fn=summary.scores["fn"], tn=summary.scores["tn"], fp=summary.scores["fp"],
                    sensitivity=summary.scores["sensitivity"], specificity=summary.scores["specificity"],
                    precision=summary.scores["precision"]),
        region_shift=region_shift,
    )

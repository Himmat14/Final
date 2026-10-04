"""
Workstream C/G, extended: the two-feature Gaussian mixture, in 2D and 3D, on the SAME data and
features as report step 6 (days 0-100 of the 400-day controlled run, TRAIN days 0-40, TEST days
40-100, step 6 reference noise 0.1 m / 0.5 mm/s unless "clean"). Library: deadband/gmm_2d.py.

The "weighting" used as a colour in the 3D figures is each sample's GMM posterior probability of
being a burn, P(burn | x) (see gmm_2d.py).

Figures (outputs/report/step6_classification/)
    gmm_density_surface_3d      log10 of the fitted mixture density over the feature plane: a broad
                                coast hill and a small, sharp burn peak far away from it
    gmm_burn_probability_3d     P(burn | x) as a surface: flat at 0 over the coast, a cliff up to 1
                                around the burn components (the decision boundary is the cliff edge)
    gmm_phase_space_3d          the system's phase space (mean SMA, d(mean SMA)/dt, along-track
                                acceleration) over two TEST cycles, every sample coloured by P(burn)
    gmm_orbit_weighting_3d      the orbit in space around a TEST burn, coloured by P(burn): the burn
                                arc lights up, the rest of the orbit stays dark
    gmm_scatter_3d              time + both features (first 15 TEST days), misclassified samples circled
    gmm_contour_ellipses        density contours, component ellipses and the data, clean vs noisy
    decision_regions_grid       GMM vs k-means regions, clean vs noisy (TEST samples)
    orbit_map_detections        where along the mean-SMA trace each method fires, clean vs noisy
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband.constants import DU, SAMPLE_STEP_S, to_days
from deadband.derivative_detection import event_scores, group_into_events
from deadband.classifiers import EVENT_TOLERANCE_S
from deadband.gmm_2d import ellipse_points, mixture_density_grid, mixture_density_on, probability_grid, refined_grid
from deadband.long_run import TRAIN_DAYS
from deadband.method_comparison import feature_grid, fit_regions_under_noise, region_labels
from deadband.derivative_detection import runs_of_ones
from .common import AMBER, GREY, NAVY, RUST, SLATE
from .report_common import report_style, save, step_dir

NOISE_CASES = {0.0: "clean", 1.0: "noisy (0.1 m, 0.5 mm/s)"}
PHASE_DAYS = (TRAIN_DAYS, TRAIN_DAYS + 8)         # two full cycles at the start of TEST
SCATTER_DAYS = 15
ELLIPSE_COLORS = ("#FFD700", "#00FFFF", "#FF69B4", "#7CFC00")


def _labels(ax3d, x, y, z):
    ax3d.set_xlabel(x, labelpad=6)
    ax3d.set_ylabel(y, labelpad=6)
    ax3d.set_zlabel(z, labelpad=4)


# ---------------------------------------------------------------------------
# 3D figures
# ---------------------------------------------------------------------------
def _burn_zoom_limits(fit, n_sd=6):
    """Feature-plane box of +-n_sd standard deviations around the main burn component."""
    model, is_burn = fit["gmm"].model, fit["gmm"].is_burn
    k = int(np.argmax(np.where(is_burn, model.means_[:, 0], -np.inf)))
    sd = np.sqrt(np.diag(model.covariances_[k]))
    return tuple((model.means_[k, d] - n_sd * sd[d], model.means_[k, d] + n_sd * sd[d]) for d in (0, 1))


def _surface(ax, x_grid, y_grid, z, cmap, floor):
    # rstride = cstride = 1: draw every grid line (the default thins the grid to 50 x 50, which turns
    # the narrow burn component into aliasing spikes)
    ax.plot_surface(x_grid, y_grid, z, cmap=cmap, rstride=1, cstride=1, lw=0, antialiased=False, alpha=0.95)
    ax.contour(x_grid, y_grid, z, zdir="z", offset=floor, levels=15, cmap=cmap)
    ax.set_zlim(floor, z.max() + 0.2 * (z.max() - floor))


def _draw_density_surface(axes, fit):
    """Left: the whole feature plane. Right: zoom on the burn component (its own shape resolved)."""
    model = fit["gmm"].model
    _, _, x_limits, y_limits = feature_grid(fit["X"])
    for ax, limits, title in ((axes[0], (x_limits, y_limits), "whole plane: broad coast hill, thin burn ridge"),
                              (axes[1], _burn_zoom_limits(fit), "zoom on the burn component (+-6 sd)")):
        x_grid, y_grid = refined_grid(model, *limits, n=160)
        z = np.log10(mixture_density_on(model, x_grid, y_grid) + 1e-12)
        floor = max(z.min(), -8)
        _surface(ax, x_grid, y_grid, np.maximum(z, floor), "viridis", floor)
        _labels(ax, "log |accel from v'|", "along-track [1e-4 m/s$^2$]", "log10 mixture density")
        ax.set_title(f"GMM density surface, {title}", fontsize=10)
        ax.view_init(elev=30, azim=-55)


def _draw_probability_surface(axes, fit):
    """P(burn | x) over the whole plane and around the burn component."""
    model = fit["gmm"].model
    _, _, x_limits, y_limits = feature_grid(fit["X"])
    (zx, zy) = _burn_zoom_limits(fit, n_sd=12)
    for ax, limits, title in ((axes[0], (x_limits, y_limits), "whole plane: 0 almost everywhere"),
                              (axes[1], (zx, zy), "around the burn component: the cliff up to 1")):
        x_grid, y_grid = refined_grid(model, *limits, n=160)
        probability = probability_grid(fit["gmm"], x_grid, y_grid)
        _surface(ax, x_grid, y_grid, probability, "magma", 0.0)
        ax.contour(x_grid, y_grid, probability, levels=[0.5], zdir="z", offset=0, colors=RUST, linewidths=2)
        ax.set_zlim(0, 1.05)
        _labels(ax, "log |accel from v'|", "along-track [1e-4 m/s$^2$]", "P(burn | x)")
        ax.set_title(f"Burn probability, {title}\n(red on the floor: the 0.5 decision boundary)", fontsize=10)
        ax.view_init(elev=30, azim=-125)


def _draw_phase_space(ax, fit):
    dataset, X = fit["dataset"], fit["X"]
    days = dataset.days
    keep = np.flatnonzero((days >= PHASE_DAYS[0]) & (days < PHASE_DAYS[1]))[::3]
    sma = dataset.mean_sma_km
    rate_m_per_hour = np.gradient(sma, SAMPLE_STEP_S / 3600) * 1000
    weight = fit["gmm"].burn_probability(X[keep])
    points = ax.scatter(sma[keep], rate_m_per_hour[keep], X[keep, 1], c=weight, cmap="plasma", s=3, vmin=0, vmax=1)
    ax.ticklabel_format(axis="x", useOffset=False)
    _labels(ax, "mean SMA [km]", "d(mean SMA)/dt [m/hour]", "measured along-track accel [1e-4 m/s$^2$]")
    ax.set_title(f"Phase space, days {PHASE_DAYS[0]}-{PHASE_DAYS[1]} (TEST), coloured by the GMM weight P(burn):\n"
                 "the coast line (dark) and the burn line (bright) are told apart sample by sample", fontsize=10)
    ax.view_init(elev=20, azim=-60)
    return points


def _draw_orbit(ax, fit):
    dataset, X = fit["dataset"], fit["X"]
    start, end = next((s, e) for s, e in runs_of_ones(dataset.t, dataset.true_on) if to_days(s) > TRAIN_DAYS)
    pad = (end - start) * 2.5
    keep = np.flatnonzero((dataset.t > start - pad) & (dataset.t < end + pad))
    r_km = dataset.r[:, keep] * DU
    weight = fit["gmm"].burn_probability(X[keep])
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    earth = 6378.0
    ax.plot_wireframe(earth * np.cos(u) * np.sin(v), earth * np.sin(u) * np.sin(v), earth * np.cos(v), color=GREY,
                      lw=0.3, alpha=0.4)
    points = ax.scatter(*r_km, c=weight, cmap="plasma", s=4, vmin=0, vmax=1)
    ax.set_box_aspect((1, 1, 1))
    _labels(ax, "x [km]", "y [km]", "z [km]")
    ax.set_title(f"The orbit around the first TEST burn (day {to_days(start):.1f}), coloured by P(burn)", fontsize=10)
    ax.view_init(elev=25, azim=-60)
    return points


def _draw_scatter_3d(ax, fit):
    dataset, X = fit["dataset"], fit["X"]
    days = dataset.days
    keep = np.flatnonzero(dataset.test & (days < TRAIN_DAYS + SCATTER_DAYS))
    coast, burn = keep[dataset.true_on[keep] == 0][::20], keep[dataset.true_on[keep] == 1]
    predicted = fit["gmm"].predict(X[keep])
    wrong = keep[predicted != dataset.true_on[keep]]
    ax.scatter(days[coast], X[coast, 0], X[coast, 1], c=GREY, s=2, alpha=0.4, label="coast (true, every 20th)")
    ax.scatter(days[burn], X[burn, 0], X[burn, 1], c=RUST, s=6, label="burn (true)")
    ax.scatter(days[wrong], X[wrong, 0], X[wrong, 1], facecolors="none", edgecolors=NAVY, s=40, lw=1,
               label=f"GMM wrong ({len(wrong)} samples)")
    _labels(ax, "time [days]", "log |accel from v'|", "along-track [1e-4 m/s$^2$]")
    ax.set_title(f"First {SCATTER_DAYS} TEST days in time + feature space", fontsize=10)
    ax.legend(fontsize=7)
    ax.view_init(elev=20, azim=-50)


# ---------------------------------------------------------------------------
# 2D figures
# ---------------------------------------------------------------------------
def _draw_contours(ax, fit, case):
    dataset, X = fit["dataset"], fit["X"]
    x_grid, y_grid, x_limits, y_limits = feature_grid(X, n=120)
    _, _, density = mixture_density_grid(fit["gmm"].model, x_limits, y_limits, n=120)
    ax.contourf(x_grid, y_grid, np.log10(density + 1e-12), levels=np.linspace(-8, np.log10(density.max()), 25),
                cmap="viridis", alpha=0.6, extend="min")
    test = np.flatnonzero(dataset.test)
    coast, burn = test[dataset.true_on[test] == 0][::30], test[dataset.true_on[test] == 1]
    ax.scatter(X[coast, 0], X[coast, 1], c=SLATE, s=2, alpha=0.4, label="coast (TEST, every 30th)")
    ax.scatter(X[burn, 0], X[burn, 1], c=RUST, s=4, label="burn (TEST)")
    model = fit["gmm"].model
    for k in range(model.n_components):
        ex, ey = ellipse_points(model.means_[k], model.covariances_[k], n_std=2.0)
        kind = "burn" if fit["gmm"].is_burn[k] else "coast"
        ax.plot(ex, ey, color=ELLIPSE_COLORS[k % 4], lw=1.8, label=f"component {k}: {kind}, w={model.weights_[k]:.3f}")
    ax.set(xlim=x_limits, ylim=y_limits, xlabel="log |unmodelled accel from v'|", ylabel="along-track [1e-4 m/s$^2$]",
           title=f"GMM density (log10), 2-sigma ellipses and TEST data, {case}")
    ax.legend(fontsize=7, loc="upper left")


def _draw_regions(axes, fits):
    shift = {}
    for column, (scale, case) in enumerate(NOISE_CASES.items()):
        fit = fits[scale]
        dataset, X = fit["dataset"], fit["X"]
        x_grid, y_grid, x_limits, y_limits = feature_grid(X, n=150)
        test = np.flatnonzero(dataset.test)
        coast, burn = test[dataset.true_on[test] == 0][::30], test[dataset.true_on[test] == 1]
        for row, (name, predict) in enumerate((("GMM", fit["gmm"].predict), ("k-means", fit["kmeans"]))):
            ax = axes[row, column]
            ax.contourf(x_grid, y_grid, region_labels(predict, x_grid, y_grid), levels=[-0.5, 0.5, 1.5],
                        colors=["#E8E8F0", "#F5C6BA"])
            ax.scatter(X[coast, 0], X[coast, 1], c=SLATE, s=2, alpha=0.4)
            ax.scatter(X[burn, 0], X[burn, 1], c=RUST, s=4)
            ax.set(xlim=x_limits, ylim=y_limits, xlabel="log |accel from v'|", ylabel="along-track [1e-4 m/s$^2$]",
                   title=f"{name} region (pink = burn), {case}")
        shift[case] = dict(agreement=float(np.mean(fit["gmm_test"] == fit["kmeans_test"])),
                           gmm_flagged=int(fit["gmm_test"].sum()), kmeans_flagged=int(fit["kmeans_test"].sum()))
    return shift


def _draw_orbit_map(axes, fits):
    scores = {}
    for ax, (scale, case) in zip(axes, NOISE_CASES.items()):
        fit = fits[scale]
        dataset = fit["dataset"]
        test = np.flatnonzero(dataset.test & (dataset.days < TRAIN_DAYS + SCATTER_DAYS))
        days = dataset.days[test]
        ax.plot(days[::20], dataset.mean_sma_km[test][::20], color="#BBBBBB", lw=1, label="mean SMA")
        all_test = dataset.test
        for name, flags, marker, color, offset in (("GMM", fit["gmm_test"], "o", NAVY, 0.02),
                                                   ("k-means", fit["kmeans_test"], "x", RUST, -0.02)):
            events = np.zeros(len(dataset.t), dtype=int)
            events[all_test] = flags
            events = group_into_events(events)
            scores[f"{name}, {case}"] = event_scores(dataset.t[all_test], events[all_test], dataset.true_on[all_test],
                                                     dataset.step_s, EVENT_TOLERANCE_S)["f1"]
            on = events[test] == 1
            ax.plot(days[on][::10], dataset.mean_sma_km[test][on][::10] + offset, marker, ms=3, color=color, ls="none",
                    label=f"{name} events (F1 {scores[f'{name}, {case}']:.2f})")
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.set(xlabel="time [days]", title=f"Where each method fires, first {SCATTER_DAYS} TEST days, {case}")
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("mean SMA [km]")
    return scores


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    """`sim` is not used: the data are the report's controlled run (detectors.window_features)."""
    folder = step_dir(out_dir, "step6_classification")
    fits = fit_regions_under_noise(tuple(NOISE_CASES))
    noisy = fits[1.0]

    with report_style():
        for draw, name in ((_draw_density_surface, "gmm_density_surface_3d.png"),
                           (_draw_probability_surface, "gmm_burn_probability_3d.png")):
            fig = plt.figure(figsize=(17, 7))
            draw([fig.add_subplot(1, 2, i, projection="3d") for i in (1, 2)], noisy)
            save(fig, folder, name)
        fig = plt.figure(figsize=(9, 7))
        _draw_scatter_3d(fig.add_subplot(111, projection="3d"), noisy)
        save(fig, folder, "gmm_scatter_3d.png")
        for draw, name in ((_draw_phase_space, "gmm_phase_space_3d.png"), (_draw_orbit, "gmm_orbit_weighting_3d.png")):
            fig = plt.figure(figsize=(9.5, 7.5))
            points = draw(fig.add_subplot(111, projection="3d"), noisy)
            fig.colorbar(points, shrink=0.6, pad=0.1, label="GMM weight P(burn | x)")
            save(fig, folder, name)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        for ax, (scale, case) in zip(axes, NOISE_CASES.items()):
            _draw_contours(ax, fits[scale], case)
        save(fig, folder, "gmm_contour_ellipses.png")
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        shift = _draw_regions(axes, fits)
        save(fig, folder, "decision_regions_grid.png")
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True)
        f1 = _draw_orbit_map(axes, fits)
        save(fig, folder, "orbit_map_detections.png")

    model = noisy["gmm"].model
    return dict(gmm_2d=dict(data="controlled run days 0-100, TRAIN 0-40, TEST 40-100, reference noise",
                            weights=model.weights_.tolist(), means=model.means_.tolist(),
                            burn_components=noisy["gmm"].is_burn.tolist(), event_f1=f1),
                region_shift=shift)

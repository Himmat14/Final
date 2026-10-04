"""
Report step 9: learn the control law with SINDy and a BINDy-style Bayesian fit, reconstruct it,
and use it to FORECAST burns on held-out data, compared with a Gaussian process and machine learning.

Data: the 400-day controlled run with the reference noise (0.1 m, 0.5 mm/s). Clusters (coast / burn)
come from the Bayesian-GMM detector of step 6. Everything is learned on TRAIN (days 0-40) and the
forecast is run, with no further data, over TEST (days 40-400); errors are reported for the burns
in the first 15, 30, 60, 120, 240 and 360 days of the forecast.

Figures (outputs/report/step9_sindy_control_law/)
    step9_library               why the library uses cluster indicators: the physics shapes are constant
    step9_coefficients          SINDy and Bayesian (BINDy-style) coefficients: real terms kept, distractors removed
    step9_physics_recovered     cd and thrust level read off the learned law, with 95% credible intervals
    step9_switching_law         memory-less sigmoid vs hysteresis (memory) for the on/off switching
    step9_forecast              free-running forecast of the mean SMA over the TEST period
    step9_forecast_errors       burn-onset timing error of every method, burn by burn, over 360 days
    step9_forecast_horizons     mean |timing error| of the burns within each forecast horizon
    step9_summary               six-panel summary
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from deadband.classifiers import MODELS, detection_dataset, evaluate, noisy_features
from deadband.constants import DU, SECONDS, SMOOTH_CD, THRUST_ACCEL, TU
from deadband.derivative_detection import runs_of_ones
from deadband.long_run import HORIZONS_DAYS
from deadband.sindy import (LIBRARY_NAMES, hysteresis_states, learn_cluster_laws, learn_switching_law, library,
                            match_onsets, memoryless_sigmoid, onset_times, physics_library, random_forest_rate,
                            run_hybrid, sindy_rate, sma_data)
from .common import AMBER, GREEN, GREY, NAVY, PURPLE, RUST, SLATE
from .report_common import panel_label, report_style, save, step_dir

FORECAST_STRIDE = 10         # forecast with 60 s Euler steps (switches land exactly on the edges)
ENSEMBLE_SIZE = 20
FORECAST_PLOT_DAYS = 30      # the SMA forecast plot shows the first 30 days (360 days is a solid block)
METHOD_COLORS = {"periodic baseline": GREY, "GP coast curve": PURPLE, "random forest": AMBER,
                 "SINDy": NAVY, "BINDy ensemble": GREEN}


# ---------------------------------------------------------------------------
# Learning
# ---------------------------------------------------------------------------
def prepare():
    dataset = detection_dataset()
    features, (r, v) = noisy_features(dataset, 1.0)
    events = evaluate(dataset, features, methods={"Bayesian GMM": MODELS["Bayesian GMM"]},
                      predict_all=True)["Bayesian GMM"].events          # TRAIN labels are needed to learn the law
    data = sma_data(dataset.t, r, v, events, dataset.step_s)
    laws = learn_cluster_laws(data, dataset.train)
    switching = learn_switching_law(dataset.t, data.a, events, dataset.train)
    return dataset, events, data, laws, switching


def switching_accuracy(dataset, data, switching):
    """Balanced accuracy on TEST of (memory-less sigmoid on a) vs (hysteresis with the learned edges)."""
    from deadband.metrics import balanced_accuracy
    sigmoid = memoryless_sigmoid(data.a[dataset.train], dataset.true_on[dataset.train])
    test = dataset.test
    memoryless = (sigmoid(data.a[test]) > 0.5).astype(int)
    hysteresis = hysteresis_states(data.a[test], switching.lower, switching.upper, s0=dataset.true_on[test][0])
    return sigmoid, dict(memoryless=balanced_accuracy(memoryless, dataset.true_on[test]),
                         hysteresis=balanced_accuracy(hysteresis, dataset.true_on[test]))


# ---------------------------------------------------------------------------
# Forecasting the test period
# ---------------------------------------------------------------------------
def forecasts(dataset, events, data, laws, switching):
    test_index = np.flatnonzero(dataset.test)[::FORECAST_STRIDE]
    t_s = data.t_s[test_index]
    h = dataset.step_s * FORECAST_STRIDE * SECONDS
    a0, s0 = data.a[test_index[0]], int(events[test_index[0]])
    n = len(test_index)
    actual = onset_times(data.t_s[dataset.test], dataset.true_on[dataset.test])
    out = {}

    # 1. periodic baseline: keep firing at the mean training interval after the last training burn
    train_starts = np.array([s for s, _ in runs_of_ones(dataset.t[dataset.train], events[dataset.train])]) * TU
    period = np.mean(np.diff(train_starts))
    predicted = train_starts[-1] + period * np.arange(1, 400)
    out["periodic baseline"] = dict(onsets=predicted[(predicted >= t_s[0]) & (predicted <= t_s[-1])], sma=None)

    # 2. GP coast curve: SMA vs time since the last burn ended (train arcs), time to reach the lower edge
    out["GP coast curve"] = gp_coast_forecast(dataset, events, data, switching, t_s)

    # 3. random forest learns da/dt from (a, cluster), same hybrid integrator
    grid = np.linspace(switching.lower - 30 / 1000 / DU, switching.upper + 30 / 1000 / DU, 400)
    a, s = run_hybrid(a0, s0, n, h, random_forest_rate(data, dataset.train, grid), switching.lower, switching.upper)
    out["random forest"] = dict(onsets=onset_times(t_s, s), sma=a)

    # 4. SINDy
    a, s = run_hybrid(a0, s0, n, h, sindy_rate(laws.sindy, laws.a_ref), switching.lower, switching.upper)
    out["SINDy"] = dict(onsets=onset_times(t_s, s), sma=a)

    # 5. BINDy ensemble: sample coefficients and band edges from their uncertainty, run each
    rng = np.random.default_rng(0)
    n_edges = len(switching.start_smas)
    ensemble_onsets, ensemble_sma = [], []
    for _ in range(ENSEMBLE_SIZE):
        xi = laws.bindy_mean + laws.bindy_std * rng.standard_normal(len(laws.bindy_mean))
        lower = switching.lower + np.std(switching.start_smas) / np.sqrt(n_edges) * rng.standard_normal()
        upper = switching.upper + np.std(switching.end_smas) / np.sqrt(n_edges) * rng.standard_normal()
        a, s = run_hybrid(a0, s0, n, h, sindy_rate(xi, laws.a_ref), lower, upper)
        ensemble_onsets.append(match_onsets(onset_times(t_s, s), actual) + actual)
        ensemble_sma.append(a)
    ensemble_onsets = np.array(ensemble_onsets)
    out["BINDy ensemble"] = dict(onsets=np.nanmean(ensemble_onsets, axis=0), sma=np.mean(ensemble_sma, axis=0),
                                 sma_low=np.percentile(ensemble_sma, 5, axis=0),
                                 sma_high=np.percentile(ensemble_sma, 95, axis=0),
                                 onset_spread_s=np.nanstd(ensemble_onsets, axis=0))

    for result in out.values():
        result["errors_s"] = match_onsets(np.asarray(result["onsets"]), actual)
    return t_s, actual, out


def gp_coast_forecast(dataset, events, data, switching, t_s):
    """GP of SMA vs time since the previous burn ended (train only); coast time = when it reaches the lower edge."""
    ends = [e for _, e in runs_of_ones(dataset.t[dataset.train], events[dataset.train])]
    starts = [s for s, _ in runs_of_ones(dataset.t[dataset.train], events[dataset.train])]
    x, y = [], []
    for end, next_start in zip(ends[:-1], starts[1:]):
        arc = (dataset.t > end) & (dataset.t < next_start)
        x.append((dataset.t[arc] - end) * TU)
        y.append(data.a[arc])
    x, y = np.concatenate(x), np.concatenate(y)
    pick = np.random.default_rng(0).choice(len(x), 600, replace=False)
    scale_x, offset_y, scale_y = 3600.0, y.mean(), y.std()
    gp = GaussianProcessRegressor(ConstantKernel() * RBF(5.0) + WhiteKernel(0.1), normalize_y=False, random_state=0)
    gp.fit(x[pick, None] / scale_x, (y[pick] - offset_y) / scale_y)
    grid = np.linspace(0, x.max() * 1.3, 40000)          # ~10 s resolution, so rounding cannot accumulate
    curve = gp.predict(grid[:, None] / scale_x) * scale_y + offset_y
    coast_s = grid[np.argmax(curve <= switching.lower)]
    burn_s = np.mean([(e - s) * TU for s, e in zip(starts, ends)])
    # free-run: from the last training burn end, alternate coast and burn
    onsets, clock = [], ends[-1] * TU
    while clock < t_s[-1]:
        clock += coast_s
        onsets.append(clock)
        clock += burn_s
    onsets = np.array(onsets)
    return dict(onsets=onsets[(onsets >= t_s[0]) & (onsets <= t_s[-1])], sma=None, coast_hours=coast_s / 3600)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_library(ax, dataset, data):
    """The drag shape D and thrust shape B move together (and with the constant): a singular library."""
    rows = data.coast & dataset.train
    shapes = physics_library(data.a[rows], data.speed[rows])
    d_ppm = (shapes[:, 0] / shapes[:, 0].mean() - 1) * 1e6
    b_ppm = (shapes[:, 1] / shapes[:, 1].mean() - 1) * 1e6
    correlation = np.corrcoef(d_ppm, b_ppm)[0, 1]
    ax.scatter(b_ppm[::50], d_ppm[::50], s=3, color=NAVY)
    ax.set(xlabel="thrust shape B: change from its mean [ppm]", ylabel="drag shape D: change from its mean [ppm]",
           title=f"D and B are collinear (corr = {correlation:.5f}), both ~constant")


def _draw_coefficients(ax, laws, data, dataset):
    """Each term's contribution to da/dt at a typical state [mm/s], SINDy bars with BINDy +-2 sigma."""
    rows = (data.coast | data.burn) & dataset.train
    theta = library(data.a[rows], data.burn[rows].astype(float), laws.a_ref)
    typical = np.sqrt(np.mean(theta**2, axis=0))                     # typical size of each column
    to_mm_s = DU * 1e6 / TU
    sindy = laws.sindy * typical * to_mm_s
    bindy = laws.bindy_mean * typical * to_mm_s
    bindy_err = 2 * laws.bindy_std * typical * to_mm_s
    y = np.arange(len(LIBRARY_NAMES))
    ax.barh(y + 0.18, sindy, 0.35, color=NAVY, label="SINDy (STLSQ)")
    ax.barh(y - 0.18, bindy, 0.35, xerr=bindy_err, color=GREEN, label="BINDy-style (ARD), +-2 sigma")
    for i, value in enumerate(sindy):
        if value == 0:
            ax.text(0.02, i + 0.18, "removed", va="center", fontsize=8, color=SLATE)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set(yticks=y, yticklabels=LIBRARY_NAMES, xlabel="contribution to da/dt [mm/s] (symlog)",
           title="Learned law: only the two cluster terms survive")
    ax.axvline(0, color="black", lw=0.8)
    ax.legend(loc="lower right")


def _draw_physics(ax, laws):
    cd_s, thrust_s = laws.physics(laws.sindy)
    cd_b, thrust_b = laws.physics(laws.bindy_mean)
    cd_sd = 2 * laws.bindy_std[0] / laws.drag_shape
    thrust_sd = 2 * np.hypot(laws.bindy_std[1], laws.bindy_std[0]) / laws.thrust_shape
    x = np.arange(2)
    ax.bar(x - 0.18, [cd_s / SMOOTH_CD, thrust_s / THRUST_ACCEL], 0.35, color=NAVY, label="SINDy")
    ax.bar(x + 0.18, [cd_b / SMOOTH_CD, thrust_b / THRUST_ACCEL], 0.35, color=GREEN,
           yerr=[cd_sd / SMOOTH_CD, thrust_sd / THRUST_ACCEL], capsize=6, label="BINDy-style, 95% interval")
    ax.axhline(1, color="black", lw=1)
    for i, (s, b) in enumerate(((cd_s / SMOOTH_CD, cd_b / SMOOTH_CD), (thrust_s / THRUST_ACCEL, thrust_b / THRUST_ACCEL))):
        ax.text(i, 1.012, f"{s:.4f} / {b:.4f}", ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=["drag cd", "thrust level T"], ylim=(0.97, 1.02), ylabel="learned / true",
           title="Physics read off the learned law")
    ax.legend(loc="upper right")


def _draw_switching(ax, dataset, data, switching, sigmoid, accuracy):
    a_km = data.a * DU
    test = dataset.test
    every = slice(None, None, 200)
    jitter = 0.03 * np.random.default_rng(0).standard_normal(test.sum())[every]
    ax.scatter(a_km[test][every], dataset.true_on[test][every] + jitter, s=2, color=GREY, alpha=0.4,
               label="TEST samples (true state)")
    grid = np.linspace(a_km.min(), a_km.max(), 300)
    ax.plot(grid, sigmoid(grid / DU), color=RUST, lw=2, label=f"memory-less sigmoid (bal. acc. {accuracy['memoryless']:.2f})")
    lower, upper = switching.lower * DU, switching.upper * DU
    ax.plot([upper, lower, lower, upper, upper], [0, 0, 1, 1, 0], color=NAVY, lw=2,
            label=f"learned hysteresis (bal. acc. {accuracy['hysteresis']:.2f})")
    ax.ticklabel_format(axis="x", useOffset=False)
    ax.set(xlabel="mean SMA [km]", ylabel="thruster on (1) / off (0)",
           title=f"Switching law: edges {lower:.4f} / {upper:.4f} km")
    ax.legend(loc="center", fontsize=7)


def _draw_forecast(ax, dataset, data, t_s, actual, out):
    days = (t_s - t_s[0]) / 86400
    shown = days < FORECAST_PLOT_DAYS
    test_days = (data.t_s[dataset.test] - t_s[0]) / 86400
    measured = test_days < FORECAST_PLOT_DAYS
    ax.plot(test_days[measured][::20], data.a[dataset.test][measured][::20] * DU, color=GREY, lw=0.6,
            label="measured (noisy), TEST")
    bindy = out["BINDy ensemble"]
    ax.fill_between(days[shown], bindy["sma_low"][shown] * DU, bindy["sma_high"][shown] * DU, color=GREEN, alpha=0.25,
                    label="BINDy 5-95%")
    for name, style in (("SINDy", "-"), ("random forest", "--")):
        ax.plot(days[shown], out[name]["sma"][shown] * DU, color=METHOD_COLORS[name], lw=1.2, ls=style,
                label=f"{name} forecast")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set(xlabel="days into the TEST period", ylabel="mean SMA [km]",
           title=f"Free-running forecast, first {FORECAST_PLOT_DAYS} of 360 days (no TEST data used)")
    ax.legend(loc="lower left", fontsize=7)


def _draw_errors(ax, out, actual, t0):
    days = (actual - t0) / 86400
    for name, result in out.items():
        errors_min = result["errors_s"] / 60
        ax.plot(days, errors_min, ".-", ms=4, color=METHOD_COLORS[name], ls="--" if name == "random forest" else "-",
                label=f"{name} (mean |err| {np.nanmean(np.abs(errors_min)):.0f} min)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="days ahead (forecast from day 40)", ylabel="predicted - true onset [min]",
           title="Forecast timing error, burn by burn")
    ax.text(0.02, 0.03, "SINDy and random forest learn the same coast rate (the cluster mean of da/dt);\n"
                        "its 2.5e-4 error shortens every coast by ~30 s and adds up over 92 cycles",
            transform=ax.transAxes, fontsize=7, color=SLATE, ha="left", va="bottom")
    ax.legend(fontsize=7, loc="upper left")


def horizon_errors(out, actual, t0):
    """Mean |onset error| [min] of the burns inside each forecast horizon."""
    days = (actual - t0) / 86400
    return {name: [float(np.nanmean(np.abs(result["errors_s"][days < h]))) / 60 for h in HORIZONS_DAYS]
            for name, result in out.items()}


def _draw_horizon_errors(ax, table):
    for name, errors in table.items():
        ax.loglog(HORIZONS_DAYS, np.maximum(errors, 1e-2), "o-", color=METHOD_COLORS[name], label=name,
                  ls="--" if name == "random forest" else "-")      # dashed: it sits on top of SINDy
    ax.set(xticks=HORIZONS_DAYS, xlabel="forecast horizon [days]", ylabel="mean |onset error| [min]",
           title="Forecast accuracy vs horizon")
    ax.set_xticklabels([str(h) for h in HORIZONS_DAYS])
    ax.minorticks_off()
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    warnings.filterwarnings("ignore")
    folder = step_dir(out_dir, "step9_sindy_control_law")
    dataset, events, data, laws, switching = prepare()
    sigmoid, accuracy = switching_accuracy(dataset, data, switching)
    t_s, actual, out = forecasts(dataset, events, data, laws, switching)
    horizon_table = horizon_errors(out, actual, t_s[0])

    with report_style():
        fig, ax = plt.subplots(figsize=(7.5, 4.5)); _draw_library(ax, dataset, data); save(fig, folder, "step9_library.png")
        fig, ax = plt.subplots(figsize=(9, 4.5)); _draw_coefficients(ax, laws, data, dataset); save(fig, folder, "step9_coefficients.png")
        fig, ax = plt.subplots(figsize=(6.5, 4.5)); _draw_physics(ax, laws); save(fig, folder, "step9_physics_recovered.png")
        fig, ax = plt.subplots(figsize=(8, 4.8)); _draw_switching(ax, dataset, data, switching, sigmoid, accuracy); save(fig, folder, "step9_switching_law.png")
        fig, ax = plt.subplots(figsize=(10, 4.5)); _draw_forecast(ax, dataset, data, t_s, actual, out); save(fig, folder, "step9_forecast.png")
        fig, ax = plt.subplots(figsize=(9, 4.5)); _draw_errors(ax, out, actual, t_s[0]); save(fig, folder, "step9_forecast_errors.png")
        fig, ax = plt.subplots(figsize=(7.5, 4.5)); _draw_horizon_errors(ax, horizon_table); save(fig, folder, "step9_forecast_horizons.png")

        fig, axes = plt.subplots(2, 3, figsize=(19, 9.5))
        _draw_library(axes[0, 0], dataset, data)
        _draw_coefficients(axes[0, 1], laws, data, dataset)
        _draw_physics(axes[0, 2], laws)
        _draw_switching(axes[1, 0], dataset, data, switching, sigmoid, accuracy)
        _draw_errors(axes[1, 1], out, actual, t_s[0])
        _draw_horizon_errors(axes[1, 2], horizon_table)
        for ax, letter in zip(axes.ravel(), "abcdef"):
            panel_label(ax, letter)
        save(fig, folder, "step9_summary.png", suptitle="Step 9: learning the control law (SINDy / BINDy) and forecasting burns")

    cd_s, thrust_s = laws.physics(laws.sindy)
    cd_b, thrust_b = laws.physics(laws.bindy_mean)
    return dict(report_step9=dict(
        sindy_coefficients=dict(zip(LIBRARY_NAMES, laws.sindy.tolist())),
        bindy_mean=dict(zip(LIBRARY_NAMES, laws.bindy_mean.tolist())), bindy_std=dict(zip(LIBRARY_NAMES, laws.bindy_std.tolist())),
        cd_ratio=dict(sindy=cd_s / SMOOTH_CD, bindy=cd_b / SMOOTH_CD),
        thrust_ratio=dict(sindy=thrust_s / THRUST_ACCEL, bindy=thrust_b / THRUST_ACCEL),
        switching_edges_km=dict(lower=switching.lower * DU, upper=switching.upper * DU),
        switching_balanced_accuracy=accuracy,
        forecast_mean_abs_error_min={name: float(np.nanmean(np.abs(r["errors_s"])) / 60) for name, r in out.items()},
        forecast_error_min_by_horizon={name: dict(zip([f"{h}d" for h in HORIZONS_DAYS], errors))
                                       for name, errors in horizon_table.items()},
    ))


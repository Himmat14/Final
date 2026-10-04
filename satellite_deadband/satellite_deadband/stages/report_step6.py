"""
Report step 6: classifying the physics into COAST and BURN, trained on one part of the data and
tested on another, scored per burn EVENT, on noiseless and on noisy measurements.

Data: the 400-day controlled run. TRAIN = days 0-40, TEST = days 40-400; every score is also
reported over the first 15, 30, 60, 120, 240 and 360 days of TEST.

Figures (outputs/report/step6_classification/)
    deriv_*                     r'' vs v' GMM on days 0-10 (main.py stage "derivative_gmm", same folder)
    step6_dataset_split         the 400-day run: true burns, train (days 0-40) / test (days 40-400)
    step6_derivative_spaces     unmodelled acceleration from positions (r'') and from velocities (v'),
                                coast vs burn, clean vs noisy: why the classifiers use v'
    step6_feature_space         the classifier features with the fitted GMM / Bayesian-GMM components
    step6_test_timeline         every method's detected events on the first 15 TEST days vs the truth
    step6_horizon_scores        event F1 and false alarms vs TEST horizon (15 ... 360 days)
    step6_event_scores          precision / recall / F1 per method, noiseless vs noisy
    step6_timing_errors         how far each detected start / end is from the true switch-on / off
    step6_summary               six-panel summary
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt

from deadband import gmm_2d
from deadband.classifiers import MODELS, REFERENCE_NOISE, detection_dataset, evaluate, noisy_features
from deadband.constants import THRUST_ACCEL_MS2, to_days
from deadband.long_run import HORIZONS_DAYS, TRAIN_DAYS
from deadband.derivative_detection import runs_of_ones
from .common import AMBER, GREEN, GREY, NAVY, PURPLE, RUST
from .report_common import panel_label, report_style, save, step_dir

METHOD_COLORS = dict(zip(MODELS, (NAVY, GREEN, RUST, PURPLE, AMBER)))
CASES = {"noiseless": 0.0, "noisy": 1.0}       # noise scale x REFERENCE_NOISE (0.1 m, 0.5 mm/s)
NOISE_LABEL = f"noisy ({REFERENCE_NOISE[0]:g} m, {REFERENCE_NOISE[1]:g} mm/s)"


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_dataset(ax, dataset):
    days = to_days(dataset.t)
    ax.plot(days[::100], dataset.mean_sma_km[::100], color=NAVY, lw=0.6)
    split = days[dataset.train][-1]
    ax.axvspan(days[0], split, color=NAVY, alpha=0.10, label=f"train (days 0-{TRAIN_DAYS})")
    ax.axvspan(split, days[-1], color=RUST, alpha=0.06, label=f"test (days {TRAIN_DAYS}-{days[-1]:.0f})")
    n_train = len(runs_of_ones(dataset.t[dataset.train], dataset.true_on[dataset.train]))
    n_test = len(runs_of_ones(dataset.t[dataset.test], dataset.true_on[dataset.test]))
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set(xlabel="time [days]", ylabel="mean SMA [km]",
           title=f"500 m band, 400 days: {n_train} train + {n_test} test burns")
    ax.legend(loc="lower left")


def _draw_spaces(ax, dataset, features):
    bins = np.linspace(-17, -6, 90)
    every = slice(None, None, 10)                       # every 10th sample is plenty for a histogram
    on = dataset.true_on[every] == 1
    for (case, feature), style in zip(features.items(), ("-", "--")):
        log_r = np.log(np.linalg.norm(feature.accel_from_r[:, every], axis=0) + 1e-12)
        log_v = np.log(np.linalg.norm(feature.accel_from_v[:, every], axis=0) + 1e-12)
        for values, color, name in ((log_r, RUST, "r''"), (log_v, NAVY, "v'")):
            ax.hist(values[~on], bins=bins, density=True, histtype="step", color=color, ls=style, lw=1.3,
                    label=f"{name} coast, {case}")
    ax.axvline(np.log(THRUST_ACCEL_MS2), color=AMBER, lw=2, label="burn level (thrust)")
    ax.set(xlabel="log |unmodelled acceleration [m/s$^2$]|", ylabel="density (coast samples)",
           title="Coast noise floor: from positions (r'') vs velocities (v')")
    ax.legend(ncol=2, fontsize=7)


def _draw_feature_space(ax, dataset, X, model, title):
    sample = np.random.default_rng(0).choice(len(X), 20000, replace=False)
    burn_rows = np.flatnonzero(dataset.true_on)[::5]
    ax.scatter(X[sample, 0], X[sample, 1], s=2, color=GREY, alpha=0.3, label="coast (true)")
    ax.scatter(X[burn_rows, 0], X[burn_rows, 1], s=4, color=RUST, label="burn (true)")
    for k in range(model.n_components):
        if model.weights_[k] < 0.005:
            continue
        ex, ey = gmm_2d.ellipse_points(model.means_[k], model.covariances_[k], n_std=2.0)
        ax.plot(ex, ey, color=NAVY, lw=1.2)
    ax.set(xlabel="log |v' residual|", ylabel="along-track v' residual [1e-4 m/s$^2$]", title=title)
    ax.legend(loc="upper left", markerscale=4)


def _draw_timeline(ax, dataset, results, title, horizon_days=15):
    window = dataset.test_horizon(horizon_days)
    test_t = dataset.t[window]
    for row, (name, result) in enumerate(results.items()):
        for start, end in runs_of_ones(test_t, result.events[window]):
            ax.plot([to_days(start), to_days(end) + 0.01], [row, row], lw=8, color=METHOD_COLORS[name],
                    solid_capstyle="butt")
    for start, end in runs_of_ones(test_t, dataset.true_on[window]):
        ax.axvspan(to_days(start), to_days(end) + 0.01, color=AMBER, alpha=0.35)
    ax.set(yticks=range(len(results)), yticklabels=list(results), xlabel="time [days]", title=title,
           ylim=(-0.7, len(results) - 0.3))


def _draw_horizon_f1(ax, scores, case):
    for name in MODELS:
        f1 = [scores[case][name].horizon_scores[h]["f1"] for h in HORIZONS_DAYS]
        ax.semilogx(HORIZONS_DAYS, f1, "o-", color=METHOD_COLORS[name], label=name)
    ax.set(xticks=HORIZONS_DAYS, ylim=(-0.05, 1.05), xlabel="TEST horizon [days after training]",
           ylabel="event F1", title=f"Event F1 vs horizon, {case}")
    ax.set_xticklabels([str(h) for h in HORIZONS_DAYS])
    ax.minorticks_off()
    ax.legend(fontsize=7)


def _draw_scores(ax, scores):
    names = list(MODELS)
    width = 0.38
    for k, (case, hatch) in enumerate((("noiseless", ""), ("noisy", "//"))):
        f1 = [scores[case][n].test_scores["f1"] for n in names]
        ax.bar(np.arange(len(names)) + (k - 0.5) * width, f1, width, color=[METHOD_COLORS[n] for n in names],
               hatch=hatch, edgecolor="white", label=f"F1, {case}")
        for i, n in enumerate(names):
            s = scores[case][n].test_scores
            ax.text(i + (k - 0.5) * width, f1[i] + 0.02, f"{s['n_found']}/{s['n_burns']}\n{s['n_false_events']} FA",
                    ha="center", fontsize=6.5)
    ax.set(xticks=range(len(names)), xticklabels=names, ylim=(0, 1.3), ylabel="event F1 on TEST",
           title="Event scores: burns found / false alarms (FA)")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(loc="upper right")


def _draw_timing(ax, scores):
    names = list(MODELS)
    for k, (case, marker) in enumerate((("noiseless", "o"), ("noisy", "s"))):
        for i, name in enumerate(names):
            per_burn = [b for b in scores[case][name].test_scores["per_burn"] if b["found"]]
            starts = [b["start_error_s"] for b in per_burn]
            ends = [b["end_error_s"] for b in per_burn]
            x = i + (k - 0.5) * 0.35
            ax.scatter([x - 0.06] * len(starts), starts, marker=marker, s=14, color=METHOD_COLORS[name],
                       label=f"start, {case}" if i == 0 else None)
            ax.scatter([x + 0.06] * len(ends), ends, marker=marker, s=14, facecolors="none",
                       edgecolors=METHOD_COLORS[name], label=f"end, {case}" if i == 0 else None)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xticks=range(len(names)), xticklabels=names, ylabel="detected - true [s]", ylim=(-200, 200),
           title="Timing of detected burn start / end (TEST)")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=7, ncol=2)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    warnings.filterwarnings("ignore", category=UserWarning)
    folder = step_dir(out_dir, "step6_classification")

    dataset = detection_dataset()
    features = {case: noisy_features(dataset, scale)[0] for case, scale in CASES.items()}
    scores = {case: evaluate(dataset, features[case], horizons=HORIZONS_DAYS) for case in CASES}

    with report_style():
        fig, ax = plt.subplots(figsize=(10, 4)); _draw_dataset(ax, dataset); save(fig, folder, "step6_dataset_split.png")
        fig, ax = plt.subplots(figsize=(8, 4.5)); _draw_spaces(ax, dataset, features); save(fig, folder, "step6_derivative_spaces.png")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for ax, case in zip(axes, CASES):
            label = case if case == "noiseless" else NOISE_LABEL
            _draw_feature_space(ax, dataset, features[case].matrix, scores[case]["Bayesian GMM"].model,
                                f"Bayesian GMM components (2 sigma), {label}")
        save(fig, folder, "step6_feature_space.png")
        fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
        for ax, case in zip(axes, CASES):
            _draw_timeline(ax, dataset, scores[case], f"Detected events, first 15 TEST days, {case} (orange = truth)")
        save(fig, folder, "step6_test_timeline.png")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for ax, case in zip(axes, CASES):
            _draw_horizon_f1(ax, scores, case)
        save(fig, folder, "step6_horizon_scores.png")
        fig, ax = plt.subplots(figsize=(9, 4.5)); _draw_scores(ax, scores); save(fig, folder, "step6_event_scores.png")
        fig, ax = plt.subplots(figsize=(9, 4.5)); _draw_timing(ax, scores); save(fig, folder, "step6_timing_errors.png")

        fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
        _draw_dataset(axes[0, 0], dataset)
        _draw_spaces(axes[0, 1], dataset, features)
        _draw_feature_space(axes[0, 2], dataset, features["noisy"].matrix, scores["noisy"]["Bayesian GMM"].model,
                            "Feature space, noisy, Bayesian GMM")
        _draw_horizon_f1(axes[1, 0], scores, "noisy")
        _draw_scores(axes[1, 1], scores)
        _draw_timing(axes[1, 2], scores)
        for ax, letter in zip(axes.ravel(), "abcdef"):
            panel_label(ax, letter)
        save(fig, folder, "step6_summary.png", suptitle="Step 6: classifying coast vs burn, trained and tested on separate data")

    def compact(s):
        return {k: v for k, v in s.items() if k != "per_burn"}

    table = {case: {name: dict(test=compact(r.test_scores),
                               by_horizon={f"{h}d": compact(r.horizon_scores[h]) for h in HORIZONS_DAYS})
                    for name, r in results.items()} for case, results in scores.items()}
    return dict(report_step6=dict(reference_noise=REFERENCE_NOISE, scores=table))


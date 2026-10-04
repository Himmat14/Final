"""
Week 5: burn detection with a GMM in "rdot space" and "vdot space" on days 0-10 of the report's
400-day controlled run (the cached Week 5 run IS that run's first 10-day chunk, so this is the same data).

Every GMM input comes straight from the state vector: positions r and velocities v are
sampled every SAMPLE_STEP_MIN (0.1 min = 6 s), finite-differenced, and the known gravity +
J2 is subtracted. GMM flags are grouped into burn EVENTS (no single-point detections) and
performance is scored per event: how many true burns were flagged, and how many false ones.

Figures
-------
deriv_residual_timeline   one burn: unmodelled acceleration from r and from v vs the true thrust,
                          raw GMM flags and the detected events (is this data representative of the thruster?)
deriv_gmm_histograms      log |unmodelled acceleration| in each space, with the fitted GMM components
deriv_gmm_2d              rdot vs vdot evidence together, 2-component GMM with 2-sigma ellipses
deriv_method_comparison   event-level scores for rdot / vdot / both, and the Week 4 features
                          (2-feature GMM on log |delta osculating SMA|, log |delta speed|) on the same samples

Comparison of derivative methods (acceleration from VELOCITY data, 5 ways), clean and noisy:
deriv_gmm_histograms_methods   log |unmodelled accel from v'| per method, with the fitted GMM components (tuned K)
deriv_gmm_2d_methods           the two velocity features per method (size, along-track) with 2-sigma ellipses
deriv_method_floor             coasting error floor per method vs the real drag and the thrust, and event F1

deriv_cadence_sweep       event scores vs sampling step (1 ... 60 s)
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture

from deadband import gmm_2d
from deadband.constants import SAMPLE_STEP_MIN, SAMPLE_STEP_S, SMOOTH_DAYS, THRUST_ACCEL_MS2, from_minutes, to_minutes
from deadband.derivative_detection import (
    DERIVATIVE_METHODS, burn_components, build_derivative_evidence, event_scores, fit_derivative_gmm, fit_method_gmm,
    group_into_events, method_evidence, runs_of_ones, velocity_features, cadence_sweep,
)
from deadband.smooth_controller import default_runs, osculating_sma_km_series
from .common import AMBER, GREEN, GREY, NAVY, PURPLE, RUST, SLATE, save_figure
from .report_common import step_dir
from deadband.settings import tuned

EVENT_TOLERANCE_S = 60.0    # a detected event within this many seconds of a burn still counts for that burn
SWEEP_STEPS_S = (1, 2, 5, 10, 20, 30, 60)
SPACES = ("rdot", "vdot", "both")
SPACE_LABELS = {"rdot": "rdot space (from r)", "vdot": "vdot space (from v)", "both": "rdot + vdot",
                "week4": "Week 4: delta SMA + delta speed"}
SPACE_COLORS = {"rdot": NAVY, "vdot": RUST, "both": GREEN, "week4": GREY}
NOISE_CASES = {"clean": (0.0, 0.0), "noisy (0.1 m, 0.5 mm/s)": (0.1, 0.5)}   # (position m, velocity mm/s)
METHOD_COLORS = dict(zip(DERIVATIVE_METHODS, (NAVY, "#3F7FBF", GREEN, PURPLE, RUST)))


# ---------------------------------------------------------------------------
# The Week 4 features on the same samples, for comparison
# ---------------------------------------------------------------------------
def _week4_style_scores(sim, evidence):
    """
    The Week 4 detector on THIS data: a 2-component GMM on [log |delta osculating SMA|, log |delta speed|]
    (sample-to-sample jumps), grouped into events and scored like the others. With J2 the osculating
    SMA swings ~12 km per orbit, so its jumps are dominated by J2, not by the thruster.
    """
    samples = sim.sample(evidence.t)

    def log_jump(values):
        step = np.abs(np.diff(values))
        return np.log(np.concatenate([[step[0]], step]) + 1e-12)

    features = np.column_stack([log_jump(osculating_sma_km_series(samples)),
                                log_jump(np.linalg.norm(samples[3:6], axis=0))])
    model = GaussianMixture(n_components=2, n_init=3, random_state=0).fit(features)
    burn = int(np.argmax(model.means_.sum(axis=1)))
    predicted = group_into_events((model.predict(features) == burn).astype(int))
    return event_scores(evidence.t, predicted, evidence.true_on, evidence.step_s, EVENT_TOLERANCE_S)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plot_residual_timeline(evidence, fits, out_dir):
    start, end = runs_of_ones(evidence.t, evidence.true_on)[0]
    window = (evidence.t > start - from_minutes(15)) & (evidence.t < end + from_minutes(15))
    minutes = to_minutes(evidence.t[window] - start)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.5), sharex=True)
    for space, accel in (("rdot", evidence.accel_from_r), ("vdot", evidence.accel_from_v)):
        axes[0].semilogy(minutes, np.linalg.norm(accel, axis=0)[window], ".-", ms=3, lw=0.8,
                         color=SPACE_COLORS[space], label=f"|unmodelled accel|, {SPACE_LABELS[space]}")
    axes[0].semilogy(minutes, evidence.true_thrust_ms2[window] + 1e-9, color=AMBER, lw=2, alpha=0.7,
                     label="true thrust (+1e-9 so it shows on a log axis)")
    axes[0].set(ylabel="m/s^2", title=f"Finite differences of state-vector r, v every {SAMPLE_STEP_MIN} min, "
                                       "minus known gravity + J2")
    axes[0].legend(fontsize=8, loc="center right")

    # along-track component: is the residual really the thruster?
    along_r = np.sum(evidence.accel_from_r * evidence.along_track, axis=0)
    along_v = np.sum(evidence.accel_from_v * evidence.along_track, axis=0)
    axes[1].plot(minutes, evidence.true_thrust_ms2[window] * 1e4, color=AMBER, lw=3, alpha=0.6, label="true thrust")
    axes[1].plot(minutes, along_r[window] * 1e4, ".", ms=4, color=SPACE_COLORS["rdot"], label="along-track residual, from r")
    axes[1].plot(minutes, along_v[window] * 1e4, "x", ms=4, color=SPACE_COLORS["vdot"], label="along-track residual, from v")
    axes[1].set(ylabel="along-track accel [1e-4 m/s^2]",
                title="Is this data representative of the thruster? Along-track residual vs true thrust")
    axes[1].legend(fontsize=8, loc="center right")

    # raw per-sample flags (thin ticks) vs the grouped events (thick bars)
    for row, space in enumerate(SPACES):
        flagged = fits[space].flagged[window] == 1
        axes[2].plot(minutes[flagged], np.full(flagged.sum(), row + 0.2), "|", ms=6, color=SPACE_COLORS[space], alpha=0.5)
        for event_start, event_end in runs_of_ones(evidence.t, fits[space].predicted):
            if event_end >= evidence.t[window][0] and event_start <= evidence.t[window][-1]:
                axes[2].plot(to_minutes(np.array([event_start, event_end]) - start), [row - 0.15] * 2,
                             lw=7, color=SPACE_COLORS[space], solid_capstyle="butt")
    axes[2].axvspan(0, to_minutes(end - start), color=AMBER, alpha=0.25, label="true burn (throttle > 1/2)")
    axes[2].axvspan(-EVENT_TOLERANCE_S / 60, 0, color=GREY, alpha=0.2, label=f"+-{EVENT_TOLERANCE_S:.0f} s tolerance")
    axes[2].axvspan(to_minutes(end - start), to_minutes(end - start) + EVENT_TOLERANCE_S / 60, color=GREY, alpha=0.2)
    axes[2].set(yticks=range(len(SPACES)), yticklabels=[SPACE_LABELS[s] for s in SPACES], ylim=(-0.7, len(SPACES) - 0.3),
                xlabel="Minutes from burn start",
                title="GMM: raw flagged samples (thin ticks) grouped into one detected event (thick bar)")
    axes[2].legend(fontsize=8, loc="center right")
    for ax in axes:
        ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "deriv_residual_timeline.png")


def _draw_histogram_with_components(ax, feature, true_on, model, is_burn, column=0):
    """
    Histogram of ALL samples (counts, log y-axis so the small burn class is visible) split by the
    truth, with every GMM component drawn as  N * weight * Gaussian * bin width , i.e. on the same
    count scale. Drawn like this, each component sits exactly where the samples it explains are.
    (The old version drew unweighted Gaussians on per-class densities and stopped the curve at the
    last sample, which made the burn component look flat and off-centre.)
    """
    margin = 0.5
    bins = np.linspace(feature.min() - margin, feature.max() + margin, 140)
    width = bins[1] - bins[0]
    ax.hist(feature[true_on == 0], bins=bins, color=SLATE, alpha=0.55, label="coast (true)")
    ax.hist(feature[true_on == 1], bins=bins, color=RUST, alpha=0.85, label="burn (true)")
    names = component_names(model, is_burn, column)
    for k in range(len(model.weights_)):
        mean, sd = model.means_[k, column], np.sqrt(np.atleast_2d(model.covariances_[k])[column, column])
        # a component can be far narrower than one bin (clean data): add points around its mean so it shows
        grid = np.sort(np.concatenate([np.linspace(bins[0], bins[-1], 800), mean + sd * np.linspace(-4, 4, 200)]))
        counts = len(feature) * model.weights_[k] * width * np.exp(-0.5 * ((grid - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        style = COMPONENT_STYLES[names[k]]
        ax.plot(grid, np.minimum(counts, 10 * len(feature)), lw=1.6, label=names[k], **style)
    ax.axvline(np.log(THRUST_ACCEL_MS2), color=AMBER, ls=":", lw=1.4, label="full thrust level")
    ax.set_yscale("log")
    ax.set_xlim(bins[0], bins[-1])                          # a wide ramp component must not stretch the axis
    ax.set_ylim(0.5, len(feature))
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))                     # one legend entry per label
    ax.legend(unique.values(), unique.keys(), fontsize=7, loc="upper center")


COMPONENT_STYLES = {"GMM burn component": dict(color=PURPLE, ls="-"),
                    "GMM ramp / transition component": dict(color=GREEN, ls="-."),
                    "GMM coast component(s)": dict(color=NAVY, ls="--")}


def component_names(model, is_burn, column=0):
    """Burn-flagged components: the highest is 'burn', any lower ones catch the thruster ramps."""
    top = int(np.argmax(np.where(is_burn, model.means_[:, column], -np.inf))) if np.any(is_burn) else -1
    return ["GMM burn component" if k == top else
            "GMM ramp / transition component" if is_burn[k] else "GMM coast component(s)"
            for k in range(len(model.weights_))]


def _plot_gmm_histograms(evidence, fits, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, space in zip(axes, ("rdot", "vdot")):
        fit = fits[space]
        is_burn = np.arange(len(fit.model.weights_)) == fit.burn_component
        _draw_histogram_with_components(ax, evidence.features(space)[:, 0], evidence.true_on, fit.model, is_burn)
        ax.set(xlabel="log |unmodelled acceleration [m/s^2]|", ylabel="number of samples (log scale)",
               title=f"{SPACE_LABELS[space]}, 2nd-order FD, every {SAMPLE_STEP_MIN} min")
        ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "deriv_gmm_histograms.png")


def _plot_gmm_2d(evidence, fit, out_dir):
    features, labels = evidence.features("both"), evidence.true_on
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(features[labels == 0, 0], features[labels == 0, 1], s=3, color=SLATE, alpha=0.3, label="coast (true)")
    ax.scatter(features[labels == 1, 0], features[labels == 1, 1], s=12, color=RUST, label="burn (true)")
    for k, color in zip(range(2), ("#FFB000", "#00A0C0")):
        ex, ey = gmm_2d.ellipse_points(fit.model.means_[k], fit.model.covariances_[k], n_std=2.0)
        name = "burn" if k == fit.burn_component else "coast"
        ax.plot(ex, ey, color=color, lw=2, label=f"GMM {name} component (2 sigma)")
    ax.set(xlabel="log |unmodelled accel| from r (rdot space)", ylabel="log |unmodelled accel| from v (vdot space)",
           title=f"GMM on rdot + vdot evidence together (every {SAMPLE_STEP_MIN} min)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    save_figure(fig, out_dir, "deriv_gmm_2d.png")


# ---------------------------------------------------------------------------
# Comparison of derivative methods (velocity data only)
# ---------------------------------------------------------------------------
def method_comparison(controlled):
    """{case: {method: (evidence, tuned-K fit, 2-component fit)}} for the clean and noisy data."""
    table = {}
    for case, (position_m, velocity_mms) in NOISE_CASES.items():
        table[case] = {}
        for method in DERIVATIVE_METHODS:
            evidence = method_evidence(controlled, SAMPLE_STEP_S, method, position_m, velocity_mms, seed=1)
            table[case][method] = (evidence, fit_method_gmm(evidence, tuned("gmm_components"), EVENT_TOLERANCE_S),
                                   fit_method_gmm(evidence, 2, EVENT_TOLERANCE_S))
    return table


def _short(method):
    return method.replace("central ", "").replace(" order", "") if method != "Savitzky-Golay" else "Sav.-Golay"


def _coast_floor(evidence):
    """Median |unmodelled acceleration from v'| while coasting [m/s^2]."""
    return float(np.median(np.exp(evidence.log_accel_from_v[evidence.true_on == 0])))


def _plot_method_histograms(table, out_dir):
    fig, axes = plt.subplots(len(table), len(DERIVATIVE_METHODS), figsize=(22, 8.5), sharey=True)
    for row, (case, methods) in enumerate(table.items()):
        for col, (method, (evidence, fit, _)) in enumerate(methods.items()):
            ax = axes[row, col]
            _draw_histogram_with_components(ax, evidence.log_accel_from_v, evidence.true_on, fit.model,
                                            burn_components(fit.model))
            ax.set_title(f"{method}, {case}\ncoast floor {_coast_floor(evidence):.1e} m/s$^2$, "
                         f"event F1 {fit.event_scores['f1']:.2f}", fontsize=9)
            ax.set_xlabel("log |unmodelled accel from v'| [m/s$^2$]", fontsize=8)
            ax.grid(alpha=0.3)
        axes[row, 0].set_ylabel("number of samples (log scale)")
    fig.suptitle(f"Acceleration from VELOCITY data by five derivative methods, {tuned('gmm_components')}-component GMM "
                 "(top: clean state vectors, bottom: 0.1 m / 0.5 mm/s noise)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir, "deriv_gmm_histograms_methods.png")


def _plot_method_2d(table, out_dir):
    fig, axes = plt.subplots(len(table), len(DERIVATIVE_METHODS), figsize=(22, 8.5))
    for row, (case, methods) in enumerate(table.items()):
        for col, (method, (evidence, fit, _)) in enumerate(methods.items()):
            ax = axes[row, col]
            features, labels = velocity_features(evidence), evidence.true_on
            ax.scatter(features[labels == 0, 0][::5], features[labels == 0, 1][::5], s=2, color=SLATE, alpha=0.3,
                       label="coast (true, every 5th)")
            ax.scatter(features[labels == 1, 0], features[labels == 1, 1], s=8, color=RUST, label="burn (true)")
            names = component_names(fit.model, burn_components(fit.model))
            colors = {"GMM burn component": "#00A0C0", "GMM ramp / transition component": GREEN,
                      "GMM coast component(s)": "#FFB000"}
            for k in range(len(fit.model.weights_)):
                ex, ey = gmm_2d.ellipse_points(fit.model.means_[k], fit.model.covariances_[k], n_std=2.0)
                ax.plot(ex, ey, color=colors[names[k]], lw=1.8, label=f"{names[k]} (2 sigma)")
                if names[k] == "GMM burn component":     # a very tight burn component is just a dot
                    ax.plot(*fit.model.means_[k], "*", ms=12, color=colors[names[k]], mec="black", mew=0.5)
            ax.set_title(f"{method}, {case}\nevent F1 {fit.event_scores['f1']:.2f}, "
                         f"false events {fit.event_scores['n_false_events']}", fontsize=9)
            ax.set_xlabel("log |unmodelled accel from v'|", fontsize=8)
            ax.grid(alpha=0.3)
            if col == 0:
                handles, names = ax.get_legend_handles_labels()
                unique = dict(zip(names, handles))
                ax.set_ylabel("along-track part [1e-4 m/s$^2$]")
                ax.legend(unique.values(), unique.keys(), fontsize=7, loc="upper left")
    fig.suptitle("The two velocity features per derivative method: thrust pushes FORWARDS (up), "
                 "truncation error and noise do not", fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir, "deriv_gmm_2d_methods.png")


def _plot_method_floor(table, out_dir, drag_ms2):
    fig, (ax_floor, ax_f1) = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(DERIVATIVE_METHODS))
    for (case, methods), offset, hatch in zip(table.items(), (-0.2, 0.2), ("", "//")):
        floors = [_coast_floor(evidence) for evidence, _, _ in methods.values()]
        ax_floor.bar(x + offset, floors, 0.4, color=[METHOD_COLORS[m] for m in methods], hatch=hatch,
                     edgecolor="black", lw=0.5, label=case)
        for xi, value in zip(x + offset, floors):
            ax_floor.text(xi, value * 1.3, f"{value:.0e}", ha="center", fontsize=7)
        for k, (label, index) in enumerate(((f"{tuned('gmm_components')} components (tuned)", 1), ("2 components", 2))):
            ax_f1.plot(x + (k - 0.5) * 0.08, [fits[index].event_scores["f1"] for fits in methods.values()],
                       marker="os"[k], ms=8 - 2 * k, ls="-" if case == "clean" else "--", color=(NAVY, RUST)[k],
                       label=f"{label}, {case}")       # small x offset so overlapping lines stay visible
    ax_floor.axhline(THRUST_ACCEL_MS2, color=AMBER, ls="--", label="thrust (2e-4 m/s$^2$)")
    ax_floor.axhline(drag_ms2, color=GREEN, ls="--", label=f"real drag ({drag_ms2:.1e} m/s$^2$)")
    ax_floor.set(yscale="log", xticks=x, xticklabels=[_short(m) for m in DERIVATIVE_METHODS],
                 ylabel="median coasting |unmodelled accel| [m/s$^2$]",
                 title="Coast error floor: truncation (clean, solid) vs noise (noisy, hatched)")
    ax_floor.set_ylim(3e-7, 2e-3)
    ax_floor.legend(fontsize=7, loc="upper center", ncol=2)
    ax_f1.set(xticks=x, xticklabels=[_short(m) for m in DERIVATIVE_METHODS], ylim=(-0.05, 1.1), ylabel="event F1",
              title="Burn detection: tuned K vs 2 GMM components")
    ax_f1.legend(fontsize=7, loc="center left")
    ax_f1.text(0.02, 0.6, "with noise only Savitzky-Golay (smoothing) + the tuned K works:\n"
                          "higher-order stencils amplify the noise",
               transform=ax_f1.transAxes, fontsize=7, color=SLATE)
    for ax in (ax_floor, ax_f1):
        ax.grid(alpha=0.3, axis="y")
    save_figure(fig, out_dir, "deriv_method_floor.png")


def _plot_cadence_sweep(table, out_dir):
    """Event-level scores vs sampling step (not part of the Week 5 talk)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for space, rows in table.items():
        steps = [row["step_s"] for row in rows]
        style = dict(color=SPACE_COLORS[space], marker="o", label=SPACE_LABELS[space])
        axes[0].loglog(steps, [row["coast_median_ms2"] for row in rows], **style)
        axes[1].semilogx(steps, [row["n_found"] for row in rows], **style)
        axes[2].semilogx(steps, [row["n_false_events"] for row in rows], **style)
    axes[0].axhline(THRUST_ACCEL_MS2, color=AMBER, ls="--", label="thrust level")
    axes[0].set(ylabel="median coasting |unmodelled accel| [m/s^2]", title="Finite-difference error grows like h^2")
    axes[1].set(ylabel="true burns flagged", title="True events flagged")
    axes[2].set(ylabel=f"false burn events in {SMOOTH_DAYS:g} days", yscale="symlog", title="False events")
    for ax in axes:
        ax.set_xlabel("sampling step h [s]"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    save_figure(fig, out_dir, "deriv_cadence_sweep.png")


def _plot_method_comparison(scores, out_dir):
    """Event-level scores for each method: burns flagged, false events, precision / recall / F1, coverage."""
    methods = list(scores)
    n_burns = scores[methods[0]]["n_burns"]
    fig, (ax_events, ax_rates) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw=dict(width_ratios=[1.2, 2]))

    x = np.arange(len(methods))
    ax_events.bar(x - 0.2, [scores[m]["n_found"] for m in methods], 0.4, color=[SPACE_COLORS[m] for m in methods],
                  label="true burns flagged")
    ax_events.bar(x + 0.2, [scores[m]["n_false_events"] for m in methods], 0.4, color="white",
                  edgecolor=[SPACE_COLORS[m] for m in methods], hatch="//", label="false events")
    ax_events.axhline(n_burns, color="black", ls="--", lw=1, label=f"true burns in the run ({n_burns})")
    for i, method in enumerate(methods):
        ax_events.text(i - 0.2, scores[method]["n_found"], str(scores[method]["n_found"]), ha="center", va="bottom")
        ax_events.text(i + 0.2, scores[method]["n_false_events"], str(scores[method]["n_false_events"]),
                       ha="center", va="bottom")
    ax_events.set(xticks=x, xticklabels=["Week 4" if m == "week4" else m for m in methods], ylabel="number of events",
                  ylim=(0, 1.25 * max(n_burns, max(scores[m]["n_false_events"] for m in methods))),
                  title="Events, not samples")
    ax_events.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax_events.grid(alpha=0.3, axis="y")

    rates = [("recall\n(burns flagged / burns)", "recall"), ("precision\n(real / detected events)", "precision"),
             ("F1", "f1"), ("coverage of\nfound burns", "mean_coverage")]
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        ax_rates.bar(np.arange(len(rates)) + (i - (len(methods) - 1) / 2) * width,
                     [scores[method][key] for _, key in rates], width, color=SPACE_COLORS[method],
                     label=SPACE_LABELS[method])
    ax_rates.set(xticks=range(len(rates)), xticklabels=[name for name, _ in rates], ylim=(0, 1.08),
                 title=f"Event-level GMM scores, days 0-{SMOOTH_DAYS:g} of the controlled run, every {SAMPLE_STEP_MIN} min")
    ax_rates.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(methods))
    ax_rates.grid(alpha=0.3, axis="y")
    save_figure(fig, out_dir, "deriv_method_comparison.png")


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    """`sim` is not used: days 0-10 of the 400-day controlled run (the cached Week 5 run); figures go to step6_classification/."""
    out_dir = step_dir(out_dir, "step6_classification")
    controlled, _ = default_runs()
    evidence = build_derivative_evidence(controlled, SAMPLE_STEP_S)
    fits = {space: fit_derivative_gmm(evidence, space, EVENT_TOLERANCE_S) for space in SPACES}

    _plot_residual_timeline(evidence, fits, out_dir)
    _plot_gmm_histograms(evidence, fits, out_dir)
    _plot_gmm_2d(evidence, fits["both"], out_dir)

    scores = {space: fits[space].event_scores for space in SPACES}
    scores = {"week4": _week4_style_scores(controlled, evidence), **scores}       # Week 4 features, same samples
    _plot_method_comparison(scores, out_dir)

    # comparison of derivative methods: acceleration from velocity data, clean and noisy
    table = method_comparison(controlled)
    _plot_method_histograms(table, out_dir)
    _plot_method_2d(table, out_dir)
    drag_ms2 = _coast_floor(table["clean"]["central 8th order"][0])   # 8th order: truncation error ~0, so this is drag
    _plot_method_floor(table, out_dir, drag_ms2)
    method_scores = {case: {method: dict(coast_floor_ms2=_coast_floor(evidence),
                                         f1_4_components=fit3.event_scores["f1"],
                                         f1_2_components=fit2.event_scores["f1"],
                                         false_events_4_components=fit3.event_scores["n_false_events"])
                            for method, (evidence, fit3, fit2) in methods.items()}
                     for case, methods in table.items()}

    sweep = cadence_sweep(controlled, steps_s=SWEEP_STEPS_S, spaces=("rdot", "vdot"), tolerance_s=EVENT_TOLERANCE_S)
    _plot_cadence_sweep(sweep, out_dir)

    return dict(derivative_gmm=dict(step_min=SAMPLE_STEP_MIN, event_tolerance_s=EVENT_TOLERANCE_S,
                                    event_scores=scores, derivative_methods=method_scores, cadence_sweep=sweep))

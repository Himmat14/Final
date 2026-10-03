"""
Week 5: burn detection with a GMM in "rdot space" and "vdot space" on the smooth J2 run.

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
deriv_method_comparison   event-level scores for rdot / vdot / both

Commented out (not part of the Week 5 talk; uncomment in `run` to bring back):
deriv_cadence_sweep       event scores vs sampling step
Week 4 features           the Week 4 delta-SMA / delta-speed GMM as an extra bar in the comparison
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband import gmm_2d
from deadband.constants import SAMPLE_STEP_MIN, SAMPLE_STEP_S, THRUST_ACCEL_MS2, from_minutes, to_minutes
from deadband.derivative_detection import (
    build_derivative_evidence, event_scores, fit_derivative_gmm, group_into_events, runs_of_ones,
    cadence_sweep,  # noqa: F401  (used by the commented-out cadence sweep in `run`)
)
from deadband.evidence import BurnEvidence
from deadband.smooth_controller import default_runs, osculating_sma_km_series
from .common import AMBER, GREEN, GREY, NAVY, PURPLE, RUST, SLATE, save_figure

EVENT_TOLERANCE_S = 60.0    # a detected event within this many seconds of a burn still counts for that burn
SWEEP_STEPS_S = (1, 2, 5, 10, 20, 30, 60)
SPACES = ("rdot", "vdot", "both")
SPACE_LABELS = {"rdot": "rdot space (from r)", "vdot": "vdot space (from v)", "both": "rdot + vdot",
                "week4": "Week 4: delta SMA + delta speed"}
SPACE_COLORS = {"rdot": NAVY, "vdot": RUST, "both": GREEN, "week4": GREY}


# ---------------------------------------------------------------------------
# Week 4 features on the new data, for comparison (currently not used in `run`)
# ---------------------------------------------------------------------------
def _week4_style_scores(sim, evidence):
    """The Week 4 detector (2-feature GMM on log |delta osculating SMA|, log |delta speed|), scored per event."""
    samples = sim.sample(evidence.t)
    osculating = osculating_sma_km_series(samples)
    speed = np.linalg.norm(samples[3:6], axis=0)

    def jump(values):
        step = np.abs(np.diff(values))
        return np.concatenate([[step[0]], step])

    week4_evidence = BurnEvidence(t=evidence.t, true_label=evidence.true_on, jump_sma=jump(osculating),
                                  jump_speed=jump(speed))
    predicted = group_into_events(gmm_2d.summarize_gmm_2d(week4_evidence).predicted)
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


def _plot_gmm_histograms(evidence, fits, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, space in zip(axes, ("rdot", "vdot")):
        feature = evidence.features(space)[:, 0]
        bins = np.linspace(feature.min(), feature.max(), 120)
        ax.hist(feature[evidence.true_on == 0], bins=bins, color=SLATE, alpha=0.6, density=True, label="coast (true)")
        ax.hist(feature[evidence.true_on == 1], bins=bins, color=RUST, alpha=0.8, density=True, label="burn (true)")
        model = fits[space].model
        grid = np.linspace(bins[0], bins[-1], 400)
        for k in range(2):
            mean, sd = model.means_[k, 0], np.sqrt(model.covariances_[k, 0, 0])
            density = np.exp(-0.5 * ((grid - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            name = "burn" if k == fits[space].burn_component else "coast"
            ax.plot(grid, density, color=PURPLE if name == "burn" else NAVY, lw=1.5, label=f"GMM {name} component")
        ax.axvline(np.log(THRUST_ACCEL_MS2), color=AMBER, ls="--", lw=1.2, label="full thrust level")
        ax.set(xlabel="log |unmodelled acceleration [m/s^2]|", ylabel="density (each class normalised)",
               title=f"{SPACE_LABELS[space]}, every {SAMPLE_STEP_MIN} min")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
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
    axes[2].set(ylabel="false burn events in 8 days", yscale="symlog", title="False events")
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
                 title=f"Event-level GMM scores, 8-day J2 run sampled every {SAMPLE_STEP_MIN} min")
    ax_rates.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(methods))
    ax_rates.grid(alpha=0.3, axis="y")
    save_figure(fig, out_dir, "deriv_method_comparison.png")


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    """`sim` (the Week 4 run) is not used: this stage uses the cached 8-day smooth J2 run."""
    controlled, _ = default_runs()
    evidence = build_derivative_evidence(controlled, SAMPLE_STEP_S)
    fits = {space: fit_derivative_gmm(evidence, space, EVENT_TOLERANCE_S) for space in SPACES}

    _plot_residual_timeline(evidence, fits, out_dir)
    _plot_gmm_histograms(evidence, fits, out_dir)
    _plot_gmm_2d(evidence, fits["both"], out_dir)

    scores = {space: fits[space].event_scores for space in SPACES}
    # Not part of the Week 5 talk -- uncomment to add the Week 4 features as an extra bar:
    # scores = {"week4": _week4_style_scores(controlled, evidence), **scores}
    _plot_method_comparison(scores, out_dir)

    # Not part of the Week 5 talk (every plot now uses one 0.1 min step) -- uncomment to bring back:
    # sweep = cadence_sweep(controlled, steps_s=SWEEP_STEPS_S, spaces=("rdot", "vdot"), tolerance_s=EVENT_TOLERANCE_S)
    # _plot_cadence_sweep(sweep, out_dir)

    return dict(derivative_gmm=dict(step_min=SAMPLE_STEP_MIN, event_tolerance_s=EVENT_TOLERANCE_S,
                                    event_scores=scores))

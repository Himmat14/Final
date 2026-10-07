"""
Report step 20: the whole analysis chain end to end (measured r, v -> GMM burn detection -> mean SMA ->
c_d, T and switching edges by SINDy / BINDy -> free-running burn forecast), with
    * the accuracy of every stage and an error budget (each stage swapped for the truth in turn),
    * the GMM step in detail (features, components, information criteria, P(burn) through a burn),
    * a sampling-robustness study of the whole chain, the Gaussian-process thrust law and the space-weather
      forecaster, and
    * closed-form checks of the headline numbers.
Library: deadband/pipeline.py, deadband/report_checks.py.

Every figure here is a single panel (no summary montage), sized to be read at page width.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from deadband import pipeline as P
from deadband.constants import THRUST_ACCEL_MS2
from deadband.long_run import HORIZONS_DAYS
from deadband.report_checks import checks, to_latex
from .common import AMBER, GREEN, GREY, NAVY, PURPLE, RUST, SKY_BLUE, SLATE
from .report_common import report_style, save, step_dir

MODEL_COLORS = dict(zip(P.MODELS, (NAVY, RUST, GREEN, PURPLE, SLATE)))
MODEL_MARKERS = dict(zip(P.MODELS, "osD^v"))
SIZE = (8, 4.8)


def _horizon_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(HORIZONS_DAYS)
    ax.set_xticklabels([str(h) for h in HORIZONS_DAYS])
    ax.minorticks_off()
    ax.set_xlabel("forecast horizon [days after day 40]")


def _step_axis(ax, steps):
    ax.set_xscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s:g} s" if s < 60 else f"{s / 60:g} min" for s in steps])
    ax.minorticks_off()
    ax.set_xlabel("sampling step")
    ax.axvline(P.DERIVATIVE_LIMIT_S * 0.9, color=GREY, ls=":", lw=1)


# ---------------------------------------------------------------------------
# Error budget
# ---------------------------------------------------------------------------
def _draw_budget(ax, rows):
    names = [r["variant"] for r in rows]
    values = [r["errors"]["360d"] for r in rows]
    colors = [NAVY] + [GREEN] * (len(rows) - 3) + [SLATE, GREY]
    bars = ax.barh(np.arange(len(rows))[::-1], values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.1f} min", va="center", fontsize=8)
    ax.set(yticks=np.arange(len(rows))[::-1], xlabel="mean |burn-onset error| over 360 days [min]",
           ylabel="pipeline variant",
           title="Error budget: each stage made perfect in turn", xlim=(0, max(values) * 1.25))
    ax.set_yticklabels(names, fontsize=8.5)


def _draw_budget_horizon(ax, rows):
    styles = ["-", "--", "-.", ":", (0, (5, 1)), "-", (0, (1, 1))]
    colors = [NAVY, AMBER, RUST, PURPLE, GREEN, SLATE, GREY]
    for r, ls, c in zip(rows, styles, colors):
        ax.plot(HORIZONS_DAYS, [max(r["errors"][f"{h}d"], 0.1) for h in HORIZONS_DAYS], ls=ls, color=c, marker="o",
                ms=4, label=f"{r['variant']} (P = {r['period_days']:.5f} d)")
    _horizon_axis(ax)
    ax.set(yscale="log", ylabel="mean |burn-onset error| [min]", title="Error budget against forecast horizon")
    ax.legend(fontsize=7.5)


def _draw_models(ax, forecasts, label):
    for m, (errors, _) in forecasts.items():
        ax.plot(HORIZONS_DAYS, [max(errors[f"{h}d"], 0.1) for h in HORIZONS_DAYS], color=MODEL_COLORS[m],
                marker=MODEL_MARKERS[m], label=m)
    _horizon_axis(ax)
    ax.set(yscale="log", ylabel="mean |burn-onset error| [min]", title=f"Forecasters on the learned pipeline ({label})")
    ax.legend(fontsize=8)


def _draw_labels_effect(ax, forecasts):
    x = np.arange(len(P.MODELS))
    for offset, (label, color) in zip((-0.2, 0.2), (("GMM labels", NAVY), ("true labels", AMBER))):
        values = [max(forecasts[label][m][0]["360d"], 0.1) for m in P.MODELS]
        ax.bar(x + offset, values, 0.38, color=color, label=f"burn labels: {label}")
        for xi, v in zip(x + offset, values):
            ax.text(xi, v * 1.1, f"{v:.0f}" if v >= 1 else f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set(yscale="log", xticks=x, xlabel="forecaster", ylabel="mean |onset error| at 360 d [min]",
           title="Does the detector limit the forecast? GMM vs true labels")
    ax.set_xticklabels([m.replace(" (", "\n(") for m in P.MODELS], fontsize=8)
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# GMM anatomy
# ---------------------------------------------------------------------------
def _draw_gmm_hist(ax, g):
    x = g["X"][:, 0]
    grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 800)
    # both histograms as densities of the WHOLE sample, so they sit on the same scale as the mixture terms
    bins = np.linspace(x.min(), x.max(), 200)
    width = bins[1] - bins[0]
    for mask, color, label in ((~g["truth"], SKY_BLUE, "true coast samples"), (g["truth"], AMBER, "true burn samples")):
        counts, _ = np.histogram(x[mask], bins)
        ax.bar(bins[:-1], counts / (x.size * width), width, align="edge", color=color, alpha=0.6, label=label)
    total = np.zeros_like(grid)
    for i, c in enumerate(g["components"]):
        sd = np.sqrt(c["cov"][0][0])
        pdf = c["weight"] * np.exp(-0.5 * ((grid - c["mean"][0]) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        total += pdf
        ax.plot(grid, pdf, color=RUST if c["burn"] else NAVY, ls="--" if c["burn"] else "-", lw=1.3,
                label=f"component {i + 1}: w = {c['weight']:.4f} ({'burn' if c['burn'] else 'coast'})")
    ax.plot(grid, total, color="black", lw=1, alpha=0.6, label="mixture density")
    ax.axvline(g["threshold"], color=GREY, ls=":", lw=1.5, label="burn rule: coast mean + 3 coast sd")
    ax.set(yscale="log", ylim=(1e-4, 5), xlabel="feature 1: log |unmodelled acceleration| [log m/s$^2$]",
           ylabel="probability density", title="The fitted mixture along the acceleration-magnitude feature")
    ax.legend(fontsize=7.5, loc="upper right")


def _draw_gmm_scatter(ax, g):
    X, truth = g["X"], g["truth"]
    ax.scatter(X[~truth, 0], X[~truth, 1], s=2, color=SKY_BLUE, alpha=0.3, label="true coast", rasterized=True)
    ax.scatter(X[truth, 0], X[truth, 1], s=6, color=AMBER, alpha=0.8, label="true burn", rasterized=True)
    for i, c in enumerate(g["components"]):
        vals, vecs = np.linalg.eigh(np.array(c["cov"]))
        angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        for n_sd in (1, 2):
            ax.add_patch(Ellipse(c["mean"], 2 * n_sd * np.sqrt(vals[1]), 2 * n_sd * np.sqrt(vals[0]), angle=angle,
                                 fill=False, color=RUST if c["burn"] else NAVY, lw=1.5 if n_sd == 1 else 0.8,
                                 ls="-" if n_sd == 1 else "--"))
        ax.annotate(f"{i + 1}", c["mean"], fontsize=9, fontweight="bold", color=RUST if c["burn"] else NAVY,
                    xytext=(6, 6), textcoords="offset points")
    ax.axvline(g["threshold"], color=GREY, ls=":", lw=1.5, label="burn rule threshold")
    ax.set(xlabel="feature 1: log |unmodelled acceleration|", ylabel="feature 2: along-track acceleration / 1e-4 m/s$^2$",
           title="Feature space with the 1- and 2-sigma ellipses of the 4 components")
    ax.legend(fontsize=8, loc="upper left", markerscale=3)


def _draw_gmm_ic(ax, g):
    k = [r["k"] for r in g["ic"]]
    bic = np.array([r["bic"] for r in g["ic"]])
    aic = np.array([r["aic"] for r in g["ic"]])
    ax.plot(k, bic - bic.min(), "o-", color=NAVY, label="BIC - min BIC")
    ax.plot(k, aic - aic.min(), "s--", color=RUST, label="AIC - min AIC")
    ax.axvline(4, color=GREY, ls=":", label="K = 4 (chosen by detection F1, step 0)")
    ax.set(yscale="symlog", xlabel="number of mixture components K", ylabel="information criterion above its minimum",
           title="Information criteria keep falling: the coast cloud is not Gaussian")
    ax.legend(fontsize=8)


def _draw_gmm_posterior(axes, g):
    """Two stacked panels: (top) the along-track signal, noisy and noise-free, the true thrust and the burn rule;
    (bottom) P(burn | x) against the true thruster state. Kept apart so neither hides the other."""
    top, bottom = axes
    tm = g["t_min"]
    top.plot(tm, g["feature"][:, 1], color=GREY, lw=0.6, alpha=0.8, label="measured along-track feature (noisy)")
    top.plot(tm, g["clean_feature"][:, 1], color="black", lw=1.6, label="noise-free signal (same derivative filter)")
    top.plot(tm, g["thrust"] / 1e-4, color=AMBER, ls="--", lw=1.6, label="true thrust acceleration")
    rule = np.exp(g["threshold"]) / 1e-4
    top.axhline(rule, color=RUST, ls=":", lw=1.5, label=f"burn rule |a| = exp(threshold) = {rule * 1e4:.1e} m/s$^2$")
    on = np.flatnonzero(np.diff(g["on"].astype(int)))
    for i, edge in enumerate(on):
        top.axvline(tm[edge + 1], color=NAVY, ls="-.", lw=1, label="true activation bounds (throttle = 1/2)" if i == 0 else None)
        bottom.axvline(tm[edge + 1], color=NAVY, ls="-.", lw=1)
    top.set(ylabel="along-track acceleration [1e-4 m/s$^2$]",
            title="One TEST burn: the signal the GMM sees (top) and its decision (bottom)")
    top.legend(fontsize=7.5, loc="center right")
    bottom.fill_between(tm, 0, g["on"], color=AMBER, alpha=0.3, step="mid", label="true thruster on")
    bottom.plot(tm, g["prob"], color=NAVY, lw=1.6, label="GMM P(burn | x)")
    bottom.set(xlabel="minutes from the true burn start", ylabel="probability", ylim=(-0.05, 1.1),
               title="Posterior burn probability")
    bottom.legend(fontsize=8, loc="center right")


# ---------------------------------------------------------------------------
# Sampling robustness
# ---------------------------------------------------------------------------
def _draw_sampling_params(ax, rows):
    steps = [r["step_s"] for r in rows]
    floor = 1e-3
    for key, label, color, marker in (("cd_error_pct", "c_d, split (D)", NAVY, "o"),
                                      ("T_error_pct", "T, split (D)", RUST, "s"),
                                      ("cd_joint_error_pct", "c_d, joint (A)", NAVY, "^"),
                                      ("T_joint_error_pct", "T, joint (A)", RUST, "v")):
        ls = "-" if "joint" not in key else "--"
        ax.plot(steps, [max(abs(r[key]), floor) for r in rows], ls=ls, marker=marker, color=color, label=label)
    ax.plot(steps, [2 * r["cd_bindy_std_pct"] for r in rows], ":", color=GREEN, label="BINDy 2-sigma for c_d")
    _step_axis(ax, steps)
    ax.set(yscale="log", ylabel="|learned / true - 1| [%]", title="Physics parameters against the sampling step")
    ax.legend(fontsize=8)


def _draw_sampling_f1_edges(ax, rows):
    steps = [r["step_s"] for r in rows]
    ax.plot(steps, [r["lower_error_m"] for r in rows], "o-", color=NAVY, label="lower edge error [m]")
    ax.plot(steps, [r["upper_error_m"] for r in rows], "s--", color=RUST, label="upper edge error [m]")
    ax.axhline(0, color="black", lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(steps, [r["f1"] for r in rows], "D:", color=GREEN, label="GMM event F1 (TEST)")
    ax2.set(ylim=(0, 1.05), ylabel="event F1")
    _step_axis(ax, steps)
    ax.set(ylabel="learned - true band edge [m]", title="Detection and switching edges against the sampling step")
    lines, lines2 = ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=8, loc="center left")


def _draw_sampling_forecast(ax, rows):
    steps = [r["step_s"] for r in rows]
    for m in P.MODELS:
        ax.plot(steps, [max(r["forecast"][m]["360d"], 0.1) for r in rows], marker=MODEL_MARKERS[m],
                color=MODEL_COLORS[m], label=m)
    _step_axis(ax, steps)
    ax.set(yscale="log", ylabel="mean |onset error| at 360 d [min]", title="The whole chain against the sampling step")
    ax.legend(fontsize=8)


def _draw_gp_profiles(ax, gp):
    colors = [NAVY, SKY_BLUE, GREEN, AMBER, RUST]
    for r, c in zip(gp, colors):
        ax.plot(np.array(r["offsets_s"]) / 60, np.array(r["mean"]) * 1e4, color=c, lw=1.4,
                label=f"step {r['step_s']:g} s (derivative window {r['window_s']:g} s)")
    ax.axhline(THRUST_ACCEL_MS2 * 1e4, color="black", ls=":", lw=1, label="true plateau")
    ax.set(xlabel="minutes from the detected burn start", ylabel="learned thrust law [1e-4 m/s$^2$]",
           title="Gaussian-process thrust law from stacked burns, by sampling step")
    ax.legend(fontsize=8)


def _draw_gp_ratios(ax, gp):
    steps = [r["step_s"] for r in gp]
    ax.plot(steps, [r["plateau_ratio"] for r in gp], "o-", color=NAVY, label="plateau / true")
    ax.plot(steps, [r["duration_ratio"] for r in gp], "s--", color=RUST, label="duration / true")
    ax.axhline(1, color="black", lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(steps, [r["rise_time_s"] for r in gp], "D:", color=GREEN, label="learned rise time [s]")
    ax2.set_ylabel("rise time 10-90 % [s]")
    ax.set_xscale("log"); ax.set_xticks(steps); ax.set_xticklabels([f"{s:g} s" for s in steps]); ax.minorticks_off()
    ax.set(xlabel="sampling step", ylabel="learned / true", title="Thrust-law numbers against the sampling step")
    lines, lines2 = ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=8, loc="lower left")


def _draw_cadence(ax, rows, horizon):
    cad = [r["cadence_s"] / 60 for r in rows]
    colors = dict(zip(rows[0]["errors"], (SLATE, GREY, NAVY, GREEN)))
    for name in rows[0]["errors"]:
        ax.plot(cad, [r["errors"][name][horizon] for r in rows], "o-", color=colors[name], label=name)
    ax.set(xscale="log", xticks=cad, xlabel="mean-SMA measurement cadence [min]", yscale="log",
           ylabel=f"mean |onset error| at {horizon[:-1]} d [min]",
           title=f"Space-weather drag: forecast error at {horizon[:-1]} days against cadence")
    ax.set_xticklabels([f"{c:g}" for c in cad]); ax.minorticks_off()
    ax.legend(fontsize=8)


def _draw_checks(ax, rows):
    eq = [r for r in rows if r["bound"] is None]
    y = np.arange(len(eq))[::-1]
    ax.barh(y, [max(100 * r["relative_difference"], 1e-3) for r in eq],
            color=[GREEN if r["passed"] else RUST for r in eq])
    for yi, r in zip(y, eq):
        ax.text(max(100 * r["relative_difference"], 1e-3) * 1.15, yi, f"{100 * r['relative_difference']:.2f} %",
                va="center", fontsize=7.5)
    ax.set(xscale="log", yticks=y, xlabel="|formula - measured| / measured [%]", ylabel="check", xlim=(1e-3, 100),
           title="Closed-form checks of the headline numbers")
    ax.set_yticklabels([r["check"] for r in eq], fontsize=8)


# ---------------------------------------------------------------------------
# Tables for the LaTeX report
# ---------------------------------------------------------------------------
def _tex_sampling(rows):
    out = [r"\begin{tabular}{lllrrrrrr}", r"\toprule",
           r"Step & Feature & F1 & $c_d$ err. & $T$ err. & lower edge & upper edge & law (D) 360\,d & GP 360\,d\\",
           r" & & & [\%] & [\%] & [m] & [m] & [min] & [min]\\", r"\midrule"]
    for r in rows:
        step = f"{r['step_s']:g}\\,s" if r["step_s"] < 60 else f"{r['step_s'] / 60:g}\\,min"
        out.append(f"{step} & {'S-G derivative' if r['feature'].startswith('Savitzky') else 'propagation'} & "
                   f"{r['f1']:.3f} & {r['cd_error_pct']:+.3f} & {r['T_error_pct']:+.2f} & {r['lower_error_m']:+.1f} & "
                   f"{r['upper_error_m']:+.1f} & {r['forecast']['SINDy split (D)']['360d']:.0f} & "
                   f"{r['forecast']['GP coast curve']['360d']:.0f}\\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


# main.py points this at the results of the stages that ran before this one (this run)
PREVIOUS = {}
CHECK_INPUTS = ("deadband", "smooth_deadband", "report_step1", "report_step3_noise", "report_step8",
                "report_step12", "report_step16", "report_step19", "spectral_nyquist_limit_min")


def _previous_results(out_dir):
    """Results of earlier stages: this run's where available, else the last outputs/results.json."""
    path = Path(out_dir) / "results.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(json.loads(json.dumps(PREVIOUS, default=_plain)))
    missing = [k for k in CHECK_INPUTS if k not in merged]
    if missing:
        raise RuntimeError(f"step 20 needs the results of earlier stages first: {missing}")
    return merged


def _plain(value):
    return value.tolist() if isinstance(value, np.ndarray) else float(value)


def run(sim, out_dir):
    folder = step_dir(out_dir, "step20_pipeline")
    budget, forecasts, metrics, learned, effective = P.error_budget()
    anatomy = P.gmm_anatomy()
    sampling = P.sampling_study()
    gp = P.thrust_law_sampling()
    cadence = P.cadence_study()
    check_rows = checks(_previous_results(out_dir))

    with report_style():
        for name, draw, args in (
                ("step20_error_budget.png", _draw_budget, (budget,)),
                ("step20_budget_horizon.png", _draw_budget_horizon, (budget,)),
                ("step20_forecasters.png", _draw_models, (forecasts["GMM labels"], "GMM labels")),
                ("step20_labels_effect.png", _draw_labels_effect, (forecasts,)),
                ("step20_gmm_histogram.png", _draw_gmm_hist, (anatomy,)),
                ("step20_gmm_feature_space.png", _draw_gmm_scatter, (anatomy,)),
                ("step20_gmm_information.png", _draw_gmm_ic, (anatomy,)),
                ("step20_sampling_parameters.png", _draw_sampling_params, (sampling,)),
                ("step20_sampling_detection_edges.png", _draw_sampling_f1_edges, (sampling,)),
                ("step20_sampling_forecast.png", _draw_sampling_forecast, (sampling,)),
                ("step20_gp_profiles.png", _draw_gp_profiles, (gp,)),
                ("step20_gp_numbers.png", _draw_gp_ratios, (gp,)),
                ("step20_cadence_15d.png", _draw_cadence, (cadence, "15d")),
                ("step20_cadence_60d.png", _draw_cadence, (cadence, "60d"))):
            fig, ax = plt.subplots(figsize=SIZE)
            draw(ax, *args)
            save(fig, folder, name)
        fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, gridspec_kw=dict(height_ratios=(1.6, 1)))
        _draw_gmm_posterior(axes, anatomy)
        save(fig, folder, "step20_gmm_posterior.png")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        _draw_checks(ax, check_rows)
        save(fig, folder, "step20_checks.png")
    (folder / "sampling_table.tex").write_text(_tex_sampling(sampling), encoding="utf-8")
    (folder / "checks_table.tex").write_text(to_latex(check_rows), encoding="utf-8")

    slim_anatomy = {k: v for k, v in anatomy.items() if k in ("components", "threshold", "ic", "confusion")}
    slim_gp = [{k: v for k, v in r.items() if k not in ("offsets_s", "mean")} for r in gp]
    return dict(report_step20=dict(error_budget=budget, forecasts=forecasts, stage_metrics=metrics,
                                   effective_truth=effective, gmm=slim_anatomy, sampling=sampling,
                                   thrust_law_sampling=slim_gp, cadence=cadence, checks=check_rows))

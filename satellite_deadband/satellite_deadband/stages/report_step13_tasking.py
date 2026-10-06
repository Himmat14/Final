"""
Report step 13: sensor tasking and track association (objectives 1 and 3 of the brief).
Library: deadband/conjunction.tasking, on the step 10 space-weather scenario.

Each day offers 6 observation opportunities at random times; the network can use 1, 2 or 4 of them.
Three ways to choose: at random, evenly spaced, or manoeuvre-aware (the first pass after each PREDICTED
burn end, from the step 10 "law + space weather" forecast). After an unobserved burn a drag-only
catalogue drifts 3 km/h; if its error at the next observation exceeds the 20 km association gate the
track cannot be linked (custody lost). A manoeuvre-aware catalogue carries the predicted burn and is only
wrong by the forecast timing error.

Figures (outputs/report/step13_tasking/)
    step13_custody        custody loss vs observation budget, every strategy, with / without a manoeuvre model
    step13_latency        time from burn end to the first observation, per strategy and budget
    step13_error_growth   along-track error after a burn: unknown burn vs predicted burn, with the gate
    step13_summary
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband import conjunction as C, forecasting as F
from .common import GREEN, GREY, NAVY, RUST
from .report_common import panel_label, report_style, save, step_dir

COLORS = dict(zip(C.STRATEGIES, (GREY, NAVY, GREEN)))


def _draw_custody(ax, rows):
    width = 0.13
    for k, strategy in enumerate(C.STRATEGIES):
        mine = [r for r in rows if r["strategy"] == strategy]
        x = np.arange(len(mine)) + (k - 1) * 2 * width
        ax.bar(x - width / 2, [100 * r["custody_lost"] for r in mine], width, color=COLORS[strategy],
               label=f"{strategy}: drag-only catalogue")
        ax.bar(x + width / 2, [100 * r["custody_lost_with_manoeuvre_model"] for r in mine], width, color=COLORS[strategy],
               hatch="//", edgecolor="white", label=f"{strategy}: + manoeuvre model")
    ax.set(xticks=range(len(C.BUDGETS)), xticklabels=[f"{b} obs/day" for b in C.BUDGETS], xlabel="observation budget",
           ylabel="burns after which custody is lost [%]", title=f"Custody loss (association gate {C.GATE_KM:g} km)")
    ax.legend(fontsize=7, ncol=2)


def _draw_latency(ax, rows):
    for strategy in C.STRATEGIES:
        mine = [r for r in rows if r["strategy"] == strategy]
        ax.plot(C.BUDGETS, [r["median_latency_h"] for r in mine], "o-", color=COLORS[strategy], label=strategy)
    ax.set(xticks=C.BUDGETS, xlabel="observations per day", ylabel="median time from burn end to next observation [h]",
           title="Observing soon after the burn")
    ax.legend()


def _draw_growth(ax, sc):
    L = sc.learned
    drift = 1.5 * C.mean_motion(L.upper) * (L.upper - L.lower) * 3600
    hours = np.linspace(0, 24, 200)
    ax.plot(hours, drift * hours, color=RUST, label=f"burn unknown: {drift:.1f} km per hour")
    timing = np.median([abs(e) for n, _, lead, e in sc.rows if n == "law + space weather" and lead <= 5]) / 3600
    ax.plot(hours, np.minimum(drift * hours, drift * timing), color=GREEN,
            label=f"burn predicted (median timing error {timing:.1f} h)")
    ax.axhline(C.GATE_KM, color="black", ls=":", label=f"association gate {C.GATE_KM:g} km")
    ax.set(xlabel="hours since the burn", ylabel="along-track error of the catalogue [km]",
           title="Why the next observation must come quickly")
    ax.legend()


def run(sim, out_dir):
    folder = step_dir(out_dir, "step13_tasking")
    sc = F.scenario()
    rows = C.tasking(sc)
    with report_style():
        fig, ax = plt.subplots(figsize=(10, 5)); _draw_custody(ax, rows); save(fig, folder, "step13_custody.png")
        fig, ax = plt.subplots(figsize=(8, 4.8)); _draw_latency(ax, rows); save(fig, folder, "step13_latency.png")
        fig, ax = plt.subplots(figsize=(8, 4.8)); _draw_growth(ax, sc); save(fig, folder, "step13_error_growth.png")
        fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
        _draw_custody(axes[0], rows); _draw_latency(axes[1], rows); _draw_growth(axes[2], sc)
        for ax, letter in zip(axes, "abc"):
            panel_label(ax, letter)
        save(fig, folder, "step13_summary.png", suptitle="Step 13: sensor tasking and track association")
    return dict(report_step13=dict(rows=rows))

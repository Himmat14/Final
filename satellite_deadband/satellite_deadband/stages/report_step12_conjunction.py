"""
Report step 12: what manoeuvre prediction buys close-approach screening (objective 1 of the brief).
Library: deadband/conjunction.py, on the step 10 space-weather scenario.

Figures (outputs/report/step12_conjunction/)
    step12_position_error     RMS along-track position error vs lead time for every prediction method
    step12_error_spread       the distribution of along-track errors at 3 days
    step12_screening          dangerous conjunctions MISSED and FALSE ALERTS vs lead time, honest and
                              catalogue-only covariance
    step12_summary
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband import conjunction as C, forecasting as F
from .common import AMBER, GREEN, GREY, NAVY, PURPLE, SLATE
from .report_common import panel_label, report_style, save, step_dir

COLORS = dict(zip(C.METHODS, (SLATE, GREY, NAVY, AMBER, GREEN, PURPLE)))
MARKERS = dict(zip(C.METHODS, "XosD^v"))


def _draw_error(ax, errors):
    for method, leads in errors.items():
        rms = [np.sqrt(np.mean(leads[l] ** 2)) for l in C.LEADS_DAYS]
        ax.loglog(C.LEADS_DAYS, rms, "-", marker=MARKERS[method], color=COLORS[method], label=method)
    ax.axhline(C.DANGER_KM, color="black", ls=":", label=f"danger radius {C.DANGER_KM * 1000:.0f} m")
    ax.set(xticks=C.LEADS_DAYS, xlabel="lead time to TCA [days]", ylabel="RMS along-track error [km]",
           title="Position error at the close approach (Figure 1 of the brief, measured)")
    ax.set_xticklabels([str(l) for l in C.LEADS_DAYS])
    ax.minorticks_off()
    ax.legend(fontsize=7)


def _draw_spread(ax, errors, lead=3):
    data = [np.abs(errors[m][lead]) + 1e-3 for m in C.METHODS]
    parts = ax.boxplot(data, patch_artist=True, showfliers=False)
    for box, m in zip(parts["boxes"], C.METHODS):
        box.set(facecolor=COLORS[m], alpha=0.6)
    ax.set(yscale="log", xticks=range(1, len(C.METHODS) + 1), xlabel="prediction method",
           ylabel="|along-track error| [km]", title=f"Error spread at {lead} days")
    ax.set_xticklabels([m.replace(" (no manoeuvres)", "\n(no manoeuvres)").replace(", ", ",\n").replace(" + ", "\n+ ")
                        for m in C.METHODS], fontsize=7)


def _draw_screening(axes, rows):
    for ax, key, ylabel in zip(axes, ("missed", "false_alerts_per_1000"),
                               ("fraction of DANGEROUS conjunctions missed", "false alerts per 1000 encounters")):
        for method in C.METHODS:
            for cov, ls in (("honest", "-"), ("catalogue only", ":")):
                mine = [r for r in rows if r["method"] == method and r["covariance"] == cov]
                ax.plot([r["lead"] for r in mine], [r[key] for r in mine], ls, marker=MARKERS[method], ms=4,
                        color=COLORS[method], label=f"{method} ({cov})" if key == "missed" else None)
        ax.set(xscale="log", xticks=C.LEADS_DAYS, xlabel="lead time to TCA [days]", ylabel=ylabel,
               title="Screening (Pc > 1e-4 alerts): solid = honest covariance, dotted = catalogue only")
        ax.set_xticklabels([str(l) for l in C.LEADS_DAYS])
        ax.minorticks_off()
    axes[0].legend(fontsize=6, ncol=2)


def run(sim, out_dir):
    folder = step_dir(out_dir, "step12_conjunction")
    errors = C.along_track_errors(F.scenario())
    rows = C.screening(errors)
    with report_style():
        fig, ax = plt.subplots(figsize=(9, 5)); _draw_error(ax, errors); save(fig, folder, "step12_position_error.png")
        fig, ax = plt.subplots(figsize=(10, 5)); _draw_spread(ax, errors); save(fig, folder, "step12_error_spread.png")
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5)); _draw_screening(axes, rows); save(fig, folder, "step12_screening.png")
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        _draw_error(axes[0, 0], errors); _draw_spread(axes[0, 1], errors)
        _draw_screening(axes[1], rows)
        for ax, letter in zip(axes.ravel(), "abcd"):
            panel_label(ax, letter)
        save(fig, folder, "step12_summary.png", suptitle="Step 12: close-approach screening with and without manoeuvre prediction")
    return dict(report_step12=dict(
        rms_along_track_km={m: {f"{l}d": float(np.sqrt(np.mean(v[l] ** 2))) for l in C.LEADS_DAYS} for m, v in errors.items()},
        screening=rows))

"""
Report step 22: the chain on a controlled run with every force (J2, drag, Moon, Sun, SRP), and the forecast
tables at every horizon for the whole report. Library: deadband/full_forcing.py, deadband/report_tables.py.

Figures (outputs/report/step22_full_forcing/), single panels:
    step22_force_magnitudes     |drag|, |Moon|, |Sun|, |SRP| along the controlled orbit: the same order of magnitude
    step22_force_along          their along-track components (what changes the SMA)
    step22_sma_signature        the mean SMA on a coast arc minus its straight line: the Moon / Sun signature
    step22_coast_cloud          the coast feature x1 with and without the third bodies, clean data
    step22_forecast_horizons    split law forecast error vs horizon for the three set-ups
    step22_forecast_models      every forecaster at 360 days for the three set-ups
    step22_error_budget         the error budget with all forces (J2-only analyst model)
Tables (outputs/report/tables/): onset error at 15-360 days for steps 9, 10, 19, 20 and 22.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from deadband import full_forcing as F
from deadband.long_run import HORIZONS_DAYS
from deadband.pipeline import MODELS
from deadband.report_tables import write_all
from .common import AMBER, GREEN, GREY, NAVY, PURPLE, RUST, SKY_BLUE, SLATE
from .report_common import report_style, save, step_dir

SIZE = (8.5, 5.0)
FORCE_STYLE = {"drag": (NAVY, "-"), "Moon": (RUST, "--"), "Sun": (AMBER, "-."), "SRP": (GREEN, ":")}
CASE_STYLE = dict(zip([c[0] for c in F.CASES], ((NAVY, "o", "-"), (RUST, "s", "--"), (GREEN, "D", "-."))))
PREVIOUS = {}


def _draw_magnitudes(ax, fs):
    for name, s in fs["series"].items():
        c, ls = FORCE_STYLE[name]
        ax.semilogy(fs["hours"], s["magnitude"], color=c, ls=ls, lw=1.6, label=name)
    ax.set(xlabel="time [hours]", ylabel="acceleration [m/s$^2$]",
           title="Drag and the third bodies along the controlled orbit: all ~1e-7 to 1e-6 m/s$^2$")
    ax.legend(fontsize=8)


def _draw_along(ax, fs, hours=6.0):
    keep = fs["hours"] < hours
    for name, s in fs["series"].items():
        c, ls = FORCE_STYLE[name]
        ax.plot(fs["hours"][keep], s["along"][keep] * 1e7, color=c, ls=ls, lw=1.8, label=name)
    ax.set(xlabel="time [hours]", ylabel="along-track component [1e-7 m/s$^2$]",
           title="Along-track parts over 4 orbits: drag is steady, Moon / Sun / SRP oscillate")
    ax.legend(fontsize=8, loc="upper right", ncol=4)


def _draw_signature(ax, sg):
    for source, color, label in (("j2_drag", NAVY, "J2 + drag"), ("full", RUST, "all forces")):
        d = sg[source]
        ax.plot(d["hours"], d["residual_m"], color=color, lw=1.2,
                label=f"{label}: coast slope {d['slope_m_per_h']:.4f} m/h, wiggle {np.ptp(d['residual_m']):.2f} m")
    ax.set(xlabel="hours into the first coast arc", ylabel="mean SMA minus straight line [m]",
           title="Moon and Sun leave a metre-level periodic signature in the mean SMA")
    ax.legend(fontsize=8)


def _draw_cloud(ax, clouds):
    bins = np.linspace(-17, -11, 160)
    for (label, X), color, ls, lw in zip(clouds.items(), (NAVY, RUST, GREEN), ("-", "-", "--"), (3.5, 1.6, 1.6)):
        ax.hist(X[:, 0], bins=bins, density=True, histtype="step", lw=lw, ls=ls, color=color, label=label)
    ax.axvline(np.log(8.1e-7), color=GREY, ls=":", lw=1, label="drag alone, ln(8.1e-7)")
    ax.set(yscale="log", xlabel="x1 = log |unmodelled acceleration| (clean data)", ylabel="probability density",
           title="Clean coast feature: third bodies spread it; the catalogue model restores the drag spike")
    ax.legend(fontsize=8)


def _draw_case_horizons(ax, cases, model="SINDy split (D)"):
    for case in cases:
        c, m, ls = CASE_STYLE[case["label"]]
        errors = case["forecasts"][model][0]
        ax.plot(HORIZONS_DAYS, [errors[f"{h}d"] for h in HORIZONS_DAYS], color=c, marker=m, ls=ls, label=case["label"])
    ax.set(xscale="log", yscale="log", xticks=HORIZONS_DAYS, xlabel="forecast horizon [days after day 40]",
           ylabel="mean |burn-onset error| [min]", title=f"{model} forecast with and without the third bodies")
    ax.set_xticklabels([str(h) for h in HORIZONS_DAYS])
    ax.minorticks_off()
    ax.legend(fontsize=8)


def _draw_case_models(ax, cases):
    x = np.arange(len(MODELS))
    width = 0.27
    for i, case in enumerate(cases):
        c, _, _ = CASE_STYLE[case["label"]]
        values = [max(case["forecasts"][m][0]["360d"], 0.1) for m in MODELS]
        ax.bar(x + (i - 1) * width, values, width, color=c, label=case["label"])
    ax.set(yscale="log", xticks=x, xlabel="forecaster", ylabel="mean |onset error| at 360 d [min]",
           title="Every forecaster at 360 days, three data / model set-ups")
    ax.set_xticklabels([m.replace(" (", "\n(") for m in MODELS], fontsize=8)
    ax.legend(fontsize=8)


def _draw_budget(ax, rows):
    names = [r["variant"] for r in rows]
    values = [r["errors"]["360d"] for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, values, color=[NAVY] + [GREEN] * (len(rows) - 3) + [SLATE, GREY])
    for yi, v in zip(y, values):
        ax.text(v + 1, yi, f"{v:.1f} min", va="center", fontsize=8)
    ax.set(yticks=y, xlabel="mean |burn-onset error| over 360 days [min]", ylabel="pipeline variant",
           title="Error budget with all forces (analyst models gravity + J2 only)", xlim=(0, max(values) * 1.25))
    ax.set_yticklabels(names, fontsize=8.5)


def run(sim, out_dir):
    folder = step_dir(out_dir, "step22_full_forcing")
    fs = F.force_series()
    sg = F.sma_signature()
    cases = F.cases()
    clouds = {label: F.coast_features(source, catalogue) for label, source, catalogue in F.CASES}
    budget, effective = F.budget_full(False)
    with report_style():
        for name, draw, args in (("step22_force_magnitudes.png", _draw_magnitudes, (fs,)),
                                 ("step22_force_along.png", _draw_along, (fs,)),
                                 ("step22_sma_signature.png", _draw_signature, (sg,)),
                                 ("step22_coast_cloud.png", _draw_cloud, (clouds,)),
                                 ("step22_forecast_horizons.png", _draw_case_horizons, (cases,)),
                                 ("step22_forecast_models.png", _draw_case_models, (cases,)),
                                 ("step22_error_budget.png", _draw_budget, (budget,))):
            fig, ax = plt.subplots(figsize=SIZE)
            draw(ax, *args)
            save(fig, folder, name)
    result = dict(report_step22=dict(
        force_median_ms2={k: float(np.median(s["magnitude"])) for k, s in fs["series"].items()},
        sma_signature={k: dict(wiggle_m=float(np.ptp(v["residual_m"])), slope_m_per_h=v["slope_m_per_h"]) for k, v in sg.items()},
        cases=cases, error_budget=budget, effective_truth=effective))
    # forecast tables at every horizon, for this and every earlier step
    path = Path(out_dir) / "results.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    previous.update(json.loads(json.dumps(PREVIOUS, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))))
    previous.update(json.loads(json.dumps(result, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))))
    write_all(previous, Path(out_dir) / "report" / "tables")
    return result

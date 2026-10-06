"""
Report step 16: calibrated burn-time intervals by split conformal prediction (deadband/forecasting.conformal).

Forecasts of the step 10 space-weather scenario made from origins in days 40-200 are the CALIBRATION set;
for each lead-time bucket the 90% quantile of their absolute onset errors is the interval half-width.
Forecasts from days 200-360 are the TEST set: their coverage shows whether the intervals hold.
Exchangeability is only approximate (the solar-cycle trend makes later errors larger), which the test
coverage reveals.

Figures (outputs/report/step16_conformal/)
    step16_coverage       test coverage per lead bucket (target 90%) for each forecaster
    step16_widths         the interval half-width per lead bucket: the price of honesty
    step16_example        one forecast with its conformal intervals against the true burns
    step16_summary
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband import forecasting as F
from deadband.mean_element import DAY_S
from .common import GREY, RUST
from .report_common import panel_label, report_style, save, step_dir
from .report_step10_space_weather import COLORS, STYLES

ALPHA = 0.1
METHODS = ("periodic baseline", "law, latest coast", "law + space weather", "law + perfect forecast")
EXAMPLE_ORIGIN_DAY = 250


def _label(row):
    return f"{row['lead_lo']}-{row['lead_hi']} d"


def _draw_coverage(ax, tables):
    for name, rows in tables.items():
        ax.plot([_label(r) for r in rows], [r["coverage"] for r in rows], STYLES[name], color=COLORS[name], label=name)
    ax.axhline(1 - ALPHA, color="black", lw=1, label=f"target {100 * (1 - ALPHA):.0f}%")
    ax.set(ylim=(0, 1.05), xlabel="lead-time bucket", ylabel="coverage on TEST origins (days 200-360)",
           title="Do the 90% intervals hold on new data?")
    ax.legend(fontsize=7)


def _draw_widths(ax, tables):
    for name, rows in tables.items():
        ax.semilogy([_label(r) for r in rows], [r["q_min"] / 60 for r in rows], STYLES[name], color=COLORS[name],
                    label=name)
    ax.set(xlabel="lead-time bucket", ylabel="90% interval half-width [hours]", title="Interval width: what each forecaster must admit")
    ax.legend(fontsize=7)


def _draw_example(ax, sc, tables):
    t0 = EXAMPLE_ORIGIN_DAY * DAY_S
    true = sc.run.onsets[(sc.run.onsets > t0) & (sc.run.onsets < t0 + 60 * DAY_S)] / DAY_S
    for y, name in enumerate(("periodic baseline", "law + space weather")):
        p = F.forecast(name, sc.learned, sc.weather, t0)
        p = p[p < t0 + 60 * DAY_S]
        for onset in p:
            lead = (onset - t0) / DAY_S
            q = next((r["q_min"] for r in tables[name] if r["lead_lo"] < lead <= r["lead_hi"]), np.nan) / 1440
            ax.plot([onset / DAY_S - q, onset / DAY_S + q], [y, y], lw=6, color=COLORS[name], alpha=0.4, solid_capstyle="butt")
            ax.plot(onset / DAY_S, y, "|", ms=14, mew=2, color=COLORS[name])
    for x in true:
        ax.axvline(x, color=RUST, lw=0.8)
    ax.set(yticks=[0, 1], yticklabels=["periodic baseline", "law + space weather"], ylim=(-0.6, 1.6),
           xlabel="time [days]", ylabel="forecaster",
           title=f"Forecast from day {EXAMPLE_ORIGIN_DAY} with 90% conformal intervals (red = true burns)")


def run(sim, out_dir):
    folder = step_dir(out_dir, "step16_conformal")
    sc = F.scenario()
    tables = {name: F.conformal(sc.rows, name, ALPHA) for name in METHODS}
    with report_style():
        fig, ax = plt.subplots(figsize=(9, 4.8)); _draw_coverage(ax, tables); save(fig, folder, "step16_coverage.png")
        fig, ax = plt.subplots(figsize=(9, 4.8)); _draw_widths(ax, tables); save(fig, folder, "step16_widths.png")
        fig, ax = plt.subplots(figsize=(12, 3.6)); _draw_example(ax, sc, tables); save(fig, folder, "step16_example.png")
        fig, axes = plt.subplots(1, 3, figsize=(21, 5))
        _draw_coverage(axes[0], tables); _draw_widths(axes[1], tables); _draw_example(axes[2], sc, tables)
        for ax, letter in zip(axes, "abc"):
            panel_label(ax, letter)
        save(fig, folder, "step16_summary.png", suptitle="Step 16: calibrated (conformal) burn-time intervals")
    return dict(report_step16=dict(alpha=ALPHA, tables=tables))

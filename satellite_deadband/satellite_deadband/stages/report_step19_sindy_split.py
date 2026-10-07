"""
Report step 19: learning the thrust control law and the natural (coasting) dynamics separately or together,
compared with the true model, and learning the natural law when no deadband is applied.
Library: deadband/sindy_split.py.

Figures (outputs/report/step19_sindy_split/)
    step19_parameters       c_d / true and T / true for the four scenarios, with BINDy 2-sigma intervals
    step19_rate_errors      RMS error of the predicted da/dt against the true model (all TEST samples and coast only)
    step19_forecast         burn-onset error of the free-running forecast vs horizon, per scenario
    step19_burn_zoom        one TEST burn: the true da/dt and what each scenario predicts
    step19_free_decay       no deadband: the free decay, the learned rate laws and their 360-day forecasts
    step19_summary
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband import sindy_split as S
from deadband.constants import DU, SMOOTH_CD, TU
from deadband.long_run import HORIZONS_DAYS, TRAIN_DAYS
from .common import AMBER, GREEN, GREY, NAVY, PURPLE, RUST, SLATE
from .report_common import panel_label, report_style, save, step_dir

COLORS = dict(zip(S.SCENARIOS, (NAVY, AMBER, GREEN, PURPLE)))
MARKERS = dict(zip(S.SCENARIOS, "osD^"))
SHORT = dict(zip(S.SCENARIOS, ("A whole system", "B thrust only\n(natural known)", "C natural only\n(thrust known)",
                               "D both, separately")))
LIB_COLORS = (GREY, SLATE, "#999999", GREEN, NAVY, RUST, AMBER, PURPLE)


def _draw_parameters(ax, rows):
    x = np.arange(len(rows))
    for offset, key, std, color, label in ((-0.18, "cd_ratio", "cd_std_pct", NAVY, "drag c_d"),
                                           (0.18, "T_ratio", "T_std_pct", RUST, "thrust T")):
        values = [r[key] for r in rows]
        errors = [2 * r[std] / 100 for r in rows]
        ax.bar(x + offset, values, 0.34, yerr=errors, capsize=4, color=color, label=f"{label} (BINDy +-2 sigma)")
        for xi, v, r in zip(x + offset, values, rows):
            given = (key == "cd_ratio" and r["scenario"].startswith("B")) or (key == "T_ratio" and r["scenario"].startswith("C"))
            ax.text(xi, 0.905, "given" if given else f"{v:.4f}", ha="center", va="bottom", fontsize=7, rotation=90,
                    color="white")
    ax.axhline(1, color="black", lw=1)
    ax.set(xticks=x, ylim=(0.9, 1.05), xlabel="scenario", ylabel="learned / true",
           title="Physics recovered by each split of the learning problem")
    ax.set_xticklabels([SHORT[r["scenario"]] for r in rows], fontsize=8)
    ax.legend(fontsize=7, loc="lower right")


def _draw_rate_errors(ax, rows):
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r["rate_rms_m_per_hour"] for r in rows], 0.34, color=GREY, label="all TEST samples")
    ax.bar(x + 0.18, [r["coast_rate_rms_m_per_hour"] for r in rows], 0.34, color=GREEN, label="coasting samples only")
    ax.set(yscale="log", xticks=x, xlabel="scenario", ylabel="RMS error of predicted da/dt [m/hour]",
           title="Predicted rate vs the true model (TEST, days 40-400)")
    ax.set_xticklabels([SHORT[r["scenario"]] for r in rows], fontsize=8)
    ax.legend(fontsize=7)


def _draw_forecast(ax, rows):
    for r in rows:
        values = [r["forecast_error_min"][f"{h}d"] for h in HORIZONS_DAYS]
        ax.loglog(HORIZONS_DAYS, values, "-", marker=MARKERS[r["scenario"]], color=COLORS[r["scenario"]],
                  label=f"{r['scenario']} (period {r['period_days']:.4f} d)")
    ax.set(xticks=HORIZONS_DAYS, xlabel="forecast horizon [days after day 40]", ylabel="mean |burn-onset error| [min]",
           title="Free-running burn forecast (true period 3.9051 d)")
    ax.set_xticklabels([str(h) for h in HORIZONS_DAYS])
    ax.minorticks_off()
    ax.legend(fontsize=7)


def _draw_burn_zoom(ax, p, learned):
    from deadband.derivative_detection import runs_of_ones
    d = p["dataset"]
    start, end = next((s, e) for s, e in runs_of_ones(d.t, d.true_on) if s * TU / 86400 > TRAIN_DAYS)
    days = p["days"]
    window = (days > start * TU / 86400 - 0.05) & (days < end * TU / 86400 + 0.05)
    minutes = (days[window] - start * TU / 86400) * 1440
    ax.plot(minutes, p["rate_true"][window] * S.TO_M_PER_HOUR, color="black", lw=2.5, label="true model")
    ax.plot(minutes, p["y"][window] * S.TO_M_PER_HOUR, ".", ms=3, color=GREY, label="measured (smoothed derivative)")
    for m in learned:
        thrust = p["theta"] if m.name.startswith("C") else p["s"]
        pred = (-m.cd * p["D"] + m.T * thrust * p["B"])[window] * S.TO_M_PER_HOUR
        ax.plot(minutes, pred, "--", color=COLORS[m.name], lw=1.3, label=m.name)
    ax.set(xlabel="minutes from the true burn start", ylabel="da/dt [m/hour]",
           title="One burn: the detected burn flags are wider than the thrust ramps")
    ax.legend(fontsize=6.5, loc="center right")


LIB_STYLES = (("-", "o"), ("--", "s"), (":", "^"), ("-", "D"), ("--", "v"), ("-.", "P"), (":", "X"), ("-.", "*"))


def _is_exponential(name):
    return name.lower().startswith("6") or "exp" in name.lower()


def _draw_free_decay(axes, studies):
    """axes: [decay, learned laws (exponential library excluded), exponential library alone, forecast errors]."""
    d, rows, true_rate = studies["drag only"]
    axes[0].plot(d["days"][::60], d["a_true"][::60] * DU, color=NAVY, label="drag only (J2 + drag)")
    dn = studies["natural"][0]
    axes[0].plot(dn["days"][::60], dn["a_true"][::60] * DU, "--", color=RUST, label="natural (+ Moon, Sun, SRP)")
    axes[0].axvspan(0, TRAIN_DAYS, color=GREY, alpha=0.15, label="TRAIN (days 0-40)")
    axes[0].ticklabel_format(axis="y", useOffset=False)
    axes[0].set(xlabel="time [days]", ylabel="mean SMA [km]", title="No deadband: 51 km of free decay in 400 days")
    axes[0].legend(fontsize=8)
    a_grid = np.linspace(d["a"].min(), d["a"].max(), 100)
    a0 = float(np.median(d["a"][d["days"] < TRAIN_DAYS]))
    truth = -SMOOTH_CD * 2 * np.sqrt(S.MU * a_grid) * S.TO_M_PER_HOUR
    for (name, library), row, color, (ls, marker) in zip(S.free_libraries(a0).items(), rows, LIB_COLORS, LIB_STYLES):
        theta, _ = library(a_grid)
        rate = theta @ np.array(row["coefficients"]) * S.TO_M_PER_HOUR
        ax = axes[2] if _is_exponential(name) else axes[1]
        ax.plot(a_grid * DU, rate, color=color, ls=ls, marker=marker, markevery=(LIB_STYLES.index((ls, marker)) * 3, 20),
                ms=5, lw=1.4, label=name)
    for ax in axes[1:3]:
        ax.plot(a_grid * DU, truth, "k:", lw=2.5, label=r"true law $-2c_d\sqrt{\mu a}$")
        ax.set(xlabel="mean SMA [km]", ylabel="da/dt [m/hour]")
        ax.legend(fontsize=7.5)
    axes[1].set_title("Learned laws over the whole decay (fitted on 40 days)")
    axes[2].set_title("Exponential-density library on its own axis: wrong physics here")
    for i, (row, color, (ls, marker)) in enumerate(zip(rows, LIB_COLORS, LIB_STYLES)):
        days = np.asarray(row["forecast_days"])
        err = np.abs(np.asarray(row["forecast_err_km"])) * 1000 + 1e-3
        axes[3].plot(days, err, color=color, ls=ls, marker=marker, markevery=(i * 7, 60), ms=6, lw=1.4,
                     label=row["library"])
    axes[3].set(yscale="log", xlabel="time [days]", ylabel="|SMA forecast error| [m]",
                title="Free-running SMA forecast from day 40 (identical laws overlap: see markers)")
    axes[3].legend(fontsize=7.5, ncol=2, loc="lower right")


def run(sim, out_dir):
    folder = step_dir(out_dir, "step19_sindy_split")
    p, learned, rows, true_period = S.run_scenarios()
    studies = {source: S.free_decay_study(source) for source in ("drag only", "natural")}
    with report_style():
        fig, ax = plt.subplots(figsize=(10, 5)); _draw_parameters(ax, rows); save(fig, folder, "step19_parameters.png")
        fig, ax = plt.subplots(figsize=(10, 5)); _draw_rate_errors(ax, rows); save(fig, folder, "step19_rate_errors.png")
        fig, ax = plt.subplots(figsize=(9, 5)); _draw_forecast(ax, rows); save(fig, folder, "step19_forecast.png")
        fig, ax = plt.subplots(figsize=(10, 5)); _draw_burn_zoom(ax, p, learned); save(fig, folder, "step19_burn_zoom.png")
        fig, axes = plt.subplots(2, 2, figsize=(16, 10.5)); _draw_free_decay(axes.ravel(), studies)
        save(fig, folder, "step19_free_decay.png")
        fig, axes = plt.subplots(2, 3, figsize=(20, 10.5))
        _draw_parameters(axes[0, 0], rows); _draw_forecast(axes[0, 1], rows); _draw_burn_zoom(axes[0, 2], p, learned)
        _draw_free_decay(list(axes[1]) + [axes[1, 2].inset_axes([0.6, 0.6, 0.38, 0.38]), axes[1, 2]], studies)
        for ax, letter in zip(axes.ravel(), "abcdef"):
            panel_label(ax, letter)
        save(fig, folder, "step19_summary.png",
             suptitle="Step 19: thrust law vs natural dynamics, learned together or apart, and with no deadband")
    free = {source: [{k: v for k, v in r.items() if k not in ("forecast_days", "forecast_err_km")} for r in st[1]]
            for source, st in studies.items()}
    return dict(report_step19=dict(true_period_days=true_period, scenarios=rows, free_decay=free))

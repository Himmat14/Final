"""
Report step 10: burn forecasting when drag VARIES (space weather), with the learned control law vs a
periodic baseline. Library: deadband/mean_element.py (model calibrated on the full 400-day run) and
deadband/forecasting.py (forecasters, rolling-origin evaluation).

Data: two 400-day mean-element runs, constant drag and space-weather drag (27-day solar rotation, a slow
solar-cycle trend and 8 geomagnetic storms), mean SMA measured every 10 min with 20 m noise. Laws are
learned on days 0-40; forecasts are made from an origin every 5 days and scored on the true burn onsets
up to 360 days ahead.

Figures (outputs/report/step10_space_weather/)
    step10_space_weather        density factor and the observable proxy; burn intervals, constant vs variable drag
    step10_model_check          the mean-element model against the full 6-s controlled run (same sawtooth)
    step10_drag_learning        coast rate of every measured coast arc vs the proxy: the learned drag law
    step10_forecast_horizons    mean |onset error| vs horizon for every forecaster, constant and variable drag
    step10_forecast_example     one forecast (from day 100): predicted vs true burn onsets for 60 days
    step10_summary
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband import forecasting as F
from deadband.long_run import HORIZONS_DAYS, controlled_run
from deadband.mean_element import DAY_S, calibrate
from .common import AMBER, GREEN, GREY, NAVY, PURPLE, RUST, SLATE
from .report_common import panel_label, report_style, save, step_dir

COLORS = dict(zip(F.FORECASTERS, (GREY, NAVY, AMBER, GREEN, PURPLE)))
STYLES = dict(zip(F.FORECASTERS, ("o:", "s-", "^--", "D-", "v-.")))
EXAMPLE_ORIGIN_DAY = 100


def _draw_weather(ax, sc):
    w = sc.weather
    d = w.t / DAY_S
    ax.plot(d[::60], w.w[::60], color=RUST, lw=0.8, label="true density factor w(t)")
    ax.plot(d[::1440], w.proxy[::1440], ".", ms=3, color=NAVY, label="daily proxy (observed, 5% noise)")
    for onset in w.storms:
        ax.axvline(onset, color=GREY, lw=0.5, ls=":")
    ax.set(xlabel="time [days]", ylabel="density / average", title="Space weather: solar rotation, trend and storms (dotted)")
    ax.legend()


def _draw_intervals(ax, constant, variable):
    for sc, color, label in ((constant, NAVY, "constant drag"), (variable, RUST, "space-weather drag")):
        on = sc.run.onsets / DAY_S
        ax.plot(on[1:], np.diff(on), "o-", ms=3, color=color, label=label)
    ax.set(xlabel="burn time [days]", ylabel="time since the previous burn [days]",
           title="Variable drag changes the burn INTERVAL, not the band edges")
    ax.legend()


def _draw_model_check(ax):
    run = controlled_run()
    c = calibrate()
    from deadband.mean_element import simulate, space_weather
    mean_run = simulate(space_weather(20, constant=True), a0=float(run.mean_sma_km[0]))   # same starting SMA
    days_full = np.arange(int(20 * 14400)) * 6.0 / DAY_S                                   # 6-s samples
    ax.plot(days_full[::50], run.mean_sma_km[: int(20 * 14400):50], color=GREY, lw=2.5, alpha=0.6,
            label="full 6-s orbit integration (step 5)")
    ax.plot(mean_run.days, mean_run.a, color=NAVY, lw=0.9, ls="--",
            label=f"mean-element model (period {np.mean(np.diff(mean_run.onsets)) / DAY_S:.3f} d vs {c['period_days']:.3f} d)")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set(xlabel="time [days]", ylabel="mean SMA [km]", title="The fast model reproduces the full simulation")
    ax.legend(loc="lower left")


def _draw_drag(ax, sc):
    L, w = sc.learned, sc.weather
    arcs = F._coast_arcs(L.t, L.a, L.onsets, L.ends)
    proxy = [np.mean(w.proxy[(w.t > s) & (w.t < e)]) for s, e, _ in arcs]
    rate = [-slope * 3.6e6 for _, _, slope in arcs]
    train = [s < 40 * DAY_S for s, _, _ in arcs]
    ax.scatter(np.array(proxy)[~np.array(train)], np.array(rate)[~np.array(train)], s=12, color=GREY, label="TEST arcs")
    ax.scatter(np.array(proxy)[train], np.array(rate)[train], s=20, color=RUST, label="TRAIN arcs (days 0-40)")
    x = np.linspace(0, max(proxy) * 1.05, 50)
    ax.plot(x, L.k_proxy * 3.6e6 * x, color=NAVY, label=f"learned: {L.k_proxy * 3.6e6:.2f} m/h per unit proxy "
                                                       f"(truth {calibrate()['k'] * 3.6e6:.2f})")
    ax.set(xlabel="mean proxy over the coast arc", ylabel="coast decay rate [m/hour]",
           title="Drag law learned from 40 days of noisy SMA")
    ax.legend()


def _draw_horizons(ax, sc, title):
    table = F.horizon_table(sc.rows)
    for name, values in table.items():
        ax.semilogy(HORIZONS_DAYS, np.maximum(values, 0.1), STYLES[name], color=COLORS[name], label=name)
    ax.set(xscale="log", xticks=HORIZONS_DAYS, xlabel="forecast horizon [days]", ylabel="mean |onset error| [min]",
           title=title)
    ax.set_xticklabels([str(h) for h in HORIZONS_DAYS])
    ax.minorticks_off()
    ax.legend(fontsize=7)
    return table


def _draw_example(ax, sc):
    t0 = EXAMPLE_ORIGIN_DAY * DAY_S
    true = sc.run.onsets[(sc.run.onsets > t0) & (sc.run.onsets < t0 + 60 * DAY_S)] / DAY_S
    for y, name in enumerate(F.FORECASTERS):
        p = F.forecast(name, sc.learned, sc.weather, t0)
        if p is None:
            continue
        p = p[p < t0 + 60 * DAY_S] / DAY_S
        ax.plot(p, np.full(p.size, y), "|", ms=14, mew=2, color=COLORS[name])
    for x in true:
        ax.axvline(x, color=RUST, lw=0.8, alpha=0.6)
    ax.set(yticks=range(len(F.FORECASTERS)), yticklabels=F.FORECASTERS, xlabel="time [days]", ylabel="forecaster",
           title=f"Forecast from day {EXAMPLE_ORIGIN_DAY}: predicted (ticks) vs true burns (lines)")


def run(sim, out_dir):
    folder = step_dir(out_dir, "step10_space_weather")
    constant, variable = F.scenario(constant_drag=True), F.scenario()
    with report_style():
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
        _draw_weather(axes[0], variable); _draw_intervals(axes[1], constant, variable)
        save(fig, folder, "step10_space_weather.png")
        fig, ax = plt.subplots(figsize=(10, 4.4)); _draw_model_check(ax); save(fig, folder, "step10_model_check.png")
        fig, ax = plt.subplots(figsize=(8, 5)); _draw_drag(ax, variable); save(fig, folder, "step10_drag_learning.png")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        t_const = _draw_horizons(axes[0], constant, "Constant drag: the periodic baseline wins")
        t_var = _draw_horizons(axes[1], variable, "Space-weather drag: the learned law wins")
        save(fig, folder, "step10_forecast_horizons.png")
        fig, ax = plt.subplots(figsize=(12, 4)); _draw_example(ax, variable); save(fig, folder, "step10_forecast_example.png")

        fig, axes = plt.subplots(2, 3, figsize=(19, 10))
        _draw_weather(axes[0, 0], variable); _draw_intervals(axes[0, 1], constant, variable); _draw_drag(axes[0, 2], variable)
        _draw_horizons(axes[1, 0], constant, "Constant drag")
        _draw_horizons(axes[1, 1], variable, "Space-weather drag")
        _draw_example(axes[1, 2], variable)
        for ax, letter in zip(axes.ravel(), "abcdef"):
            panel_label(ax, letter)
        save(fig, folder, "step10_summary.png", suptitle="Step 10: burn forecasting under space-weather drag")

    L = variable.learned
    return dict(report_step10=dict(
        calibration=calibrate(), learned_edges_km=[L.lower, L.upper], k_proxy_over_true=L.k_proxy / calibrate()["k"],
        mean_abs_error_min_constant={n: dict(zip([f"{h}d" for h in HORIZONS_DAYS], v)) for n, v in t_const.items()},
        mean_abs_error_min_variable={n: dict(zip([f"{h}d" for h in HORIZONS_DAYS], v)) for n, v in t_var.items()}))

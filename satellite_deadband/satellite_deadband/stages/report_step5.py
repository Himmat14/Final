"""
Report step 5: the deadband controller as a dynamic model, and the sigmoid (tanh) thruster.

The smooth-deadband stage (main.py stage "smooth_deadband", run right after this one) adds Figure 1
of the proposal over days 0-10, the tanh switches, one burn up close, the slope sweep, J2 mean vs
osculating SMA and the integrator step sizes to the same folder. This step adds the 400-day picture
(long_run.controlled_run and its drag-only twin):

Figures (outputs/report/step5_deadband/)
    step5_long_term             400 days: mean SMA (controlled vs drag-only), miss distance, and
                                burns / delta-v / burn interval at 15, 30, 60, 120, 240, 360 days
    step5_impulsive_vs_smooth   delta-v delivery: Week 4 "if statement" impulses vs the Week 5 tanh thruster
    step5_thrust_profiles       how the switch time constant shapes (attenuates) the thrust ramp
    step5_hysteresis_loop       thruster state vs mean SMA: the loop IS the controller's memory
    step5_fuel_vs_band          total delta-v and number of burns vs deadband width (60-day runs)
    step5_summary               six-panel summary
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband.long_run import HORIZONS_DAYS, controlled_run, drag_only_run
from deadband.constants import (DU, METRES, PERIOD, SAMPLE_STEP_S, SECONDS, SMOOTH_A_LOWER, SMOOTH_A_LOWER_KM,
                                SMOOTH_A_UPPER_KM, SMOOTH_DAYS, SWITCH_SMA_WIDTH_M, THRUST_ACCEL_MS2, VU,
                                from_days, from_hours, to_days, to_minutes)
from deadband.controller import simulate_deadband
from deadband.smooth_controller import (default_runs, mean_sma_km_series, rising_switch, rising_switch_slope,
                                        simulate_smooth_deadband, smooth_initial_state, throttle)
from .common import AMBER, GREEN, NAVY, PURPLE, RUST, SLATE
from .report_common import panel_label, report_style, save, step_dir

SWITCH_TIMES_S = (5, 15, 60, 180, 600)
BAND_WIDTHS_KM = (0.25, 0.5, 1.0)
FUEL_DAYS = 60.0
STEP = SAMPLE_STEP_S * SECONDS


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def thrust_profiles():
    """Thrust [m/s^2] around one burn for several switch time constants, aligned on the throttle crossing 1/2."""
    profiles = {}
    for switch_time_s in SWITCH_TIMES_S:
        sim = simulate_smooth_deadband(from_hours(1.2), switch_time=switch_time_s * SECONDS,
                                       state0=smooth_initial_state(SMOOTH_A_LOWER + 3 * METRES))
        t = sim.sample_times(STEP)
        thrust = throttle(sim.sample(t)[6]) * THRUST_ACCEL_MS2
        start = t[np.argmax(thrust > 0.5 * THRUST_ACCEL_MS2)]
        profiles[switch_time_s] = (to_minutes(t - start), thrust)
    return profiles


def fuel_vs_band():
    """Total delta-v [m/s] and number of burns over FUEL_DAYS for several band widths (same lower edge)."""
    rows = []
    for width_km in BAND_WIDTHS_KM:
        if np.isclose(width_km, SMOOTH_A_UPPER_KM - SMOOTH_A_LOWER_KM):   # the 400-day run already has this band
            run = controlled_run()
            keep = run.t < from_days(FUEL_DAYS)
            thrust, on = run.thrust_ms2[keep], run.true_on[keep]
        else:
            lower, upper = SMOOTH_A_LOWER_KM, SMOOTH_A_LOWER_KM + width_km
            sim = simulate_smooth_deadband(from_days(FUEL_DAYS), a_lower=lower / DU, a_upper=upper / DU,
                                           state0=smooth_initial_state((upper - 0.3 * width_km) / DU))
            states = sim.sample(sim.sample_times(STEP))
            thrust = throttle(states[6]) * THRUST_ACCEL_MS2
            on = (thrust > 0.5 * THRUST_ACCEL_MS2).astype(int)
        rows.append(dict(width_km=width_km, delta_v_ms=float(np.sum(thrust) * SAMPLE_STEP_S),
                         n_burns=int(np.sum(np.diff(on) == 1))))
    return rows


def long_term_metrics():
    """Per horizon: burns, delta-v, burn interval and duration, SMA range, miss distance vs the drag-only twin."""
    run = controlled_run()
    days = to_days(run.t)
    starts = days[1:][np.diff(run.true_on) == 1]
    ends = days[1:][np.diff(run.true_on) == -1]
    drag_t, drag_states = drag_only_run()
    minute = slice(None, None, 10)                                    # controlled run every 6 s -> every 1 min
    miss_km = np.linalg.norm(run.r[:, minute][:, :drag_states.shape[1]] - drag_states[0:3, :run.r[:, minute].shape[1]],
                             axis=0) * DU
    miss_days = to_days(drag_t[:len(miss_km)])
    rows = []
    for horizon in HORIZONS_DAYS:
        keep = days < horizon
        in_horizon = starts < horizon
        durations = [(e - s) * 1440 for s, e in zip(starts, ends) if s < horizon and e > s]
        rows.append(dict(days=horizon, n_burns=int(in_horizon.sum()),
                         delta_v_ms=float(np.sum(run.thrust_ms2[keep]) * SAMPLE_STEP_S),
                         mean_interval_days=float(np.mean(np.diff(starts[in_horizon]))) if in_horizon.sum() > 1 else float("nan"),
                         mean_burn_minutes=float(np.mean(durations)) if durations else float("nan"),
                         sma_min_km=float(run.mean_sma_km[keep].min()), sma_max_km=float(run.mean_sma_km[keep].max()),
                         miss_distance_km=float(miss_km[min(np.searchsorted(miss_days, horizon), len(miss_km) - 1)])))
    drag_sma = mean_sma_km_series(drag_states)
    return rows, dict(days=days[::600], sma=run.mean_sma_km[::600], drag_days=to_days(drag_t)[::60],
                      drag_sma=drag_sma[::60], miss_days=miss_days[::60], miss_km=miss_km[::60])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_figure_one(ax, controlled, drag_only):
    times = controlled.sample_times(STEP)
    days = SMOOTH_DAYS - to_days(times)
    ax.plot(days, mean_sma_km_series(drag_only.sample(times)), color="#6A6AE0", label="SMA (drag only)")
    ax.plot(days, mean_sma_km_series(controlled.sample(times)), color="#F07070", label="SMA (control)")
    for edge in (SMOOTH_A_LOWER_KM, SMOOTH_A_UPPER_KM):
        ax.axhline(edge, color="gray", ls="--", lw=1)
    miss = np.linalg.norm(controlled.sample(times)[0:3] - drag_only.sample(times)[0:3], axis=0) * DU
    twin = ax.twinx()
    twin.plot(days, miss, color="black", lw=1.8, label="miss distance")
    twin.set(ylabel="miss distance [km]", ylim=(0, 600))
    twin.grid(False)
    ax.set(xlim=(SMOOTH_DAYS, 0), ylim=(SMOOTH_A_LOWER_KM - 0.3, SMOOTH_A_UPPER_KM + 0.2), xlabel="TCA [days]",
           ylabel="mean SMA [km]", title="500 m deadband with J2 + drag")
    ax.legend(loc="lower left")


def _draw_switch(ax):
    offset = np.linspace(-40, 40, 400)
    ax.plot(offset, rising_switch(-offset, SWITCH_SMA_WIDTH_M), color=NAVY, lw=2, label="tanh switch")
    ax.plot(offset, (offset < 0).astype(float), color=SLATE, ls=":", label="if statement")
    twin = ax.twinx()
    twin.plot(offset, rising_switch_slope(-offset, SWITCH_SMA_WIDTH_M), color=RUST, lw=1.5, label="slope sech$^2$/2w")
    twin.set_ylabel("slope [1/m]")
    twin.grid(False)
    ax.set(xlabel="mean SMA - lower edge [m]", ylabel="turn-on switch", title=f"tanh switch (w = {SWITCH_SMA_WIDTH_M:g} m)")
    ax.legend(loc="center left")


def _draw_hysteresis(ax, controlled):
    times = controlled.sample_times(STEP)
    states = controlled.sample(times)
    sma = mean_sma_km_series(states)
    sc = ax.scatter(sma, states[6], c=to_days(times), s=3, cmap="viridis")
    for edge in (SMOOTH_A_LOWER_KM, SMOOTH_A_UPPER_KM):
        ax.axvline(edge, color="gray", ls="--", lw=1)
    ax.annotate("drag lowers the orbit (s = 0)", xy=(6922.2, 0.02), xytext=(6922.05, 0.25), fontsize=8,
                arrowprops=dict(arrowstyle="<-", color=SLATE))
    ax.annotate("burn raises it (s = 1)", xy=(6922.15, 0.98), xytext=(6922.0, 0.7), fontsize=8,
                arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.set(xlabel="mean SMA [km]", ylabel="thruster state s", title="Hysteresis loop = the controller's memory")
    plt.colorbar(sc, ax=ax, label="time [days]", fraction=0.046)


def _draw_profiles(ax, profiles):
    colors = (RUST, AMBER, NAVY, GREEN, PURPLE)
    for (switch_time_s, (minutes, thrust)), color in zip(profiles.items(), colors):
        ax.plot(minutes, thrust * 1e4, color=color, lw=1.5, label=f"switch time {switch_time_s} s")
    ax.set(xlim=(-3, 30), xlabel="minutes from throttle = 1/2", ylabel="thrust [1e-4 m/s$^2$]",
           title="Thruster attenuation over short time scales")
    ax.legend()


def _draw_impulsive_vs_smooth(ax, controlled):
    """Delta-v delivered during one burn: Week 4 impulses (instant) vs the Week 5 tanh thruster (spread out)."""
    week4 = simulate_deadband(PERIOD * 2)
    raise_ms, trim_ms = week4.raise_dv[0] * VU * 1000, week4.trim_dv[0] * VU * 1000
    # the impulses happen at one instant; draw them a hair apart so the overshoot is visible
    ax.step([-5, -0.01, 0, 0.3, 30], [0, 0, raise_ms, raise_ms + trim_ms, raise_ms + trim_ms], where="post",
            color=SLATE, lw=2, label=f"Week 4 if statement: raise {raise_ms:.2f} then trim {trim_ms:.2f} m/s, instantly")
    start, end = controlled.burn_intervals(STEP)[0]
    times = np.arange(start - from_hours(0.1), end + from_hours(0.2), STEP)
    thrust = throttle(controlled.sample(times)[6]) * THRUST_ACCEL_MS2
    ax.plot(to_minutes(times - start), np.cumsum(thrust) * SAMPLE_STEP_S, color=NAVY, lw=2,
            label=f"Week 5 tanh thruster: {np.sum(thrust) * SAMPLE_STEP_S:.2f} m/s over {to_minutes(end - start):.0f} min")
    ax.set(xlim=(-5, 30), xlabel="minutes from burn start", ylabel="delta-v delivered [m/s]",
           title="Jump vs smooth: how the delta-v arrives")
    ax.legend(loc="center right")


def _draw_fuel(ax, rows):
    widths = [row["width_km"] * 1000 for row in rows]
    ax.plot(widths, [row["delta_v_ms"] for row in rows], "o-", color=NAVY, label="total delta-v")
    ax.set(xlabel="deadband width [m]", ylabel=f"delta-v over {FUEL_DAYS:g} days [m/s]", title="Fuel vs control accuracy",
           ylim=(0, None))
    twin = ax.twinx()
    twin.plot(widths, [row["n_burns"] for row in rows], "s--", color=RUST, label="number of burns")
    twin.set_ylabel("number of burns", color=RUST)
    twin.grid(False)
    ax.legend(loc="upper center")


def _draw_long_sma(ax, trace):
    ax.plot(trace["drag_days"], trace["drag_sma"], color="#6A6AE0", lw=1.2, label="drag only")
    ax.plot(trace["days"], trace["sma"], color="#F07070", lw=0.6, label="controlled (500 m deadband)")
    for edge in (SMOOTH_A_LOWER_KM, SMOOTH_A_UPPER_KM):
        ax.axhline(edge, color="gray", ls="--", lw=0.8)
    for horizon in HORIZONS_DAYS:
        ax.axvline(horizon, color="gray", ls=":", lw=0.6)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set(xlabel="time [days]", ylabel="mean SMA [km]", title="400 days of station-keeping")
    ax.legend(loc="lower left")


def _draw_miss(ax, trace):
    ax.plot(trace["miss_days"], trace["miss_km"], color="black", lw=1)
    for horizon in HORIZONS_DAYS:
        ax.axvline(horizon, color="gray", ls=":", lw=0.6)
    ax.set(xlabel="time [days]", ylabel="miss distance [km]",
           title="Controlled vs drag-only position (saturates near the orbit diameter)")


def _draw_horizon_metrics(ax, rows):
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [row["n_burns"] for row in rows], 0.4, color=NAVY, label="burns")
    ax.set(xticks=x, xticklabels=[f"{row['days']} d" for row in rows], ylabel="number of burns",
           title="Burns and fuel by horizon")
    twin = ax.twinx()
    twin.bar(x + 0.2, [row["delta_v_ms"] for row in rows], 0.4, color=RUST, label="delta-v")
    twin.set_ylabel("total delta-v [m/s]", color=RUST)
    twin.grid(False)
    for i, row in enumerate(rows):
        ax.text(i, row["n_burns"] + 1, f"every {row['mean_interval_days']:.2f} d", ha="center", fontsize=7, rotation=90)
    ax.legend(loc="upper left")


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    folder = step_dir(out_dir, "step5_deadband")
    controlled, drag_only = default_runs()
    profiles = thrust_profiles()
    fuel = fuel_vs_band()
    horizon_rows, trace = long_term_metrics()

    with report_style():
        fig, axes = plt.subplots(1, 3, figsize=(19, 4.6))
        _draw_long_sma(axes[0], trace); _draw_miss(axes[1], trace); _draw_horizon_metrics(axes[2], horizon_rows)
        save(fig, folder, "step5_long_term.png")
        fig, ax = plt.subplots(figsize=(8, 4.5)); _draw_impulsive_vs_smooth(ax, controlled); save(fig, folder, "step5_impulsive_vs_smooth.png")
        fig, ax = plt.subplots(figsize=(8, 4.5)); _draw_profiles(ax, profiles); save(fig, folder, "step5_thrust_profiles.png")
        fig, ax = plt.subplots(figsize=(8, 4.8)); _draw_hysteresis(ax, controlled); save(fig, folder, "step5_hysteresis_loop.png")
        fig, ax = plt.subplots(figsize=(7.5, 4.5)); _draw_fuel(ax, fuel); save(fig, folder, "step5_fuel_vs_band.png")

        fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
        _draw_figure_one(axes[0, 0], controlled, drag_only)
        _draw_long_sma(axes[0, 1], trace)
        _draw_horizon_metrics(axes[0, 2], horizon_rows)
        _draw_hysteresis(axes[1, 0], controlled)
        _draw_profiles(axes[1, 1], profiles)
        _draw_fuel(axes[1, 2], fuel)
        for ax, letter in zip(axes.ravel(), "abcdef"):
            panel_label(ax, letter)
        save(fig, folder, "step5_summary.png", suptitle="Step 5: deadband control with a smooth (tanh) thruster, 8 and 400 days")

    return dict(report_step5=dict(fuel_vs_band=fuel, long_term_by_horizon=horizon_rows))

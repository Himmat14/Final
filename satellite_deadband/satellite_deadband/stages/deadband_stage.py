"""
Workstream B: the deadband controller, on the SAME data as report step 5 (the 400-day smooth-tanh
controlled run, long_run.controlled_run, 500 m band, J2 + realistic drag, sampled every 6 s).

The Week 4 version used its own 8-orbit run with impulsive raise + trim burns and the demo drag
(CD_TRUE, ~45 km/day). That controller and drag are no longer used anywhere in the report, so the
figures below are rebuilt from the report's run.

Figures (outputs/report/step5_deadband/)
    deadband_state_dependent      the sawtooth: mean SMA between the band edges, 400 days and a 30-day zoom
    deadband_raise_trim_dv        every burn's delta-v and duration over 400 days (one smooth burn per
                                  cycle replaces the old raise + trim pair)
    deadband_hysteresis_3d        3D phase portrait: mean SMA, its rate of change and the thruster state.
                                  Coast and burn trace one closed loop: the controller's memory
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband.constants import SAMPLE_STEP_S, SMOOTH_A_LOWER_KM, SMOOTH_A_UPPER_KM, to_days
from deadband.derivative_detection import runs_of_ones
from deadband.long_run import HORIZONS_DAYS, controlled_run
from deadband.smooth_controller import throttle
from .common import AMBER, NAVY, RUST, SLATE
from .report_common import report_style, save, step_dir

ZOOM_DAYS = 30
LOOP_DAYS = (40, 48)          # two full cycles for the 3D phase portrait (inside the TEST period)


def burn_table(run):
    """Start day, duration [min] and delta-v [m/s] of every burn in the run."""
    days = to_days(run.t)
    rows = []
    for start, end in runs_of_ones(run.t, run.true_on):
        inside = (run.t >= start) & (run.t <= end)
        rows.append(dict(start_day=float(to_days(start)), minutes=float(to_days(end - start) * 1440),
                         delta_v_ms=float(np.sum(run.thrust_ms2[inside]) * SAMPLE_STEP_S)))
    return rows, days


def _draw_sawtooth(axes, run, days):
    every = 600                                              # one point per hour for the 400-day view
    axes[0].plot(days[::every], run.mean_sma_km[::every], color=NAVY, lw=0.6)
    for horizon in HORIZONS_DAYS:
        axes[0].axvline(horizon, color=SLATE, ls=":", lw=0.6)
    zoom = days < ZOOM_DAYS
    axes[1].plot(days[zoom][::10], run.mean_sma_km[zoom][::10], color=NAVY, lw=0.8, label="mean SMA")
    on = run.true_on[zoom][::10] == 1
    axes[1].plot(days[zoom][::10][on], run.mean_sma_km[zoom][::10][on], ".", ms=2, color=RUST, label="thruster on")
    for ax in axes:
        for edge in (SMOOTH_A_LOWER_KM, SMOOTH_A_UPPER_KM):
            ax.axhline(edge, color=RUST, ls="--", lw=0.8)
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.set(xlabel="time [days]", ylabel="mean SMA [km]")
    axes[0].set_title("400 days: the band holds for every cycle (dotted = report horizons)")
    axes[1].set_title(f"First {ZOOM_DAYS} days: coast down, burn up, no timer")
    axes[1].legend(loc="center right")


def _draw_burns(axes, rows):
    start = [row["start_day"] for row in rows]
    axes[0].bar(start, [row["delta_v_ms"] for row in rows], width=1.5, color=NAVY)
    axes[0].set(xlabel="burn start [day]", ylabel="delta-v [m/s]",
                title=f"{len(rows)} burns, {np.sum([r['delta_v_ms'] for r in rows]):.1f} m/s in total")
    axes[1].plot(start, [row["minutes"] for row in rows], "o", ms=3, color=RUST)
    axes[1].set(xlabel="burn start [day]", ylabel="burn duration [min]",
                title=f"Every burn lasts {np.mean([r['minutes'] for r in rows]):.1f} +- "
                      f"{np.std([r['minutes'] for r in rows]):.2f} min")


def _draw_hysteresis_3d(ax, run, days):
    """The controller's phase portrait: (mean SMA, d(mean SMA)/dt, on/off state s), drawn as one line."""
    keep = (days >= LOOP_DAYS[0]) & (days < LOOP_DAYS[1])
    sma = run.mean_sma_km[keep]
    rate_m_per_hour = np.gradient(sma, SAMPLE_STEP_S / 3600) * 1000
    state = run.s[keep]                                    # the 7th state: moves smoothly from 0 to 1 and back
    on = throttle(state) > 0.5
    for mask, color, label in ((~on, NAVY, "coast (throttle closed)"), (on, RUST, "burn (throttle open)")):
        ax.plot(np.where(mask, sma, np.nan), np.where(mask, rate_m_per_hour, np.nan), np.where(mask, state, np.nan),
                color=color, lw=1.5, label=label)
    switching = np.abs(np.diff(on.astype(int), prepend=on[0])) == 1
    ax.scatter(sma[switching], rate_m_per_hour[switching], state[switching], color=AMBER, s=40, depthshade=False,
               label="switch points")
    for edge in (SMOOTH_A_LOWER_KM, SMOOTH_A_UPPER_KM):
        ax.plot([edge, edge], [rate_m_per_hour.min(), rate_m_per_hour.max()], [0, 0], color=SLATE, ls="--", lw=1)
    ax.set_xlabel("mean SMA [km]", labelpad=8)
    ax.set_ylabel("d(mean SMA)/dt [m/hour]", labelpad=8)
    ax.set_zlabel("thruster on/off state s")
    ax.ticklabel_format(axis="x", useOffset=False)
    ax.set_title(f"Hysteresis loop in 3D (days {LOOP_DAYS[0]}-{LOOP_DAYS[1]}): coast slides down at s = 0,\n"
                 "the burn climbs at s = 1; switch on at the lower edge, off at the upper edge", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=22, azim=-60)


def run(sim, out_dir):
    """`sim` is not used: everything comes from long_run.controlled_run (the report's data)."""
    folder = step_dir(out_dir, "step5_deadband")
    controlled = controlled_run()
    rows, days = burn_table(controlled)

    with report_style():
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
        _draw_sawtooth(axes, controlled, days)
        save(fig, folder, "deadband_state_dependent.png")

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
        _draw_burns(axes, rows)
        save(fig, folder, "deadband_raise_trim_dv.png")

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        _draw_hysteresis_3d(ax, controlled, days)
        save(fig, folder, "deadband_hysteresis_3d.png")

    delta_v = np.array([row["delta_v_ms"] for row in rows])
    minutes = np.array([row["minutes"] for row in rows])
    return dict(deadband=dict(
        data="long_run.controlled_run (400 days, 6 s)", n_burns=len(rows),
        delta_v_mean_ms=float(delta_v.mean()), delta_v_std_ms=float(delta_v.std()), delta_v_total_ms=float(delta_v.sum()),
        burn_minutes_mean=float(minutes.mean()), burn_minutes_std=float(minutes.std()),
        mean_interval_days=float(np.mean(np.diff([row["start_day"] for row in rows]))),
        sim_days=float(days[-1])))

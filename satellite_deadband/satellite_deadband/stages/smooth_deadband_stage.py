"""
Week 5: the smooth (tanh) deadband controller with J2.

Figures
-------
smooth_deadband_mean_sma_miss   mean SMA (controlled vs drag-only) + miss distance: Figure 1 of the proposal
smooth_switch_functions         the tanh on/off switches and their sech^2 slopes for several widths
smooth_burn_zoom                one burn up close: mean SMA, thruster state, throttle, solver step size
smooth_slope_sweep              overshoot / burn length as the switch width and switch time are varied
smooth_j2_mean_vs_osculating    why the controller must watch the MEAN SMA once J2 is on, with the
                                12 km osculating swing checked against J2 theory
smooth_integrator_steps         what the integrator does: step size over the whole run

Every curve is sampled from the continuous solution every SAMPLE_STEP_MIN (0.1 min = 6 s).
The only exception is the solver step size, which is shown at the solver's own steps.
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband.constants import (
    DU, EARTH_EQUATORIAL_RADIUS_KM, INCLINATION, J2, METRES, PERIOD, SAMPLE_STEP_MIN, SAMPLE_STEP_S, SECONDS, SMOOTH_A_LOWER, SMOOTH_A_LOWER_KM, SMOOTH_A_UPPER_KM,
    SMOOTH_DAYS, SMOOTH_SOLVER,
    SMOOTH_SOLVER_TOLERANCE, SWITCH_SMA_WIDTH_M, SWITCH_TIME_S, THROTTLE_OPENS_AT, THRUST_ACCEL_MS2, from_days, from_hours,
    from_minutes, to_days, to_minutes,
)
from deadband.smooth_controller import (
    default_runs, mean_sma_km_series, osculating_sma_km_series, rising_switch, rising_switch_slope,
    simulate_smooth_deadband, smooth_initial_state, throttle,
)
from .common import AMBER, GREEN, NAVY, PURPLE, RUST, SLATE, save_figure
from .report_common import step_dir

SWEEP_SMA_WIDTHS_M = (1, 2, 5, 10, 20, 50)        # tanh switch widths to try (SWITCH_TIME fixed)
SWEEP_SWITCH_TIMES_S = (5, 15, 60, 180, 600)     # switch time constants to try (width fixed)
SINGLE_BURN_HOURS = 1.5                          # length of each sweep run (one burn)
OSCULATING_TRIGGER_DAYS = 1.0                    # length of the "controller watches osculating SMA" run
SAMPLE_STEP = SAMPLE_STEP_S * SECONDS            # 0.1 min, non-dimensional: used for every curve
BURN_RESOLUTION = SAMPLE_STEP                    # time resolution used to find burn start/end


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _burn_delta_v_ms(sim, start, end):
    """Delta-v of one burn [m/s]: thrust acceleration integrated over the burn (+-10 min margin)."""
    times = np.arange(start - from_minutes(10), end + from_minutes(10), SAMPLE_STEP)
    return float(np.trapezoid(throttle(sim.sample(times)[6]) * THRUST_ACCEL_MS2, times / SECONDS))


def _sampled_mean_sma_km(sim, times):
    return mean_sma_km_series(sim.sample(times))


def _miss_distance_km(controlled, drag_only, times):
    return np.linalg.norm(controlled.sample(times)[0:3] - drag_only.sample(times)[0:3], axis=0) * DU


def _single_burn(sma_width_m=SWITCH_SMA_WIDTH_M, switch_time_s=SWITCH_TIME_S):
    """
    One burn starting 3 m above the lower edge. Returns, in metres (signed):
        peak_minus_upper_m   > 0: overshot the upper edge,  < 0: switched off before reaching it
        lowest_minus_lower_m < 0: dipped below the lower edge, > 0: switched on before reaching it
    """
    sim = simulate_smooth_deadband(from_hours(SINGLE_BURN_HOURS), sma_width=sma_width_m * METRES,
                                   switch_time=switch_time_s * SECONDS,
                                   state0=smooth_initial_state(SMOOTH_A_LOWER + 3 * METRES))
    burns = sim.burn_intervals(BURN_RESOLUTION)
    sma_km = _sampled_mean_sma_km(sim, sim.sample_times(SAMPLE_STEP))
    return dict(peak_minus_upper_m=float((sma_km.max() - SMOOTH_A_UPPER_KM) * 1000),
                lowest_minus_lower_m=float((sma_km.min() - SMOOTH_A_LOWER_KM) * 1000),
                burn_minutes=float(to_minutes(burns[0][1] - burns[0][0])) if burns else 0.0,
                n_steps=len(sim.t))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _plot_figure_one(controlled, drag_only, out_dir):
    """Mean SMA of both orbits on the left axis, miss distance on the right, time counting down to TCA."""
    times = controlled.sample_times(SAMPLE_STEP)
    days_to_tca = SMOOTH_DAYS - to_days(times)
    miss_km = _miss_distance_km(controlled, drag_only, times)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(days_to_tca, _sampled_mean_sma_km(drag_only, times), color="#6A6AE0", lw=1.4, label="SMA (drag)")
    ax.plot(days_to_tca, _sampled_mean_sma_km(controlled, times), color="#F07070", lw=1.4, label="SMA (control)")
    ax.axhline(SMOOTH_A_LOWER_KM, color="gray", ls="--", lw=1, label="Deadband")
    ax.axhline(SMOOTH_A_UPPER_KM, color="gray", ls="--", lw=1)
    ax.set(xlabel="TCA (days)", ylabel="Mean SMA (km)", xlim=(SMOOTH_DAYS, 0),
           ylim=(SMOOTH_A_LOWER_KM - 0.3, SMOOTH_A_UPPER_KM + 0.2))
    ax.grid(alpha=0.3)

    miss_ax = ax.twinx()
    miss_ax.plot(days_to_tca, miss_km, color="black", lw=2, label="Miss Distance")
    miss_ax.set(ylabel="Miss distance (km)", ylim=(0, max(600, 1.15 * miss_km.max())))

    lines = ax.get_legend_handles_labels()[0] + miss_ax.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + miss_ax.get_legend_handles_labels()[1]
    miss_ax.legend(lines, labels, loc="lower right", fontsize=9)
    ax.set_title("Starlink-style SMA station-keeping (500 m deadband, J2 + drag, smooth tanh thruster,\n"
                 f"sampled every {SAMPLE_STEP_MIN} min)", fontsize=10)
    save_figure(fig, out_dir, "smooth_deadband_mean_sma_miss.png")
    return miss_km


def _plot_switch_functions(out_dir):
    """The tanh switch and its sech^2 slope, for several widths, drawn around the lower edge."""
    offset_m = np.linspace(-60, 60, 600)
    fig, (ax_switch, ax_slope) = plt.subplots(1, 2, figsize=(11, 4.2))
    for width_m, color in zip((2, 5, 20), (RUST, NAVY, GREEN)):
        ax_switch.plot(offset_m, rising_switch(-offset_m, width_m), color=color, lw=2, label=f"width {width_m} m")
        ax_slope.plot(offset_m, rising_switch_slope(-offset_m, width_m), color=color, lw=2, label=f"width {width_m} m")
    ax_switch.plot(offset_m, (offset_m < 0).astype(float), color=SLATE, ls=":", lw=1.5, label="if statement (step)")
    ax_switch.set(xlabel="mean SMA - lower edge [m]", ylabel="turn-on switch  (1 + tanh(-x / width)) / 2",
                  title="Turn-on switch: tanh instead of an if statement")
    ax_slope.set(xlabel="mean SMA - lower edge [m]", ylabel="switch slope  sech^2(x / width) / (2 width)  [1/m]",
                 title="Its slope: sech^2. Narrow width = fast switch")
    for ax in (ax_switch, ax_slope):
        ax.axvline(0, color="gray", lw=0.8); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save_figure(fig, out_dir, "smooth_switch_functions.png")


def _plot_burn_zoom(controlled, first_burn, out_dir):
    start, end = first_burn
    times = np.arange(start - from_minutes(20), end + from_minutes(25), SAMPLE_STEP)
    states = controlled.sample(times)
    sma_km = mean_sma_km_series(states)
    minutes = to_minutes(times - start)

    step_starts = controlled.t[:-1]
    in_window = (step_starts >= times[0]) & (step_starts <= times[-1])
    step_minutes = to_minutes(step_starts[in_window] - start)
    step_seconds = controlled.step_sizes[in_window] / SECONDS

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(minutes, sma_km, color=NAVY, lw=1.6, label="mean SMA (smooth tanh controller)")
    axes[0].axhline(SMOOTH_A_LOWER_KM, color=RUST, ls="--", lw=1, label="deadband edges")
    axes[0].axhline(SMOOTH_A_UPPER_KM, color=RUST, ls="--", lw=1)
    axes[0].set(ylabel="Mean SMA [km]",
                title=f"One burn up close (sampled every {SAMPLE_STEP_MIN} min): thrust ramps on and off smoothly")
    axes[0].legend(fontsize=8, loc="center right")

    axes[1].plot(minutes, states[6], color=PURPLE, lw=1.6, label="thruster on/off state s")
    axes[1].plot(minutes, throttle(states[6]), color=AMBER, lw=1.6, ls="--", label="throttle = tanh switch of s")
    axes[1].axhline(THROTTLE_OPENS_AT, color="gray", lw=0.8, ls=":")
    axes[1].set(ylabel="0 = off, 1 = on", ylim=(-0.05, 1.1))
    axes[1].legend(fontsize=8, loc="center right")

    axes[2].semilogy(step_minutes, step_seconds, ".", ms=3, color=SLATE)
    axes[2].set(xlabel="Minutes from burn start", ylabel="Solver step [s]",
                title=f"{SMOOTH_SOLVER} picks smaller steps by itself while the thruster switches (solver steps)")
    for ax in axes:
        ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "smooth_burn_zoom.png")


def _plot_slope_sweep(width_rows, time_rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = [(axes[0], width_rows, "sma_width_m", "tanh switch width [m of SMA]",
               f"Varying the SMA slope (switch time {SWITCH_TIME_S:.0f} s)"),
              (axes[1], time_rows, "switch_time_s", "switch time constant [s]",
               f"Varying the switching speed (width {SWITCH_SMA_WIDTH_M:.0f} m)")]
    for ax, rows, key, xlabel, title in panels:
        x = [row[key] for row in rows]
        ax.plot(x, [row["peak_minus_upper_m"] for row in rows], "o-", color=NAVY,
                label="peak SMA - upper edge  (> 0 = overshoot)")
        ax.plot(x, [row["lowest_minus_lower_m"] for row in rows], "s-", color=RUST,
                label="lowest SMA - lower edge  (< 0 = dipped below)")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(xscale="log", xlabel=xlabel, title=title)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    axes[0].set_ylabel("metres from the deadband edge")
    save_figure(fig, out_dir, "smooth_slope_sweep.png")


def j2_osculating_theory_km(states):
    """
    What J2 does to the osculating SMA, from energy conservation alone.

    The osculating SMA uses the point-mass energy:  1/a_osc = 2/r - v^2/mu.
    The total energy also contains the J2 potential  U = mu J2 R^2 (3 sin^2(lat) - 1) / (2 r^3),
    and the TOTAL energy is what stays constant (it defines the mean SMA a_E). Taking U out gives
        a_osc ~ a_E - (J2 R^2 / a_E) (3 sin^2(lat) - 1)
    i.e. +J2 R^2/a = +6.4 km over the equator (stronger pull -> faster -> SMA reads high) and
    -(3 sin^2 i - 1) J2 R^2/a = -5.8 km at the highest latitude (i = 53 deg). Twice per orbit.
    """
    a_e = mean_sma_km_series(states)
    sin_lat_sq = (states[2] / np.linalg.norm(states[0:3], axis=0)) ** 2
    return a_e - J2 * EARTH_EQUATORIAL_RADIUS_KM**2 / a_e * (3 * sin_lat_sq - 1)


def _orbit_average(values, step):
    """Running mean over exactly one orbital period (edges where the window does not fit are NaN)."""
    n = int(round(PERIOD / step))
    out = np.full(len(values), np.nan)
    out[n // 2:n // 2 + len(values) - n + 1] = np.convolve(values, np.ones(n) / n, mode="valid")
    return out


def _plot_j2_mean_vs_osculating(controlled, osculating_run, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (ax_signal, ax_latitude), (ax_residual, ax_run) = axes
    times = np.arange(0, from_days(0.4), SAMPLE_STEP)
    states = controlled.sample(times)
    hours = to_days(times) * 24
    osculating = osculating_sma_km_series(states)
    theory = j2_osculating_theory_km(states)
    mean = mean_sma_km_series(states)
    averaged = _orbit_average(osculating, SAMPLE_STEP)
    scale = J2 * EARTH_EQUATORIAL_RADIUS_KM**2 / np.mean(mean)             # J2 R^2 / a  [km]
    offset = scale * (1 - 1.5 * np.sin(INCLINATION) ** 2)                 # orbit average of -(3 sin^2 lat - 1)

    # (a) the swing, with the theory on top and the two kinds of "mean"
    ax_signal.plot(hours, osculating, color=SLATE, lw=2.5, alpha=0.6, label="osculating SMA (simulation)")
    ax_signal.plot(hours, theory, color=AMBER, lw=1, ls="--", label="J2 theory: a_E - (J2 R^2/a)(3 sin^2 lat - 1)")
    ax_signal.plot(hours, mean, color=NAVY, lw=2, label="mean SMA a_E (energy incl. J2): what the controller watches")
    ax_signal.plot(hours, averaged, color=GREEN, lw=2, ls="-.",
                   label=f"osculating SMA averaged over one orbit (= a_E + {1000 * offset:.0f} m)")
    for level, text in ((np.mean(mean) + scale, "equator: a_E + J2 R^2/a"),
                        (np.mean(mean) - scale * (3 * np.sin(INCLINATION) ** 2 - 1), "max latitude 53 deg")):
        ax_signal.axhline(level, color=AMBER, lw=0.6, ls=":")
        ax_signal.text(hours[-1], level, f" {text}", fontsize=7, va="center", ha="left", color=AMBER)
    ax_signal.set(xlabel="Time [hours]", ylabel="SMA [km]", xlim=(0, hours[-1] * 1.25), ylim=(6910, 6930),
                  title=f"J2 swings the osculating SMA by {np.ptp(osculating):.1f} km, twice per orbit")
    ax_signal.legend(fontsize=7, loc="lower right")

    # (b) the swing depends on latitude only: it IS J2
    latitude = np.degrees(np.arcsin(states[2] / np.linalg.norm(states[0:3], axis=0)))
    ax_latitude.plot(latitude, osculating - mean, ".", ms=2, color=SLATE, label="simulation")
    grid = np.linspace(-np.degrees(INCLINATION), np.degrees(INCLINATION), 200)
    ax_latitude.plot(grid, -scale * (3 * np.sin(np.radians(grid)) ** 2 - 1), color=AMBER, lw=2, ls="--",
                     label="theory -(J2 R^2/a)(3 sin^2 lat - 1)")
    ax_latitude.set(xlabel="latitude [deg]", ylabel="osculating - mean SMA [km]",
                    title="The swing is a function of latitude only (J2 is zonal)")
    ax_latitude.legend(fontsize=8)

    # (c) how good is the theory: the residual is second order (r is not exactly a)
    ax_residual.plot(hours, (osculating - theory) * 1000, color=PURPLE, lw=1)
    ax_residual.set(xlabel="Time [hours]", ylabel="simulation - theory [m]",
                    title=f"Theory matches to {np.max(np.abs(osculating - theory)) * 1000:.0f} m out of "
                          f"{np.ptp(osculating):.1f} km (first-order J2, r ~ a)")
    ax_residual.set_ylim(top=np.max(np.abs(osculating - theory)) * 1000 * 1.8)
    ax_residual.text(0.01, 0.97, "the 500 m deadband is applied to a_E (navy), never to the osculating SMA;\n"
                                 "a_E and the orbit-averaged SMA differ by a fixed "
                                 f"(J2 R^2/a)(1 - 1.5 sin^2 i) = {1000 * offset:.0f} m: a definition, not an error",
                     transform=ax_residual.transAxes, fontsize=8, color=SLATE, va="top")

    n_burns = len(osculating_run.burn_intervals(BURN_RESOLUTION))
    times = osculating_run.sample_times(SAMPLE_STEP)
    ax_run.plot(to_days(times) * 24, _sampled_mean_sma_km(osculating_run, times), color=RUST, lw=1.5,
                label=f"controller watching OSCULATING SMA ({n_burns} burns)")
    ax_run.plot(to_days(times) * 24, _sampled_mean_sma_km(controlled, times), color=NAVY, lw=1.5,
                label="controller watching MEAN SMA (0 burns yet)")
    ax_run.axhspan(SMOOTH_A_LOWER_KM, SMOOTH_A_UPPER_KM, color=RUST, alpha=0.15)
    ax_run.set(xlabel="Time [hours]", ylabel="Mean SMA [km]",
               title="Triggering on the osculating SMA fires every orbit and climbs out")
    ax_run.legend(fontsize=8, loc="upper left")
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir, "smooth_j2_mean_vs_osculating.png")
    return n_burns


def _plot_integrator_steps(controlled, burns, out_dir):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.semilogy(to_days(controlled.t[:-1]), controlled.step_sizes / SECONDS, ".", ms=1.5, color=SLATE)
    for start, end in burns:
        ax.axvspan(to_days(start), to_days(end), color=RUST, alpha=0.4)
    ax.set(xlabel="Time [days]", ylabel="Solver step [s]",
           title=(f"Inside the integrator: {SMOOTH_SOLVER}, rtol = atol = {SMOOTH_SOLVER_TOLERANCE:g}, "
                  f"{len(controlled.t)} steps, {controlled.n_rhs_evaluations} RHS calls (burns shaded)"))
    ax.grid(alpha=0.3, which="both")
    save_figure(fig, out_dir, "smooth_integrator_steps.png")


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    """`sim` is not used: days 0-10 of the controlled run (the cached Week 5 run); figures go to step5_deadband/."""
    out_dir = step_dir(out_dir, "step5_deadband")
    controlled, drag_only = default_runs()
    burns = controlled.burn_intervals(BURN_RESOLUTION)
    times = controlled.sample_times(SAMPLE_STEP)
    controlled_sma_km = _sampled_mean_sma_km(controlled, times)
    first_hours = times[times <= from_days(0.4)]

    miss_km = _plot_figure_one(controlled, drag_only, out_dir)
    _plot_switch_functions(out_dir)
    _plot_burn_zoom(controlled, burns[0], out_dir)

    width_rows = [dict(sma_width_m=w, **_single_burn(sma_width_m=w)) for w in SWEEP_SMA_WIDTHS_M]
    time_rows = [dict(switch_time_s=s, **_single_burn(switch_time_s=s)) for s in SWEEP_SWITCH_TIMES_S]
    _plot_slope_sweep(width_rows, time_rows, out_dir)

    osculating_run = simulate_smooth_deadband(from_days(OSCULATING_TRIGGER_DAYS), use_mean_sma=False)
    osculating_burns = _plot_j2_mean_vs_osculating(controlled, osculating_run, out_dir)
    _plot_integrator_steps(controlled, burns, out_dir)

    return dict(smooth_deadband=dict(
        n_burns=len(burns),
        burn_start_days=[float(to_days(start)) for start, _ in burns],
        burn_minutes=[float(to_minutes(end - start)) for start, end in burns],
        burn_delta_v_ms=[_burn_delta_v_ms(controlled, start, end) for start, end in burns],
        sample_step_min=SAMPLE_STEP_MIN,
        mean_sma_min_km=float(controlled_sma_km.min()), mean_sma_max_km=float(controlled_sma_km.max()),
        drag_only_final_mean_sma_km=float(_sampled_mean_sma_km(drag_only, times[-1:])[0]),
        miss_distance_final_km=float(miss_km[-1]),
        osculating_sma_swing_km=float(np.ptp(osculating_sma_km_series(controlled.sample(first_hours)))),
        osculating_trigger_burns_per_day=osculating_burns / OSCULATING_TRIGGER_DAYS,
        solver=SMOOTH_SOLVER, solver_steps=len(controlled.t), rhs_evaluations=int(controlled.n_rhs_evaluations),
        sweep_sma_width=width_rows, sweep_switch_time=time_rows,
    ))

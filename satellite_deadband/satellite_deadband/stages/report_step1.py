"""
Report step 1: data generation, the perturbations, and the dynamic / state-space model.
All data comes from the 400-day runs in deadband/long_run.py.

Figures (outputs/report/step1_dynamics/)
    step1_state_space_equations   the full model written out (state vector, every force, the thruster state)
    step1_orbit_3d_groundtrack    one day of the orbit in 3D and as a ground track
    step1_ground_tracks           ground tracks: 1 day, 15-day coverage, drift over a year, with vs without J2
    step1_j2_check                J2 verified: nodal regression vs theory over 400 days, mean SMA conserved
    step1_force_budget            size of every force along the orbit
    step1_perturbation_divergence what each extra force does to the trajectory, at 15 ... 360 days
    step1_augmented_state         all 7 states [r, v, s] through one burn of the smooth deadband controller
    step1_solver_crosscheck       three different ODE solvers agree (so later errors are not integrator errors)
    step1_summary                 six-panel summary
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband.constants import (ACCEL_UNIT_MS2, DU, EARTH_EQUATORIAL_RADIUS_KM, EARTH_MU_KM3_S2, J2, PERIOD,
                                SMOOTH_CD, VU, from_days, from_minutes, to_days)
from deadband.long_run import HORIZONS_DAYS, controlled_run, divergence_runs, natural_run
from deadband.perturbations import force_accelerations, propagate_full
from deadband.smooth_controller import mean_sma_km_series, osculating_sma_km_series, throttle
from .common import AMBER, GREEN, NAVY, PURPLE, RUST, SLATE, save_figure
from .report_common import panel_label, report_style, save, step_dir

FORCE_COLORS = {"Gravity": "black", "Drag": RUST, "J2": NAVY, "Moon": GREEN, "Sun": AMBER, "SRP": PURPLE}
SOLVER_DAYS = 2.0
SAMPLES_PER_DAY = 1440          # the long runs are sampled every minute
SAMPLES_PER_ORBIT = 95          # ~95.5 min per orbit


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def latitude_longitude(r, t):
    """Ground-track latitude / longitude [deg]. In these units Earth turns exactly t radians in time t."""
    latitude = np.degrees(np.arcsin(r[2] / np.linalg.norm(r, axis=0)))
    longitude = np.degrees(np.arctan2(r[1], r[0]) - t)
    return latitude, (longitude + 180) % 360 - 180


def day_slice(day, days=1):
    return slice(int(day * SAMPLES_PER_DAY), int((day + days) * SAMPLES_PER_DAY))


def j2_check(t, states):
    """
    Simulated nodal regression vs the textbook J2 rate  dRAAN/dt = -1.5 n J2 (Re/a)^2 cos i.
    The theory needs the MEAN a and i, so both are averaged over one orbit (95 samples). Drag lowers
    the orbit through the year, which speeds the regression up, so the theory rate is evaluated at
    every sample and integrated, rather than taken once at the start.
    """
    r, v = states[0:3], states[3:6]
    h = np.cross(r.T, v.T).T
    raan = np.degrees(np.unwrap(np.arctan2(h[0], -h[1])))
    one_orbit = np.ones(SAMPLES_PER_ORBIT) / SAMPLES_PER_ORBIT
    a_mean = np.convolve(osculating_sma_km_series(states), one_orbit, mode="same")
    inclination = np.convolve(np.arccos(h[2] / np.linalg.norm(h, axis=0)), one_orbit, mode="same")
    n = np.sqrt(EARTH_MU_KM3_S2 / a_mean**3)
    rate = -1.5 * n * J2 * (EARTH_EQUATORIAL_RADIUS_KM / a_mean) ** 2 * np.cos(inclination)    # rad/s
    dt_s = (t[1] - t[0]) * 13713.441
    theory = raan[0] + np.degrees(np.concatenate([[0.0], np.cumsum(rate[:-1]) * dt_s]))
    days = to_days(t)
    valid = slice(SAMPLES_PER_ORBIT, -SAMPLES_PER_ORBIT)          # the one-orbit averages are clipped at the ends
    first, last = days < 10, days > days[-1] - 10
    return dict(days=days[valid], raan=raan[valid], analytic=theory[valid],
                simulated_rate=np.polyfit(days[first], raan[first], 1)[0],
                simulated_rate_end=np.polyfit(days[last], raan[last], 1)[0],
                relative_difference=(raan[valid][-1] - theory[valid][-1]) / (theory[valid][-1] - raan[0]))


def force_history(t, states):
    """|acceleration| of every force [m/s^2] at each sample."""
    history = {}
    for i in range(len(t)):
        for name, accel in force_accelerations(t[i], states[0:3, i], states[3:6, i], cd=SMOOTH_CD).items():
            history.setdefault(name, []).append(np.linalg.norm(accel) * ACCEL_UNIT_MS2)
    return {name: np.array(values) for name, values in history.items()}


def divergence_km(runs):
    """|r - r_baseline| [km] for each extra force (and all combined)."""
    return {name: np.linalg.norm(positions - runs["baseline"], axis=0) * DU
            for name, positions in runs.items() if name != "baseline"}


def solver_runs():
    """Position difference [m] of three solvers against a very tight DOP853 reference (gravity + drag + J2)."""
    t = np.linspace(0, from_days(SOLVER_DAYS), 1500)
    reference = propagate_full(t, cd=SMOOTH_CD, forces=("J2",), tolerance=1e-13)
    return t, {method: np.linalg.norm(propagate_full(t, cd=SMOOTH_CD, forces=("J2",), method=method, tolerance=1e-10)[0:3]
                                      - reference[0:3], axis=0) * DU * 1000
               for method in ("DOP853", "LSODA", "Radau")}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
EQUATIONS = [
    r"$\mathbf{State\ vector:}\quad \mathbf{x} = [\,\mathbf{r};\ \mathbf{v};\ s\,]$   (position, velocity, thruster on/off state)",
    r"$\mathbf{State\!-\!space\ form:}\quad \dot{\mathbf{r}} = \mathbf{v},\qquad \dot{\mathbf{v}} = \mathbf{a}_{grav}+\mathbf{a}_{drag}+\mathbf{a}_{J2}+\mathbf{a}_{moon}+\mathbf{a}_{sun}+\mathbf{a}_{SRP}+\mathbf{a}_{thrust}$",
    r"$\mathbf{a}_{grav} = -\mu\,\mathbf{r}/|\mathbf{r}|^3 \qquad \mathbf{a}_{drag} = -c_d\,|\mathbf{v}|\,\mathbf{v}$",
    r"$\mathbf{a}_{J2} = \frac{3}{2}J_2\,\mu\,\frac{R_{eq}^2}{r^4}\left[\frac{x}{r}\left(5\frac{z^2}{r^2}-1\right),\ \frac{y}{r}\left(5\frac{z^2}{r^2}-1\right),\ \frac{z}{r}\left(5\frac{z^2}{r^2}-3\right)\right],\quad R_{eq} = 6378.137\ \mathrm{km}$",
    r"$\mathbf{a}_{moon/sun} = \mu_b\left(\frac{\mathbf{r}_b-\mathbf{r}}{|\mathbf{r}_b-\mathbf{r}|^3}-\frac{\mathbf{r}_b}{|\mathbf{r}_b|^3}\right) \qquad \mathbf{a}_{SRP} = P\,C_R\,\frac{A}{m}\,\hat{\mathbf{u}}_{away\ from\ sun}$",
    r"$\mathbf{a}_{thrust} = T\ \mathrm{throttle}(s)\ \hat{\mathbf{v}}, \qquad \mathrm{throttle}(s)=\frac{1}{2}\left[1+\tanh\frac{s-0.9}{0.02}\right]$",
    r"$\dot{s} = \frac{1}{\tau}\left[\sigma_{on}(\bar a)(1-s) - \sigma_{off}(\bar a)\,s + s(1-s)(2s-1)\right], \quad \sigma_{on}=\frac{1}{2}\left[1+\tanh\frac{a_{low}-\bar a}{w}\right], \ \sigma_{off}=\frac{1}{2}\left[1+\tanh\frac{\bar a-a_{up}}{w}\right]$",
    r"$\bar a = -\mu / (2E),\quad E = \frac{|\mathbf{v}|^2}{2}-\frac{\mu}{|\mathbf{r}|}+U_{J2}(\mathbf{r})$   (mean SMA: constant under J2, changed only by drag and thrust)",
    r"$\mathbf{Units:}\ \ DU = 6371\ \mathrm{km},\ TU = 1/\omega_e = 13713\ \mathrm{s},\ \mu = 289.873;\ \ $drag decays the orbit ~0.13 km/day; all runs 400 days",
]


def _plot_equations(folder):
    fig = plt.figure(figsize=(12, 6.2))
    fig.text(0.02, 0.95, "The dynamic model, written as one state-space system", fontsize=14, fontweight="bold")
    for i, line in enumerate(EQUATIONS):
        fig.text(0.03, 0.85 - i * 0.095, line, fontsize=11)
    save_figure(fig, folder, "step1_state_space_equations.png", tight=False)   # text only: no layout to tighten


def _draw_orbit_3d(ax3d, states):
    r_km = states[0:3] * DU
    u, w = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    ax3d.plot_surface(6371 * np.cos(u) * np.sin(w), 6371 * np.sin(u) * np.sin(w), 6371 * np.cos(w),
                      color="#B9D2E0", alpha=0.5, linewidth=0)
    ax3d.plot(*r_km, color=NAVY, lw=0.8)
    ax3d.set(xlabel="x [km]", ylabel="y [km]", zlabel="z [km]", title="One day of orbit (550 km, 53 deg)")


def _draw_track(ax, t, positions, title, color=NAVY, label=None, size=1.0, colour_by_time=False):
    latitude, longitude = latitude_longitude(positions, t)
    if colour_by_time:
        ax.scatter(longitude, latitude, s=size, c=to_days(t), cmap="viridis")
    else:
        ax.scatter(longitude, latitude, s=size, color=color, label=label)
    ax.set(xlabel="Longitude [deg]", ylabel="Latitude [deg]", xlim=(-180, 180), ylim=(-90, 90), title=title)


def _first_orbit_of_day(day):
    start = int(day * SAMPLES_PER_DAY)
    return slice(start, start + SAMPLES_PER_ORBIT)


def _plot_ground_tracks(folder, t_runs, runs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    _draw_track(axes[0, 0], t_runs[day_slice(0)], runs["J2"][:, day_slice(0)], "Day 1 (colour = time)",
                colour_by_time=True)
    fifteen = day_slice(0, 15)
    latitude, longitude = latitude_longitude(runs["J2"][:, fifteen], t_runs[fifteen])
    axes[0, 1].hist2d(longitude, latitude, bins=[120, 60], cmap="Blues")
    axes[0, 1].set(xlabel="Longitude [deg]", ylabel="Latitude [deg]", title="15-day coverage (time spent per cell)")
    for day, color in ((0, NAVY), (15, GREEN), (120, AMBER), (359, RUST)):
        part = _first_orbit_of_day(day)
        _draw_track(axes[1, 0], t_runs[part], runs["J2"][:, part], "First orbit of day 1 / 16 / 121 / 360",
                    color=color, label=f"day {day + 1}", size=4)
    axes[1, 0].legend(markerscale=4)
    part = _first_orbit_of_day(15)
    _draw_track(axes[1, 1], t_runs[part], runs["baseline"][:, part], "", color=SLATE, label="no J2", size=4)
    _draw_track(axes[1, 1], t_runs[part], runs["J2"][:, part], "Day 16: with vs without J2", color=NAVY,
                label="with J2", size=4)
    axes[1, 1].legend(markerscale=4)
    save(fig, folder, "step1_ground_tracks.png", suptitle="Ground track of the satellite")


def _draw_j2_check(ax, check):
    ax.plot(check["days"], check["raan"], color=NAVY, lw=2,
            label=f"simulated: {check['simulated_rate']:.3f} -> {check['simulated_rate_end']:.3f} deg/day (drag lowers a)")
    ax.plot(check["days"], check["analytic"], color=AMBER, ls="--", lw=1.5, label="theory, integrated with the mean a(t)")
    ax.set(xlabel="time [days]", ylabel="RAAN [deg, unwrapped]", title="J2 check: nodal regression over 400 days")
    ax.legend()


def _plot_j2_check(folder, check, natural_t, natural_states):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    _draw_j2_check(axes[0], check)
    axes[1].plot(check["days"], check["raan"] - check["analytic"], color=RUST)
    axes[1].set(xlabel="time [days]", ylabel="simulated - theory [deg]",
                title=f"Difference: {100 * check['relative_difference']:.3f}% after 400 days")
    window = slice(0, int(0.4 * SAMPLES_PER_DAY))
    hours = to_days(natural_t[window]) * 24
    axes[2].plot(hours, osculating_sma_km_series(natural_states[:, window]), color=SLATE, lw=0.8, label="osculating SMA")
    axes[2].plot(hours, mean_sma_km_series(natural_states[:, window]), color=NAVY, lw=2, label="mean SMA (energy incl. J2)")
    axes[2].set(xlabel="time [hours]", ylabel="SMA [km]", title="J2 energy is conserved: mean SMA flat")
    axes[2].legend()
    save(fig, folder, "step1_j2_check.png",
         suptitle="Checking the J2 implementation (J2 now uses the equatorial radius 6378.137 km)")


def _draw_force_budget(ax, forces):
    names = list(forces)
    means = [np.mean(forces[n]) for n in names]
    ax.barh(names, means, color=[FORCE_COLORS[n] for n in names])
    ax.set_xscale("log")
    for i, value in enumerate(means):
        ax.text(value * 1.3, i, f"{value:.1e}", va="center", fontsize=8)
    ax.set(xlabel="mean |acceleration| along the orbit [m/s$^2$]", title="Force budget at 550 km",
           xlim=(min(means) / 5, max(means) * 50))


def _draw_divergence(ax, t, divergence):
    days = to_days(t)
    for name, km in divergence.items():
        combined = name == "All combined"
        ax.semilogy(days[::30], np.maximum(km[::30], 1e-6), color="black" if combined else FORCE_COLORS[name],
                    lw=1.2, ls="--" if combined else "-", label=name)   # dashed so J2 stays visible beneath
    for horizon in HORIZONS_DAYS:
        ax.axvline(horizon, color="gray", lw=0.6, ls=":")
    ax.set(xlabel="Time [days]", ylabel="|r - r_baseline| [km]", ylim=(1e-3, 3e4),
           title="Trajectory change vs gravity + drag")
    ax.legend(ncol=2)


def _draw_divergence_horizons(ax, t, divergence):
    days = to_days(t)
    names = list(divergence)
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        values = [divergence[name][days <= h].mean() for h in HORIZONS_DAYS]
        ax.bar(np.arange(len(HORIZONS_DAYS)) + (k - (len(names) - 1) / 2) * width, values, width,
               color="black" if name == "All combined" else FORCE_COLORS[name], label=name)
    ax.set(xticks=range(len(HORIZONS_DAYS)), xticklabels=[f"{h} d" for h in HORIZONS_DAYS], yscale="log",
           ylabel="average |r - r_baseline| [km]", title="Average effect of each force, by horizon")
    ax.legend(ncol=3, fontsize=7)


def _draw_solvers(ax, t, runs):
    for (method, metres), color in zip(runs.items(), (NAVY, RUST, GREEN)):
        ax.semilogy(to_days(t), np.maximum(metres, 1e-6), color=color, lw=1.2, label=method)
    ax.set(xlabel="Time [days]", ylabel="position difference [m]", title="Solver cross-check (rtol = 1e-10)")
    ax.legend()


def _augmented_state_data():
    run = controlled_run()
    start = run.t[np.argmax(run.true_on)]
    keep = (run.t > start - from_minutes(30)) & (run.t < start + from_minutes(40))
    states = np.vstack([run.r[:, keep], run.v[:, keep], run.s[keep]])
    return (run.t[keep] - start) / from_minutes(1), states


def _plot_augmented_state(folder, minutes, states):
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for k, (label, color) in enumerate(zip("xyz", (NAVY, RUST, GREEN))):
        axes[0].plot(minutes, states[k] * DU, color=color, label=f"r_{label}")
        axes[1].plot(minutes, states[3 + k] * VU, color=color, label=f"v_{label}")
    axes[0].set(ylabel="position [km]", title="All seven states through one burn (sampled every 0.1 min)")
    axes[1].set(ylabel="velocity [km/s]")
    axes[2].plot(minutes, mean_sma_km_series(states), color=NAVY)
    axes[2].ticklabel_format(axis="y", useOffset=False)
    axes[2].set(ylabel="mean SMA [km]")
    axes[3].plot(minutes, states[6], color=PURPLE, label="thruster state s")
    axes[3].plot(minutes, throttle(states[6]), color=AMBER, ls="--", label="throttle(s)")
    axes[3].set(ylabel="0 = off, 1 = on", xlabel="minutes from burn start")
    for ax in (axes[0], axes[1], axes[3]):
        ax.legend(loc="upper right", ncol=3)
    save(fig, folder, "step1_augmented_state.png")


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    folder = step_dir(out_dir, "step1_dynamics")
    natural_t, natural_states = natural_run()
    t_runs, runs = divergence_runs()
    divergence = divergence_km(runs)
    check = j2_check(natural_t, natural_states)
    two_orbits = slice(0, 2 * SAMPLES_PER_ORBIT)
    forces = force_history(natural_t[two_orbits], natural_states[:, two_orbits])
    minutes, burn_states = _augmented_state_data()
    t_solver, solver_errors = solver_runs()

    with report_style():
        _plot_equations(folder)
        one_day = day_slice(0)
        fig = plt.figure(figsize=(12, 4.8))
        _draw_orbit_3d(fig.add_subplot(1, 2, 1, projection="3d"), natural_states[:, one_day])
        _draw_track(fig.add_subplot(1, 2, 2), natural_t[one_day], natural_states[0:3, one_day], "Ground track over one day")
        save(fig, folder, "step1_orbit_3d_groundtrack.png")
        _plot_ground_tracks(folder, t_runs, runs)
        _plot_j2_check(folder, check, natural_t, natural_states)

        fig, ax = plt.subplots(figsize=(8, 4)); _draw_force_budget(ax, forces); save(fig, folder, "step1_force_budget.png")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        _draw_divergence(axes[0], t_runs, divergence)
        _draw_divergence_horizons(axes[1], t_runs, divergence)
        save(fig, folder, "step1_perturbation_divergence.png")
        _plot_augmented_state(folder, minutes, burn_states)
        fig, ax = plt.subplots(figsize=(8, 4)); _draw_solvers(ax, t_solver, solver_errors); save(fig, folder, "step1_solver_crosscheck.png")

        # ---- summary -------------------------------------------------------
        fig = plt.figure(figsize=(17, 9.5))
        ax_orbit = fig.add_subplot(2, 3, 1, projection="3d")
        axes = [fig.add_subplot(2, 3, k) for k in (2, 3, 4, 5, 6)]
        _draw_orbit_3d(ax_orbit, natural_states[:, one_day])
        _draw_track(axes[0], natural_t[day_slice(0, 3)], natural_states[0:3, day_slice(0, 3)],
                    "Ground track, first 3 days (colour = time)", colour_by_time=True)
        _draw_j2_check(axes[1], check)
        _draw_force_budget(axes[2], forces)
        _draw_divergence(axes[3], t_runs, divergence)
        _draw_divergence_horizons(axes[4], t_runs, divergence)
        ax_orbit.text2D(-0.05, 1.02, "(a)", transform=ax_orbit.transAxes, fontsize=12, fontweight="bold")
        for ax, letter in zip(axes, "bcdef"):
            panel_label(ax, letter)
        save(fig, folder, "step1_summary.png", suptitle="Step 1: dynamic model, perturbations and 400-day data generation")

    days = to_days(t_runs)
    return dict(report_step1=dict(
        mean_force_ms2={name: float(np.mean(values)) for name, values in forces.items()},
        divergence_avg_km_by_horizon={name: {f"{h}d": float(km[days <= h].mean()) for h in HORIZONS_DAYS}
                                      for name, km in divergence.items()},
        divergence_at_horizon_km={name: {f"{h}d": float(km[np.searchsorted(days, h)]) for h in HORIZONS_DAYS}
                                  for name, km in divergence.items()},
        j2_raan_check=dict(rate_first_10_days=float(check["simulated_rate"]), rate_last_10_days=float(check["simulated_rate_end"]),
                           relative_difference_vs_theory=float(check["relative_difference"])),
        solver_max_difference_m={name: float(m.max()) for name, m in solver_errors.items()},
    ))

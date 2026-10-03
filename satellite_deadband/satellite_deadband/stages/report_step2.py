"""
Report step 2: learning the coefficients of the equations of motion by regression.

Data: the 400-day natural run (every force, realistic drag), sampled every minute (long_run.py).
Windows of the first 15, 30, 60, 120, 240 and 360 days show how much longer data helps.
J2 is treated as KNOWN physics in the mu and cd fits (subtracted first); the joint fit estimates
cd, J2, mu_moon, mu_sun and the SRP acceleration together.

Figures (outputs/report/step2_regression/)
    step2_mu_vs_cadence       mu error vs sampling step (15-day window): too slow breaks Nyquist
    step2_fd_error_vs_step    finite-difference acceleration error vs step for stencil orders 2, 6, 8,
                              compared with the (realistic, much smaller) drag signal
    step2_stencil_order       mu and cd error for each stencil order (noisy, 1 min, 15 days)
    step2_drag_fit            the drag regression: unmodelled along-track acceleration vs the drag model
    step2_joint_vs_horizon    every joint-fit coefficient's error vs data length: 1 min clean / noisy, 5 min clean
    step2_joint_correlation   how correlated the five fit columns are (360 days)
    step2_summary             six-panel summary
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from deadband.constants import ACCEL_UNIT_MS2, J2, MU, SMOOTH_CD, from_days
from deadband.long_run import HORIZONS_DAYS, natural_run
from deadband.perturbations import propagate_full
from deadband.physics import drag_accel, gravity_accel, j2_accel
from deadband.regression import (TRUE_PERTURBATION_VALUES, add_position_noise, fd_acceleration_error,
                                 fd_velocity_and_acceleration, fit_all_perturbations, fit_cd, fit_mu, gravity_shape,
                                 known_j2)
from .common import AMBER, GREEN, NAVY, PURPLE, RUST, SLATE
from .report_common import panel_label, report_style, save, step_dir

NOISE_KM = 0.01                              # 10 m position noise for the "noisy" curves
CADENCES_MIN = (1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120)
FD_STEPS_S = (10, 20, 30, 60, 120, 300, 600, 1200, 1800, 3600)
ORBIT_NYQUIST_MIN = 95.5 / 2
TERM_COLORS = {"cd": RUST, "J2": NAVY, "mu_moon": GREEN, "mu_sun": AMBER, "SRP (P*CR*A/m)": PURPLE}
TRUTHS = {"cd": SMOOTH_CD, **TRUE_PERTURBATION_VALUES}


def _percent(estimate, truth):
    return 100 * abs(estimate - truth) / abs(truth)


def _window(days, stride_min=1):
    """(t, positions, h) for the first `days` of the natural run, keeping every stride-th minute."""
    t, states = natural_run()
    n = int(days * 1440)
    t, positions = t[:n:stride_min], states[0:3, :n:stride_min]
    return t, positions, t[1] - t[0]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def mu_cadence_sweep(days=15):
    rows = []
    for step_min in CADENCES_MIN:
        t, positions, h = _window(days, step_min)
        noisy = add_position_noise(positions, NOISE_KM, seed=1)
        rows.append(dict(step_min=step_min, clean=_percent(fit_mu(positions, h, j2=J2), MU),
                         noisy=_percent(fit_mu(noisy, h, j2=J2), MU)))
    return rows


def fd_error_sweep():
    """RMS error of FD acceleration [m/s^2] vs step for each order, against the exact acceleration (1 day)."""
    rows = {order: [] for order in (2, 6, 8)}
    drag_level = None
    for step_s in FD_STEPS_S:
        t = np.arange(0, from_days(1), step_s / 13713.441)
        states = propagate_full(t, cd=SMOOTH_CD, forces=("J2",))
        exact = np.column_stack([gravity_accel(states[0:3, i]) + drag_accel(states[3:6, i], SMOOTH_CD)
                                 + j2_accel(states[0:3, i]) for i in range(len(t))])
        drag_level = np.mean(np.linalg.norm(states[3:6], axis=0) ** 2) * SMOOTH_CD * ACCEL_UNIT_MS2
        for order in rows:
            rows[order].append(fd_acceleration_error(states[0:3], t[1] - t[0], exact, order) * ACCEL_UNIT_MS2)
    return rows, drag_level


def stencil_order_comparison(days=15):
    t, positions, h = _window(days)
    noisy = add_position_noise(positions, NOISE_KM, seed=2)
    return {order: dict(mu=_percent(fit_mu(noisy, h, order, j2=J2), MU),
                        cd=_percent(fit_cd(noisy, h, order, j2=J2), SMOOTH_CD)) for order in (2, 6, 8)}


def drag_fit_data(days=1):
    """Along-track unmodelled acceleration (gravity and J2 removed) vs the fitted drag model, 1-min data."""
    t, positions, h = _window(days)
    cd_hat = fit_cd(positions, h, j2=J2)
    r, v, accel = fd_velocity_and_acceleration(positions, h)
    residual = accel - MU * gravity_shape(r) - known_j2(r, J2)
    along = v / np.linalg.norm(v, axis=0)
    measured = np.sum(residual * along, axis=0) * ACCEL_UNIT_MS2
    model = -cd_hat * np.linalg.norm(v, axis=0) ** 2 * ACCEL_UNIT_MS2
    return measured, model, cd_hat


def joint_vs_horizon():
    """{case: {term: [error % at each horizon]}} for 1 min clean, 1 min noisy, 5 min clean."""
    cases = {"1 min, clean": (1, 0.0), f"1 min, {NOISE_KM * 1000:.0f} m noise": (1, NOISE_KM), "5 min, clean": (5, 0.0)}
    table, correlation = {}, None
    for case, (stride, noise) in cases.items():
        table[case] = {term: [] for term in TRUTHS}
        for horizon in HORIZONS_DAYS:
            t, positions, h = _window(horizon, stride)
            estimates, corr = fit_all_perturbations(t, add_position_noise(positions, noise, seed=3), h)
            for term, value in estimates.items():
                table[case][term].append(_percent(value, TRUTHS[term]))
            if stride == 1 and noise == 0.0:
                correlation = corr
    return table, correlation


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_mu_cadence(ax, rows):
    steps = [row["step_min"] for row in rows]
    ax.loglog(steps, [max(row["clean"], 1e-9) for row in rows], "o-", color=NAVY, label="clean")
    ax.loglog(steps, [max(row["noisy"], 1e-9) for row in rows], "s-", color=RUST, label=f"{NOISE_KM * 1000:.0f} m noise")
    ax.axvline(ORBIT_NYQUIST_MIN, color=SLATE, ls="--", lw=1, label="Nyquist (T/2 = 47.75 min)")
    ax.set(xlabel="sampling step [min]", ylabel="mu error [%]", title="Finding mu (15 days): the sampling step decides it")
    ax.legend()


def _draw_fd_error(ax, rows, drag_level):
    for (order, errors), color in zip(rows.items(), (RUST, NAVY, GREEN)):
        ax.loglog(FD_STEPS_S, errors, "o-", color=color, label=f"order {order}")
    ax.axhline(drag_level, color=AMBER, ls="--", label="size of the (realistic) drag signal")
    ax.set(xlabel="sampling step h [s]", ylabel="FD acceleration error [m/s$^2$]", title="Finite-difference error vs step")
    ax.legend()


def _draw_orders(ax, orders):
    x = np.arange(3)
    ax.bar(x - 0.2, [orders[o]["mu"] for o in (2, 6, 8)], 0.4, color=NAVY, label="mu")
    ax.bar(x + 0.2, [orders[o]["cd"] for o in (2, 6, 8)], 0.4, color=RUST, label="cd")
    ax.set(xticks=x, xticklabels=["order 2", "order 6", "order 8"], yscale="log", ylabel="error [%]",
           title=f"Stencil order ({NOISE_KM * 1000:.0f} m noise, 1 min, 15 days)")
    ax.legend()


def _draw_drag_fit(ax, measured, model, cd_hat):
    hours = np.arange(len(measured)) / 60
    ax.plot(hours, measured * 1e7, ".", ms=2, color=SLATE, label="FD residual (along-track)")
    ax.plot(hours, model * 1e7, color=RUST, lw=2, label=f"fit: cd = {cd_hat / SMOOTH_CD:.4f} x true")
    ax.set(xlabel="time [hours]", ylabel="along-track accel [1e-7 m/s$^2$]",
           title="Drag regression on the gravity- and J2-free residual")
    ax.legend()


def _draw_joint_horizon(ax, table, case):
    for term, errors in table[case].items():
        ax.loglog(HORIZONS_DAYS, np.maximum(errors, 1e-7), "o-", color=TERM_COLORS[term], label=term.split(" ")[0])
    ax.axhline(10, color=SLATE, ls=":", lw=1)
    ax.set(xlabel="data length [days]", ylabel="error [%]", title=f"Joint fit vs data length ({case})",
           xticks=HORIZONS_DAYS)
    ax.set_xticklabels([str(h) for h in HORIZONS_DAYS])
    ax.xaxis.set_minor_formatter(NullFormatter())       # only label the horizons themselves
    ax.legend(ncol=2)


def _draw_correlation(ax, correlation):
    names = [n.split(" ")[0] for n in TRUTHS]
    image = ax.imshow(np.abs(correlation), cmap="Reds", vmin=0, vmax=1)
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
           title="|correlation| between fit columns (360 d)")
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{abs(correlation[i, j]):.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.046)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    folder = step_dir(out_dir, "step2_regression")
    mu_rows = mu_cadence_sweep()
    fd_rows, drag_level = fd_error_sweep()
    orders = stencil_order_comparison()
    measured, model, cd_hat = drag_fit_data()
    joint, correlation = joint_vs_horizon()
    cases = list(joint)

    with report_style():
        for draw, args, filename, size in (
                (_draw_mu_cadence, (mu_rows,), "step2_mu_vs_cadence.png", (7.5, 4.5)),
                (_draw_fd_error, (fd_rows, drag_level), "step2_fd_error_vs_step.png", (7.5, 4.5)),
                (_draw_orders, (orders,), "step2_stencil_order.png", (6.5, 4.2)),
                (_draw_drag_fit, (measured, model, cd_hat), "step2_drag_fit.png", (8, 4.2)),
                (_draw_correlation, (correlation,), "step2_joint_correlation.png", (6.5, 5))):
            fig, ax = plt.subplots(figsize=size)
            draw(ax, *args)
            save(fig, folder, filename)
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
        for ax, case in zip(axes, cases):
            _draw_joint_horizon(ax, joint, case)
        save(fig, folder, "step2_joint_vs_horizon.png")

        fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
        _draw_mu_cadence(axes[0, 0], mu_rows)
        _draw_fd_error(axes[0, 1], fd_rows, drag_level)
        _draw_correlation(axes[0, 2], correlation)
        for ax, case in zip(axes[1], cases):
            _draw_joint_horizon(ax, joint, case)
        for ax, letter in zip(axes.ravel(), "abcdef"):
            panel_label(ax, letter)
        save(fig, folder, "step2_summary.png", suptitle="Step 2: learning the coefficients by regression (400-day data)")

    return dict(report_step2=dict(
        mu_cadence=mu_rows, stencil_orders=orders, drag_fit_cd_ratio=float(cd_hat / SMOOTH_CD),
        joint_fit_error_pct_by_horizon={case: {term: dict(zip([f"{h}d" for h in HORIZONS_DAYS], errors))
                                               for term, errors in terms.items()} for case, terms in joint.items()}))

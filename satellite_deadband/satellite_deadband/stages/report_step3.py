"""
Report step 3: determining drag, and how robust / sensitive each estimator is to noise and to
how much data is available (15 ... 360 days of the 400-day natural run).

Three ways to get cd from noisy positions (J2 is known physics and is removed first):
    1. FD regression   : two finite-difference derivatives, then least squares   (step 2)
    2. energy method   : one derivative, then the SLOPE of the mean-SMA history     (Week 2 Step 10)
    3. shooting method : propagate the orbit and fit the WHOLE trajectory (Levenberg-Marquardt),
                         started from the energy-method answer (a warm start avoids false minima)

Every noise level is repeated with several random seeds (Monte Carlo) so the bands show how much
the answer depends on the particular noise draw.

Figures (outputs/report/step3_drag_noise/)
    step3_noise_monte_carlo     mu and cd error vs noise (15 days, 1-min data)
    step3_cd_vs_horizon         cd error vs data length for the three methods (10 m noise)
    step3_energy_method         what the energy method fits: noisy mean-SMA history and its trend line
    step3_sensitivity_map       cd error vs (sampling step x noise), FD and energy methods (15 days)
    step3_shooting_landscape    whole-trajectory cost vs trial cd (3 days): narrow basin, rugged far away
    step3_summary               six-panel summary
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.optimize import least_squares

from deadband.constants import DU, J2, MU, SMOOTH_CD, to_days
from deadband.long_run import HORIZONS_DAYS, natural_run
from deadband.physics import propagate
from deadband.regression import add_position_noise, fit_cd, fit_cd_energy, fit_mu
from .common import AMBER, GREEN, NAVY, RUST, SLATE
from .report_common import panel_label, report_style, save, step_dir

NOISE_KM = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1)       # 0 ... 100 m
SEEDS = range(6)
HORIZON_NOISE_KM = 0.01                                # 10 m for the data-length study
HORIZON_SEEDS = range(4)
SHOOTING_HORIZONS = (15, 30)                           # each fit propagates the whole window several times
SHOOTING_STEP_MIN = 30                                 # observations used by the shooting fit
MAP_STEPS_MIN = (1, 2, 5, 10, 15, 30)
MAP_NOISE_KM = (0.0, 0.01, 0.05, 0.1, 0.5)
LANDSCAPE_DAYS = 3


def _percent(estimate, truth):
    return 100 * abs(estimate - truth) / truth


def _window(days, stride_min=1):
    t, states = natural_run()
    n = int(days * 1440)
    return t[:n:stride_min], states[0:3, :n:stride_min], (t[1] - t[0]) * stride_min


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def monte_carlo(days=15):
    t, positions, h = _window(days)
    rows = {key: np.zeros((len(NOISE_KM), len(SEEDS))) for key in ("mu", "cd_fd", "cd_energy")}
    for i, noise in enumerate(NOISE_KM):
        for j, seed in enumerate(SEEDS):
            noisy = add_position_noise(positions, noise, seed)
            rows["mu"][i, j] = _percent(fit_mu(noisy, h, j2=J2), MU)
            rows["cd_fd"][i, j] = _percent(fit_cd(noisy, h, j2=J2), SMOOTH_CD)
            rows["cd_energy"][i, j] = _percent(fit_cd_energy(t, noisy, h, j2=J2)[0], SMOOTH_CD)
    return rows


def shooting_fit(t, observed, cd_guess):
    """Fit cd by matching the propagated (gravity + J2 + drag) trajectory to the observed positions."""
    state0 = natural_run()[1][:, 0]

    def residuals(params):
        return (propagate(params[0] * SMOOTH_CD, t, j2=J2, state0=state0)[0:3] - observed).ravel() * DU

    return least_squares(residuals, x0=[cd_guess / SMOOTH_CD], method="lm", max_nfev=30).x[0] * SMOOTH_CD


def horizon_study():
    """cd error [%] vs data length for FD, energy (several seeds) and shooting (one seed, short horizons)."""
    fd = np.zeros((len(HORIZONS_DAYS), len(HORIZON_SEEDS)))
    energy = np.zeros_like(fd)
    shooting = {}
    for i, horizon in enumerate(HORIZONS_DAYS):
        t, positions, h = _window(horizon)
        for j, seed in enumerate(HORIZON_SEEDS):
            noisy = add_position_noise(positions, HORIZON_NOISE_KM, seed)
            fd[i, j] = _percent(fit_cd(noisy, h, j2=J2), SMOOTH_CD)
            cd_energy = fit_cd_energy(t, noisy, h, j2=J2)[0]
            energy[i, j] = _percent(cd_energy, SMOOTH_CD)
            if horizon in SHOOTING_HORIZONS and seed == 0:
                coarse = slice(None, None, SHOOTING_STEP_MIN)
                shooting[horizon] = _percent(shooting_fit(t[coarse], noisy[:, coarse], cd_energy), SMOOTH_CD)
    return fd, energy, shooting


def sensitivity_maps(days=15):
    fd = np.zeros((len(MAP_STEPS_MIN), len(MAP_NOISE_KM)))
    energy = np.zeros_like(fd)
    for i, step_min in enumerate(MAP_STEPS_MIN):
        t, positions, h = _window(days, step_min)
        for j, noise in enumerate(MAP_NOISE_KM):
            noisy = add_position_noise(positions, noise, seed=0)
            fd[i, j] = _percent(fit_cd(noisy, h, j2=J2), SMOOTH_CD)
            energy[i, j] = _percent(fit_cd_energy(t, noisy, h, j2=J2)[0], SMOOTH_CD)
    return fd, energy


def shooting_landscape():
    """Whole-trajectory RMSE [km] for trial cd values (clean data, 3 days, 10-min observations)."""
    t, positions, _ = _window(LANDSCAPE_DAYS, 10)
    state0 = natural_run()[1][:, 0]
    trials = np.linspace(0.0, 3.0, 31) * SMOOTH_CD
    rmse = [np.sqrt(np.mean((propagate(cd, t, j2=J2, state0=state0)[0:3] - positions) ** 2)) * DU for cd in trials]
    return trials, np.array(rmse)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _band(ax, x, errors, color, label, marker="o"):
    """Median line with the seed-to-seed spread (min..max) shaded."""
    errors = np.maximum(errors, 1e-7)
    ax.plot(x, np.median(errors, axis=1), marker=marker, color=color, label=label)
    ax.fill_between(x, errors.min(axis=1), errors.max(axis=1), color=color, alpha=0.2)


def _draw_monte_carlo(ax, rows):
    metres = [n * 1000 for n in NOISE_KM]
    _band(ax, metres, rows["mu"], NAVY, "mu")
    _band(ax, metres, rows["cd_fd"], RUST, "cd, FD regression", "s")
    _band(ax, metres, rows["cd_energy"], GREEN, "cd, energy method", "^")
    ax.set(yscale="log", xlabel="position noise sigma [m]", ylabel="error [%]",
           title=f"Noise study ({len(SEEDS)} seeds, 15 days, 1-min data)")
    ax.legend()


def _draw_horizon(ax, fd, energy, shooting):
    _band(ax, HORIZONS_DAYS, fd, RUST, "FD regression", "s")
    _band(ax, HORIZONS_DAYS, energy, GREEN, "energy method", "^")
    if shooting:
        ax.plot(list(shooting), [max(v, 1e-7) for v in shooting.values()], "D-", color=NAVY, ms=8,
                label="shooting (warm start)")
    ax.set(xscale="log", yscale="log", xlabel="data length [days]", ylabel="cd error [%]", xticks=HORIZONS_DAYS,
           title=f"cd vs data length ({HORIZON_NOISE_KM * 1000:.0f} m noise, 1-min data)")
    ax.set_xticklabels([str(h) for h in HORIZONS_DAYS])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend()


def _draw_energy(ax):
    t, positions, h = _window(60)
    for noise, color, label in ((0.1, SLATE, "100 m noise"), (0.0, NAVY, "clean")):
        cd_hat, times, sma = fit_cd_energy(t, add_position_noise(positions, noise, seed=4), h, j2=J2)
        ax.plot(to_days(times)[::10], sma[::10] * DU, ".", ms=1.5, color=color, alpha=0.4 if noise else 1,
                label=f"{label}: cd = {cd_hat / SMOOTH_CD:.3f} x true")
        slope, intercept = np.polyfit(times, sma, 1)
        ax.plot(to_days(times), (slope * times + intercept) * DU, color=RUST if noise else AMBER, lw=2)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set(xlabel="time [days]", ylabel="mean SMA [km]", title="Energy method: cd from the mean-SMA trend (60 days)")
    ax.legend(markerscale=6)


def _draw_map(ax, grid, title):
    image = ax.imshow(np.log10(np.maximum(grid, 1e-4)), cmap="viridis_r", aspect="auto", vmin=-3, vmax=3)
    ax.set(xticks=range(len(MAP_NOISE_KM)), xticklabels=[f"{n * 1000:g}" for n in MAP_NOISE_KM],
           yticks=range(len(MAP_STEPS_MIN)), yticklabels=[f"{s:g}" for s in MAP_STEPS_MIN],
           xlabel="noise sigma [m]", ylabel="sampling step [min]", title=title)
    ax.grid(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.2g}", ha="center", va="center", fontsize=7,
                    color="white" if grid[i, j] > 1 else "black")
    return image


def _draw_landscape(ax, trials, rmse):
    ax.semilogy(trials / SMOOTH_CD, np.maximum(rmse, 1e-6), "o-", color=NAVY, ms=3)
    ax.axvline(1.0, color=GREEN, ls="--", label="true cd")
    ax.set(xlabel="trial cd / true cd", ylabel="trajectory RMSE [km]", title=f"Shooting cost over {LANDSCAPE_DAYS} days")
    ax.legend()


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    folder = step_dir(out_dir, "step3_drag_noise")
    rows = monte_carlo()
    fd_h, energy_h, shooting = horizon_study()
    fd_map, energy_map = sensitivity_maps()
    trials, rmse = shooting_landscape()

    with report_style():
        fig, ax = plt.subplots(figsize=(7.5, 4.5)); _draw_monte_carlo(ax, rows); save(fig, folder, "step3_noise_monte_carlo.png")
        fig, ax = plt.subplots(figsize=(7.5, 4.5)); _draw_horizon(ax, fd_h, energy_h, shooting); save(fig, folder, "step3_cd_vs_horizon.png")
        fig, ax = plt.subplots(figsize=(8, 4.5)); _draw_energy(ax); save(fig, folder, "step3_energy_method.png")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        _draw_map(axes[0], fd_map, "cd error [%]: FD regression (15 d)")
        image = _draw_map(axes[1], energy_map, "cd error [%]: energy method (15 d)")
        fig.colorbar(image, ax=axes, label="log10 cd error [%]")
        fig.savefig(folder / "step3_sensitivity_map.png", dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.5, 4.5)); _draw_landscape(ax, trials, rmse); save(fig, folder, "step3_shooting_landscape.png")

        fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
        _draw_monte_carlo(axes[0, 0], rows)
        _draw_horizon(axes[0, 1], fd_h, energy_h, shooting)
        _draw_energy(axes[0, 2])
        _draw_map(axes[1, 0], fd_map, "cd error [%]: FD regression (15 d)")
        _draw_map(axes[1, 1], energy_map, "cd error [%]: energy method (15 d)")
        _draw_landscape(axes[1, 2], trials, rmse)
        for ax, letter in zip(axes.ravel(), "abcdef"):
            panel_label(ax, letter)
        save(fig, folder, "step3_summary.png", suptitle="Step 3: drag estimation, noise robustness and data length")

    def medians(errors, keys):
        return {key: float(np.median(errors[i])) for i, key in enumerate(keys)}

    noise_keys = [f"{n * 1000:g}m" for n in NOISE_KM]
    horizon_keys = [f"{h}d" for h in HORIZONS_DAYS]
    return dict(report_step3=dict(
        noise_study_15d=dict(mu=medians(rows["mu"], noise_keys), cd_fd=medians(rows["cd_fd"], noise_keys),
                             cd_energy=medians(rows["cd_energy"], noise_keys)),
        cd_error_pct_by_horizon=dict(fd=medians(fd_h, horizon_keys), energy=medians(energy_h, horizon_keys),
                                     shooting={f"{h}d": v for h, v in shooting.items()})))

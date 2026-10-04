"""
Workstream E: recover cd (and cd + J2 together) by whole-trajectory fitting ("shooting"), on the
SAME data as report step 3: the 400-day natural run (every force, realistic drag SMOOTH_CD).

The model being fitted is gravity + J2 + drag (deadband.physics.propagate), exactly as in step 3's
shooting method. The "observed" orbit also feels the Moon, Sun and SRP, which the model leaves out,
so even the true (cd, J2) leaves a small RMSE floor: that is model error, not noise.

The Week 4 version fitted its own 3-orbit run with the demo drag (CD_TRUE, ~45 km/day) and km-level
noise. Here the noise is in metres and the window is the step 3 landscape window (3 days, 10-min data).

Figures (outputs/report/step3_drag_noise/)
    inverse_cd_sweep              cd error of the shooting fit vs position noise (start 50% away)
    inverse_cost_surface_3d       3D cost surface log10 RMSE(cd, J2): a narrow valley at the true cd that
                                  runs along J2 (a 0.2% change of J2 costs less than a 25% change of cd),
                                  with a sharp pit where both are right
    inverse_cost_contours         the same surface as contours, with the joint-fit path end point
"""
from dataclasses import asdict, dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from deadband.constants import DU, J2, SMOOTH_CD
from deadband.long_run import natural_run
from deadband.physics import propagate
from .common import AMBER, NAVY, RUST
from .report_common import report_style, save, step_dir

WINDOW_DAYS = 3                       # same window as step 3's shooting landscape
OBSERVATION_STEP_MIN = 10
NOISE_LEVELS_M = (0.0, 10.0, 100.0, 1000.0)
JOINT_NOISE_M = 100.0
CD_GUESS_FACTOR = 1.5                 # start the fit 50% away from the true cd
J2_GUESS_FACTOR = 0.999               # J2 is so strong that 0.1% off is already a large error
SURFACE_CD = np.linspace(0.0, 3.0, 13)              # x true cd
SURFACE_J2 = np.linspace(0.998, 1.002, 13)          # x true J2


@dataclass
class Fit:
    noise_m: float
    cd_hat: float
    cd_err_pct: float
    j2_hat: float
    j2_err_pct: float
    rmse_m: float
    evaluations: int


def observations(noise_m, seed=0):
    """(t, observed positions, initial state): 3 days of the natural run every 10 min, plus noise."""
    t, states = natural_run()
    keep = slice(0, int(WINDOW_DAYS * 1440), OBSERVATION_STEP_MIN)
    positions = states[0:3, keep] + noise_m / 1000 / DU * np.random.default_rng(seed).standard_normal((3, len(t[keep])))
    return t[keep], positions, states[:, 0]


def rmse_m(cd, j2, t, observed, state0):
    return float(np.sqrt(np.mean((propagate(cd, t, j2=j2, state0=state0)[0:3] - observed) ** 2)) * DU * 1000)


def fit(noise_m, joint=False, seed=0):
    """Levenberg-Marquardt on the positions; parameters scaled to O(1) (cd / SMOOTH_CD, J2 / J2)."""
    t, observed, state0 = observations(noise_m, seed)

    def residuals(p):
        cd, j2 = p[0] * SMOOTH_CD, (p[1] * J2 if joint else J2)
        return (propagate(cd, t, j2=j2, state0=state0)[0:3] - observed).ravel() * DU

    start = [CD_GUESS_FACTOR, J2_GUESS_FACTOR] if joint else [CD_GUESS_FACTOR]
    result = least_squares(residuals, x0=start, method="lm", max_nfev=60)
    cd_hat = result.x[0] * SMOOTH_CD
    j2_hat = result.x[1] * J2 if joint else J2
    return Fit(noise_m=noise_m, cd_hat=float(cd_hat), cd_err_pct=float(100 * abs(cd_hat / SMOOTH_CD - 1)),
               j2_hat=float(j2_hat), j2_err_pct=float(100 * abs(j2_hat / J2 - 1)),
               rmse_m=rmse_m(cd_hat, j2_hat, t, observed, state0), evaluations=int(result.nfev))


def cost_surface():
    """RMSE [m] on the (cd, J2) grid, clean observations."""
    t, observed, state0 = observations(0.0)
    return np.array([[rmse_m(c * SMOOTH_CD, j * J2, t, observed, state0) for c in SURFACE_CD] for j in SURFACE_J2])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_sweep(ax, fits):
    noise = [max(f.noise_m, 1.0) for f in fits]
    ax.loglog(noise, [max(f.cd_err_pct, 1e-4) for f in fits], "o-", color=NAVY, label="cd error [%]")
    ax.set(xlabel="position noise [m] (0 m drawn at 1 m)", ylabel="cd error [%]",
           title=f"Shooting fit of cd ({WINDOW_DAYS} days, {OBSERVATION_STEP_MIN}-min data, start 50% away)")
    for f, x in zip(fits, noise):
        ax.annotate(f"{f.evaluations} runs", (x, max(f.cd_err_pct, 1e-4)), textcoords="offset points", xytext=(5, 5),
                    fontsize=7)
    ax.legend()


def _draw_surface_3d(ax, surface):
    cd_grid, j2_grid = np.meshgrid(SURFACE_CD, (SURFACE_J2 - 1) * 100)
    z = np.log10(surface)
    ax.plot_surface(cd_grid, j2_grid, z, cmap="viridis", alpha=0.9, linewidth=0, antialiased=True)
    ax.contour(cd_grid, j2_grid, z, zdir="z", offset=z.min() - 0.3, cmap="viridis", levels=15)
    ax.scatter([1], [0], [np.log10(surface[len(SURFACE_J2) // 2, np.argmin(np.abs(SURFACE_CD - 1))])], color=RUST,
               s=60, depthshade=False, label="truth")
    ax.set_xlabel("cd / true cd", labelpad=6)
    ax.set_ylabel("J2 - true J2 [%]", labelpad=6)
    ax.set_zlabel("log10 RMSE [m]")
    ax.set_zlim(z.min() - 0.3, z.max())
    ax.set_title("Whole-trajectory cost surface (3 days, clean data):\n"
                 "a narrow valley at the true cd, running along J2, with a pit at the truth", fontsize=10)
    ax.view_init(elev=28, azim=-130)
    ax.legend(loc="upper left")


def _draw_contours(ax, surface, joint):
    cd_grid, j2_grid = np.meshgrid(SURFACE_CD, (SURFACE_J2 - 1) * 100)
    filled = ax.contourf(cd_grid, j2_grid, np.log10(surface), levels=20, cmap="viridis")
    plt.colorbar(filled, ax=ax, label="log10 RMSE [m]")
    ax.plot(1, 0, "*", ms=14, color=RUST, mec="white", label="truth")
    ax.plot(CD_GUESS_FACTOR, (J2_GUESS_FACTOR - 1) * 100, "o", color=AMBER, mec="white", label="start")
    ax.plot(joint.cd_hat / SMOOTH_CD, (joint.j2_hat / J2 - 1) * 100, "X", ms=10, color="white", mec="black",
            label=f"joint fit ({JOINT_NOISE_M:g} m noise)")
    ax.set(xlabel="cd / true cd", ylabel="J2 - true J2 [%]",
           title=f"Joint cd + J2 fit: cd {joint.cd_err_pct:.3f}% off, J2 {joint.j2_err_pct:.4f}% off")
    ax.legend(loc="upper right", fontsize=8)


def run(sim, out_dir):
    """`sim` is not used: the observations are the report's natural run."""
    folder = step_dir(out_dir, "step3_drag_noise")
    fits = [fit(noise) for noise in NOISE_LEVELS_M]
    joint = fit(JOINT_NOISE_M, joint=True, seed=1)
    surface = cost_surface()

    with report_style():
        fig, ax = plt.subplots(figsize=(8, 4.5)); _draw_sweep(ax, fits); save(fig, folder, "inverse_cd_sweep.png")
        fig = plt.figure(figsize=(9, 7))
        _draw_surface_3d(fig.add_subplot(111, projection="3d"), surface)
        save(fig, folder, "inverse_cost_surface_3d.png")
        fig, ax = plt.subplots(figsize=(8, 5.5)); _draw_contours(ax, surface, joint)
        save(fig, folder, "inverse_cost_contours.png")

    return dict(cd_sweep=[asdict(f) for f in fits], cd_j2_joint=asdict(joint),
                inverse_model_floor_rmse_m=float(surface[len(SURFACE_J2) // 2, np.argmin(np.abs(SURFACE_CD - 1))]))

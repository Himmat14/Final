"""
Inverse problem (Workstream E): recover drag (cd) and J2 by matching a propagated
trajectory to observed positions with Levenberg-Marquardt least squares.

Everything except the parameters being fitted is frozen. The fit criterion is the
mean-squared difference between the propagated and observed positions.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .constants import CD_TRUE, DU, J2, PERIOD
from .physics import propagate

CD_INITIAL_GUESS_FACTOR = 1.5   # start the fit 50% away from the true cd
J2_INITIAL_GUESS_FACTOR = 0.5   # start the J2 fit at half the true J2


@dataclass
class CdFit:
    noise_km: float
    cd_true: float
    cd_hat: float
    err_pct: float
    rmse_km: float
    nfev: int


@dataclass
class CdJ2Fit:
    cd_true: float
    cd_hat: float
    cd_err_pct: float
    j2_true: float
    j2_hat: float
    j2_err_pct: float
    nfev: int


def sample_times(n_orbits, samples_per_orbit=19):
    """Evenly spaced observation times covering `n_orbits` orbits."""
    return np.linspace(0, PERIOD * n_orbits, int(n_orbits * samples_per_orbit))


def make_observations(t_eval, cd=CD_TRUE, j2=0.0, noise_km=0.0, seed=0):
    """Positions from the true model, with optional Gaussian position noise [km]."""
    positions = propagate(cd, t_eval, j2=j2)[0:3]
    if noise_km > 0:
        noise = np.random.default_rng(seed).standard_normal(positions.shape)
        positions = positions + (noise_km / DU) * noise
    return positions


def _percent_error(estimate, truth):
    return 100 * abs(estimate - truth) / truth


def fit_cd(t_eval, observed_positions, cd_guess, j2=0.0):
    """Fit cd alone, with J2 frozen at `j2`."""
    def residuals(params):
        return (propagate(params[0], t_eval, j2=j2)[0:3] - observed_positions).ravel()

    return least_squares(residuals, x0=[cd_guess], method="lm", max_nfev=200)


def fit_cd_and_j2(t_eval, observed_positions, cd_guess, j2_guess):
    """Fit cd and J2 together."""
    def residuals(params):
        cd, j2 = params
        return (propagate(cd, t_eval, j2=j2)[0:3] - observed_positions).ravel()

    return least_squares(residuals, x0=[cd_guess, j2_guess], method="lm", max_nfev=300)


def position_rmse_km(cd, t_eval, observed_positions, j2=0.0):
    """RMSE between the model (at `cd`) and the observations [km]."""
    model = propagate(cd, t_eval, j2=j2)[0:3]
    return np.sqrt(np.mean((model - observed_positions) ** 2)) * DU


def run_cd_noise_sweep(noise_levels_km, n_orbits=3, seed0=0):
    """Recover cd from observations at several position-noise levels."""
    t_eval = sample_times(n_orbits)
    fits = []
    for i, noise_km in enumerate(noise_levels_km):
        observed = make_observations(t_eval, noise_km=noise_km, seed=seed0 + i)
        result = fit_cd(t_eval, observed, cd_guess=CD_TRUE * CD_INITIAL_GUESS_FACTOR)
        cd_hat = result.x[0]
        fits.append(CdFit(noise_km=noise_km, cd_true=CD_TRUE, cd_hat=cd_hat,
                          err_pct=_percent_error(cd_hat, CD_TRUE),
                          rmse_km=position_rmse_km(cd_hat, t_eval, observed),
                          nfev=result.nfev))
    return fits


def run_cd_j2_joint_fit(noise_km=1.0, n_orbits=5, seed=1):
    """Recover cd and J2 together from noisy observations that include J2."""
    t_eval = sample_times(n_orbits)
    observed = make_observations(t_eval, j2=J2, noise_km=noise_km, seed=seed)
    result = fit_cd_and_j2(t_eval, observed, cd_guess=CD_TRUE * CD_INITIAL_GUESS_FACTOR,
                           j2_guess=J2 * J2_INITIAL_GUESS_FACTOR)
    cd_hat, j2_hat = result.x
    return CdJ2Fit(cd_true=CD_TRUE, cd_hat=cd_hat, cd_err_pct=_percent_error(cd_hat, CD_TRUE),
                   j2_true=J2, j2_hat=j2_hat, j2_err_pct=_percent_error(j2_hat, J2),
                   nfev=result.nfev)

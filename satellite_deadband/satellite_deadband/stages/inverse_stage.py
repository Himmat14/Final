"""Workstream E: recover cd (and then cd + J2 together) by whole-trajectory optimisation."""
from dataclasses import asdict

import matplotlib.pyplot as plt

from deadband import inverse
from .common import NAVY, save_figure

NOISE_LEVELS_KM = [0.0, 0.5, 1.0, 2.0, 5.0]


def run(sim, out_dir):
    cd_fits = inverse.run_cd_noise_sweep(NOISE_LEVELS_KM)
    joint_fit = inverse.run_cd_j2_joint_fit(noise_km=1.0)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.semilogy([f.noise_km for f in cd_fits], [max(f.err_pct, 1e-6) for f in cd_fits], "o-", color=NAVY)
    ax.set(xlabel="Position noise sigma [km]", ylabel="c_d error [%] (log scale)",
           title="Single-parameter inverse fit: c_d recovered via whole-trajectory optimisation")
    ax.grid(alpha=0.3, which="both")
    save_figure(fig, out_dir, "inverse_cd_sweep.png")

    return dict(cd_sweep=[asdict(f) for f in cd_fits], cd_j2_joint=asdict(joint_fit))

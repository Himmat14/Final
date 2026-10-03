"""Workstream B: the state-dependent deadband controller (sawtooth trace and burn sizes)."""
import numpy as np
import matplotlib.pyplot as plt

from deadband.constants import A_LOWER_KM, A_UPPER_KM, VU, to_hours
from .common import NAVY, RUST, save_figure


def run(sim, out_dir):
    hours = to_hours(sim.t)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(hours, sim.sma_km, "-", lw=1, color=NAVY)
    ax.axhline(A_LOWER_KM, color=RUST, ls="--", lw=1, label="deadband edges")
    ax.axhline(A_UPPER_KM, color=RUST, ls="--", lw=1)
    ax.set(xlabel="Time [hours]", ylabel="Osculating SMA [km]",
           title="State-dependent deadband: lower raise burn + upper trim burn, no timer")
    ax.legend(); ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "deadband_state_dependent.png")

    raise_dv_ms = sim.raise_dv * VU * 1000
    trim_dv_ms = sim.trim_dv * VU * 1000
    pair_index = np.arange(1, len(raise_dv_ms) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(pair_index - 0.17, raise_dv_ms, width=0.34, color=NAVY, label="raise burn")
    ax.bar(pair_index + 0.17, trim_dv_ms, width=0.34, color=RUST, label="trim burn")
    ax.set(xlabel="Burn-pair index", ylabel="Delta-v [m/s]",
           title="Each lower-bound event fires two impulses: a raise, then an opposing trim")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    save_figure(fig, out_dir, "deadband_raise_trim_dv.png")

    return dict(deadband=dict(
        n_burn_pairs=int(sim.n_burns),
        raise_dv_mean_ms=float(np.mean(raise_dv_ms)), raise_dv_std_ms=float(np.std(raise_dv_ms)),
        trim_dv_mean_ms=float(np.mean(trim_dv_ms)), trim_dv_std_ms=float(np.std(trim_dv_ms)),
        sim_hours=float(hours[-1]),
    ))

"""Workstream D: does a time-indexed neural network extrapolate? (It does not.)"""
import matplotlib.pyplot as plt

from deadband.neural_net import run_long_horizon_test
from .common import NAVY, REFERENCE_LINE, RUST, save_figure


def run(sim, out_dir):
    test = run_long_horizon_test()

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(test.t_hours, test.sma_true_km, "-", color=NAVY, lw=1.3, label="true SMA")
    ax.plot(test.t_hours, test.sma_pred_km, "-", color=RUST, lw=1.3, label="NN prediction")
    ax.axvline(test.train_end_hours, color=REFERENCE_LINE, ls="--", lw=1, label="end of training window")
    ax.set(xlabel="Time [hours]", ylabel="Osculating SMA [km]",
           title="NN fit as r(t): accurate inside training, degrades sharply beyond it")
    ax.legend(); ax.grid(alpha=0.3)
    save_figure(fig, out_dir, "nn_long_horizon.png")

    return dict(nn_horizon=test.rows)

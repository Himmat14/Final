"""
Workstream D: does a time-indexed neural network extrapolate? (It does not.) On the SAME data as the
report: the 400-day natural run, trained on days 0-40 (the steps 6-9 training period) and checked
at 15, 30, 60, 120, 240 and 360 days past the end of training.

Figures (outputs/report/step2_regression/)
    nn_long_horizon        mean SMA: truth, the NN and a straight line, all fitted on days 0-40 only
    nn_horizon_errors      SMA error vs days past training: the NN's error grows, the physics line's
                           stays small (drag is nearly constant over a year)
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband.long_run import HORIZONS_DAYS
from deadband.neural_net import run_long_horizon_test
from .common import GREEN, NAVY, RUST, SLATE
from .report_common import report_style, save, step_dir


def _draw_series(ax, test):
    ax.plot(test.days, test.sma_true_km, color=NAVY, lw=1.2, label="true mean SMA")
    ax.plot(test.days, test.sma_nn_km, color=RUST, lw=1.2, label="NN: SMA = f(t)")
    ax.plot(test.days, test.sma_line_km, color=GREEN, lw=1, ls="--", label="straight line (constant decay rate)")
    ax.axvspan(0, test.train_days, color=SLATE, alpha=0.1, label=f"training data (days 0-{test.train_days:g})")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set(xlabel="time [days]", ylabel="mean SMA [km]",
           title="A network that learns SMA(t) flattens out once it leaves its training window")
    ax.legend()


def _draw_errors(ax, test):
    horizons = [row["days_after_training"] for row in test.rows]
    ax.loglog(horizons, [max(row["nn_err_km"], 1e-4) for row in test.rows], "o-", color=RUST, label="NN")
    ax.loglog(horizons, [max(row["line_err_km"], 1e-4) for row in test.rows], "s--", color=GREEN, label="straight line")
    ax.set(xticks=horizons, xlabel="days past the end of training", ylabel="|SMA error| [km]",
           title="Extrapolation error vs horizon")
    ax.set_xticklabels([str(h) for h in HORIZONS_DAYS])
    ax.minorticks_off()
    ax.legend()


def run(sim, out_dir):
    """`sim` is not used: the data are the report's natural run."""
    folder = step_dir(out_dir, "step2_regression")
    test = run_long_horizon_test()
    with report_style():
        fig, ax = plt.subplots(figsize=(10, 4.6)); _draw_series(ax, test); save(fig, folder, "nn_long_horizon.png")
        fig, ax = plt.subplots(figsize=(7, 4.5)); _draw_errors(ax, test); save(fig, folder, "nn_horizon_errors.png")
    return dict(nn_horizon=test.rows)

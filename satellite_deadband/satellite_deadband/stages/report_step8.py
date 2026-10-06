"""
Report step 8: use the repeating structure to average the noise away and learn an average thrust law.

Data: the 400-day controlled run with 2 x the reference noise (0.2 m position, 1 mm/s velocity), so
each single burn is clearly noisy. Burn starts come from the Bayesian-GMM detector (step 6, trained
on days 0-40, applied to days 40-400), not from the truth: this could be done on real tracking data.
The horizon study repeats the learning with only the first 15, 30, ... 360 days of that data.

Figures (outputs/report/step8_repetition_gp/)
    step8_autocorrelation        autocorrelation of the along-track residual: peaks every burn period
    step8_stacked_bursts         every noisy burn (grey) lined up on its start, their average, and the truth
    step8_noise_vs_bursts        error of the average vs number of burns stacked (1/sqrt(N) law)
    step8_gp_thrust_law          Gaussian-process thrust law with +-2 sigma band vs the true thrust profile
    step8_coast_fold             every coast arc of the mean SMA, overlaid from the end of the burn before it
    step8_horizon                learned thrust level / duration / period vs days of data used
    step8_summary                six-panel summary
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt

from deadband.burn_folding import (autocorrelation, dominant_period, fit_thrust_law, noise_vs_bursts, stack_bursts)
from deadband.classifiers import MODELS, build_features, detection_dataset, evaluate, noisy_features
from deadband.settings import tuned
from deadband.constants import TU, to_days
from deadband.long_run import HORIZONS_DAYS
from deadband.derivative_detection import runs_of_ones
from deadband.smooth_controller import mean_sma_km_series
from .common import AMBER, GREY, NAVY, RUST, SLATE
from .report_common import panel_label, report_style, save, step_dir

NOISE_SCALE = 2.0
MAX_LAG_DAYS = 10.0      # must exceed the ~3.9-day burn period


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def prepare():
    dataset = detection_dataset()
    features, (r, v) = noisy_features(dataset, NOISE_SCALE)
    events = evaluate(dataset, features, methods={"Bayesian GMM": MODELS["Bayesian GMM"]})["Bayesian GMM"].events
    # the stacked signal uses its own tuned derivative window: averaging N burns removes variance, not bias,
    # so a shorter (lower-bias) window than the classifier's is optimal here (deadband/tuning.stacked_choice)
    stacked_features = build_features(r, v, dataset.step_s, window=tuned("sg_window_stacked"),
                                      order=tuned("sg_order_stacked"), method="Savitzky-Golay")
    signal = stacked_features.along_track                 # noisy unmodelled along-track acceleration [m/s^2]
    step = dataset.step_s

    acf = autocorrelation(signal[dataset.test], int(MAX_LAG_DAYS * 86400 / step))
    period_s = dominant_period(acf, step, min_lag_s=12 * 3600)
    true_starts = [start for start, _ in runs_of_ones(dataset.t, dataset.true_on)]
    true_period_s = float(np.mean(np.diff(true_starts)) * TU)

    stack = stack_bursts(dataset.t, signal, events, step)
    truth = stack_bursts(dataset.t, dataset.thrust_ms2, events, step)
    law = fit_thrust_law(stack)
    measured_sma = mean_sma_km_series(np.vstack([r, v]))
    return dict(dataset=dataset, events=events, acf=acf, period_s=period_s, true_period_s=true_period_s,
                stack=stack, truth=truth, law=law, measured_sma=measured_sma,
                noise_rows=noise_vs_bursts(stack, truth.mean, counts=(1, 2, 4, 8, 16, 32, 64, len(stack.windows))),
                horizon_rows=horizon_study(dataset, signal, events, truth))


def horizon_study(dataset, signal, events, truth_all):
    """Learn the thrust law and the period from only the first H days of TEST data, for each horizon H."""
    truth_numbers = true_law_numbers(truth_all)
    rows = []
    for horizon in HORIZONS_DAYS:
        window = dataset.test_horizon(horizon)
        events_h = events * window
        stack = stack_bursts(dataset.t, signal, events_h, dataset.step_s)
        law = fit_thrust_law(stack)
        acf = autocorrelation(signal[window], min(int(MAX_LAG_DAYS * 86400 / dataset.step_s), window.sum() - 1))
        rows.append(dict(days=horizon, bursts=len(stack.windows),
                         plateau_ratio=law.plateau / truth_numbers["plateau"],
                         duration_ratio=law.duration_s / truth_numbers["duration_s"],
                         period_days=dominant_period(acf, dataset.step_s, min_lag_s=12 * 3600) / 86400))
    return rows


def true_law_numbers(truth):
    plateau = float(truth.mean.max())
    above = truth.offsets_s[truth.mean > 0.5 * plateau]
    rising = (truth.offsets_s > -60) & (truth.offsets_s < 120)
    t10 = truth.offsets_s[rising][np.argmax(truth.mean[rising] > 0.1 * plateau)]
    t90 = truth.offsets_s[rising][np.argmax(truth.mean[rising] > 0.9 * plateau)]
    return dict(plateau=plateau, duration_s=float(above.max() - above.min()), rise_time_s=float(t90 - t10))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _draw_acf(ax, d):
    lags_days = np.arange(len(d["acf"])) * d["dataset"].step_s / 86400
    ax.plot(lags_days, d["acf"], color=NAVY, lw=0.8)
    ax.axvline(d["period_s"] / 86400, color=RUST, ls="--", label=f"ACF peak: {d['period_s'] / 86400:.3f} days")
    ax.axvline(d["true_period_s"] / 86400, color=AMBER, ls=":", lw=2, label=f"true burn period: {d['true_period_s'] / 86400:.3f} days")
    ax.set(xlabel="lag [days]", ylabel="autocorrelation", title="The signal repeats: autocorrelation (360 d)",
           ylim=(-0.05, 0.5))
    ax.legend()


def _draw_stack(ax, d):
    stack, truth = d["stack"], d["truth"]
    minutes = stack.offsets_s / 60
    for window in stack.windows:
        ax.plot(minutes, window * 1e4, color=GREY, lw=0.4, alpha=0.4)
    ax.plot(minutes, stack.mean * 1e4, color=NAVY, lw=2, label=f"average of {len(stack.windows)} burns")
    ax.plot(minutes, truth.mean * 1e4, color=AMBER, lw=1.5, ls="--", label="true thrust")
    ax.set(xlabel="minutes from detected burn start", ylabel="along-track accel [1e-4 m/s$^2$]",
           title="Stacking noisy burns (grey = single burns)")
    ax.legend()


def _draw_noise_vs_n(ax, d):
    counts = np.array([n for n, _ in d["noise_rows"]])
    errors = np.array([e for _, e in d["noise_rows"]]) * 1e4
    ax.loglog(counts, errors, "o-", color=NAVY, label="error of the average")
    ax.loglog(counts, errors[0] / np.sqrt(counts), "--", color=SLATE, label="1 / sqrt(N)")
    # slower than 1/sqrt(N): the derivative window correlates neighbouring samples, and the
    # smoothing at the burn edges leaves a fixed bias that no amount of averaging removes
    ax.set(xlabel="number of burns stacked N", ylabel="RMS error [1e-4 m/s$^2$]",
           title="Averaging beats the noise (floor: edge smoothing)")
    ax.legend()


def _draw_gp(ax, d):
    law, truth = d["law"], d["truth"]
    minutes = law.offsets_s / 60
    ax.fill_between(minutes, (law.mean - 2 * law.std) * 1e4, (law.mean + 2 * law.std) * 1e4, color=NAVY, alpha=0.2,
                    label="GP +-2 sigma (spread of single burns)")
    ax.plot(minutes, law.mean * 1e4, color=NAVY, lw=2, label="GP thrust law")
    ax.plot(minutes, truth.mean * 1e4, color=AMBER, ls="--", lw=1.5, label="true thrust")
    ax.set(xlabel="minutes from burn start", ylabel="thrust [1e-4 m/s$^2$]",
           title=f"Learned law: {law.plateau * 1e4:.2f}e-4 m/s$^2$ for {law.duration_s:.0f} s, rise {law.rise_time_s:.0f} s")
    ax.legend()


def _draw_coast_fold(ax, d):
    dataset, sma = d["dataset"], d["measured_sma"]
    ends = [end for _, end in runs_of_ones(dataset.t, d["events"])]
    starts = [start for start, _ in runs_of_ones(dataset.t, d["events"])]
    for end, next_start in zip(ends[:-1], starts[1:]):
        arc = (dataset.t > end) & (dataset.t < next_start)
        hours = (dataset.t[arc] - end) * TU / 3600
        ax.plot(hours[::10], (sma[arc][::10] - 6921.9) * 1000, color=NAVY, lw=0.3, alpha=0.25)
    ax.set(xlabel="hours since the burn ended", ylabel="measured mean SMA - 6921.9 km [m]",
           title=f"{len(ends) - 1} coast arcs overlaid: one repeating decay law")


def _draw_horizon(ax, d):
    rows = d["horizon_rows"]
    days = [row["days"] for row in rows]
    ax.semilogx(days, [row["plateau_ratio"] for row in rows], "o-", color=RUST, label="thrust level")
    ax.semilogx(days, [row["duration_ratio"] for row in rows], "s-", color=AMBER, label="burn duration")
    ax.semilogx(days, [row["period_days"] / (d["true_period_s"] / 86400) for row in rows], "^-", color=NAVY,
                label="burn period")
    ax.axhline(1, color="black", lw=0.8)
    for row in rows:
        ax.text(row["days"], 1.12, f"{row['bursts']}", ha="center", fontsize=7, color=SLATE)
    ax.set(xticks=days, ylim=(0.8, 1.2), xlabel="days of TEST data used (numbers = burns stacked)",
           ylabel="learned / true", title="Learning improves with more repetitions")
    ax.set_xticklabels([str(h) for h in days])
    ax.minorticks_off()
    ax.legend()


def _draw_parameters(ax, d):
    learned = dict(plateau=d["law"].plateau, duration_s=d["law"].duration_s, rise_time_s=d["law"].rise_time_s)
    truth = true_law_numbers(d["truth"])
    labels = ["burn period", "thrust level", "burn duration", "rise time"]
    values = [d["period_s"] / d["true_period_s"], learned["plateau"] / truth["plateau"],
              learned["duration_s"] / truth["duration_s"], learned["rise_time_s"] / max(truth["rise_time_s"], 1)]
    ax.barh(labels, values, color=[NAVY, RUST, AMBER, SLATE])
    ax.axvline(1, color="black", lw=1)
    for i, value in enumerate(values):
        ax.text(value + 0.02, i, f"{value:.3f}", va="center")
    ax.set(xlim=(0, max(1.6, max(values) * 1.25)), xlabel="learned / true", ylabel="quantity", title="Thrust law learned from noisy data")


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def run(sim, out_dir):
    warnings.filterwarnings("ignore", category=UserWarning)
    folder = step_dir(out_dir, "step8_repetition_gp")
    d = prepare()

    with report_style():
        for draw, name, size in ((_draw_acf, "step8_autocorrelation.png", (8, 4)),
                                 (_draw_stack, "step8_stacked_bursts.png", (8, 4.5)),
                                 (_draw_noise_vs_n, "step8_noise_vs_bursts.png", (6.5, 4.5)),
                                 (_draw_gp, "step8_gp_thrust_law.png", (8, 4.5)),
                                 (_draw_coast_fold, "step8_coast_fold.png", (8, 4.5)),
                                 (_draw_horizon, "step8_horizon.png", (8, 4.5))):
            fig, ax = plt.subplots(figsize=size)
            draw(ax, d)
            save(fig, folder, name)

        fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
        for ax, draw, letter in zip(axes.ravel(), (_draw_acf, _draw_stack, _draw_noise_vs_n, _draw_gp,
                                                   _draw_horizon, _draw_parameters), "abcdef"):
            draw(ax, d)
            panel_label(ax, letter)
        save(fig, folder, "step8_summary.png",
             suptitle="Step 8: repetition, autocorrelation and a Gaussian-process thrust law")

    law = d["law"]
    return dict(report_step8=dict(
        noise_scale=NOISE_SCALE, bursts_stacked=len(d["stack"].windows),
        acf_period_days=d["period_s"] / 86400, true_period_days=d["true_period_s"] / 86400,
        learned=dict(plateau_ms2=law.plateau, duration_s=law.duration_s, rise_time_s=law.rise_time_s, kernel=law.kernel),
        truth=true_law_numbers(d["truth"]), noise_vs_bursts=d["noise_rows"], by_horizon=d["horizon_rows"],
        days=float(to_days(d["dataset"].t[-1]))))

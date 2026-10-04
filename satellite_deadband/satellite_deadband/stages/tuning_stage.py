"""
Step 0: bias-variance tuning of every adjustable setting, and the walk-forward / walk-backward
choice of the GMM's derivative method (library: deadband/tuning.py). It runs FIRST: the settings it
chooses are saved (deadband/settings.py) and every later step reads them.

The studies are cached in .cache/tuned_settings_v1.json; delete that file to re-tune.

Figures (outputs/report/step0_tuning/)
    tuning_fd_order            bias^2, variance and MSE of cd vs FD stencil order (1, 2, 5-min data)
    tuning_sg_window           bias^2, variance and MSE of the classifier's v' vs Savitzky-Golay window
    tuning_walk_forward        event F1 of each derivative method over 15 ... 360 days, trained on days
                               0-40 and tested forward, and trained on days 360-400 and tested backward
    tuning_walk_forward_table  mean F1, false events and start error per method: the winner is carried forward
    tuning_gmm_components      bias^2 / variance of P(burn | x), train / test log-likelihood and event F1 vs K
    tuning_sindy               SMA-derivative window and STLSQ threshold: bias^2, variance, MSE
    tuning_summary             six-panel summary with every chosen setting
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband import settings, tuning
from deadband.long_run import HORIZONS_DAYS
from .common import AMBER, GREEN, GREY, METHOD_COLORS, NAVY, PURPLE, RUST, SLATE
from .report_common import panel_label, report_style, save, step_dir

BIAS, VARIANCE, MSE = RUST, NAVY, SLATE
tuning_styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1))]


def _bias_variance(ax, x, rows, chosen, xlabel, title, ylabel, log_x=False, total="mse"):
    """The standard three curves: bias^2, variance and their sum, with the chosen setting marked."""
    ax.plot(x, [r["bias_sq"] for r in rows], "o-", color=BIAS, label="bias$^2$")
    ax.plot(x, [r["variance"] for r in rows], "s--", color=VARIANCE, label="variance")
    ax.plot(x, [r[total] for r in rows], "^-", color=MSE, lw=2.2, label="MSE = bias$^2$ + variance")
    ax.axvline(chosen, color=GREEN, lw=2, alpha=0.6, label=f"chosen: {chosen:g}")
    ax.set(yscale="log", xlabel=xlabel, ylabel=ylabel, title=title)
    if log_x:
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlim(left=0)                      # thresholds are never negative
    ax.legend()


def _draw_fd(ax, rows, chosen, step_min=1):
    mine = [r for r in rows if r["step_min"] == step_min and r["noise_km"] == tuning.FD_NOISE_KM]
    _bias_variance(ax, [r["order"] for r in mine], mine, chosen, "FD stencil order",
                   f"cd from 10 m noisy positions ({step_min}-min data, 15 days, 20 seeds)", "cd error$^2$ [%$^2$]")
    ax.set_xticks(tuning.FD_ORDERS)


def _draw_fd_steps(ax, rows):
    """MSE vs order for every noise level and sampling step: the optimum moves."""
    for noise, color in zip(tuning.FD_NOISES_KM, (GREEN, AMBER, RUST)):
        for step, style in zip(tuning.FD_STEPS_MIN, ("o-", "s--", "^:")):
            mine = [r for r in rows if r["step_min"] == step and r["noise_km"] == noise]
            ax.plot([r["order"] for r in mine], [max(r["mse"], 1e-8) for r in mine], style, color=color,
                    label=f"{noise * 1000:g} m noise, {step}-min data")
    ax.set(yscale="log", xticks=tuning.FD_ORDERS, xlabel="FD stencil order", ylabel="MSE of cd [%$^2$]",
           title="The best order moves with noise and sampling step\n"
                 "(1-min data: order 4 even when clean; 5-min clean data: order 8)")
    ax.legend(fontsize=7, ncol=3)


def _draw_sg(ax, rows, window, order):
    mine = [r for r in rows if r["order"] == order]
    _bias_variance(ax, [r["window_s"] for r in mine], mine, window * 6.0,
                   "Savitzky-Golay window [s]",
                   f"Classifier derivative v' (order {order}), coast + burn weighted equally",
                   "error$^2$ [(1e-4 m/s$^2$)$^2$]")
    ax.set_xscale("log")


def _draw_sg_orders(ax, rows):
    for order, color, style in zip(tuning.SG_ORDERS, (NAVY, RUST, GREEN), ("o-", "s--", "^-.")):
        mine = [r for r in rows if r["order"] == order]
        ax.plot([r["window_s"] for r in mine], [r["mse"] for r in mine], style, color=color, label=f"order {order}")
    ax.set(xscale="log", yscale="log", xlabel="Savitzky-Golay window [s]", ylabel="MSE [(1e-4 m/s$^2$)$^2$]",
           title="Savitzky-Golay: window x polynomial order")
    ax.legend()


def _draw_walk(axes, rows, best):
    """Rows of panels: noise level; columns: walk-forward / walk-backward."""
    methods = list(dict.fromkeys(r["method"] for r in rows))
    for row_axes, scale in zip(axes, tuning.BACKTEST_NOISE_SCALES):
        for ax, direction in zip(row_axes, ("forward", "backward")):
            for i, (method, color) in enumerate(zip(methods, METHOD_COLORS)):
                mine = [r for r in rows if r["method"] == method and r["direction"] == direction
                        and r["noise_scale"] == scale]
                f1 = np.array([r["f1"] for r in mine]) + (i - 2) * 0.012     # tiny offset: overlapping lines stay visible
                ax.semilogx([r["horizon_days"] for r in mine], f1, "os^Dv"[i], ls=tuning_styles[i], color=color,
                            lw=2.6 if method == best else 1.4, label=method + (" (chosen)" if method == best else ""))
            what = ("trained on days 0-40, tested on the NEXT h days" if direction == "forward"
                    else "trained on days 360-400, tested on the h days BEFORE day 360")
            ax.set(xticks=HORIZONS_DAYS, ylim=(-0.08, 1.08), xlabel="out-of-sample horizon h [days]",
                   ylabel="event F1", title=f"Walk-{direction}, {scale:g} x reference noise: {what}")
            ax.set_xticklabels([str(h) for h in HORIZONS_DAYS])
            ax.minorticks_off()
    axes[0][0].legend(fontsize=7, loc="center left")


def _draw_walk_table(ax, summary, best):
    methods = list(summary)
    x = np.arange(len(methods))
    ax.bar(x, [summary[m]["mean_f1"] for m in methods], 0.6, yerr=[summary[m]["std_f1"] for m in methods], capsize=5,
           color=[GREEN if m == best else GREY for m in methods])
    for i, m in enumerate(methods):
        start = summary[m]["start_error_s"]
        start_text = f"{start:.0f} s" if np.isfinite(start) else "no burns found"
        ax.text(i, 0.04, f"false events {summary[m]['false_events']}\nstart error {start_text}",
                ha="center", fontsize=7, color="white" if summary[m]["mean_f1"] > 0.2 else SLATE)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, fontsize=8)
    ax.set(ylim=(0, 1.1), ylabel="mean event F1 (12 out-of-sample periods, +-1 sd)",
           title=f"Walk-forward / backward winner (reference noise): {best}")


def _draw_gmm(axes, rows, chosen):
    k = [r["k"] for r in rows]
    _bias_variance(axes[0], k, rows, chosen, "number of GMM components K",
                   "Burn probability P(burn | x) on TEST (6 bootstrap re-fits)", "Brier score parts", total="brier")
    axes[0].errorbar(k, [r["brier"] for r in rows], yerr=[r["brier_se"] for r in rows], fmt="none", ecolor=SLATE,
                     capsize=3)
    axes[0].set_xticks(k)
    axes[1].plot(k, [r["train_nll"] for r in rows], "o-", color=NAVY, label="TRAIN (days 0-40)")
    axes[1].errorbar(k, [r["test_nll"] for r in rows], yerr=[r["test_nll_std"] for r in rows], fmt="s--", color=RUST,
                     capsize=3, label="TEST (days 40-100)")
    axes[1].axvline(chosen, color=GREEN, lw=2, alpha=0.6)
    axes[1].set(xticks=k, xlabel="number of GMM components K", ylabel="negative log-likelihood per sample",
                title="Fit quality: TRAIN keeps improving, TEST levels off")
    axes[1].legend()
    axes[2].errorbar(k, [r["f1_mean"] for r in rows], yerr=[r["f1_std"] for r in rows], fmt="o-", color=PURPLE,
                     capsize=3, label="event F1 on TEST (mean +- sd)")
    axes[2].axvline(chosen, color=GREEN, lw=2, alpha=0.6, label=f"chosen K = {chosen}")
    axes[2].set(xticks=k, ylim=(-0.05, 1.05), xlabel="number of GMM components K", ylabel="event F1",
                title="Burn detection vs K")
    axes[2].legend()


def _draw_sindy(axes, sma_rows, sma_window, stlsq_rows, threshold):
    _bias_variance(axes[0], [r["window_s"] for r in sma_rows], sma_rows, sma_window * 6.0,
                   "mean-SMA derivative window [s]", "SINDy input da/dt (coast + burn weighted equally)",
                   "error$^2$ [(m/hour)$^2$]")
    axes[0].set_xscale("log")
    _bias_variance(axes[1], [r["threshold"] for r in stlsq_rows], stlsq_rows, threshold, "STLSQ threshold",
                   "SINDy law: predicted da/dt on TEST", "error$^2$ [(m/hour)$^2$]", log_x=True)
    twin = axes[1].twinx()
    twin.plot([r["threshold"] for r in stlsq_rows], [r["terms_kept"] for r in stlsq_rows], "x:", color=AMBER)
    twin.set_ylabel("library terms kept", color=AMBER)
    twin.grid(False)


def _draw_chosen(ax, chosen):
    ax.axis("off")
    lines = ["Settings carried forward to every later step", ""]
    lines += [f"{name:<20} {settings.DEFAULTS[name]!s:>16}  ->  {value!s}" for name, value in chosen.items()]
    lines += ["", "(left: the hand-picked default, right: the tuned value)"]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)


def run(sim, out_dir):
    folder = step_dir(out_dir, "step0_tuning")
    studies = settings.stored_studies()
    if studies is None:                                   # first run: do the studies (several minutes)
        chosen, studies = tuning.run_all()
    chosen = {name: settings.tuned(name) for name in settings.DEFAULTS}

    with report_style():
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
        _draw_fd(axes[0], studies["fd_order"], chosen["fd_order"]); _draw_fd_steps(axes[1], studies["fd_order"])
        save(fig, folder, "tuning_fd_order.png")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        _draw_sg(axes[0], studies["sg"], chosen["sg_window"], chosen["sg_order"]); _draw_sg_orders(axes[1], studies["sg"])
        save(fig, folder, "tuning_sg_window.png")
        fig, axes = plt.subplots(len(tuning.BACKTEST_NOISE_SCALES), 2, figsize=(16, 12), sharey=True)
        _draw_walk(axes, studies["walk_forward"], chosen["derivative_method"])
        save(fig, folder, "tuning_walk_forward.png")
        fig, ax = plt.subplots(figsize=(10, 4.8))
        _draw_walk_table(ax, studies["walk_forward_summary"], chosen["derivative_method"])
        save(fig, folder, "tuning_walk_forward_table.png")
        fig, axes = plt.subplots(1, 3, figsize=(19, 4.8))
        _draw_gmm(axes, studies["gmm_components"], chosen["gmm_components"])
        save(fig, folder, "tuning_gmm_components.png")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        _draw_sindy(axes, studies["sma_window"], chosen["sma_window"], studies["stlsq"], chosen["stlsq_threshold"])
        save(fig, folder, "tuning_sindy.png")

        fig, axes = plt.subplots(2, 3, figsize=(19, 10))
        _draw_fd(axes[0, 0], studies["fd_order"], chosen["fd_order"])
        _draw_sg(axes[0, 1], studies["sg"], chosen["sg_window"], chosen["sg_order"])
        _draw_walk_table(axes[0, 2], studies["walk_forward_summary"], chosen["derivative_method"])
        _bias_variance(axes[1, 0], [r["k"] for r in studies["gmm_components"]], studies["gmm_components"],
                       chosen["gmm_components"], "number of GMM components K", "GMM size: P(burn | x) bias-variance",
                       "Brier score parts", total="brier")
        _bias_variance(axes[1, 1], [r["threshold"] for r in studies["stlsq"]], studies["stlsq"],
                       chosen["stlsq_threshold"], "STLSQ threshold", "SINDy threshold: predicted da/dt",
                       "error$^2$ [(m/hour)$^2$]", log_x=True)
        _draw_chosen(axes[1, 2], chosen)
        for ax, letter in zip(axes.ravel()[:5], "abcde"):
            panel_label(ax, letter)
        save(fig, folder, "tuning_summary.png", suptitle="Step 0: bias-variance tuning and walk-forward model choice")

    return dict(tuning=dict(chosen=chosen, defaults=settings.DEFAULTS,
                            walk_forward_summary=studies["walk_forward_summary"]))

"""
Report step 14: several control laws on one satellite (orbit raising, station keeping, a band change and
collision-avoidance burns), learned and told apart from the noisy SMA alone. Library: deadband/regimes.py.

Figures (outputs/report/step14_regimes/)
    step14_timeline        400 days of mean SMA, every detected burn coloured by its LEARNED regime
    step14_features        the burns in feature space (duration x SMA gain), true regime as colour
    step14_confusion       learned vs true regime: rules with a running edge, and an unsupervised GMM
    step14_edges           the running lower edge learned from station-keeping burns, and the changepoint
    step14_summary
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband import regimes as R
from deadband.mean_element import DAY_S
from .common import AMBER, GREEN, GREY, NAVY, RUST
from .report_common import panel_label, report_style, save, step_dir

REGIME_COLORS = (RUST, NAVY, AMBER)


def _draw_timeline(ax, an):
    ax.plot(an.t / DAY_S, an.a, color=GREY, lw=0.5)
    for k, name in enumerate(R.REGIMES):
        mine = an.rule_labels == k
        ax.plot(an.onsets[mine] / DAY_S, an.a_on[mine], "o", ms=5, color=REGIME_COLORS[k], label=f"{name} ({mine.sum()})")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set(ylim=(6921, 6926), xlabel="time [days]", ylabel="mean SMA [km]",
           title="Burns labelled by the learned regime (orbit raising from 6890 km is off the top of the scale)")
    ax.legend(loc="lower right")


def _draw_features(ax, an):
    duration, gain = (an.ends - an.onsets) / 60, (an.a_off - an.a_on) * 1000
    for k, name in enumerate(R.REGIMES):
        mine = an.true_labels == k
        ax.loglog(duration[mine], np.maximum(gain[mine], 1), "o", color=REGIME_COLORS[k], label=f"true: {name}")
    ax.set(xlabel="burn duration [min]", ylabel="SMA gain [m]", title="Three regimes, three clusters")
    ax.legend()


def _draw_confusion(axes, an):
    for ax, labels, title in ((axes[0], an.rule_labels, "Rules + running edge"), (axes[1], an.gmm_labels, "Unsupervised GMM")):
        m = R.confusion(labels, an.true_labels)
        ax.imshow(m, cmap="Blues")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, str(m[i, j]), ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
        short = ["transfer", "station\nkeeping", "collision\navoidance"]
        ax.set(xticks=range(3), yticks=range(3), xticklabels=short, yticklabels=short, xlabel="learned regime",
               ylabel="true regime", title=f"{title}: {np.trace(m)}/{m.sum()} correct")
        ax.grid(False)


def _draw_edges(ax, an):
    sk = an.rule_labels == 1
    ax.plot(an.onsets[sk] / DAY_S, an.a_on[sk], "o", ms=4, color=NAVY, label="station-keeping burn start SMA")
    ax.plot(an.onsets / DAY_S, an.running_edge, "-", color=GREEN, label="running lower edge (median of the last 7)")
    for cp in an.changepoints:
        ax.axvline(cp, color=RUST, ls="--", label=f"changepoint found at day {cp:.1f}")
    ax.axvline(R.BAND_SHIFT_DAY, color=GREY, lw=3, alpha=0.4, label=f"true band change (day {R.BAND_SHIFT_DAY:g})")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set(xlabel="time [days]", ylabel="SMA at burn start [km]", title="Learning where the band is, and noticing when it moves")
    ax.legend(fontsize=7)


def run(sim, out_dir):
    folder = step_dir(out_dir, "step14_regimes")
    rr = R.simulate_regimes()
    an = R.analyse(rr)
    with report_style():
        fig, ax = plt.subplots(figsize=(12, 4.6)); _draw_timeline(ax, an); save(fig, folder, "step14_timeline.png")
        fig, ax = plt.subplots(figsize=(8, 5)); _draw_features(ax, an); save(fig, folder, "step14_features.png")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5)); _draw_confusion(axes, an); save(fig, folder, "step14_confusion.png")
        fig, ax = plt.subplots(figsize=(11, 4.6)); _draw_edges(ax, an); save(fig, folder, "step14_edges.png")
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        _draw_timeline(axes[0, 0], an); _draw_features(axes[0, 1], an); _draw_edges(axes[0, 2], an)
        _draw_confusion(axes[1, :2], an)
        axes[1, 2].axis("off")
        axes[1, 2].text(0, 1, "Learned laws (SMA rate per regime)\n\n" + "\n".join(
            f"{name:<22} {law['rate_km_per_day']:9.3f} km/day   (n = {law['n']})" for name, law in an.laws.items()),
            va="top", family="monospace", fontsize=10)
        for ax, letter in zip(list(axes[0]) + list(axes[1, :2]), "abcde"):
            panel_label(ax, letter)
        save(fig, folder, "step14_summary.png", suptitle="Step 14: several control laws on one satellite")
    return dict(report_step14=dict(
        burns_true=int(len(rr.labels)), burns_detected=int(an.onsets.size),
        confusion_rules=R.confusion(an.rule_labels, an.true_labels).tolist(),
        confusion_gmm=R.confusion(an.gmm_labels, an.true_labels).tolist(), changepoints_day=an.changepoints, laws=an.laws))

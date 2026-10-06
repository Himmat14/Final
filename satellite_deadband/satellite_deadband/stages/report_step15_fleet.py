"""
Report step 15: fleet-level learning on 40 satellites of one shell (deadband/fleet.py): pattern-of-life
change detection against each satellite's own history (space weather removed using the fleet), and
pooled (empirical-Bayes) drag estimates for satellites with little data.

Figures (outputs/report/step15_fleet/)
    step15_fleet_sma          the mean SMA of a few satellites, normal and anomalous
    step15_change_scores      largest |z| per satellite and window (heat map), with the true change times
    step15_detections         detection day vs truth for the three changes, and false flags
    step15_pooling            drag-coefficient error vs days of data: individual vs pooled
    step15_summary
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband import fleet as Fl
from deadband.mean_element import DAY_S
from .common import GREEN, GREY, NAVY, RUST
from .report_common import panel_label, report_style, save, step_dir

SHOWN = (0, 7, 12, 20)


def _draw_sma(ax, f):
    days = f["t_meas"] / DAY_S
    for sat, color in zip(SHOWN, (GREY, RUST, NAVY, GREEN)):
        label = f"sat {sat}" + (f": {Fl.ANOMALIES[sat][0]} (day {Fl.ANOMALIES[sat][1]:g})" if sat in Fl.ANOMALIES else ": normal")
        ax.plot(days[::6], f["a_meas"][sat, ::6] - f["centre"][sat], lw=0.6, color=color, label=label)
    ax.set(xlabel="time [days]", ylabel="SMA - satellite's band centre [km]", title="Four of the 40 satellites")
    ax.legend(fontsize=7)


def _draw_scores(ax, starts, z):
    score = np.nan_to_num(np.nanmax(np.abs(z), axis=2), nan=0.0)
    image = ax.imshow(np.log10(score + 1), aspect="auto", cmap="magma", origin="lower",
                      extent=(starts[0], starts[-1] + Fl.WINDOW_DAYS, -0.5, Fl.N_SATS - 0.5))
    for sat, (_, day) in Fl.ANOMALIES.items():
        ax.plot(day, sat, "c*", ms=12, mec="white")
    ax.set(xlabel="window end [days]", ylabel="satellite", title="Change score log10(1 + max |z|) (stars: true changes)")
    ax.grid(False)
    return image


def _draw_detections(ax, flags):
    for sat, (name, day) in Fl.ANOMALIES.items():
        found = flags.get(sat)
        ax.barh(f"sat {sat}: {name}", (found - day) if found else 0, color=NAVY if found else RUST)
        ax.text(max((found - day) if found else 0, 0) + 1, f"sat {sat}: {name}",
                f"flagged {found - day:.0f} days after the change" if found else "missed", va="center", fontsize=8)
    false = [s for s in flags if s not in Fl.ANOMALIES]
    ax.set(xlabel="detection delay [days] (30-day windows, 10-day steps)", ylabel="planted change",
           title=f"Detection delay; false flags on the 37 normal satellites: {len(false)}")


def _draw_pooling(ax, rows):
    days = [r["days"] for r in rows]
    ax.plot(days, [r["individual_rms_pct"] for r in rows], "o-", color=GREY, label="each satellite on its own")
    ax.plot(days, [r["pooled_rms_pct"] for r in rows], "s--", color=GREEN, label="pooled (empirical Bayes)")
    ax.set(xscale="log", xticks=days, xlabel="days of data per satellite", ylabel="RMS error of the drag coefficient [%]",
           title="Borrowing strength from the fleet")
    ax.set_xticklabels([str(d) for d in days])
    ax.minorticks_off()
    ax.legend()


def run(sim, out_dir):
    folder = step_dir(out_dir, "step15_fleet")
    f = Fl.simulate_fleet()
    starts, laws = Fl.window_laws(f)
    z = Fl.change_scores(starts, laws)
    flags = Fl.detections(starts, z)
    pooling = Fl.pooled_drag(f)
    with report_style():
        fig, ax = plt.subplots(figsize=(11, 4.6)); _draw_sma(ax, f); save(fig, folder, "step15_fleet_sma.png")
        fig, ax = plt.subplots(figsize=(10, 6)); image = _draw_scores(ax, starts, z)
        fig.colorbar(image, ax=ax, label="log10(1 + max |z|)"); save(fig, folder, "step15_change_scores.png")
        fig, ax = plt.subplots(figsize=(10, 3.8)); _draw_detections(ax, flags); save(fig, folder, "step15_detections.png")
        fig, ax = plt.subplots(figsize=(8, 4.6)); _draw_pooling(ax, pooling); save(fig, folder, "step15_pooling.png")
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        _draw_sma(axes[0, 0], f); _draw_scores(axes[0, 1], starts, z); _draw_detections(axes[1, 0], flags)
        _draw_pooling(axes[1, 1], pooling)
        for ax, letter in zip(axes.ravel(), "abcd"):
            panel_label(ax, letter)
        save(fig, folder, "step15_summary.png", suptitle="Step 15: fleet-level pattern-of-life learning (40 satellites)")
    return dict(report_step15=dict(flags={str(k): v for k, v in flags.items()},
                                   anomalies={str(k): v for k, v in Fl.ANOMALIES.items()}, pooling=pooling))

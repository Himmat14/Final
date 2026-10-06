"""
Report step 11: realistic observations. The satellite is seen in 8-minute tracking passes (positions every
12 s with 10 m ... 1 km noise), 1 to 8 times a day; an orbit is fitted to each pass and burns are found
in the gaps between passes. Library: deadband/observation.py.

Data: days 0-120 of the 400-day controlled run (TRAIN 0-40 for the drag rate and noise level).
Real ephemerides: any CCSDS OEM file put in data/ephemeris/ (e.g. Starlink ephemerides from
space-track.org) is read and its mean SMA plotted and run through the same burn test.

Figures (outputs/report/step11_observations/)
    step11_pass_od            one pass: noisy positions and the fitted orbit; per-pass SMA error vs noise
    step11_detection_grid     gap-test F1 for every (passes per day x noise): pass-to-pass and 1-day multi-pass
    step11_gap_residuals      residual SMA jump per gap with the threshold, 8 passes/day at 10 m and 100 m
    step11_real_ephemeris     (only if data/ephemeris/ holds OEM files) mean SMA of the real satellite
    step11_summary
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband import observation as O
from .common import AMBER, GREEN, GREY, NAVY, RUST
from .report_common import panel_label, report_style, save, step_dir


def _draw_sma_error(ax, rows):
    for per_day, color in zip(O.PASSES_PER_DAY, (GREY, NAVY, GREEN, RUST)):
        mine = [r for r in rows if r["passes_per_day"] == per_day and r["test"] == "pass-to-pass"]
        ax.loglog([r["noise_km"] * 1000 for r in mine], [r["sma_error_m"] for r in mine], "o-", color=color,
                  label=f"{per_day} passes/day")
    ax.axhline(500, color=AMBER, ls="--", label="one station-keeping burn (500 m)")
    ax.set(xlabel="position noise per measurement [m]", ylabel="RMS error of one pass's mean SMA [m]",
           title="One 8-min pass pins the SMA to ~2 x the position noise")
    ax.legend()


def _draw_grid(axes, rows):
    for ax, test in zip(axes, O.TESTS):
        grid = np.array([[next(r["f1"] for r in rows if r["test"] == test and r["passes_per_day"] == p
                               and r["noise_km"] == n) for n in O.NOISES_KM] for p in O.PASSES_PER_DAY])
        image = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, origin="lower", aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set(xticks=range(len(O.NOISES_KM)), xticklabels=[f"{n * 1000:g}" for n in O.NOISES_KM],
               yticks=range(len(O.PASSES_PER_DAY)), yticklabels=[str(p) for p in O.PASSES_PER_DAY],
               xlabel="position noise [m]", ylabel="passes per day", title=f"Burn F1 in the gaps: {test}")
        ax.grid(False)
    return image


def _draw_residuals(axes, examples, onsets):
    for ax, noise in zip(axes, (0.01, 0.1)):
        t, sma, truth, residual, flags, in_gap = examples[("1-day multi-pass", 8, noise)]
        mid = 0.5 * (t[1:] + t[:-1])
        ax.plot(mid, residual * 1000, ".", ms=2, color=GREY, label="gap residual (1-day windows)")
        ax.plot(mid[flags], residual[flags] * 1000, "o", ms=5, color=RUST, label="flagged burn")
        for o in onsets[onsets < t[-1]]:
            ax.axvline(o, color=NAVY, lw=0.6, alpha=0.5)
        ax.set(xlabel="time [days]", ylabel="SMA jump after drag correction [m]",
               title=f"8 passes/day, {noise * 1000:g} m noise (blue lines = true burns)")
        ax.legend(fontsize=7)


def _draw_ephemeris(ax, files):
    for path in files[:4]:
        t, sma = O.oem_mean_sma(path)
        ax.plot(t, sma, lw=0.8, label=path.name)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set(xlabel="time [days]", ylabel="mean SMA [km]", title="Real ephemerides (data/ephemeris/)")
    ax.legend(fontsize=7)


def run(sim, out_dir):
    folder = step_dir(out_dir, "step11_observations")
    rows, examples, onsets = O.stress()
    files = O.find_ephemerides()
    with report_style():
        fig, ax = plt.subplots(figsize=(8, 4.8)); _draw_sma_error(ax, rows); save(fig, folder, "step11_pass_od.png")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        image = _draw_grid(axes, rows); fig.colorbar(image, ax=axes, label="F1")
        from .common import save_figure
        save_figure(fig, folder, "step11_detection_grid.png", tight=False)
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        _draw_residuals(axes, examples, onsets); save(fig, folder, "step11_gap_residuals.png")
        if files:
            fig, ax = plt.subplots(figsize=(10, 4.5)); _draw_ephemeris(ax, files); save(fig, folder, "step11_real_ephemeris.png")
        fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
        _draw_sma_error(axes[0, 0], rows)
        _draw_grid([axes[0, 1], axes[1, 0]], rows)
        _draw_residuals([axes[1, 1]], examples, onsets)
        for ax, letter in zip(axes.ravel(), "abcd"):
            panel_label(ax, letter)
        save(fig, folder, "step11_summary.png", suptitle="Step 11: burns from realistic tracking passes")
    return dict(report_step11=dict(rows=rows, real_ephemeris_files=[str(f) for f in files]))

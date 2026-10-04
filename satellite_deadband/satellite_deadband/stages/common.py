"""
Shared plotting helpers: ONE style for every figure, the colour palette, and a save-and-close function.

Style (applied to every figure the moment this module is imported, and enforced again in save_figure):
    * one font family (DejaVu Sans, also used for maths text) and fixed sizes for titles, labels,
      ticks and legends, so every graph in every folder looks the same;
    * a high-contrast, colour-blind-safe palette (Okabe-Ito);
    * major gridlines plus light minor gridlines;
    * when one chart holds three or more solid labelled lines, they get different line styles
      (solid, dashed, dash-dot, dotted, ...) so they can be told apart even in grey-scale.
save_figure also records any chart with a missing title or axis label (see `LABEL_REPORT`), so the
pipeline can list them at the end of a run.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files only; no window needed
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, NullLocator  # noqa: E402

# --- Palette (Okabe & Ito, colour-blind safe, high contrast) ----------------
NAVY = "#0072B2"         # blue
RUST = "#D55E00"         # vermilion
GREEN = "#009E73"        # bluish green
AMBER = "#E69F00"        # orange
PURPLE = "#CC79A7"       # reddish purple
SKY_BLUE = "#56B4E9"
GREY = "#7A7A7A"
REFERENCE_LINE = "#555555"
BURN_GREY = "#9A9A9A"
SLATE = "#222222"
SKY = "#D6EAF8"          # light background fill (coast spans)
METHOD_COLORS = [NAVY, RUST, GREEN, AMBER, PURPLE, SKY_BLUE, SLATE]
LINE_STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)), (0, (8, 2)), (0, (1, 1))]

# --- Fonts and sizes ---------------------------------------------------------
FONT = "DejaVu Sans"
TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, SUPTITLE_SIZE = 10.5, 9.5, 8.5, 8, 13

STYLE = {
    "font.family": FONT,
    "mathtext.fontset": "dejavusans",
    "font.size": LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.titleweight": "bold",
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.which": "both",
    "grid.color": "#B0B0B0",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.5,
    "lines.linewidth": 1.6,
    "axes.prop_cycle": cycler(color=METHOD_COLORS) + cycler(linestyle=LINE_STYLES),
    "figure.dpi": 100,
}
plt.rcParams.update(STYLE)

LABEL_REPORT = []        # (file, panel title or index, what is missing), filled by save_figure


# ---------------------------------------------------------------------------
# Enforcing the style on a finished figure
# ---------------------------------------------------------------------------
def _is_numeric(labels):
    texts = [label.get_text().replace("−", "-") for label in labels if label.get_text()]
    try:
        [float(text) for text in texts]
        return True
    except ValueError:
        return False


def _style_axes(ax):
    """Fonts, minor ticks / gridlines and distinct line styles for one 2-D chart."""
    ax.title.set_fontsize(TITLE_SIZE)
    ax.xaxis.label.set_fontsize(LABEL_SIZE)
    ax.yaxis.label.set_fontsize(LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    for axis, scale, labels in ((ax.xaxis, ax.get_xscale(), ax.get_xticklabels()),
                                (ax.yaxis, ax.get_yscale(), ax.get_yticklabels())):
        # minor ticks on numeric linear axes (not on category axes such as bar-chart names);
        # log axes keep their own minor ticks unless a stage switched them off on purpose
        if scale == "linear" and _is_numeric(labels) and isinstance(axis.get_minor_locator(), NullLocator):
            axis.set_minor_locator(AutoMinorLocator())
    if ax.axison:
        ax.grid(True, which="major", color="#B0B0B0", lw=0.6, alpha=0.55)
        ax.grid(True, which="minor", color="#D0D0D0", lw=0.4, alpha=0.45, ls=":")
        ax.set_axisbelow(True)
    solid = [line for line in ax.get_lines()
             if not line.get_label().startswith("_") and line.get_linestyle() == "-" and len(line.get_xdata()) > 2]
    if len(solid) >= 3:                       # three or more solid labelled lines: make them distinguishable
        for line, style in zip(solid[1:], LINE_STYLES[1:] * 3):
            line.set_linestyle(style)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(min(text.get_fontsize(), LEGEND_SIZE))


def _check_labels(fig, filename):
    """Record charts that show data but have no title or no axis label."""
    for i, ax in enumerate(fig.axes):
        if not ax.axison or not (ax.lines or ax.collections or ax.patches or ax.images):
            continue
        if ax.get_label() == "<colorbar>":
            continue
        name = ax.get_title() or f"panel {i}"
        missing = [what for what, text in (("title", ax.get_title()), ("x label", ax.get_xlabel()),
                                           ("y label", ax.get_ylabel())) if not text]
        if getattr(ax, "name", "") == "3d" and not ax.get_zlabel():
            missing.append("z label")
        if missing and not (fig._suptitle is not None and missing == ["title"]):
            LABEL_REPORT.append((filename, name, ", ".join(missing)))


def apply_style(fig, filename=""):
    for ax in fig.axes:
        if getattr(ax, "name", "") == "3d":
            ax.title.set_fontsize(TITLE_SIZE)
            continue
        if ax.get_label() == "<colorbar>":
            continue
        _style_axes(ax)
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(SUPTITLE_SIZE)
    _check_labels(fig, filename)


def save_figure(fig, out_dir, filename, dpi=200, tight_rect=None, tight=True):
    """Apply the common style, tighten the layout (unless tight=False), write `out_dir/filename`, close."""
    apply_style(fig, filename)
    if not tight:
        pass                                   # layouts with a shared colour bar place themselves
    elif tight_rect:
        fig.tight_layout(rect=tight_rect)
    else:
        fig.tight_layout()
    fig.savefig(Path(out_dir) / filename, dpi=dpi)
    plt.close(fig)

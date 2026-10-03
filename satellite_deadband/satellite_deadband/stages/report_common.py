"""
Shared helpers for the nine report stages (report_step1 ... report_step9).

Every report stage writes into  <out>/report/stepN_<topic>/  and produces
    * several step-by-step figures (one idea per figure), and
    * one  stepN_summary.png  : a multi-panel figure condensing the step for the report.
"""
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt

from .common import save_figure

# A consistent, clean look for every report figure (applied only inside `report_style()`).
REPORT_RC = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 100,
}


@contextmanager
def report_style():
    """Use the report look for every figure made inside this `with` block."""
    with plt.rc_context(REPORT_RC):
        yield


def step_dir(out_dir, name):
    """Create and return <out_dir>/report/<name>/."""
    folder = Path(out_dir) / "report" / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def panel_label(ax, letter):
    """Bold '(a)', '(b)', ... in the top-left corner of a summary panel."""
    ax.text(-0.12, 1.06, f"({letter})", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom")


def save(fig, folder, filename, suptitle=None):
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
        save_figure(fig, folder, filename, tight_rect=(0, 0, 1, 0.96))
    else:
        save_figure(fig, folder, filename)

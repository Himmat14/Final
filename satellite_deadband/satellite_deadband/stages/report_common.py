"""
Shared helpers for the nine report stages (report_step1 ... report_step9).

Every report stage writes into  <out>/report/stepN_<topic>/  and produces
    * several step-by-step figures (one idea per figure), and
    * one  stepN_summary.png  : a multi-panel figure condensing the step for the report.
"""
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt

from .common import STYLE, SUPTITLE_SIZE, save_figure

# The report look is the common style of every figure (stages/common.py); kept as a name so the
# `with report_style():` blocks in the report stages stay readable.
REPORT_RC = STYLE


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
        fig.suptitle(suptitle, fontsize=SUPTITLE_SIZE, fontweight="bold")
        save_figure(fig, folder, filename, tight_rect=(0, 0, 1, 0.96))
    else:
        save_figure(fig, folder, filename)

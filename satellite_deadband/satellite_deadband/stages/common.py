"""Shared plotting helpers: the colour palette and a save-and-close function."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files only; no window needed
import matplotlib.pyplot as plt  # noqa: E402

NAVY = "#1E2761"
RUST = "#B85042"
GREEN = "#2C5F2D"
AMBER = "#A86B21"
PURPLE = "#6B3FA0"
GREY = "#888888"
REFERENCE_LINE = "gray"
BURN_GREY = "#9A9A9A"
SLATE = "#444444"
SKY = "#B9D2E0"
METHOD_COLORS = [NAVY, RUST, GREEN, AMBER, PURPLE]


def save_figure(fig, out_dir, filename, dpi=200, tight_rect=None):
    """Tighten the layout, write `out_dir/filename`, and close the figure."""
    if tight_rect:
        fig.tight_layout(rect=tight_rect)
    else:
        fig.tight_layout()
    fig.savefig(Path(out_dir) / filename, dpi=dpi)
    plt.close(fig)

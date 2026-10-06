"""
The settings chosen by the bias-variance tuning stage (deadband/tuning.py, main.py stage "tuning"),
read by every later step so the optimal values are carried forward automatically.

    tuned("fd_order")           finite-difference stencil order for 1-min data (steps 2-4)
    fd_order_for(step_min)      the stencil order tuned for the sampling step nearest to step_min
                                (the best order depends on the step: truncation error grows as h^order)
    tuned("sg_window")          Savitzky-Golay window (samples) of the classifier derivatives (steps 6-9)
    tuned("sg_order")           Savitzky-Golay polynomial order
    tuned("sg_window_stacked")  Savitzky-Golay window / order for step 8, which averages many burns
    tuned("sg_order_stacked")
    tuned("derivative_method")  how the classifier differentiates the velocity data (walk-forward winner)
    tuned("gmm_components")     number of Gaussians in the burn GMMs
    tuned("sma_window")         Savitzky-Golay window of the mean-SMA derivative (step 9)
    tuned("stlsq_threshold")    SINDy sparsity threshold (step 9)

Until the tuning stage has run (or if its file is deleted) the DEFAULTS below are used: they are the
hand-picked values the code used before tuning existed.
"""
import json
from functools import lru_cache
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
SETTINGS_FILE = CACHE_DIR / "tuned_settings_v1.json"

DEFAULTS = dict(fd_order=6, fd_order_by_step={}, sg_window=21, sg_order=5, sg_window_stacked=21, sg_order_stacked=5, derivative_method="Savitzky-Golay", gmm_components=4,
                sma_window=41, stlsq_threshold=0.05)


@lru_cache(maxsize=1)
def _load():
    if SETTINGS_FILE.exists():
        stored = json.loads(SETTINGS_FILE.read_text())
        return {**DEFAULTS, **stored.get("chosen", {})}
    return dict(DEFAULTS)


def tuned(name):
    """The chosen value of one setting (the default if tuning has not run yet)."""
    return _load()[name]


def fd_order_for(step_min):
    """Tuned stencil order for this sampling step [min]: the choice made at the nearest studied step (log scale)."""
    by_step = {float(k): v for k, v in tuned("fd_order_by_step").items()}
    if not by_step:
        return tuned("fd_order")
    import math
    nearest = min(by_step, key=lambda s: abs(math.log(s) - math.log(max(step_min, 1e-6))))
    return int(by_step[nearest])


def all_settings():
    return dict(_load())


def is_tuned():
    return SETTINGS_FILE.exists()


def save(chosen, studies):
    """Write the chosen settings (and the study tables behind them) and make tuned() use them from now on."""
    CACHE_DIR.mkdir(exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(dict(chosen=chosen, studies=studies), indent=1))
    _load.cache_clear()


def stored_studies():
    return json.loads(SETTINGS_FILE.read_text())["studies"] if SETTINGS_FILE.exists() else None

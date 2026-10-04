"""
The settings chosen by the bias-variance tuning stage (deadband/tuning.py, main.py stage "tuning"),
read by every later step so the optimal values are carried forward automatically.

    tuned("fd_order")           finite-difference stencil order for the position regressions (steps 2-4)
    tuned("sg_window")          Savitzky-Golay window (samples) of the classifier derivatives (steps 6-9)
    tuned("sg_order")           Savitzky-Golay polynomial order
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

DEFAULTS = dict(fd_order=6, sg_window=21, sg_order=5, derivative_method="Savitzky-Golay", gmm_components=4,
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

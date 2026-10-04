"""
Run the whole satellite-deadband pipeline from one place.

    python main.py                          # every stage, figures + results.json -> ./outputs
    python main.py --out my_results         # choose the output folder
    python main.py --stages step6 gmm       # only some stages
    python main.py --list                   # show the available stages

Each stage lives in `stages/` and calls the library code in `deadband/`. Every stage uses the same
cached 400-day simulations (deadband/long_run.py), built on the first run (~20 minutes).
"""
import argparse
import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from deadband import settings
from stages import (deadband_stage, derivative_gmm_stage, detection_stage, gmm_stage, heldout_stage, inverse_stage,
                    neural_net_stage, smooth_deadband_stage, spectral_stage, tuning_stage)
from stages.common import LABEL_REPORT
from stages import (report_step1, report_step2, report_step3, report_step3_noise, report_step4,
                    report_step4_spectroscopy, report_step5, report_step6, report_step7, report_step8, report_step9)

# name -> (description, module), in the order they run. Every stage reads the SAME cached 400-day
# simulations (deadband/long_run.py) and writes into the folder of the report step it belongs to,
# outputs/report/stepN_*/. The workstream stages (Weeks 2-5) run right after their report step.
STAGES = {
    "tuning": ("Step 0    bias-variance tuning + walk-forward choice of the GMM derivative (runs first)", tuning_stage),
    "step1": ("Report 1  dynamic model, perturbations, data generation", report_step1),
    "step2": ("Report 2  learning coefficients by regression (mu, cd, joint fit)", report_step2),
    "neural_net": ("  + Workstream D  time-indexed neural network vs a straight line (natural run)", neural_net_stage),
    "step3": ("Report 3  drag estimation and noise robustness (FD, energy, shooting)", report_step3),
    "inverse": ("  + Workstream E  whole-trajectory fit of cd and J2, 3D cost surface", inverse_stage),
    "step3n": ("Report 3b realistic (coloured, multi-rate) noise vs white noise for every estimator", report_step3_noise),
    "step4": ("Report 4  spectral analysis, filtering, Kalman filter for cd", report_step4),
    "spectral": ("  + Workstream A  J2 SNR vs noise / cadence, 3D SNR surface and waterfall", spectral_stage),
    "step4s": ("Report 4b force spectroscopy: which perturbations, in what mixture, from the FFT", report_step4_spectroscopy),
    "step5": ("Report 5  deadband control with a smooth tanh thruster", report_step5),
    "smooth_deadband": ("  + Week 5         Figure 1, tanh switches, J2 mean vs osculating SMA (days 0-10)", smooth_deadband_stage),
    "deadband": ("  + Workstream B  400-day sawtooth, every burn, 3D hysteresis loop", deadband_stage),
    "step6": ("Report 6  coast/burn classifiers, train/test split, event scores", report_step6),
    "derivative_gmm": ("  + Week 5         GMM in rdot / vdot space, five derivative methods (days 0-10)", derivative_gmm_stage),
    "gmm": ("  + Workstream C/G 2-feature GMM in 2D and 3D, phase space coloured by GMM weight", gmm_stage),
    "heldout": ("  + Workstream C/G seven detectors on held-out days, GP coast curve, 3D phase fold", heldout_stage),
    "step7": ("Report 7  classifier robustness: noise, clusters, training size, sampling", report_step7),
    "detection": ("  + Workstream C/G five basic detectors vs noise and sampling step", detection_stage),
    "step8": ("Report 8  repetition: autocorrelation, stacking, GP thrust law", report_step8),
    "step9": ("Report 9  SINDy / BINDy control law and burn forecasting", report_step9),
}


HEARTBEAT_SECONDS = 5   # how often to print a status line when output is not a terminal


def _format_elapsed(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes:02d}:{seconds:02d}"


@contextmanager
def status(label, run_started):
    """Show how long `label` (and the whole run) has been going while the block executes.

    In a terminal the line updates in place every second. Where output is not a terminal
    (e.g. the VS Code Code Runner output panel) a status line is printed every few seconds.
    """
    live = sys.stdout.isatty()
    stage_started = time.time()
    stop = threading.Event()

    def tick():
        interval = 1 if live else HEARTBEAT_SECONDS
        while not stop.wait(interval):
            now = time.time()
            text = (f"    ... {label}: {_format_elapsed(now - stage_started)}"
                    f"  (total {_format_elapsed(now - run_started)})")
            print(f"\r{text}" if live else text, end="" if live else "\n", flush=True)

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        if live:
            print("\r\033[K", end="", flush=True)   # clear the in-place status line


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) 
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value)}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="outputs", help="folder for figures and results.json (default: outputs)")
    parser.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES),
                        help="stages to run (default: all)")
    parser.add_argument("--list", action="store_true", help="list the stages and exit")
    parser.add_argument("--retune", action="store_true",
                        help="delete the tuned settings and redo the bias-variance studies (several minutes)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.list:
        for name, (description, _) in STAGES.items():
            print(f"{name:<16} {description}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.retune and settings.SETTINGS_FILE.exists():
        settings.SETTINGS_FILE.unlink()
    if not settings.is_tuned() and "tuning" not in args.stages:
        print("Note: the tuning stage has not run yet, so the hand-picked default settings are used.\n")

    run_started = time.time()
    sim = None   # no stage uses the old Week 4 run any more: every stage reads deadband/long_run.py

    results = {}
    for index, name in enumerate(args.stages, start=1):
        description, module = STAGES[name]
        print(f"[{index}/{len(args.stages)}] [{name}] {description}", flush=True)
        started = time.time()
        with status(name, run_started):
            results.update(module.run(sim, out_dir))
        print(f"    done in {time.time() - started:.1f} s  (total {_format_elapsed(time.time() - run_started)})")

    results["settings_used"] = settings.all_settings()
    if LABEL_REPORT:
        report = "\n".join(f"{name}: '{panel}' has no {missing}" for name, panel, missing in LABEL_REPORT)
        (out_dir / "figure_label_check.txt").write_text(report)
        print(f"\n{len(LABEL_REPORT)} chart(s) lack a title or axis label: see figure_label_check.txt")
    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=_json_default))
    print(f"\nFigures and {results_path.name} written to {out_dir.resolve()}")
    print(f"Total run time {_format_elapsed(time.time() - run_started)}")


if __name__ == "__main__":
    started = time.time()
    try:
        main()
    except KeyboardInterrupt:
        print(f"\nStopped by user after {_format_elapsed(time.time() - started)}.")
        sys.exit(130)

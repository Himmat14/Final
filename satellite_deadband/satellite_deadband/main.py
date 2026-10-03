"""
Run the whole satellite-deadband pipeline from one place.

    python main.py                          # every stage, figures + results.json -> ./outputs
    python main.py --out my_results         # choose the output folder
    python main.py --stages deadband gmm    # only some stages
    python main.py --list                   # show the available stages

Each stage lives in `stages/` and calls the library code in `deadband/`.
"""
import argparse
import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from deadband.constants import PERIOD
from deadband.controller import simulate_deadband
from stages import (deadband_stage, detection_stage, gmm_stage, heldout_stage, inverse_stage,  # noqa: F401
                    neural_net_stage, spectral_stage)   # (some are commented out of STAGES below)
from stages import derivative_gmm_stage, smooth_deadband_stage   # noqa: F401  (run inside report steps 5-6)
from stages import (report_step1, report_step2, report_step3, report_step4, report_step5, report_step6,
                    report_step7, report_step8, report_step9)

N_ORBITS = 8   # length of the main simulated run used by most stages

# name -> (description, module). Order matters only for the printed output.
STAGES = {
    # Week 4 stages not used in the Week 5 talk -- uncomment a line to bring it back.
    # "deadband": ("Workstream B   state-dependent deadband controller", deadband_stage),
    # "inverse": ("Workstream E   recover cd and J2 from trajectory data", inverse_stage),
    # "detection": ("Workstream C/G  burn vs coast: five basic detectors, noise and cadence sweeps", detection_stage),
    # "neural_net": ("Workstream D   neural-network long-horizon extrapolation test", neural_net_stage),
    # Run inside report steps 4, 5 and 6 now (their figures land in outputs/report/...):
    # "spectral": ("Workstream A   FFT / J2 spectral-peak robustness", spectral_stage),
    # "gmm": ("Workstream C/G  2-feature GMM visuals and GMM vs k-means regions", gmm_stage),
    # "heldout": ("Workstream C/G  60-orbit train/test split, more models, Gaussian Process", heldout_stage),
    # "smooth_deadband": ("Week 5          smooth tanh deadband with J2, Figure 1", smooth_deadband_stage),
    # "derivative_gmm": ("Week 5          burn detection: GMM in rdot / vdot space", derivative_gmm_stage),
    # Report: nine steps, each writes step-by-step figures + a summary figure to outputs/report/stepN_*/
    "step1": ("Report 1  dynamic model, perturbations, data generation", report_step1),
    "step2": ("Report 2  learning coefficients by regression (mu, cd, joint fit)", report_step2),
    "step3": ("Report 3  drag estimation and noise robustness (FD, energy, shooting)", report_step3),
    "step4": ("Report 4  spectral analysis, filtering, Kalman filter for cd", report_step4),
    "step5": ("Report 5  deadband control with a smooth tanh thruster", report_step5),
    "step6": ("Report 6  coast/burn classifiers, train/test split, event scores", report_step6),
    "step7": ("Report 7  classifier robustness: noise, clusters, training size, sampling", report_step7),
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
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.list:
        for name, (description, _) in STAGES.items():
            print(f"{name:<16} {description}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_started = time.time()
    print(f"Simulating {N_ORBITS} orbits under the deadband controller...", flush=True)
    with status("simulation", run_started):
        sim = simulate_deadband(PERIOD * N_ORBITS)
    print(f"  {len(sim.t)} samples, {sim.n_burns} burn pairs\n")

    results = {}
    for index, name in enumerate(args.stages, start=1):
        description, module = STAGES[name]
        print(f"[{index}/{len(args.stages)}] [{name}] {description}", flush=True)
        started = time.time()
        with status(name, run_started):
            results.update(module.run(sim, out_dir))
        print(f"    done in {time.time() - started:.1f} s  (total {_format_elapsed(time.time() - run_started)})")

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

# Satellite deadband control: physics, burn detection and parameter recovery

Simulates a Starlink-style satellite (550 km, 53 deg) under gravity, drag and J2 with a
state-dependent **deadband station-keeping controller**, then asks how well simple
machine-learning and estimation methods can recover what is going on from the trajectory:

* **Inverse problem** - recover the drag coefficient `cd` (and `cd` + J2 together).
* **Burn detection** - separate thruster (burn) samples from coasting using Gaussian mixture
  models, k-means, Isolation Forest, a Gaussian Process, and others.
* **Spectral method** - find the J2 signature at twice the orbital frequency, and see when
  noise or slow sampling hides it.
* **Neural network** - show that a time-indexed network cannot extrapolate.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python main.py                    # run everything -> ./outputs  (about 30 s)
python main.py --list             # see the stages
python main.py --stages deadband gmm --out my_results
python -m unittest discover tests # quick sanity checks
```

`main.py` writes 24 figures plus `results.json` (every number quoted in the Week 4 deck) to
the output folder. Nothing is hand-edited.

## Layout

```
main.py                      single entry point
deadband/                    the library (no plotting)
    constants.py             constants, unit system (DU/TU/VU), orbit + controller settings
    physics.py               accelerations, osculating SMA, propagation
    controller.py            state-dependent raise + trim deadband controller
    inverse.py               fit cd / J2 to trajectory data (Levenberg-Marquardt)
    spectral.py              FFT / periodogram robustness study
    neural_net.py            long-horizon MLP extrapolation test
    evidence.py              burn evidence signals (log |delta SMA|, log |delta speed|)
    metrics.py               sensitivity / precision / balanced accuracy
    detectors.py             five basic 1-D detectors + noise sweep
    gmm_2d.py                2-feature GMM and density / ellipse geometry
    method_comparison.py     GMM vs k-means regions, precision/recall, cadence sweep
    heldout_models.py        60-orbit train/test split, more models, Gaussian Process
stages/                      one module per workstream: run -> save figures -> return numbers
tests/test_core.py           sanity checks on the physics and controller
```

| Stage (`--stages`) | Workstream | Figures |
|---|---|---|
| `deadband` | B: controller fix | `deadband_state_dependent`, `deadband_raise_trim_dv` |
| `inverse` | E: cd, cd + J2 | `inverse_cd_sweep` |
| `detection` | C/G: basic detectors | `detect_*`, `activation_timeline`, `jump_signal_histograms`, `methods_timeline_comparison`, `cadence_detection_sweep`, `precision_recall_noise` |
| `neural_net` | D | `nn_long_horizon` |
| `spectral` | A | `spectral_*` |
| `gmm` | C/G extended | `gmm_*`, `decision_regions_grid`, `orbit_map_detections` |
| `heldout` | C/G further extended | `traintest_split_timeline`, `clearer_cluster_comparison`, `gp_*`, `method_comparison_bars` |

## Conventions

* The simulation is **non-dimensional**: length unit `DU` = Earth radius, time unit `TU` =
  1 / Earth spin rate. Convert with `to_hours`, `to_minutes`, `from_hours`, `from_minutes`
  in `deadband/constants.py`. Names ending in `_km`, `_ms`, `_hours` are dimensional.
* The controller has **one trigger** (osculating SMA <= lower edge) and fires a raise burn
  then a small opposing trim burn at the same instant. No timers.
* Burn detectors work on `log(|jump| + eps)`; a Gaussian mixture on the raw jumps collapses.
* Held-out scores use a **chronological** split (first 70% train, last 30% test).

## Week 5 additions

Two new stages, built on two new library modules. Nothing from Week 4 was changed, so
every earlier figure and number is reproduced exactly.

| Stage (`--stages`) | What it shows | Figures |
|---|---|---|
| `smooth_deadband` | tanh deadband controller with J2; Figure 1 of the proposal | `smooth_*` |
| `derivative_gmm` | burn detection by a GMM on finite-difference accelerations | `deriv_*` |

**`deadband/smooth_controller.py` - the deadband law with no `if` statement.**
The state gains a 7th entry, the thruster on/off state `s`. It moves smoothly from 0 to 1
when the mean SMA reaches the lower edge, holds at 1 until the upper edge, then returns to 0.
Every switch is `(1 + tanh(x / width)) / 2`, whose slope is `sech^2(x / width) / (2 width)`.
`a_thrust` applies the thrust along the velocity. The whole run is one `solve_ivp` call
(LSODA, which switches to a stiff method by itself, `rtol = atol = 1e-11`) with no events.
With J2 the osculating SMA swings ~12 km per orbit, so the controller watches a
**mean SMA**: the SMA computed from the energy *including* the J2 potential. That stays
constant under J2 and only drag and thrust change it.

**`deadband/derivative_detection.py` - detection in rdot / vdot space.**
Raw r and v are sampled at a fixed cadence and finite-differenced
(`r'' ~ (r[k+1] - 2r[k] + r[k-1]) / h^2`, `v' ~ (v[k+1] - v[k-1]) / 2h`). The known
gravity + J2 is subtracted, and a 2-component GMM is fitted on log |what is left|.
Scores are given per sample *and* per burn event (found? fraction covered? start/end
timing error? false events?) with a timing tolerance.

Settings (band edges, drag, thrust, switch width and time, solver) are at the bottom
of `deadband/constants.py`.

### Week 5 talk configuration

* `main.py` runs only `spectral`, `smooth_deadband` and `derivative_gmm`. The other Week 4
  stages are commented out in `STAGES`; uncomment a line to bring one back. Their code is unchanged.
* Every Week 5 plot and the GMM detector sample the state every `SAMPLE_STEP_MIN = 0.1` minutes
  (6 s; `deadband/constants.py`). The one exception is the solver step-size plot.
* The GMM inputs are r and v taken straight from the state vector, finite-differenced.
* GMM flags are grouped into **events**: gaps of up to 2 samples are filled and runs shorter
  than 10 samples (1 min) are dropped, so single points are never detections. Performance is
  scored per event only: true burns flagged, false events, precision / recall / F1.
* The spectral SNR now takes its noise floor from 0-6 cycles/orbit only
  (`BACKGROUND_MAX_CYCLES_PER_ORBIT` in `stages/spectral_stage.py`). Without that, a 0.1 min
  cadence fills the median with high-frequency finite-difference noise and the SNR wrongly
  collapses.

## Report figures (nine steps)

`python main.py` now runs nine report stages. Each one writes its step-by-step figures and a
six-panel `stepN_summary.png` into `outputs/report/stepN_<topic>/`. The whole run takes about 9
minutes. The first run also builds a cached 20-day dataset in `.cache/`, which takes about 1 minute.

| Step | Folder | What it shows | Library code |
|---|---|---|---|
| 1 | `step1_dynamics` | state-space equations, orbit and ground track, force budget, one-week effect of J2/Moon/Sun/SRP, the 7 states through a burn, solver cross-check | `perturbations.py` |
| 2 | `step2_regression` | mu vs sampling step, FD error vs step and stencil order, drag fit, five-term joint fit and column correlation | `regression.py` |
| 3 | `step3_drag_noise` | Monte Carlo noise study, FD vs energy vs shooting for cd, cadence x noise sensitivity maps, shooting cost landscape | `regression.py`, `inverse.py` |
| 4 | `step4_spectral_kalman` | spectrum of each force, J2 SNR studies, band/low-pass filtering, Kalman filter for cd vs the other methods | `spectral.py`, `kalman.py` |
| 5 | `step5_deadband` | Figure 1 reproduction, tanh switch, hysteresis loop, thrust attenuation vs switch time, impulse vs smooth delta-v, fuel vs band width | `smooth_controller.py` |
| 6 | `step6_classification` | 20-day dataset with train/test split, r'' vs v' noise floors, feature space, per-method test events, event F1, timing errors | `classifiers.py` |
| 7 | `step7_robustness` | F1 and false alarms vs noise, method x noise heatmap, number of GMM components, training-set size, sampling step | `classifiers.py` |
| 8 | `step8_repetition_gp` | autocorrelation period, stacked noisy burns, 1/sqrt(N) averaging, GP thrust law, folded coast arcs | `burn_folding.py` |
| 9 | `step9_sindy_control_law` | library collinearity, SINDy and BINDy-style coefficients, recovered cd and thrust, switching law with and without memory, free-running burn forecast vs GP / random forest / periodic baseline | `sindy.py` |

Every module starts with a plain-English docstring explaining the method. Read those first when
reviewing the code.

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

python main.py                    # run everything -> ./outputs  (first run ~20 min to build the 400-day caches)
python main.py --list             # see the stages
python main.py --stages step6 gmm --out my_results
python -m unittest discover tests # quick sanity checks
```

`main.py` runs 34 stages (step 0 tuning first, steps 1-22 with the workstream stages after their step) and writes every figure
into `outputs/report/stepN_*/` and every number into `outputs/results.json`. Nothing is hand-edited.
Every multi-panel figure is also cut into its single panels (`outputs/report/stepN_*/panels/<name>_a.png`, ...), and `outputs/report/figure_index.json` lists each figure's panels with their titles and axis labels. The LaTeX report `report/briefing.tex` uses only single-panel figures.

## Step 0: bias-variance tuning (runs first, results carried forward)

`python main.py` starts with the `tuning` stage (`deadband/tuning.py`, figures in
`outputs/report/step0_tuning/`). Each adjustable setting is repeated over many noise seeds or bootstrap
re-fits, and its error is split into **bias^2 + variance = MSE**. The setting with the lowest MSE is
saved to `.cache/tuned_settings_v1.json`. Every later step reads it through `deadband/settings.py`
(`tuned("...")`), so the optimum is carried forward automatically. Delete that file or run
`python main.py --retune` to tune again.

| Setting | Default | Tuned | How it was chosen |
|---|---|---|---|
| `fd_order` (steps 2-4) | 6 | per step: 1 min 4, 2 min 2, 5 min 6, 10 min 6 | MSE of cd from 10 m noisy positions at each sampling step (`settings.fd_order_for(step)`; one global order was wrong for the 10-min Kalman comparison) |
| `sg_window`, `sg_order` (steps 6-9) | 21, 5 | 61, 5 | MSE of the classifier's v' (coast and burn weighted equally) |
| `sg_window_stacked` (step 8) | 21, 5 | 41, 5 | bias^2 + noise^2 x variance / N: averaging N = 92 burns removes variance, not bias, so a shorter window wins |
| `derivative_method` (steps 6-9) | Savitzky-Golay | Savitzky-Golay | walk-forward / walk-backward event F1 (below) |
| `gmm_components` (all GMMs) | 4 | 4 | bias^2 + variance of P(burn \| x) (Brier score), one-standard-error rule |
| `sma_window` (step 9) | 41 | 41 | MSE of the learned coast and burn RATES (what drives the burn forecast) |
| `stlsq_threshold` (step 9) | 0.05 | 0.05 | MSE of the SINDy law's predicted da/dt on TEST (the sparsest law within 1%) |

**Choosing the GMM's derivative method by walk-forward / walk-backward testing.** For each way of
differentiating the velocity data (Savitzky-Golay, central 2nd/4th/6th/8th order), a GMM is trained on
a 40-day window and tested out of sample over the 15, 30, 60, 120, 240 and 360 days that follow it
(**forward**: train on days 0-40) or that come before it (**backward**: train on days 360-400). This is
done at 0.25, 0.5 and 1 x the reference noise. At the reference noise only Savitzky-Golay keeps
F1 = 1.0 in all 12 periods. The central differences already fail at 0.5 x (8th order even at 0.25 x),
because they amplify the noise.

## Step 6 stress test: largest usable sampling step

`deadband/stress.py`, figures `step6_step_stress*`. The step 6 GMM is run on noisy data (0 to 4 x the reference
noise) sampled every 6 s up to 60 min, with two ways of measuring the unmodelled acceleration:
* the step 6 Savitzky-Golay derivative works up to a **2 min** step (12 s at 4 x noise);
* propagating each state one step with gravity + J2 and dividing the velocity mismatch by the step works
  from about 1 min up to **20 min** at every noise level, because its noise shrinks as the step grows. Beyond
  one burn length (23 min) the burn is diluted into one interval and false events appear.

Switching between the two features at about 2 min gives F1 >= 0.9 from 6 s to 20 min at every noise level tested.

## Extensions towards the UK Space Agency brief (steps 10-18)

Steps 10-16 run on a **mean-element model** (`deadband/mean_element.py`): only the mean SMA and the
thruster state, `da/dt = -k w(t) exp(-(a - a_ref)/H) + T s` with the same hysteresis law. It is calibrated
on the full 6-s controlled run (`calibrate()`: coast rate, thrust and the effective band edges) and
reproduces it to 0.01% in burn period (`step10_model_check.png`, `tests/test_extensions.py`). It makes 400-day
scenarios run in about a second.

| Step | Folder | What it shows | Library |
|---|---|---|---|
| 10 | `step10_space_weather` | variable drag (27-day rotation, trend, storms): with constant drag the periodic baseline wins (2-3 min vs 70-630 min); with space-weather drag it misses by ~27 h, the law with a space-weather proxy ~12 h at 15 days, and with a perfect space-weather forecast ~3 h | `mean_element.py`, `forecasting.py` |
| 11 | `step11_observations` | 8-min tracking passes (12-s positions), per-pass orbit fits (batched Gauss-Newton), burns found in the gaps: F1 ~1 at 10 m noise for 1-8 passes/day; at 100 m only the multi-pass test finds some (F1 0.47 at 8/day); OEM reader for real ephemerides (`data/ephemeris/`) | `observation.py` |
| 12 | `step12_conjunction` | along-track error at TCA (drag-only prediction: 19 km at 1 day, ~1500 km at 14 days; learned law + space weather: 6-12 km) and Pc screening: missed dangerous conjunctions and false alerts, honest vs catalogue-only covariance | `conjunction.py` |
| 13 | `step13_tasking` | custody loss after burns (20 km association gate) vs observation budget: random 74% at 1 obs/day, manoeuvre-aware 48%, and 15-19% when the catalogue also carries the predicted burn | `conjunction.py` |
| 14 | `step14_regimes` | orbit raising, station keeping, a band change and collision-avoidance burns on one satellite: 105/105 burns labelled correctly; the band change found at day 150.0 | `regimes.py` |
| 15 | `step15_fleet` | 40 satellites: the three planted pattern-of-life changes flagged, no false flags; pooled (empirical-Bayes) drag estimates help with 4-8 days of data | `fleet.py` |
| 16 | `step16_conformal` | split-conformal 90% burn-time intervals: 93-99% coverage on new data, interval widths per forecaster | `forecasting.py` |
| 17 | `step17_geo` | GEO east-west / north-south boxes, chemical and electric propulsion; SINDy recovers the triaxiality law (A 0.00171 vs 0.0017, lambda_s 74.98 vs 75.07 deg) from a free drift; impulses detected from 6-hourly longitudes; E/W burns forecast 0.1-1.3 days off over 230 days | `geo.py` |
| 18 | `step18_sindy_zoo` | ten candidate libraries per dataset (full-physics LEO, space-weather LEO, GEO): only the proxy libraries forecast variable drag; extra terms are pruned on the full-physics data; collinear "kitchen sink" libraries diverge; relative vs significance STLSQ thresholds fail in opposite ways | `sindy_zoo.py` |
| 19 | `step19_sindy_split` | thrust law vs natural dynamics learned together (A), thrust only with drag known (B), drag only with thrust known (C) or separately on clean arcs (D), against the true model: joint learning biases T by -7.6%; separate learning recovers c_d and T to <0.1%; knowing the drag gives a 26-min error after 360 days. No deadband: the sqrt(a) / physics-shape law recovers c_d to 0.01% and the SMA a year ahead to 1 m | `sindy_split.py` |
| 20 | `step20_pipeline` | the whole chain end to end (measured r, v -> GMM -> mean SMA -> c_d, T, edges -> forecast): stage accuracy, an error budget swapping the truth into one stage at a time (all-true sanity check 1.4 min; the band edges dominate), the GMM components in detail, the chain re-run at 6 s to 10 min (F1 = 1 and c_d within 0.08% at every step; edges degrade from 5 min), GP thrust law vs step, space-weather forecast vs cadence, and 18 closed-form checks of the headline numbers. Every figure is a single panel | `pipeline.py`, `report_checks.py` |
| 21 | `step21_deep_dive` | the subtraction against machine epsilon (numerical floor 1e-8 m/s^2, 80x below drag, set by the Savitzky-Golay truncation), the SG filter in the frequency domain, why (x1, x2) and the hat-box theorem, the log-chi (fat-left-tailed) coast and better mixtures (log-chi coast + Gaussian burn: better BIC than the K = 4 GMM with 10 instead of 23 parameters), BIC / AIC vs K, the sigmoidal thrust model vs the GP, the controller as a 2-state ODE with its hysteresis direction, the SMA sawtooth, and PCA frames for the coasting dynamics (sparse but linear: they drift by tens of km a day; the 3-term physics library stays within 0.65 km after 10 days) | `gmm_physics.py`, `deep_dive.py`, `pca_coast.py` |
| 22 | `step22_full_forcing` | a second 400-day controlled run with every force (J2, drag, Moon, Sun, SRP): detection unchanged (F1 = 1); the third bodies modulate each coast, so the split-law forecast error at 360 d rises from 66 to 132 min (86 min when the analyst also models Moon / Sun / SRP) and even the periodic schedule drifts by 45 min. Also writes the onset-error tables at every horizon (outputs/report/tables/) | `full_forcing.py`, `report_tables.py` |

Also fixed: the GP detector in the `heldout` stage now picks its training coast samples from the GMM's
labels, not from the true labels.

## Figure style

Every figure goes through `stages/common.save_figure`:
* one font (DejaVu Sans, also for maths text) and fixed sizes for titles, labels, ticks and legends;
* the colour-blind-safe Okabe-Ito palette;
* major plus light minor gridlines;
* different line styles whenever a chart has three or more solid lines.

Charts missing a title or axis label are listed in `outputs/figure_label_check.txt`.

## One data source for everything

Every stage, including the Week 2-5 workstream stages, reads the same cached 400-day simulations
in `deadband/long_run.py`:

| Data | Used by |
|---|---|
| `natural_run()`: every force, realistic drag, 1-min samples | steps 1-4, `neural_net`, `inverse`, `spectral` |
| `controlled_run()`: smooth tanh deadband, 6-s samples (102 burns) | steps 5-9, `deadband`, `detection`, `gmm`, `heldout` |
| its first 10-day chunk (`smooth_controller.default_runs`) | `smooth_deadband`, `derivative_gmm` |

The settings are shared as well:
* Train/test split: TRAIN = days 0-40 and TEST = days 40+ (`TRAIN_DAYS`).
* Noise: in metres and mm/s, with the step 6 reference of 0.1 m and 0.5 mm/s.
* Scoring: per burn **event**. The classifier features are [log |v' residual|, along-track v' residual].

The old Week 4 run (8 orbits, impulsive raise + trim burns, demo drag `CD_TRUE`, 70/30 split, noise
as a fraction of the signal) is no longer used by any stage. `deadband/controller.py` stays only
for step 5's impulsive vs smooth comparison.

## Layout

```
main.py                      single entry point (python main.py --list shows the 20 stages in run order)
deadband/                    the library (no plotting)
    constants.py             constants, unit system (DU/TU/VU), orbit + controller settings
    long_run.py              the cached 400-day simulations every stage uses
    physics.py, perturbations.py   accelerations (gravity, J2, drag, Moon, Sun, SRP) and propagation
    smooth_controller.py     the tanh deadband controller (7th state = thruster on/off)
    controller.py            the Week 4 impulsive controller (step 5 comparison only)
    regression.py, kalman.py, inverse.py   learning cd, J2, ... from positions (steps 2-4)
    noise_models.py          realistic multi-frequency noise (step 3b)
    spectral.py, force_spectroscopy.py     FFT / J2 SNR and "force spectroscopy" (step 4)
    derivative_detection.py  derivative methods, derivative GMM, events and event scores
    classifiers.py           step 6-7 dataset, features, models, evaluation
    detectors.py             five basic detectors on the step 6 features (workstream C/G)
    gmm_2d.py                2-feature GMM, burn probability, density / ellipse geometry
    method_comparison.py     GMM vs k-means decision regions, clean vs noisy
    heldout_models.py        seven detectors on held-out days, GP coast-curve anomaly detector
    neural_net.py            time-indexed MLP vs a straight line (natural run)
    burn_folding.py, sindy.py   repetition / GP thrust law (step 8), SINDy / BINDy (step 9)
stages/                      one module per stage: run -> save figures -> return numbers
tests/                       sanity checks (python -m unittest discover tests)
```

| Stage (`--stages`) | Runs after | Folder | Figures |
|---|---|---|---|
| `neural_net` | step 2 | `step2_regression` | `nn_long_horizon`, `nn_horizon_errors` |
| `inverse` | step 3 | `step3_drag_noise` | `inverse_cd_sweep`, `inverse_cost_surface_3d`, `inverse_cost_contours` |
| `spectral` | step 4 | `step4_spectral_kalman` | `spectral_snr_vs_*`, `spectral_periodogram_examples`, `spectral_snr_surface_3d`, `spectral_waterfall_3d` |
| `smooth_deadband` | step 5 | `step5_deadband` | `smooth_*` (Figure 1, switches, J2 mean vs osculating SMA, ...) |
| `deadband` | step 5 | `step5_deadband` | `deadband_state_dependent`, `deadband_raise_trim_dv`, `deadband_hysteresis_3d` |
| `derivative_gmm` | step 6 | `step6_classification` | `deriv_*` (r'' / v' GMM, five derivative methods, Week 4 features, cadence sweep) |
| `gmm` | step 6 | `step6_classification` | `gmm_density_surface_3d`, `gmm_burn_probability_3d`, `gmm_phase_space_3d`, `gmm_orbit_weighting_3d`, `gmm_scatter_3d`, `gmm_contour_ellipses`, `decision_regions_grid`, `orbit_map_detections` |
| `heldout` | step 6 | `step6_classification` | `traintest_split_timeline`, `clearer_cluster_comparison`, `gp_phase_fold`, `gp_phase_fold_3d`, `gp_zscore_timeline`, `method_comparison_bars` |
| `detection` | step 7 | `step7_robustness` | `detect_*`, `precision_recall_noise`, `cadence_detection_sweep`, `activation_timeline`, `jump_signal_histograms`, `methods_timeline_comparison` |

### 3D figures

| Figure | What it shows |
|---|---|
| `gmm_density_surface_3d` | the fitted mixture density over the feature plane, plus a zoom on the burn component |
| `gmm_burn_probability_3d` | P(burn \| x): 0 over the coast, a cliff up to 1 at the burn component |
| `gmm_phase_space_3d` | the phase space (mean SMA, d(mean SMA)/dt, along-track acceleration), each sample coloured by its GMM weight P(burn \| x) |
| `gmm_orbit_weighting_3d` | the orbit in space around a TEST burn, coloured by P(burn \| x) |
| `deadband_hysteresis_3d` | the controller's hysteresis loop in (mean SMA, rate, on/off state) |
| `inverse_cost_surface_3d` | the whole-trajectory cost over (cd, J2) |
| `spectral_snr_surface_3d` | J2 SNR over (sampling step x noise) |
| `spectral_waterfall_3d` | the spectrum near 2 cycles/orbit vs record length |
| `gp_phase_fold_3d` | every coast arc side by side |
| `gmm_scatter_3d` | time + both features |

## Conventions

* The simulation is **non-dimensional**: length unit `DU` = Earth radius, time unit `TU` =
  1 / Earth spin rate. Convert with `to_hours`, `to_minutes`, `from_hours`, `from_minutes`
  in `deadband/constants.py`. Names ending in `_km`, `_ms`, `_hours` are dimensional.
* The controller watches the **mean SMA**: the energy including the J2 potential. Its on/off state
  is a 7th state variable with tanh switches (hysteresis, no timers).
* Detectors work on `log |unmodelled acceleration|`, because a Gaussian mixture on the raw values collapses.
  A burn component must sit 3 coast standard deviations above the coast mean.
* All scores use a **chronological** split (TRAIN days 0-40) and are counted per burn **event**.

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

* Every Week 5 plot and the GMM detector sample the state every `SAMPLE_STEP_MIN = 0.1` minutes
  (6 s; `deadband/constants.py`). The one exception is the solver step-size plot.
* The GMM inputs are r and v taken straight from the state vector, finite-differenced.
* GMM flags are grouped into **events**: gaps of up to 2 samples are filled and runs shorter
  than 10 samples (1 min) are dropped, so single points are never detections. Performance is
  scored per event only: true burns flagged, false events, precision / recall / F1.
* The spectral SNR now takes its noise floor from 0-6 cycles/orbit only
  (`BACKGROUND_MAX_CYCLES_PER_ORBIT` in `stages/report_step4.py`, shared by `spectral_stage.py`). Without that, a 0.1 min
  cadence fills the median with high-frequency finite-difference noise and the SNR wrongly
  collapses.

## Report figures (nine steps)

`python main.py` runs nine report stages. Each one writes its step-by-step figures and a
six-panel `stepN_summary.png` into `outputs/report/stepN_<topic>/`.

### 400-day data (`deadband/long_run.py`)

Every step now works on **400-day** simulations, built once and cached in `.cache/*.npz`
(about 500 MB in total; the first run takes roughly 20 minutes to build them, later runs read the cache).
Delete `.cache/` to rebuild, or change `CACHE_VERSION` in `long_run.py`.

| Run | Function | Content |
|---|---|---|
| natural | `natural_run()` | all forces (gravity, J2, drag, Moon, Sun, SRP), sampled every 1 min |
| divergence | `divergence_runs()` | a two-body baseline plus each force on its own, every 1 min |
| controlled | `controlled_run()` | smooth tanh deadband controller with J2 and realistic drag, every 6 s (102 burns) |
| drag only | `drag_only_run()` | the same orbit without the thruster, every 1 min |

The drag in these runs is `SMOOTH_CD` (about 0.13 km/day of decay). The Week 4 value `CD_TRUE`
lowers the orbit by about 45 km/day, so it cannot be flown for 400 days.

Metrics are reported over **15, 30, 60, 120, 240 and 360 days** (`HORIZONS_DAYS`). The
classification and learning steps (6-9) train on days 0-40 (`TRAIN_DAYS`) and test on days 40-400.
Each horizon is counted from the start of the test period.

### J2 check

J2 now uses the **equatorial** radius 6378.137 km (`EARTH_EQUATORIAL_RADIUS_KM`). Before this fix
it used the mean radius 6371 km, which made J2 about 0.22% too weak. Step 1 compares the simulated
nodal regression with the theory dOmega/dt = -1.5 n J2 (R_eq/a)^2 cos i, with a(t) taken from the run
because drag slowly lowers the orbit. After 400 days they agree to 0.04%. Step 1 also plots ground tracks.

| Step | Folder | What it shows | Library code |
|---|---|---|---|
| 1 | `step1_dynamics` | state-space equations, orbit, ground tracks, J2 nodal-regression check, force divergence vs horizon, the 7 states through a burn, solver cross-check | `perturbations.py`, `long_run.py` |
| 2 | `step2_regression` | mu vs sampling step, FD error vs step and stencil order, drag fit, five-term joint fit vs horizon, column correlation | `regression.py` |
| 3 | `step3_drag_noise` | Monte Carlo noise study, cd vs horizon (FD, energy, shooting), cadence x noise sensitivity maps, shooting cost landscape | `regression.py`, `inverse.py` |
| 3b | `step3_drag_noise` (`step3_noise_*`) | realistic noise (white at several rates + 1/f + random walk + periodic) vs white noise of the same RMS, for cd, J2, Moon, Sun, SRP and the energy cd | `noise_models.py` |
| 4 | `step4_spectral_kalman` | spectrum of each force at 15 and 360 days, J2 SNR vs horizon and cadence, filtering, Kalman filter for cd | `spectral.py`, `kalman.py` |
| 4b | `step4_spectral_kalman` (`step4_spectroscopy_*`) | "force spectroscopy": per-force spectra with diagnostic bands, spectral composition vs the force budget, band reading vs peak fit vs regression, speed | `force_spectroscopy.py` |
| 5 | `step5_deadband` | Figure 1 reproduction, 400-day station keeping, burns / delta-v / miss distance per horizon, hysteresis loop, fuel vs band width | `smooth_controller.py` |
| 6 | `step6_classification` | 400-day train/test split, feature space, per-method events, event F1 vs horizon | `classifiers.py` |
| 7 | `step7_robustness` | F1 and false alarms vs noise, method x noise heatmap, GMM components, training size, sampling step (days 0-60) | `classifiers.py` |
| 8 | `step8_repetition_gp` | autocorrelation period, stacked noisy burns, 1/sqrt(N) averaging, GP thrust law, learning vs horizon | `burn_folding.py` |
| 9 | `step9_sindy_control_law` | SINDy and BINDy-style coefficients, recovered cd and thrust, switching law, 360-day free-running forecast and its error vs horizon | `sindy.py` |

### Derivative methods, J2 check, spectroscopy and realistic noise

* **Burn GMM and derivative methods** (`derivative_detection.py`, figures `deriv_*_methods`, `deriv_method_floor`
  in step 6). With the 2nd-order stencil at 6 s, the coasting "unmodelled acceleration" is the stencil's own
  truncation error (6e-5 m/s^2 from v), not drag. 4th/6th/8th-order central differences and Savitzky-Golay
  bring it down to the real drag (8.1e-7 m/s^2). With 0.1 m / 0.5 mm/s noise, higher orders amplify the
  noise and only Savitzky-Golay works. The GMM uses velocity data only, with 2 features: log |v' residual|
  and its along-track part (thrust pushes forwards, noise does not). It has **4 components**: with 2,
  the burn Gaussian must also cover the thruster ramp samples, so it sits off the burn.
* **J2 figure** (`smooth_j2_mean_vs_osculating`). The 12.2 km osculating swing is real. It follows
  a_E - (J2 R^2/a)(3 sin^2 lat - 1) to within 11 m. The 500 m band applies to the mean SMA a_E, which sits a fixed
  (J2 R^2/a)(1 - 1.5 sin^2 i) = 275 m below the orbit-averaged osculating SMA. A unit test checks that the
  J2 force is minus the gradient of the J2 potential used for a_E.
* **Force spectroscopy** (`force_spectroscopy.py`, stage `step4s`). Spectral POWER shares do not match the
  force budget (power ~ amplitude^2). sqrt(power) shares do (Parseval). Reading diagnostic bands IR-style is
  30-40x faster than regression and reliable for drag, J2 and SRP. The Moon reads ~40% high and the Sun is
  not separable (their tidal lines overlap). A phase-aware fit is exact but needs the model, like regression.
* **Realistic noise** (`noise_models.py`, stage `step3n`). At the same RMS, realistic noise gives
  slightly better FD cd but is 2-18x worse for J2, Moon, Sun, SRP and the energy cd.

Every module starts with a plain-English docstring explaining the method. Read those first when
reviewing the code.

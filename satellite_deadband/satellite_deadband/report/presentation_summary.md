# Presentation summary: Manoeuvre Pattern-of-Life Prediction for SDA

**File:** `report/presentation.pptx` (37 slides, plain white, every slide numbered and titled, speaker notes on every
slide: 5 short points each). **Length:** 45 minutes of talk + 15 minutes of questions.

The talk follows the processing chain in the order the data flows through it:

```
0 Simulate  ->  1 Natural dynamics (c_d)  ->  2 Detect burns (GMM)  ->  3 Learn laws (GP, SINDy, BINDy)  ->  4 Forecast  ->  use in SDA
```

## Suggested timing

| Part | Slides | Minutes | Content |
|---|---|---|---|
| Opening | 1-3 | 4 | title, roadmap, the brief |
| A. Physics and simulation code | 4-14 | 14 | forces, J2 check, ODE solvers, mean SMA, deadband law and its code, 400-day data, state space, calculation checks |
| B. Drag and burn detection | 15-22 | 11 | drag estimators, the subtraction, Savitzky-Golay, features, GMM code and results, fat tails, robustness |
| C. Learning the laws | 23-26 | 6 | GP / sigmoid thrust law, SINDy / BINDy code, the four splits, switching law and forecast code |
| D. Forecast assessment | 27-31 | 6 | error budget, sampling robustness, all forces, operational extensions, negative results |
| E. Assessment and close | 32-37 | 4 | ranking, holistic assessment, limitations, conclusions, references, questions |

Code slides (5, 7, 8, 10, 11, 12, 16, 19, 22, 23, 24, 26) are meant to be walked through quickly: point at the 2-3
lines that matter (named in the right-hand bullets) rather than reading the code.

---

## Opening (slides 1-3)

**1. Title.** The project in one line: learn a satellite's station-keeping control law from noisy tracking data and use
it to forecast future burns. The figure is the deadband "sawtooth" of the mean semi-major axis (SMA).

**2. Roadmap.** The five boxes are the chain the whole talk follows. Stage 0 creates a known truth; stages 1-3 estimate
from noisy data; stage 4 forecasts; the bottom box is where the forecasts are used. The agenda lists which slides cover
which part.

**3. The brief.** The UK Space Agency challenge: unpredicted manoeuvres break conjunction screening, track association
and sensor scheduling. The brief's Figure 1 (a 500 m deadband moving the miss distance by hundreds of km a week) is
reproduced from our own simulation. The research question: can the control law be learned well enough to forecast?

## Part A: physics and simulation (slides 4-14)

**4. Physics model.** The equations of motion: two-body gravity, J2, drag, Moon, Sun, solar radiation pressure (SRP) and
thrust. The bar chart is the force budget: J2 is four orders above drag and the third bodies; the thrust (2e-4 m/s^2)
sits between J2 and drag, which is why it becomes visible once J2 is removed.

**5. Code: the force model.** The three core force functions from `deadband/physics.py`. Each is a pure function of the
state; the J2 term uses the equatorial radius (6378.137 km) - using the mean radius made J2 0.22% too weak, a bug the J2
check caught.

**6. J2 check.** The simulated node angle (from the angular momentum vector) against the analytical secular rate
dOmega/dt = -(3/2) n J2 (Re/p)^2 cos i, integrated along the run with one-orbit means of a and i. Agreement 0.04% after
400 days. The rate changes from -4.526 to -4.642 deg/day because drag lowers the free orbit by 51 km, which is why it is
integrated rather than evaluated once.

**7. ODE solvers.** DOP853 (explicit 8th-order Runge-Kutta, rtol = atol = 1e-10) for free flight; LSODA (automatic
switch between non-stiff Adams and stiff BDF, 1e-11) for the controlled run, because the tanh switches are stiff for
seconds at each burn edge. A 2-day cross-check against a 1e-13 reference gives 0.27 m (DOP853), 28 m (LSODA) and 0.013 m
(Radau). States are read from the solver's dense output.

**8. The mean SMA.** The osculating SMA swings 12.2 km per orbit under J2 (25 times the band), so a controller watching it
fires every orbit. Adding the J2 potential to the orbital energy gives a "mean" SMA that only drag and thrust change. The
code (`j2_potential`, `mean_sma`) is four lines. Theory check of the swing: 12.17 vs 12.22 km.

**9. The deadband law.** The ideal operator law is a piecewise switch with memory (on below the lower edge, off above the
upper edge, otherwise keep). Each Heaviside step is replaced by a tanh sigmoid (5 m wide); the thruster state s gets its
own ODE (60 s time constant) with a latch term that makes s = 0 and s = 1 the two stable states. The throttle opens only
past s = 0.9 so a burn cannot stall half-way.

**10. Code: the deadband logic.** The four functions that implement slide 9: `rising_switch`, `throttle`, `a_thrust`,
`thruster_state_rate`. Three sigmoids replace three if-statements.

**11. Code: the right-hand side.** The full 7-state ODE f(t, [r, v, s]): a physics block (gravity, drag, J2, and Moon /
Sun / SRP when requested, evaluated at absolute time) and a control block (thrust and ds/dt from the mean SMA). The
`controlled` flag is a model choice made once, not a discontinuity.

**12. 400 days of data.** The run is integrated in 40 chunks of 10 days, sampled every 6 s from the dense output and cached.
Outcome: 102 burns, one every 3.9051 days, 22.8 min each, 0.273 m/s per burn. Days 0-40 are used for training, days
40-400 for testing.

**13. State space and hysteresis.** Reduced to the two slow states (a, s), the controller is a 2-D ODE with two stable
branches (coast at -5.3 m/h, burn at +1311 m/h); at the band edges one branch loses stability and the state jumps. The
vector field and the true cycle show a clockwise loop. Because of this memory, a memory-less law s = f(a) scores 0.50
balanced accuracy (chance) against 1.00 for the hysteresis.

**14. Calculation checks.** Sixteen headline numbers recomputed from closed-form physics: decay rate, burn climb, burn
duration, period, delta-v, J2 swing, Nyquist limit, the drift of a missed burn (495 km/week bound vs 396 km simulated),
the forecast-error law. All 16 pass (14 within 1%).

## Part B: drag estimation and burn detection (slides 15-22)

**15. Drag coefficient.** Four estimators: finite-difference regression (two derivatives, noise amplified ~ sigma/h^2),
the energy method (one slope through the mean SMA), shooting and an extended Kalman filter. At 10 m noise FD is 6.3% off,
the energy method 0.035%; the EKF on 10-min data reaches 0.04% / 0.01% after 15 / 30 days.

**16. The unmodelled acceleration.** The feature is the measured dv/dt minus modelled gravity and J2: what remains is drag +
thrust + noise + numerical error. Because this cancels seven decades, every numerical error source was measured: machine
epsilon contributes 1.8e-15, the Savitzky-Golay truncation 8.3e-9, the total floor 1.0e-8 m/s^2 - 80 times below the
drag. The code is `build_features`.

**17. Savitzky-Golay.** A least-squares polynomial fit through a window, differentiated at the centre, is a fixed
convolution with a known frequency response: it equals the ideal differentiator at low frequency (no bias on the orbit)
and rolls off above a cut-off (~2.8-min period for the tuned 61 samples, order 5), rejecting the noise that finite
differences amplify. The price is that 30-s thrust ramps appear 60 s long.

**18. Features x1, x2.** x1 = log of the acceleration magnitude (compresses four decades so both clusters have similar
widths); x2 = along-track component (thrust points along the velocity; noise does not). Every sample lies in the wedge
|x2| <= e^x1 / 1e-4; burns sit on its upper edge.

**19. Code: the GMM.** `GaussianMixture` with K = 4, fitted by expectation-maximisation on 20,000 training samples
(unsupervised). The burn rule: a component is "burn" if its mean x1 lies more than 3 coast standard deviations above the
coast mean. `group_into_events` turns sample flags into burn events (fill gaps of <= 2 samples, drop runs < 1 min).

**20. GMM results.** Three coast components and one burn component; the burn weight (0.0039) matches the burn duty cycle
(0.0041). All 92 test burns found with no false events (F1 = 1.0), mean start error 14 s. k-means fails under noise
(F1 0.001). The posterior figure shows the noisy signal, the noise-free signal, the true thrust and P(burn).

**21. Fat tails and better mixtures.** The log of a 3-D Gaussian vector's length has a "log-chi" distribution: a fat
exponential left tail and a very thin right tail - exactly the skewed coast cloud the GMM approximates with three
Gaussians. A physics-shaped mixture (log-chi coast + Gaussian burn, 10 parameters) beats the 23-parameter GMM on BIC. BIC
keeps falling with K because extra Gaussians only reshape the skewed coast, which is why K was chosen by detection
quality.

**22. Detection robustness.** The GMM keeps F1 = 1 from 0.25 to 8 times the reference noise. The SG derivative works up to a
2-min step; a one-step propagation feature (propagate each state one step with RK4 under gravity + J2 and compare with the
next measurement; code shown) works from ~1 to 20 min. The whole chain detects every burn from 6 s to 10 min.

## Part C: learning the laws (slides 23-26)

**23. Thrust law from repetition.** Burns repeat, so all detected burns are stacked on their start time. A Gaussian process
(RBF + white-noise kernel, hyper-parameters by marginal likelihood, 1,500 random points because the cost is O(n^3)) gives
a smooth thrust law with an uncertainty band: plateau 0.999 of the truth, duration exact. A 6-parameter sigmoid law (the
same form as the simulator's thrust model) fits 14% better and is interpretable.

**24. SINDy and BINDy.** SINDy writes the SMA rate as a sparse combination of library terms, found by sequentially
thresholded least squares (code shown). The physics library [drag shape, thrust shape, constant] is singular over a 500 m
band, so cluster indicators [1-s, s, ...] are used: c_d 1.0005 and T 0.9997 of the truth. BINDy here is ARD regression,
which adds posterior standard deviations.

**25. Four ways to split the problem.** A: one regression over everything (biased: T -7.6%, 1668 min error after a year);
B: drag known (26 min); C: thrust known (302 min); D: both learned separately on clean arcs (262 min). The coast is 99.6% of
each cycle, so the natural (drag) law dominates the forecast; splitting removes the bias from mislabelled ramp samples.

**26. Switching law and forecast.** The band edges are read from the SMA just before each detected burn and just after it;
`hysteresis_states` is the memory law; `law_forecast` turns the learned rates and edges into a period and burn times. With
constant drag the periodic schedule is best (0.7 min), then the GP coast curve (19 min), the split law (66 min); the joint
law drifts by 23 hours.

## Part D: forecast assessment (slides 27-31)

**27. Error budget.** The chain is run with everything learned, then the truth is substituted into one stage at a time. All
stages true gives 1.4 min (a sanity check); true edges remove two thirds of the error, true labels a third; a true c_d alone
makes it worse because its small error was cancelling the edge error. The edges (learned to ~2 m) dominate.

**28. Sampling robustness of the chain.** Re-sampled at 6 s to 10 min: detection stays perfect and c_d stays within 0.08%,
but the edges and thrust degrade from 5 min (a 23-min burn has only a few samples), so the forecast error grows.

**29. Every force in the loop.** A second 400-day run with the Moon, Sun and SRP. They add a 2.1 m oscillation to the mean
SMA; detection is unchanged; the split-law error doubles to 132 min, and removing a catalogue Moon / Sun / SRP model brings
it to 86 min. Even the periodic schedule drifts (45 min), because the coasts are no longer identical.

**30. Towards operations.** A calibrated fast model (period within 0.01%) for many scenarios. With space-weather drag the
learned law with a drag proxy halves the error of a periodic schedule; the forecasts cut missed dangerous conjunctions
from 68% to 25% and lost tracks after a burn from 74% to 15-19%; conformal intervals are well calibrated.

**31. Negative results.** PCA gives a sparse frame for the coasting state, but a linear model in any frame drifts tens of
km per day, while a 3-term physics library stays within 0.65 km after 10 days. Also: the joint regression is biased,
k-means fails, a memory-less switch fails and a plausible but wrong library (exponential density) extrapolates badly.

## Part E: assessment and close (slides 32-37)

**32. Method ranking.** Best / also good / avoid for every stage, from the evidence in the talk (judged on accuracy,
robustness to noise and sampling, cost and whether the method reports its uncertainty).

**33. Holistic assessment.** Benefits (physics-based features, unsupervised detection, split learning, uncertainty,
interpretable laws), improvements (learn the edges with uncertainty, event-driven forecasts with modelled perturbations,
log-chi detector, real data, repeated noise seeds) and best practices (known-truth tests, oracle swaps, event scoring,
walk-forward testing, bias-variance tuning, closed-form checks, 53 unit tests).

**34. Limitations and future work.** Idealised measurements, constant density in the core run, a fast model for the
extensions, one noise seed per pipeline study, and a judgement-based ranking; next steps are real ephemerides and
event-driven forecasts coupled to screening and tasking.

**35. Conclusions.** Five numbers to remember: simulation verified (J2 0.04%, 16/16 checks); detection F1 = 1 from 6 s to
10 min; c_d +0.03%, T -0.005%, edges ~2 m; 66 min forecast error after a year with constant drag; operational gains in
screening and custody.

**36. References.** The main sources; the full numbered list is in the report.

**37. Questions.** Where to find the code and the report, and the likely question topics (solver choice, mean SMA, GMM
burn rule, BIC vs Brier, split learning, the all-forces run).

---

## Verification notes

* Every number on the slides was taken from `outputs/results.json` of the latest run and cross-checked against the report.
* Every code excerpt is copied from the current source (some docstrings shortened; no logic changed). Slide 19's
  `fit_gmm` and slide 24's `_ard` are shown in part (the remaining lines only rescale the outputs).
* Three report errors found while preparing the slides were corrected in `report/briefing.tex`:
  1. the free-flight solver tolerance is 1e-10, not 1e-12 (and the solver cross-check numbers were added);
  2. the GP is fitted to a random 1,500 of the ~33,000 stacked samples (`GP_MAX_POINTS`), not all of them;
  3. scenario C's period is 3.9005 d (3.90055), not 3.9006 d.

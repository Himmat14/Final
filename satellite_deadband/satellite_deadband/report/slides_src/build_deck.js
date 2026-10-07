// Plain, white, numbered deck for the 45-minute talk (+15 min questions).
// Every slide: a title, short bullets, a figure or raw code, and 5-point speaker notes.
const pptxgen = require("./node_modules/pptxgenjs");
const path = require("path");
const fs = require("fs");

const FIG = "E:/Y3/UROP_Lloyd_Fung/Week 5/Final/satellite_deadband/satellite_deadband/outputs/report/";
const SIZES = JSON.parse(fs.readFileSync(path.join(__dirname, "figs.json"), "utf8"));
const OUT = process.argv[2] || path.join(__dirname, "presentation.pptx");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                 // 13.33 x 7.5 in
pres.title = "Manoeuvre Pattern-of-Life Prediction for SDA";
pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };

pres.defineSlideMaster({
  title: "PLAIN",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 12.33, h: 0.75,
                                fontSize: 28, bold: true, color: "000000", fontFace: "Calibri", valign: "middle" },
                     text: "" } },
  ],
  slideNumber: { x: 12.33, y: 7.0, w: 0.6, h: 0.35, fontSize: 12, color: "000000", fontFace: "Calibri", align: "right" },
});

const BODY = 16, CODE = 10.5;

function newSlide(title, notes) {
  const s = pres.addSlide({ masterName: "PLAIN" });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes.map((n) => "- " + n).join("\n"));
  return s;
}

function bullets(s, items, x, y, w, h, size = BODY) {
  const runs = items.map((it, i) => {
    const sub = typeof it === "object";
    const text = (sub ? it.t : it).replace(/ -> /g, " → ");
    return { text, options: { bullet: sub ? { indent: 18 } : true, indentLevel: sub ? 1 : 0,
                              fontSize: sub ? size - 2 : size, breakLine: i < items.length - 1, paraSpaceAfter: 6 } };
  });
  s.addText(runs, { x, y, w, h, valign: "top", fontFace: "Calibri", color: "000000", isTextBox: true, margin: 4 });
}

function text(s, str, x, y, w, h, opts = {}) {
  s.addText(str, Object.assign({ x, y, w, h, fontFace: "Calibri", fontSize: BODY, color: "000000", valign: "top",
                                 isTextBox: true, margin: 4 }, opts));
}

function img(s, file, x, y, w, h, caption) {
  const [pw, ph] = SIZES[file] || [1600, 1000];
  const capH = caption ? 0.35 : 0;
  const boxH = h - capH;
  let iw = w, ih = (w * ph) / pw;
  if (ih > boxH) { ih = boxH; iw = (boxH * pw) / ph; }
  const ix = x + (w - iw) / 2, iy = y + (boxH - ih) / 2;
  s.addImage({ path: FIG + file, x: ix, y: iy, w: iw, h: ih });
  if (caption) text(s, caption, x, y + boxH, w, capH, { fontSize: 11, color: "444444", align: "center", valign: "top", italic: true });
}

function code(s, src, x, y, w, h, size = CODE) {
  s.addText(src.replace(/^\n/, ""), { x, y, w, h, fontFace: "Courier New", fontSize: size, color: "000000",
    fill: { color: "F2F2F2" }, line: { color: "BFBFBF", width: 0.75 }, valign: "top", isTextBox: true, margin: 6 });
}

function box(s, label, x, y, w, h) {
  s.addText(label, { shape: pres.ShapeType.rect, x, y, w, h, fontFace: "Calibri", fontSize: 14, color: "000000",
    align: "center", valign: "middle", line: { color: "000000", width: 1 }, fill: { color: "FFFFFF" }, isTextBox: true });
}

function arrow(s, x1, y1, x2, y2) {
  s.addShape(pres.ShapeType.line, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color: "000000", width: 1.5, endArrowType: "triangle" } });
}

function table(s, rows, x, y, w, colW, size = 12) {
  const data = rows.map((r, i) => r.map((c) => ({ text: String(c), options: { bold: i === 0, fontSize: size, fontFace: "Calibri",
    color: "000000", fill: { color: i === 0 ? "E7E6E6" : "FFFFFF" }, border: { type: "solid", pt: 0.75, color: "999999" } } })));
  s.addTable(data, { x, y, w, colW, margin: 0.04 });
}

// =========================================================================================================
// 1. Title
// =========================================================================================================
{
  const s = newSlide("Manoeuvre Pattern-of-Life Prediction for Space Domain Awareness", [
    "Topic: learning a satellite's station-keeping control law from noisy tracking data and forecasting its burns.",
    "Setting: UK Space Agency challenge brief; UROP project at Imperial College London.",
    "Approach: a known-truth simulation, then a chain of estimators scored against that truth.",
    "Talk: 45 minutes in process order (physics, code, detection, learning, forecasting, assessment).",
    "15 minutes for questions at the end; backup detail is in the written report.",
  ]);
  text(s, "Learning a station-keeping control law from noisy tracking data:\nsimulation, burn detection, SINDy / BINDy, Gaussian processes and forecasting",
       0.5, 1.3, 12.33, 1.0, { fontSize: 20 });
  text(s, "UROP project, Imperial College London\n45 min presentation + 15 min questions", 0.5, 2.4, 6, 0.9, { fontSize: 16 });
  img(s, "step21_deep_dive/step21_sawtooth.png", 6.4, 2.3, 6.4, 4.6, "The simulated deadband sawtooth: mean SMA vs time (20 days)");
}

// =========================================================================================================
// 2. Roadmap
// =========================================================================================================
{
  const s = newSlide("Roadmap: the analysis chain, in process order", [
    "The talk follows the processing chain from left to right.",
    "Stage 0 builds the truth: a 400-day simulation with a deadband controller.",
    "Stages 1-3 estimate the natural dynamics, detect burns and learn the laws.",
    "Stage 4 forecasts burns; extensions test the forecasts in operational settings.",
    "We close with an error budget, method ranking and best practices.",
  ]);
  const labels = ["0. Simulate\n(physics + deadband\ncontroller, ODE)", "1. Natural dynamics\n(c_d, J2: FD, energy,\nEKF, spectra)",
                  "2. Detect burns\n(unmodelled accel.,\nSG filter, GMM)", "3. Learn laws\n(SINDy / BINDy,\nGP, hysteresis)",
                  "4. Forecast\n(next burns,\nintervals)"];
  labels.forEach((l, i) => {
    box(s, l, 0.5 + i * 2.55, 1.5, 2.15, 1.35);
    if (i < labels.length - 1) arrow(s, 0.5 + i * 2.55 + 2.15, 2.175, 0.5 + (i + 1) * 2.55, 2.175);
  });
  box(s, "Use: space weather, tracking passes, conjunction screening, sensor tasking, fleets, GEO", 3.05, 3.3, 9.8, 0.6);
  arrow(s, 11.2, 2.85, 11.2, 3.3);
  bullets(s, [
    "Part A - Physics and simulation code (slides 3-14)",
    "Part B - Estimating drag and detecting burns: GMM (slides 15-22)",
    "Part C - Learning the thrust, natural and switching laws: GP, SINDy, BINDy (slides 23-26)",
    "Part D - Forecasting, error budget, robustness, all forces, extensions (slides 27-31)",
    "Part E - Ranking, holistic assessment, limitations, conclusions (slides 32-37)",
  ], 0.5, 4.2, 12.3, 2.7);
}

// =========================================================================================================
// 3. The brief
// =========================================================================================================
{
  const s = newSlide("The brief: predict future manoeuvres, not just detect past ones", [
    "Of ~50,000 tracked objects ~15,000 are active and may manoeuvre; mega-constellations will multiply this.",
    "Unpredicted manoeuvres break conjunction screening, track association and sensor scheduling.",
    "Brief Figure 1: a 500 m deadband shifts the miss distance by hundreds of km in a week.",
    "The gap: reconstruct the control law from measurements and use it to forecast.",
    "Our figure reproduces Figure 1 from our own simulation (controlled vs drag-only twin).",
  ]);
  bullets(s, [
    "UK Space Agency challenge: Manoeuvre Pattern of Life Prediction for SDA",
    "Unpredicted burns break three SDA functions:",
    { t: "conjunction screening (predicted orbits wrong)" },
    { t: "track association (observations not linked)" },
    { t: "sensor scheduling (uncertainty must be re-acquired)" },
    "Question: can the control law be learned from noisy data well enough to forecast the next burns?",
    "Method: known-truth simulation + scored chain of estimators",
  ], 0.5, 1.25, 5.3, 5.9);
  img(s, "step5_deadband/smooth_deadband_mean_sma_miss.png", 6.0, 1.2, 6.83, 5.95, "Our reproduction of the brief's Figure 1");
}

// =========================================================================================================
// 4. Physics model
// =========================================================================================================
{
  const s = newSlide("Physics model: forces on a 550 km Starlink-like orbit", [
    "State x = (r, v) in non-dimensional units: DU = 6371 km, TU = 1/omega_Earth = 13,713.44 s, mu = 289.873.",
    "Forces: two-body gravity, J2 (equatorial radius 6378.137 km), quadratic drag, Moon, Sun, SRP, thrust.",
    "Mean accelerations: gravity 8.3, J2 1.2e-2, Moon 8.8e-7, drag 8.1e-7, Sun 4.0e-7, SRP 1.1e-7 m/s^2.",
    "Thrust 2e-4 m/s^2 sits between J2 and drag: detectable once J2 is removed.",
    "Drag tuned to 5.3 m/h of SMA decay (realistic order for 550 km); constant density in the core run.",
  ]);
  bullets(s, [
    "dr/dt = v",
    "dv/dt = -mu r / r^3 + a_J2 + a_drag + a_Moon + a_Sun + a_SRP + a_thrust",
    "a_drag = -c_d |v| v ;  a_J2 = -grad U_J2,  U_J2 = (mu J2 Re^2 / 2r^3)(3 sin^2(lat) - 1)",
    "Third bodies: tidal form  mu_b [ (r_b - r)/|r_b - r|^3 - r_b/|r_b|^3 ]",
    "Units: DU = 6371 km, TU = 13,713.44 s (non-dimensional state ~ 1)",
    "Thrust (2e-4 m/s^2) is 250x drag but 60x below J2",
  ], 0.5, 1.25, 5.6, 5.9, 15);
  img(s, "step1_dynamics/step1_force_budget.png", 6.2, 1.2, 6.63, 5.95, "Mean acceleration of each force (log scale)");
}

// =========================================================================================================
// 5. Code: forces
// =========================================================================================================
{
  const s = newSlide("Code: the force model (deadband/physics.py)", [
    "Each force is a small pure function of the state; the right-hand side just sums them.",
    "Everything is non-dimensional; J2 uses the EQUATORIAL radius (a bug found and fixed: 0.22% too weak before).",
    "Drag is -c_d |v| v with a lumped ballistic coefficient c_d.",
    "Vectorised versions of the same functions are reused by the classifier (known_accel).",
    "Unit test: a_J2 equals minus the gradient of the J2 potential.",
  ]);
  code(s, String.raw`
def gravity_accel(r):
    """Two-body gravity."""
    return -MU * r / np.linalg.norm(r) ** 3


def drag_accel(v, cd):
    """Quadratic drag opposing the velocity: -cd * |v| * v."""
    return -cd * np.linalg.norm(v) * v


def j2_accel(r, j2=J2):
    """Acceleration from Earth's oblateness (J2)."""
    radius = np.linalg.norm(r)
    x, y, z = r
    earth_radius = EARTH_EQUATORIAL_RADIUS_KM / DU   # J2 reference radius
    factor = 1.5 * j2 * MU * earth_radius**2 / radius**4
    z_ratio_sq = (z / radius) ** 2
    return factor * np.array([
        x / radius * (5 * z_ratio_sq - 1),
        y / radius * (5 * z_ratio_sq - 1),
        z / radius * (5 * z_ratio_sq - 3),
    ])`, 0.5, 1.25, 7.9, 5.85, 12);
  bullets(s, [
    "Pure functions: easy to test and to vectorise",
    "Non-dimensional state keeps all components ~1 (good for solvers and least squares)",
    "J2 reference radius = equatorial 6378.137 km",
    "Verified: a_J2 = -grad U_J2 (unit test)",
    "Moon, Sun, SRP: same pattern in perturbations.py",
  ], 8.6, 1.25, 4.25, 5.85, 15);
}

// =========================================================================================================
// 6. J2 check
// =========================================================================================================
{
  const s = newSlide("Verifying J2: nodal regression against the analytical rate", [
    "Simulated node from the angular momentum h = r x v: Omega = atan2(h_x, -h_y), cos i = h_z/|h|.",
    "Theory: dOmega/dt = -3/2 n J2 (Re/p)^2 cos i, with one-orbit means of a and i, p = a(1-e^2) ~ a.",
    "The rate is integrated along the run (cumulative sum, 1-min steps) because drag lowers the orbit 51 km.",
    "Agreement after 400 days: 0.04% (rate -4.526 deg/day at start, -4.642 deg/day at end).",
    "This check exposed the mean-radius bug (J2 was 0.22% too weak) before any learning was done.",
  ]);
  bullets(s, [
    "Simulated node: Omega = atan2(h_x, -h_y),  h = r x v",
    "Theory (secular J2):",
    { t: "dOmega/dt = -(3/2) n J2 (Re / p)^2 cos i" },
    { t: "Omega_th(t) = Omega(0) + integral of dOmega/dt along the run" },
    { t: "n = sqrt(mu / a^3), one-orbit means of a and i, p ~ a (e ~ 1e-4)" },
    "Agreement after 400 days: 0.04%",
    "Rate -4.526 -> -4.642 deg/day as drag lowers the free orbit by 51 km",
  ], 0.5, 1.25, 5.6, 5.9, 15);
  img(s, "step1_dynamics/panels/step1_j2_check_a.png", 6.2, 1.2, 6.63, 5.95, "Simulated RAAN vs integrated J2 theory, 400 days");
}

// =========================================================================================================
// 7. ODE solvers
// =========================================================================================================
{
  const s = newSlide("ODE solvers: DOP853 for free flight, LSODA for the controller", [
    "Free runs: DOP853, explicit 8th-order Runge-Kutta, rtol = atol = 1e-10 (non-dimensional).",
    "Controlled runs: LSODA, which switches automatically between non-stiff Adams and stiff BDF methods.",
    "The tanh switches make the controlled ODE stiff for seconds at each burn edge; LSODA shrinks its steps there.",
    "Cross-check over 2 days vs a 1e-13 DOP853 reference: DOP853 0.27 m, LSODA 28 m, Radau 0.013 m.",
    "States are read from the solver's dense output every 6 s; 10-day chunks keep memory low.",
  ]);
  code(s, String.raw`
# free flight (perturbations.py): explicit Runge-Kutta 8(5,3)
solution = solve_ivp(full_rhs(cd, forces), [t_eval[0], t_eval[-1]], state0,
                     t_eval=t_eval, method="DOP853",
                     rtol=tolerance, atol=tolerance)      # 1e-10

# controlled run (smooth_controller.py): stiff / non-stiff switching
result = solve_ivp(rhs, [0.0, t_end], state0, method="LSODA",
                   rtol=tolerance, atol=tolerance,        # 1e-11
                   dense_output=True)`, 0.5, 1.25, 7.4, 2.6, 11);
  bullets(s, [
    "DOP853: high order, efficient for smooth free flight",
    "LSODA: Adams (non-stiff) <-> BDF (stiff) chosen automatically",
    "Stiffness: 60 s switch time and 5 m switch width at each burn edge",
    "Cross-check (2 days, vs 1e-13 reference): 0.27 m / 28 m / 0.013 m",
    "10-day Week 5 run: 55,138 steps, 157,404 RHS calls",
  ], 0.5, 4.0, 7.4, 3.1, 15);
  img(s, "step1_dynamics/step1_solver_crosscheck.png", 8.1, 1.2, 4.75, 5.95, "Solver cross-check");
}

// =========================================================================================================
// 8. Mean SMA
// =========================================================================================================
{
  const s = newSlide("Why the controller watches the J2 mean SMA", [
    "Osculating SMA from two-body energy swings 12.2 km per orbit under J2: 25x the 500 m band.",
    "A controller on the osculating SMA fires every orbit and climbs out of the band.",
    "Adding the J2 potential to the energy gives a quantity that is constant under J2.",
    "It then changes only through drag and thrust: exactly what the deadband must see.",
    "Check: first-order swing 3 J2 Re^2 sin^2 i / a = 12.17 km vs 12.22 km simulated (0.4%).",
  ]);
  code(s, String.raw`
def j2_potential(r, j2=J2):
    """Potential energy per unit mass from J2 (its gradient is physics.j2_accel)."""
    radius = np.linalg.norm(r)
    earth_radius = EARTH_EQUATORIAL_RADIUS_KM / DU   # J2 reference radius
    return MU * j2 * earth_radius**2 / (2 * radius**3) * (3 * (r[2] / radius) ** 2 - 1)


def mean_sma(r, v, j2=J2):
    """SMA from the total energy (kinetic + point-mass + J2 potential). Constant under J2."""
    energy = np.dot(v, v) / 2 - MU / np.linalg.norm(r) + j2_potential(r, j2)
    return -MU / (2 * energy)`, 0.5, 1.25, 8.6, 3.0, 10);
  bullets(s, [
    "a_osc = -mu / (2 eps), eps = v^2/2 - mu/r: swings 12.2 km per orbit",
    "a_mean = -mu / (2 eps_J2), eps_J2 = eps + U_J2: flat between burns",
    "Theory check: 12.17 vs 12.22 km (0.4%)",
  ], 0.5, 4.4, 8.6, 2.7, 15);
  img(s, "step5_deadband/panels/smooth_j2_mean_vs_osculating_a.png", 9.2, 1.2, 3.63, 5.95, "Osculating vs mean SMA");
}

// =========================================================================================================
// 9. Deadband law: piecewise -> sigmoid
// =========================================================================================================
{
  const s = newSlide("The deadband law: a piecewise switch made smooth with sigmoids", [
    "Ideal law: s = 1 when a <= a_L, s = 0 when a >= a_U, otherwise keep s (memory).",
    "Heaviside steps are discontinuous: a solver needs events and thrust would jump instantly.",
    "Replace each step with sigma(x) = (1 + tanh(x/w))/2, w = 5 m; s gets its own ODE with time constant 60 s.",
    "Latch term s(1-s)(2s-1) makes s = 0 and s = 1 the stable states: the memory (hysteresis).",
    "Throttle opens only past s = 0.9 so the burn cannot stall half-way.",
  ]);
  bullets(s, [
    "Ideal: s+ = 1 if a <= a_L;  0 if a >= a_U;  s- otherwise",
    "Smooth: tau ds/dt = sigma(a_L - a)(1 - s) - sigma(a - a_U) s + lambda s(1 - s)(2s - 1)",
    "a_thrust = T theta(s) v_hat,  theta(s) = sigma((s - 0.9) / 0.02)",
    "a_L = 6921.9 km, a_U = 6922.4 km, w = 5 m, tau = 60 s, lambda = 1, T = 2e-4 m/s^2",
  ], 0.5, 1.25, 12.3, 1.9, 15);
  img(s, "step21_deep_dive/step21_switches.png", 0.5, 3.15, 6.1, 4.0, "Ideal (dotted) vs tanh switches");
  img(s, "step21_deep_dive/step21_throttle.png", 6.73, 3.15, 6.1, 4.0, "Throttle and latch");
}

// =========================================================================================================
// 10. Code: deadband logic
// =========================================================================================================
{
  const s = newSlide("Code: the deadband logic inside the ODE (smooth_controller.py)", [
    "rising_switch is the tanh sigmoid; it replaces every 'if' of the ideal law.",
    "thruster_state_rate is ds/dt: switch on below the lower edge, off above the upper edge, latch in between.",
    "throttle maps the state s to how open the engine is; a_thrust points along the velocity.",
    "The controller reads the J2 mean SMA, so it ignores the 12 km osculating swing.",
    "No events, no discontinuities: one smooth 7-state ODE [r, v, s].",
  ]);
  code(s, String.raw`
def rising_switch(x, width):
    """Smooth step: ~0 for x << -width, 1/2 at x = 0, ~1 for x >> width."""
    return 0.5 * (1 + np.tanh(x / width))


def throttle(s):
    """How open the thruster is (0..1): opens steeply as s passes THROTTLE_OPENS_AT."""
    return rising_switch(s - THROTTLE_OPENS_AT, THROTTLE_WIDTH)


def a_thrust(v, s, thrust_accel=THRUST_ACCEL):
    """Thrust acceleration: along the velocity, magnitude THRUST_ACCEL * throttle(s)."""
    return thrust_accel * throttle(s) * v / np.linalg.norm(v)


def thruster_state_rate(sma, s, sma_width=SWITCH_SMA_WIDTH, switch_time=SWITCH_TIME,
                        a_lower=SMOOTH_A_LOWER, a_upper=SMOOTH_A_UPPER):
    """ds/dt for the thruster on/off state s."""
    turn_on = rising_switch(a_lower - sma, sma_width)    # 1 once the SMA is below the lower edge
    turn_off = rising_switch(sma - a_upper, sma_width)   # 1 once the SMA is above the upper edge
    latch = LATCH_GAIN * s * (1 - s) * (2 * s - 1)       # holds s at 0 or 1 inside the band
    return (turn_on * (1 - s) - turn_off * s + latch) / switch_time`, 0.5, 1.25, 8.6, 5.85, 10);
  bullets(s, [
    "Three sigmoids replace three 'if' statements",
    "turn_on / turn_off: 5 m wide",
    "latch: zeros at s = 0, 1/2, 1",
    "throttle: opens over s = 0.88-0.92 (~30 s ramp)",
    "THROTTLE_OPENS_AT = 0.9, THROTTLE_WIDTH = 0.02",
  ], 9.3, 1.25, 3.55, 5.85, 15);
}

// =========================================================================================================
// 11. Code: right-hand side
// =========================================================================================================
{
  const s = newSlide("Code: the 7-state right-hand side f(t, [r, v, s])", [
    "One closure returns d/dt of [r, v, s] for solve_ivp.",
    "Physics part: gravity + drag + J2, plus Moon / Sun / SRP when requested (evaluated at absolute time).",
    "Control part: thrust from s, and ds/dt from the J2 mean SMA.",
    "controlled=False gives the drag-only twin; use_mean_sma=False reproduces the osculating failure case.",
    "Same function drives the J2+drag run and the all-forces run (step 22).",
  ]);
  code(s, String.raw`
def rhs(t, state):
    r, v, s = state[0:3], state[3:6], state[6]
    accel = gravity_accel(r) + drag_accel(v, cd) + j2_accel(r, j2)
    if extra_forces:
        clock = t + t_offset
        if "Moon" in extra_forces:
            accel = accel + third_body_accel(r, moon_position(clock), MU_MOON)
        if "Sun" in extra_forces or "SRP" in extra_forces:
            r_sun = sun_position(clock)
            if "Sun" in extra_forces:
                accel = accel + third_body_accel(r, r_sun, MU_SUN)
            if "SRP" in extra_forces:
                accel = accel + srp_accel(r, r_sun)
    ds_dt = 0.0
    if controlled:   # a fixed choice of model, not a switch inside the dynamics
        accel = accel + a_thrust(v, s)
        sma = mean_sma(r, v, j2) if use_mean_sma else osculating_sma(r, v)
        ds_dt = thruster_state_rate(sma, s, sma_width, switch_time, a_lower, a_upper)
    return np.concatenate([v, accel, [ds_dt]])`, 0.5, 1.25, 8.6, 5.85, 11.5);
  bullets(s, [
    "State: [x, y, z, vx, vy, vz, s]",
    "Physics block + control block",
    "Third bodies use t + t_offset (10-day chunks keep the true Moon / Sun phase)",
    "The 'if controlled' is a model choice made once, not a discontinuity",
  ], 9.3, 1.25, 3.55, 5.85, 15);
}

// =========================================================================================================
// 12. 400-day data + chunk loop
// =========================================================================================================
{
  const s = newSlide("Generating 400 days of data: chunked integration and results", [
    "The 400-day run is integrated in 40 chunks of 10 days; each chunk starts from the last state.",
    "Samples every 6 s come from the LSODA dense output; the result is cached to disk (~370 MB).",
    "Outcome: 102 burns, one every 3.9051 days, 22.8 min each, 0.273 m/s per burn, 27.8 m/s in total.",
    "Mean SMA stays inside [6921.904, 6922.402] km for the whole run.",
    "TRAIN = days 0-40, TEST = days 40-400, horizons 15-360 days throughout the study.",
  ]);
  code(s, String.raw`
def _controlled(name, extra_forces=()):
    def build():
        step = SAMPLE_STEP_S * SECONDS
        chunk = from_days(CHUNK_DAYS)
        state = smooth_initial_state()
        times, samples = [], []
        for k in range(int(np.ceil(LONG_DAYS / CHUNK_DAYS))):
            sim = simulate_smooth_deadband(chunk, state0=state,
                                           extra_forces=extra_forces,
                                           t_offset=k * chunk)
            local = np.arange(0, chunk, step)
            times.append(local + k * chunk)
            samples.append(sim.sample(local))     # dense output
            state = sim.states[:, -1]
        return dict(t=np.concatenate(times),
                    states=np.concatenate(samples, axis=1))
    data = _cached(name, build)                    # .cache/*.npz`, 0.5, 1.25, 6.4, 4.4, 10.5);
  bullets(s, [
    "102 burns, period 3.9051 d, 22.8 min, 0.273 m/s each",
    "Band kept: [6921.904, 6922.402] km",
    "TRAIN days 0-40, TEST days 40-400",
  ], 0.5, 5.8, 6.4, 1.35, 15);
  img(s, "step21_deep_dive/step21_sawtooth.png", 7.1, 1.2, 5.73, 5.95, "Mean-SMA sawtooth, first 20 days");
}

// =========================================================================================================
// 13. State space / hysteresis
// =========================================================================================================
{
  const s = newSlide("The controller as a two-state system: hysteresis in (a, s)", [
    "Reduced to the slow states x = (a, s): da/dt = -c_d D(a,v) + T theta(s) B(a,v); ds/dt from the switches.",
    "Inside the band only the latch acts: s = 0 (coast) and s = 1 (burn) are both stable.",
    "At a_L the coast branch loses stability and s jumps to 1; at a_U the burn branch drops to 0.",
    "Result: a clockwise hysteresis loop; the same SMA can be coasting or burning.",
    "So a memory-less law s = f(a) cannot work (balanced accuracy 0.50 vs 1.00 for hysteresis).",
  ]);
  bullets(s, [
    "da/dt = -c_d D(a,v) + T theta(s) B(a,v)",
    { t: "D = (2a^2/mu)|v|^3,  B = (2a^2/mu)|v|" },
    "ds/dt = [sigma(a_L - a)(1 - s) - sigma(a - a_U) s + lambda s(1 - s)(2s - 1)] / tau",
    "Two stable branches: coast -5.3 m/h, burn +1311 m/h",
    "Memory-less s = f(a): balanced accuracy 0.50; hysteresis edges: 1.00",
  ], 0.5, 1.25, 5.0, 5.9, 15);
  img(s, "step21_deep_dive/step21_hysteresis_field.png", 5.6, 1.2, 7.23, 5.95, "Vector field (directions) and one true cycle");
}

// =========================================================================================================
// 14. Calculation checks
// =========================================================================================================
{
  const s = newSlide("Checking the numbers against closed-form physics", [
    "Every headline number is recomputed from a formula (deadband/report_checks.py).",
    "Examples: decay -2 c_d sqrt(mu a) = 5.333 vs 5.335 m/h; burn 1370 vs 1368 s; period exact.",
    "Delta-v per burn v Delta-a / 2a = 0.2741 vs 0.2730 m/s; J2 Nyquist T/4 = 23.88 min.",
    "Forecast error from a period error: (1/2) H |dP/P|, within 0.9% for the SINDy scenarios.",
    "All 16 checks pass (14 within 1%; stacking noise within 11%; two inequality bounds hold).",
  ]);
  bullets(s, [
    "Coast decay: da/dt = -2 c_d sqrt(mu a): 5.333 vs 5.335 m/h",
    "Burn climb: 2aT/v - k: 1308 vs 1311 m/h",
    "Burn: Delta-v / T = 1370 vs 1368 s",
    "Period: w/k + t_b = 3.9051 d (exact)",
    "J2 swing: 3 J2 Re^2 sin^2 i / a = 12.17 vs 12.22 km",
    "Missed burn drift: (3/2) n Delta-a t = 495 km/week (upper bound; simulated 396)",
    "16 / 16 checks pass",
  ], 0.5, 1.25, 5.6, 5.9, 15);
  img(s, "step20_pipeline/step20_checks.png", 6.2, 1.2, 6.63, 5.95, "Relative difference, formula vs measured");
}

// =========================================================================================================
// 15. Stage 1: drag
// =========================================================================================================
{
  const s = newSlide("Stage 1: estimating the drag coefficient c_d", [
    "FD regression: differentiate positions twice and solve least squares; noise amplified as sigma/h^2.",
    "Energy method: one slope through the mean-SMA history; averages the noise of every sample.",
    "Shooting (Levenberg-Marquardt) and an EKF with state (r, v, c_d).",
    "15 days, 1-min data, 10 m noise: FD 6.3%, energy 0.035%; EKF on 10-min data 0.04% (15 d), 0.01% (30 d).",
    "Inside the full chain c_d is learned on clean coast arcs to 0.03-0.08% at every sampling step.",
  ]);
  bullets(s, [
    "FD regression: r'' = mu shapes . coefficients (least squares, unit-scaled columns)",
    "Energy method: da/dt = -(2a^2/mu) c_d |v|^3, fit one slope",
    "Shooting: fit the whole trajectory (Levenberg-Marquardt)",
    "EKF: state (r, v, c_d), STM propagation",
    "c_d error at 10 m noise, 15 d: FD 6.3%, energy 0.035%",
    "EKF (10-min data): 0.04% / 0.01% after 15 / 30 days",
  ], 0.5, 1.25, 5.3, 5.9, 15);
  img(s, "step3_drag_noise/panels/step3_noise_estimators_a.png", 5.9, 1.2, 3.45, 5.95, "FD regression c_d error vs noise");
  img(s, "step3_drag_noise/panels/step3_noise_estimators_f.png", 9.4, 1.2, 3.45, 5.95, "Energy method c_d error vs noise");
}

// =========================================================================================================
// 16. The unmodelled acceleration
// =========================================================================================================
{
  const s = newSlide("Stage 2: the unmodelled acceleration (subtraction)", [
    "Measured dv/dt minus modelled gravity and J2 leaves drag + thrust + noise + numerical error.",
    "This cancels seven decades (8.3 m/s^2 down to 8e-7), so numerical error was checked explicitly.",
    "Machine epsilon x |g| = 1.8e-15; SG truncation 8.3e-9; total numerical floor 1.0e-8 m/s^2.",
    "Floor is 80x below drag and 460x below the derivative noise at reference noise.",
    "So the GMM sees physics + measurement noise, not solver artefacts.",
  ]);
  code(s, String.raw`
def build_features(r, v, step_s, window=None, order=None, method=None):
    h = step_s * SECONDS
    window, order = window or tuned("sg_window"), order or tuned("sg_order")
    r_second_derivative = savgol_filter(r, window, order, deriv=2, delta=h, axis=1)
    v_first_derivative = savgol_filter(v, window, order, deriv=1, delta=h, axis=1)
    modelled = known_accel(r)                        # gravity + J2
    from_r = (r_second_derivative - modelled) * ACCEL_UNIT_MS2
    from_v = (v_first_derivative - modelled) * ACCEL_UNIT_MS2
    along = np.sum(from_v * v, axis=0) / np.linalg.norm(v, axis=0)
    return Features(accel_from_r=from_r, accel_from_v=from_v, along_track=along)`, 0.5, 1.25, 7.6, 3.0, 10);
  bullets(s, [
    "a_u = dv/dt (measured) - [ -mu r / r^3 + a_J2 ]",
    "a_u ~ drag (8e-7) + thrust (2e-4) + noise (4.6e-6) + numerics (1e-8)",
    "Velocity-based (v') features: r'' of noisy positions is pure noise",
  ], 0.5, 4.4, 7.6, 2.7, 15);
  img(s, "step21_deep_dive/step21_subtraction_cascade.png", 8.2, 1.2, 4.63, 5.95, "Size of every term and error");
}

// =========================================================================================================
// 17. Savitzky-Golay
// =========================================================================================================
{
  const s = newSlide("The derivative: Savitzky-Golay filtering", [
    "SG fits a degree-p polynomial to 2m+1 samples by least squares and differentiates it at the centre.",
    "Linear in the data, so it is a fixed convolution with known frequency response H(f).",
    "Exact for polynomials up to order p (no bias on the orbit), low-pass above the cut-off (rejects noise).",
    "Tuned 61 samples (6 min), order 5: cut-off ~2.8-min period; noise gain falls ~ window^-3/2.",
    "Cost: fast features (30-s thrust ramps) are smeared to ~60 s.",
  ]);
  bullets(s, [
    "y'_k = sum_j c_j y_(k+j),  c = row 1 of (A^T A)^-1 A^T,  A_ji = (j h)^i",
    "H(f) = sum_j c_j exp(2 pi i f j h);  ideal = 2 pi i f",
    "Exact for polynomials up to order p",
    "Tuned: 61 samples (6 min), order 5 (bias-variance)",
    "Cut-off ~ 2.8 min period; orbit and J2 lines far below it",
    "Central differences keep amplifying noise to Nyquist",
  ], 0.5, 1.25, 5.3, 5.9, 15);
  img(s, "step21_deep_dive/step21_sg_response.png", 5.9, 1.2, 6.93, 5.95, "Frequency response of derivative filters (6-s data)");
}

// =========================================================================================================
// 18. Features x1 x2
// =========================================================================================================
{
  const s = newSlide("Features: a physics-based change of basis (x1, x2)", [
    "Raw 3-D acceleration: coast (~1e-6) collapses to a dot next to burns (2e-4).",
    "x1 = ln|a_u|: compresses four decades so both clusters get comparable widths; frame independent.",
    "x2 = a_u . v_hat / 1e-4: thrust is along the velocity, noise points anywhere.",
    "Geometry: |x2| <= e^x1 / 1e-4, so data fill a wedge; burns sit on its upper edge.",
    "For isotropic noise the along-track share is uniform on [-1, 1] (Archimedes' hat-box theorem).",
  ]);
  bullets(s, [
    "x1 = ln |a_u|  (how much)",
    "x2 = a_u . v_hat / 1e-4 m/s^2  (which way)",
    "Wedge: |x2| <= exp(x1) / 1e-4",
    "Burns: x2 ~ 2, all along-track",
    "Coast: x2 ~ 0; its mean -0.009e-4 m/s^2 is the drag",
    "Clusters ~4 decades apart in |a|",
  ], 0.5, 1.25, 5.0, 5.9, 15);
  img(s, "step21_deep_dive/step21_basis_features.png", 5.6, 1.2, 7.23, 5.95, "Samples in the (x1, x2) basis");
}

// =========================================================================================================
// 19. GMM code
// =========================================================================================================
{
  const s = newSlide("Code: the Gaussian mixture and the burn rule", [
    "GMM: p(x) = sum_k pi_k N(x; mu_k, Sigma_k), fitted by EM on 20,000 TRAIN samples (3 restarts).",
    "E-step: responsibilities gamma_ik; M-step: weighted weights, means and covariances.",
    "Unsupervised: never sees labels. K = 4 chosen by Brier score / event F1 (step 0).",
    "Burn rule: a component is 'burn' if its mean x1 is > 3 coast standard deviations above the coast mean.",
    "Flags become events: fill gaps <= 2 samples, drop runs < 10 samples (1 min); score per event (+-60 s).",
  ]);
  code(s, String.raw`
def mixture_burn_components(model, n_sigma=3.0):
    """The heaviest component is coast. A component is burn if its mean log|v'|
    sits more than n_sigma coast standard deviations above the coast mean."""
    coast = int(np.argmax(model.weights_))
    coast_sd = np.sqrt(np.atleast_2d(model.covariances_[coast])[0, 0])
    return model.means_[:, 0] > model.means_[coast, 0] + n_sigma * coast_sd


def fit_gmm(X_train, n_components=None):
    n_components = n_components or tuned("gmm_components")       # 4
    model = GaussianMixture(n_components=n_components, n_init=3, random_state=0).fit(X_train)
    burn = mixture_burn_components(model)
    return lambda X: burn[model.predict(X)].astype(int), model


def group_into_events(flags, max_gap=MAX_GAP_SAMPLES, min_samples=MIN_EVENT_SAMPLES):
    """Fill short gaps inside runs, then drop runs shorter than min_samples."""
    events = np.asarray(flags, dtype=int).copy()
    for first, last in _runs(1 - events):                 # runs of 0s
        is_inside = first > 0 and last < len(events) - 1
        if is_inside and last - first + 1 <= max_gap:
            events[first:last + 1] = 1
    for first, last in _runs(events):
        if last - first + 1 < min_samples:
            events[first:last + 1] = 0
    return events`, 0.5, 1.25, 8.9, 5.85, 10);
  bullets(s, [
    "EM: E-step gamma_ik = pi_k N_k / sum_j pi_j N_j",
    "M-step: pi_k, mu_k, Sigma_k from weighted samples",
    "P(burn | x) = sum of burn responsibilities",
    "K = 4, n_init = 3, 20k TRAIN samples",
    "Events: max gap 2, min length 10 samples",
  ], 9.6, 1.25, 3.25, 5.85, 14);
}

// =========================================================================================================
// 20. GMM results
// =========================================================================================================
{
  const s = newSlide("GMM results: four components, perfect event detection", [
    "Fitted components: three coast components (weights 0.42, 0.15, 0.42) and one burn component (0.0039).",
    "Check: burn weight should equal the duty cycle 1368 s / 3.905 d = 0.0041.",
    "92 / 92 TEST burns, 0 false events, F1 = 1.0; mean start-time error 14 s at reference noise.",
    "499 false-positive samples out of 5.18 million, all on thrust ramps; they merge into true events.",
    "Alternatives: Bayesian GMM and MAD threshold 1.0; Isolation Forest 0.85; k-means 0.001 (136,662 false events).",
  ]);
  img(s, "step20_pipeline/step20_gmm_histogram.png", 0.5, 1.2, 6.1, 3.9, "Mixture along x1 (log density)");
  img(s, "step20_pipeline/step20_gmm_posterior.png", 6.73, 1.2, 6.1, 5.95, "One TEST burn: signal (top) and P(burn) (bottom)");
  bullets(s, [
    "92/92 burns, 0 false events, F1 = 1.0",
    "Burn weight 0.0039 vs duty cycle 0.0041",
    "k-means: F1 0.001 with noise",
  ], 0.5, 5.2, 6.1, 1.95, 15);
}

// =========================================================================================================
// 21. Fat tails and alternative mixtures
// =========================================================================================================
{
  const s = newSlide("Why the coast cloud is fat-tailed, and better mixtures", [
    "If coast noise is 3-D Gaussian, |n|/sigma ~ chi_3 and x1 = ln|n| is log-chi distributed.",
    "Log-chi: left tail ~ exp(k x1) (fat, exponential), right tail ~ exp(-e^(2 x1)) (thinner than Gaussian).",
    "Data: skewness -0.89, left-tail slope 2.54 (theory 3), fitted k = 2.75; beats a Gaussian by 500 in log-likelihood.",
    "Physics mixture (log-chi coast + Gaussian burn, 10 parameters) beats the 23-parameter K = 4 GMM on BIC.",
    "BIC = -2 ln L + p ln n keeps falling with K because extra Gaussians only re-shape the skewed coast.",
  ]);
  img(s, "step21_deep_dive/step21_coast_tail.png", 0.5, 1.2, 6.1, 4.1, "Coast x1 with Poisson error bars");
  img(s, "step21_deep_dive/step21_mixture_fits.png", 6.73, 1.2, 6.1, 4.1, "GMM vs log-chi vs Student-t mixtures");
  table(s, [
    ["Model", "Params", "BIC", "Test log-lik / sample", "False samples"],
    ["GMM, K = 4 (used)", "23", "-67,079", "1.689", "499"],
    ["GMM, K = 7", "41", "-68,454", "1.728", "463"],
    ["log-chi coast + Gaussian burn", "10", "-67,804", "1.706", "492"],
    ["Student-t coast + Gaussian burn", "12", "-60,509", "1.523", "0"],
  ], 0.5, 5.4, 12.33, [4.33, 1.4, 2.0, 2.6, 2.0], 13);
}

// =========================================================================================================
// 22. Detection robustness
// =========================================================================================================
{
  const s = newSlide("Detection robustness: noise and sampling step", [
    "GMM keeps F1 = 1.0 from 0.25x to 8x the reference noise (0.1 m, 0.5 mm/s) at 6-s sampling.",
    "SG derivative feature works up to a 2-min step (curvature truncation dominates beyond).",
    "One-step propagation feature: propagate each state one step under gravity + J2 (RK4) and compare.",
    "Its noise scales as 1/h, so it works from ~1 to 20 min; switching at 2 min covers 6 s-20 min up to 2x noise.",
    "Limit: at 4x noise there is a gap near 1 min; above one burn length (23 min) a burn is diluted.",
  ]);
  code(s, String.raw`
def propagate_rk4(r, v, h, substep):
    """Every column of (r, v) propagated by h under gravity + J2 only."""
    n = max(1, int(np.ceil(h / substep)))
    dt = h / n
    for _ in range(n):
        k1r, k1v = v, known_accel(r)
        k2r, k2v = v + 0.5 * dt * k1v, known_accel(r + 0.5 * dt * k1r)
        k3r, k3v = v + 0.5 * dt * k2v, known_accel(r + 0.5 * dt * k2r)
        k4r, k4v = v + dt * k3v, known_accel(r + dt * k3r)
        r = r + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r, v

# feature: a = (v[k+1] - v_propagated[k -> k+1]) / h`, 0.5, 1.25, 6.4, 4.2, 10);
  bullets(s, [
    "SG feature: 6 s - 2 min",
    "Propagation feature: ~1 - 20 min",
    "Whole chain: F1 = 1.0 from 6 s to 10 min",
  ], 0.5, 5.6, 6.4, 1.5, 15);
  img(s, "step6_classification/panels/step6_step_stress_a.png", 7.0, 1.2, 5.83, 2.95, "SG derivative feature");
  img(s, "step6_classification/panels/step6_step_stress_b.png", 7.0, 4.2, 5.83, 2.95, "One-step propagation feature");
}

// =========================================================================================================
// 23. Thrust law: stacking, GP, sigmoid
// =========================================================================================================
{
  const s = newSlide("Stage 3a: the thrust law from repetition (GP and sigmoid)", [
    "Autocorrelation of the along-track signal peaks at 3.9051 days, the true period.",
    "Stack all detected burns on their start time: noise falls as 1/sqrt(N) to a smoothing floor.",
    "GP (RBF + white kernel, hyper-parameters by marginal likelihood, 1500 random samples): plateau 0.999 of truth, duration exact.",
    "6-parameter sigmoid law P sigma((t-t_on)/r) sigma((t_off-t)/f) + b fits 14% better (RMS) and is interpretable.",
    "Learned rise time 60 s vs true 30 s: the derivative filter, not the thruster.",
  ]);
  code(s, String.raw`
def fit_thrust_law(stack, max_points=GP_MAX_POINTS):      # 1500
    offsets = np.tile(stack.offsets_s, len(stack.windows))
    values = stack.windows.ravel()
    pick = np.random.default_rng(0).choice(offsets.size,
                       size=min(max_points, offsets.size), replace=False)
    scale = np.std(values)
    kernel = ConstantKernel(1.0) * RBF(length_scale=60.0) \
             + WhiteKernel(noise_level=0.5)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  random_state=0)
    gp.fit(offsets[pick, None], values[pick] / scale)
    mean, std = gp.predict(stack.offsets_s[:, None], return_std=True)`, 0.5, 1.25, 6.4, 3.6, 10);
  bullets(s, [
    "Why GP: no assumed shape, uncertainty band, O(n^3) so subsample",
    "Why sigmoid: same form as the simulator law, 6 parameters",
    "RMS vs truth: GP 1.03e-5, sigmoid 0.88e-5 m/s^2",
  ], 0.5, 5.0, 6.4, 2.1, 15);
  img(s, "step21_deep_dive/step21_thrust_models.png", 7.0, 1.2, 5.83, 5.95, "Stacked burns: truth, GP, sigmoid");
}

// =========================================================================================================
// 24. SINDy / BINDy code
// =========================================================================================================
{
  const s = newSlide("Stage 3b: SINDy and BINDy for the rate laws", [
    "SINDy: da/dt = Theta(a, s) xi, solved by sequentially thresholded least squares on unit-scaled columns.",
    "Physics library [D, B, 1] is singular over a 500 m band (correlation 0.997): use cluster indicators [1-s, s, ...].",
    "STLSQ keeps only the coast and burn terms: c_d 1.0005, T 0.9997 of the truth.",
    "BINDy here = ARD regression: Gaussian priors with learned precisions give posterior standard deviations.",
    "Threshold 0.05 chosen by test MSE of the predicted rate (step 0).",
  ]);
  code(s, String.raw`
def stlsq(theta, target, threshold=None, iterations=10):
    """Sequentially thresholded least squares on unit-scaled columns."""
    threshold = threshold if threshold is not None else tuned("stlsq_threshold")
    scale = np.linalg.norm(theta, axis=0)
    empty = scale == 0
    scale = np.where(empty, 1.0, scale)
    X = theta / scale
    active = ~empty
    xi = np.zeros(X.shape[1])
    for _ in range(iterations):
        xi[:] = 0
        xi[active] = np.linalg.lstsq(X[:, active], target, rcond=None)[0]
        new_active = np.abs(xi) >= threshold * np.max(np.abs(xi))
        if np.array_equal(new_active, active):
            break
        active = new_active
    return xi / scale


def _ard(X, y):          # BINDy-style: posterior mean and std of every coefficient
    scale = np.sqrt(np.mean(X**2, axis=0))
    y_scale = np.std(y) or 1.0
    model = ARDRegression(fit_intercept=False).fit(X / scale, y / y_scale)`, 0.5, 1.25, 6.9, 5.85, 9);
  img(s, "step9_sindy_control_law/step9_coefficients.png", 7.55, 1.2, 5.28, 5.95, "SINDy / BINDy coefficients");
}

// =========================================================================================================
// 25. Four splits
// =========================================================================================================
{
  const s = newSlide("Learning the thrust and natural laws together or apart", [
    "A: one regression over all samples with detected thruster flags; B: c_d known; C: thrust known; D: both on clean arcs.",
    "Joint regression A is biased: ramp samples mislabelled, T -7.6%, c_d -0.8%, 1668 min after a year.",
    "D (split on clean arcs): c_d +0.06%, T -0.05%; 262 min after a year.",
    "B (natural law known) is best: 26 min, even though its T is 7.6% low.",
    "Lesson: the coast is 99.6% of each cycle, so the natural (drag) law governs the forecast.",
  ]);
  table(s, [
    ["Scenario", "c_d / true", "T / true", "Period [d]", "Error 15 d", "Error 120 d", "Error 360 d"],
    ["A whole system", "0.9922", "0.924", "3.9351", "107 min", "691 min", "1668 min"],
    ["B natural known", "given", "0.924", "3.9047", "2.4 min", "9.4 min", "26 min"],
    ["C thrust known", "1.0015", "given", "3.9005", "15 min", "103 min", "302 min"],
    ["D both, separately", "1.0006", "0.9995", "3.9011", "13 min", "89 min", "262 min"],
    ["Truth", "1", "1", "3.9051", "-", "-", "-"],
  ], 0.5, 1.3, 12.33, [3.0, 1.5, 1.4, 1.5, 1.6, 1.6, 1.73], 13);
  bullets(s, [
    "Split learning removes the edge-leakage bias",
    "Knowing c_d matters more than knowing T",
    "The coast is 99.6% of every cycle",
  ], 0.5, 3.9, 5.0, 3.2, 15);
  img(s, "step19_sindy_split/step19_parameters.png", 5.6, 3.75, 7.23, 3.4, "c_d and T per scenario (BINDy +-2 sigma)");
}

// =========================================================================================================
// 26. Switching law and forecast code
// =========================================================================================================
{
  const s = newSlide("The switching law and the forecast", [
    "Edges: SMA read a settle time before each detected burn start (lower) and after each end (upper).",
    "Hysteresis states: on at or below the lower edge, off at or above the upper, otherwise keep the last state.",
    "Law forecast: coast time = band / coast rate, burn time = band / net burn rate; onsets every period.",
    "Forecasters compared: split law, joint law, BINDy, GP coast curve, periodic schedule.",
    "Constant drag: periodic 0.7 min, GP 19 min, split law 66 min, joint law 1391 min at 360 days.",
  ]);
  code(s, String.raw`
def hysteresis_states(a, lower, upper, s0=0):
    """The deadband WITH memory: on below lower, off above upper, else keep."""
    s = np.zeros(len(a), dtype=int)
    state = s0
    for i, value in enumerate(a):
        if value <= lower:
            state = 1
        elif value >= upper:
            state = 0
        s[i] = state
    return s


def law_forecast(law, shapes, start_day):
    D_bar, B_bar = shapes
    width = law.upper_km - law.lower_km
    coast_rate = law.cd * D_bar * DU / TU * DAY                 # km/day
    burn_rate = (law.T * B_bar - law.cd * D_bar) * DU / TU * DAY
    period = width / coast_rate + width / burn_rate
    first = start_day + width / coast_rate
    return first + period * np.arange(0, int(400 / period) + 2), period`, 0.5, 1.25, 7.3, 5.85, 10);
  img(s, "step20_pipeline/step20_forecasters.png", 7.95, 1.2, 4.88, 5.95, "Forecasters on the fully learned chain");
}

// =========================================================================================================
// 27. Error budget
// =========================================================================================================
{
  const s = newSlide("End-to-end error budget: which stage limits the forecast?", [
    "Run everything learned, then swap the truth into one stage at a time (6-s data, reference noise).",
    "All stages true: 1.4 min after 360 days (sanity check of the method).",
    "True edges: 66 -> 21 min; true burn labels: 66 -> 47 min. Edges dominate.",
    "True c_d alone makes it worse (84 min): the small c_d error was cancelling the edge error.",
    "Stage accuracies: c_d +0.029%, T -0.005%, edges -2.2 / -2.4 m, F1 1.000.",
  ]);
  bullets(s, [
    "Pipeline as learned: 66 min at 360 d",
    "+ true edges: 21 min",
    "+ true labels: 47 min",
    "+ true c_d: 84 min (compensating errors)",
    "All true: 1.4 min (sanity check)",
    "Edges learned to ~2 m (0.4% of the band) = 22 min of coast",
  ], 0.5, 1.25, 5.0, 5.9, 15);
  img(s, "step20_pipeline/step20_error_budget.png", 5.6, 1.2, 7.23, 5.95, "Forecast error with each stage made perfect");
}

// =========================================================================================================
// 28. Sampling robustness of the chain
// =========================================================================================================
{
  const s = newSlide("Sampling robustness of the whole chain (6 s to 10 min)", [
    "Same 400 days re-sampled at 6 s, 30 s, 1, 2, 5, 10 min; SG feature below 2 min, propagation above.",
    "Detection: F1 = 1.0 at every step.",
    "c_d within 0.08% at every step: the mean SMA averages the noise of every sample.",
    "Edges and T degrade from 5 min (few samples per 23-min burn): forecast error grows.",
    "The periodic schedule stays best here only because drag is constant.",
  ]);
  table(s, [
    ["Step", "F1", "c_d err", "T err", "Lower edge", "Law 360 d", "GP 360 d"],
    ["6 s", "1.000", "+0.029%", "-0.00%", "-2.2 m", "66 min", "19 min"],
    ["30 s", "1.000", "+0.065%", "+0.02%", "-2.3 m", "226 min", "11 min"],
    ["1 min", "1.000", "+0.074%", "-0.07%", "-2.3 m", "238 min", "150 min"],
    ["2 min", "1.000", "+0.007%", "+0.04%", "-3.3 m", "398 min", "479 min"],
    ["5 min", "1.000", "+0.071%", "-4.81%", "+5.2 m", "1455 min", "1361 min"],
    ["10 min", "1.000", "+0.077%", "-8.65%", "+13.9 m", "1371 min", "1352 min"],
  ], 0.5, 1.3, 6.6, [0.8, 0.75, 0.95, 0.9, 1.1, 1.05, 1.05], 12);
  img(s, "step20_pipeline/step20_sampling_forecast.png", 7.3, 1.2, 5.53, 5.95, "Year-ahead error vs sampling step");
}

// =========================================================================================================
// 29. All forces
// =========================================================================================================
{
  const s = newSlide("Every force in the loop: Moon, Sun and SRP", [
    "Second 400-day controlled run with J2 + drag + Moon + Sun + SRP (correct absolute time per chunk).",
    "Moon (9.3e-7), Sun (4.2e-7), SRP (1.1e-7 m/s^2) are drag-sized; they add a 2.1 m wiggle to the mean SMA.",
    "Detection unchanged (F1 = 1.0); burn period 3.9034 d instead of 3.9051 d.",
    "Split-law forecast 66 -> 132 min at 360 d; subtracting a catalogue Moon/Sun/SRP model gives 86 min.",
    "Even the periodic schedule drifts (45 min): a constant-rate law cannot follow periodic modulation.",
  ]);
  img(s, "step22_full_forcing/step22_sma_signature.png", 0.5, 1.2, 6.1, 4.2, "Mean SMA minus its trend on a coast arc");
  img(s, "step22_full_forcing/step22_forecast_models.png", 6.73, 1.2, 6.1, 4.2, "Every forecaster at 360 days, three set-ups");
  table(s, [
    ["360-day onset error", "J2 + drag", "All forces, J2 model", "All forces, catalogue model"],
    ["Split law (SINDy D)", "66 min", "132 min", "86 min"],
    ["GP coast curve", "19 min", "54 min", "54 min"],
    ["Periodic schedule", "0.7 min", "45 min", "45 min"],
  ], 0.5, 5.6, 12.33, [3.6, 2.4, 3.0, 3.33], 13);
}

// =========================================================================================================
// 30. Extensions
// =========================================================================================================
{
  const s = newSlide("Towards operations: space weather, screening and tasking", [
    "Fast mean-element model calibrated on the full run (period within 0.01%) for hundreds of scenarios.",
    "Space-weather drag: periodic schedule 1642 min at 15 d; law + drag proxy 730 min; perfect forecast 164 min.",
    "Screening: along-track error at 7 d drag-only 396 km vs 11.9 km with the learned law + space weather.",
    "Tasking: custody lost after a burn 74% (1 random obs/day) -> 15-19% with the predicted burn in the catalogue.",
    "Conformal 90% intervals cover 93-99% of new burns.",
  ]);
  img(s, "step10_space_weather/panels/step10_forecast_horizons_b.png", 0.4, 1.15, 4.15, 4.3, "Space-weather drag: forecast error");
  img(s, "step12_conjunction/step12_position_error.png", 4.6, 1.15, 4.15, 4.3, "Along-track error vs prediction time");
  img(s, "step13_tasking/step13_custody.png", 8.8, 1.15, 4.15, 4.3, "Custody lost after a burn");
  bullets(s, [
    "Variable drag changes the burn interval, not the edges: the learned law wins over a schedule",
    "Missed dangerous conjunctions at 1 day: 68% (drag-only) -> 25% (best model)",
    "Also: tracking passes, several control laws, fleet change detection, GEO station keeping",
  ], 0.5, 5.5, 12.33, 1.6, 15);
}

// =========================================================================================================
// 31. What did not work
// =========================================================================================================
{
  const s = newSlide("Negative results worth reporting", [
    "PCA frame for the coast: sparse (4 vs 42 terms) but linear models drift tens of km per day.",
    "Physics library [r/r^3, a_J2, -|v|v]: 0.65 km after 10 days; mu to 2e-8 - the physical frame wins.",
    "Joint regression: biased by mislabelled ramp samples (T -7.6%).",
    "k-means: two clusters split the coast cloud under noise (F1 0.001).",
    "Exponential-density library: physically plausible but wrong here (51 km after a year).",
  ]);
  bullets(s, [
    "PCA + SINDy on the coasting state:",
    { t: "2 modes hold 99.9% of variance (one oscillator pair)" },
    { t: "linear dynamics sparse, eigenvalues = orbital rate" },
    { t: "but a linear change of frame cannot fix nonlinear physics" },
    "Physics library: 0.65 km after 10 days",
    "Joint regression biased; k-means fails; memory-less switching fails",
    "Free-running forecasts, not training fit, decide",
  ], 0.5, 1.25, 5.3, 5.9, 15);
  img(s, "step21_deep_dive/step21_pca_forecast.png", 5.9, 1.2, 6.93, 5.95, "Coast learned in different frames: position error");
}

// =========================================================================================================
// 32. Ranking
// =========================================================================================================
{
  const s = newSlide("Method ranking by stage of the pipeline", [
    "Scores combine accuracy, robustness to noise and sampling, cost and whether uncertainty is reported.",
    "Drag: EKF on sparse data, energy / mean-SMA coast fit on dense data; avoid FD on noisy data.",
    "Detection: GMM / Bayesian GMM; derivative SG up to 2 min, propagation feature to 20 min.",
    "Laws: split learning on clean arcs + hysteresis edges; GP or sigmoid for the thrust shape.",
    "Forecast: law + space-weather proxy with conformal intervals; a periodic schedule only for constant drag.",
  ]);
  table(s, [
    ["Stage", "Best", "Also good", "Avoid"],
    ["Drag c_d", "EKF (sparse data); energy / mean-SMA coast fit", "Shooting; spectroscopy (survey)", "FD regression on noisy data"],
    ["Derivative", "Savitzky-Golay (<= 2 min); propagation (2-20 min)", "High-order FD on clean data", "2nd-order FD"],
    ["Burn detection", "GMM / Bayesian GMM", "MAD threshold; log-chi mixture", "k-means; Isolation Forest"],
    ["Thrust law", "Split regression on burn arcs + sigmoid shape", "GP through stacked burns", "Joint regression"],
    ["Natural law", "Split on coast arcs; physics-shape library", "BINDy for uncertainty", "Wrong-physics or linear-frame libraries"],
    ["Switching", "Hysteresis edges (accuracy 1.00)", "-", "Memory-less sigmoid (0.50)"],
    ["Forecast", "Law + space-weather proxy + conformal", "GP coast curve (fine data)", "Periodic schedule under variable drag"],
  ], 0.5, 1.3, 12.33, [2.0, 4.0, 3.2, 3.13], 13);
}

// =========================================================================================================
// 33. Holistic assessment
// =========================================================================================================
{
  const s = newSlide("Holistic assessment: benefits, improvements, best practices", [
    "Benefit: physics-based features make an unsupervised clusterer near perfect; splitting removes bias.",
    "Benefit: uncertainty everywhere it matters (BINDy, GP, conformal intervals).",
    "Improve: edges dominate the error; event-driven forecasts with modelled third bodies and space weather.",
    "Improve: real ephemerides, multi-seed confidence intervals, angles-only orbit determination.",
    "Practice: known-truth tests, oracle swaps, event scoring, walk-forward tests, closed-form checks, unit tests.",
  ]);
  const cols = [
    ["Benefits", ["Physics features: 4 decades separation", "Unsupervised detection, F1 = 1", "Split learning removes bias",
                  "Uncertainty: BINDy, GP, conformal", "Interpretable laws (edges, c_d, T)"]],
    ["Improvements", ["Learn edges with uncertainty (largest error)", "Event-driven forecast with third bodies", "Log-chi mixture as default detector",
                      "Real ephemerides and passes", "Multi-seed confidence intervals"]],
    ["Best practices", ["Known-truth simulation + oracle swaps", "Score per event, not per sample", "Walk-forward / backward testing",
                        "Bias-variance tuning per sampling step", "Closed-form checks + 53 unit tests"]],
  ];
  cols.forEach(([head, items], i) => {
    const x = 0.5 + i * 4.18;
    s.addText(head, { shape: pres.ShapeType.rect, x, y: 1.3, w: 3.97, h: 0.6, fontFace: "Calibri", fontSize: 18, bold: true,
      color: "000000", align: "center", valign: "middle", line: { color: "000000", width: 1 }, fill: { color: "FFFFFF" }, isTextBox: true });
    bullets(s, items, x, 2.1, 3.97, 5.0, 17);
  });
}

// =========================================================================================================
// 34. Limitations and future work
// =========================================================================================================
{
  const s = newSlide("Limitations and future work", [
    "Inputs are full states at 6 s with 0.1 m / 0.5 mm/s noise: far better than real tracking.",
    "Core run has constant density; space weather is modelled, not measured.",
    "Extensions use a calibrated mean-element model rather than full integration.",
    "No real ephemerides analysed yet (OEM reader exists); no angles-only orbit determination.",
    "Next: real Starlink OEMs, event-driven forecasts with modelled perturbations, closed loop with screening and tasking.",
  ]);
  bullets(s, [
    "Limitations",
    { t: "Idealised measurements (full state, 6 s, small noise)" },
    { t: "Constant density in the core run" },
    { t: "Mean-element model for steps 10-16" },
    { t: "Single noise seed per pipeline study" },
    { t: "Method ranking is a judgement from the numbers" },
    "Future work",
    { t: "Real ephemerides (CCSDS OEM)" },
    { t: "Event-driven forecast with modelled third bodies and density" },
    { t: "Couple forecasts to screening and sensor tasking" },
  ], 0.5, 1.25, 5.5, 5.9, 15);
  img(s, "step20_pipeline/step20_sampling_detection_edges.png", 6.2, 1.2, 6.63, 5.95, "Edges are the weak point at coarse sampling");
}

// =========================================================================================================
// 35. Conclusions
// =========================================================================================================
{
  const s = newSlide("Conclusions", [
    "A smooth, ODE-friendly deadband controller reproduces the brief's Figure 1 and passes 16 closed-form checks.",
    "Burns are detected perfectly (92/92, 0 false) from 6-s to 10-min data with a GMM on physics-based features.",
    "Drag and thrust are learned to < 0.1% when the problem is split onto clean arcs; memory (hysteresis) is essential.",
    "Year-ahead forecast ~1 hour with constant drag; edges dominate the remaining error.",
    "With variable drag and third bodies, a learned law with a drag proxy beats a periodic schedule.",
  ]);
  bullets(s, [
    "Simulation verified: J2 to 0.04%, 16/16 checks",
    "Detection: F1 = 1.0, 6 s - 10 min, 0.25-8x noise",
    "Laws: c_d +0.03%, T -0.005%, edges ~2 m",
    "Forecast: 66 min after a year (constant drag); edges dominate",
    "Operations: screening misses 68% -> 25%; custody lost 74% -> 15%",
  ], 0.5, 1.25, 6.0, 5.9, 16);
  img(s, "step21_deep_dive/step21_hysteresis_rate.png", 6.7, 1.2, 6.13, 5.95, "The learned loop, classified by the GMM");
}

// =========================================================================================================
// 36. References
// =========================================================================================================
{
  const s = newSlide("Key references", [
    "Brief and SDA context: UKSA challenge, Vallado 2025, Marsillach and Holzinger 2021, Bar-Shalom 2011.",
    "Dynamics and solvers: Vallado 2013; Montenbruck and Gill 2000; Petzold 1983 (LSODA).",
    "Signal processing and detection: Savitzky and Golay 1964; Schafer 2011; Dempster et al. 1977 (EM).",
    "Learning: Brunton et al. 2016 (SINDy); Tipping 2001 (ARD); Fung, Fasel and Juniper 2024 (BINDy); Rasmussen and Williams 2006.",
    "Full numbered list with URLs is in the written report.",
  ]);
  bullets(s, [
    "UK Space Agency. Manoeuvre Pattern of Life Prediction for SDA; 2025. Challenge brief.",
    "Vallado DA. Fundamentals of Astrodynamics and Applications, 4th ed.; 2013. Microcosm Press.",
    "Montenbruck O, Gill E. Satellite Orbits; 2000. Springer.",
    "Petzold L. Automatic selection of methods for stiff and nonstiff ODEs; 1983. SIAM J Sci Stat Comput 4(1).",
    "Savitzky A, Golay MJE. Smoothing and differentiation of data; 1964. Analytical Chemistry 36(8).",
    "Dempster AP, Laird NM, Rubin DB. Maximum likelihood via the EM algorithm; 1977. JRSS B 39(1).",
    "Brunton SL, Proctor JL, Kutz JN. Sparse identification of nonlinear dynamics; 2016. PNAS 113(15).",
    "Tipping ME. Sparse Bayesian learning and the relevance vector machine; 2001. JMLR 1.",
    "Fung L, Fasel U, Juniper MP. Rapid Bayesian identification of sparse nonlinear dynamics; 2024. arXiv 2402.15357.",
    "Rasmussen CE, Williams CKI. Gaussian Processes for Machine Learning; 2006. MIT Press.",
  ], 0.5, 1.25, 12.33, 5.9, 14);
}

// =========================================================================================================
// 37. Questions
// =========================================================================================================
{
  const s = newSlide("Questions", [
    "Invite questions; 15 minutes reserved.",
    "Likely topics: why LSODA, why the mean SMA, how the GMM labels burns without supervision.",
    "Likely topics: why split learning beats joint regression; why edges dominate the forecast error.",
    "Likely topics: realism of the noise model and how the method would run on real ephemerides.",
    "Point to the report sections and the code modules for detail.",
  ]);
  bullets(s, [
    "Code: deadband/ (library) and stages/ (figures); python main.py reruns everything",
    "Report: report/briefing.pdf (methods, checks, all horizons)",
    "Backup topics: solver choice, mean SMA, GMM burn rule, BIC vs Brier, split learning, all-forces run",
  ], 0.5, 1.25, 6.2, 5.9, 16);
  img(s, "step1_dynamics/step1_force_budget.png", 6.9, 1.2, 5.93, 5.95, "Force budget (for discussion)");
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("written", f));

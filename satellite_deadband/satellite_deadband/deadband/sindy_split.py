"""
Splitting the learning problem (step 19): learn the THRUST CONTROL LAW and the NATURAL (coasting) DYNAMICS
separately or together, and compare every variant with the true model.

The mean SMA obeys (non-dimensional units, step 9 shapes)
    da/dt = -c_d D(a, |v|)  +  T theta(t) B(a, |v|)          D = (2 a^2/mu)|v|^3,  B = (2 a^2/mu)|v|
    theta = 1 between the switching edges (on at a <= lower, off at a >= upper), else 0
so the whole system is three things: the natural law (c_d), the thrust law (T) and the switching law
(the two edges). Data: step 9's noisy measurements of the 400-day controlled run (reference noise),
the derivative da/dt from the tuned smoothing filter, and the burn/coast clusters from the GMM. Laws are
learned on TRAIN (days 0-40), scored on TEST (days 40-400).

Five scenarios
--------------
    A  whole system      one regression of da/dt on [-D, s B] over ALL samples (s = detected burn flags),
                         edges from the detected burns: everything learned at once, as it comes
    B  thrust only       natural dynamics KNOWN (true c_d): learn T from da/dt + c_d D = T s B, and the edges
    C  natural only      thrust KNOWN (true T and true throttle profile): learn c_d from da/dt - T theta B
                         = -c_d D over ALL samples (burns included); the edges are the known controller's
    D  both, separately  c_d from the clean COAST arcs only, then T from the clean BURN arcs given that c_d
    (each also as BINDy: ARD regression gives a posterior standard deviation for every coefficient)

Scored against the truth: c_d / true, T / true, the edges [m], the RMS error of the predicted da/dt on
TEST against the true (noise-free) da/dt, and the burn-onset error of a free-running 360-day forecast.

Experiment E: the natural law WITHOUT a deadband
------------------------------------------------
The drag-only run (gravity + J2 + drag, no thruster) and the natural run (plus Moon, Sun, SRP) decay
freely over 400 days (~51 km). With constant drag density the true law is
    da/dt = -c_d (2 a^2/mu) |v|^3 = -2 c_d sqrt(mu a)      (circular orbit)
Candidate libraries (constant, polynomials, sqrt(a), the physics shape D, exponential density, power
laws, a kitchen sink) are fitted on days 0-40 and tested on days 40-400, including a free-running
forecast of the SMA to day 400.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import ARDRegression

from .constants import DU, J2, MU, SECONDS, SMOOTH_CD, THRUST_ACCEL, THRUST_ACCEL_MS2, TU
from .long_run import HORIZONS_DAYS, TRAIN_DAYS, drag_only_run, natural_run
from .sindy import drag_shape, stlsq, thrust_shape
from .smooth_controller import mean_sma_km_series

SCENARIOS = ("A. whole system", "B. thrust law only (natural known)", "C. natural law only (thrust known)",
             "D. both, learned separately")
STRIDE = 10                                   # use every 10th 6-s sample (60 s)
TO_M_PER_HOUR = DU * 1000 * 3600 / TU         # non-dimensional da/dt -> m/hour
DAY = 86400.0


def _ard(X, y):
    """BINDy-style ARD fit through the origin: (mean, std) of every coefficient (columns scaled to unit RMS)."""
    scale = np.sqrt(np.mean(X**2, axis=0))
    y_scale = np.std(y) or 1.0
    model = ARDRegression(fit_intercept=False).fit(X / scale, y / y_scale)
    mean = model.coef_ * y_scale / scale
    std = np.zeros(X.shape[1])
    kept = np.abs(model.coef_) > 0
    if model.sigma_.shape[0] == kept.sum():
        std[kept] = np.sqrt(np.diag(model.sigma_)) * y_scale / scale[kept]
    return mean, std


def _ls(X, y):
    """Ordinary least squares via the step 9 STLSQ with the tuned threshold (no column is ever dropped here)."""
    return stlsq(X, y)


@dataclass
class Learned:
    name: str
    cd: float
    T: float
    lower_km: float
    upper_km: float
    cd_std: float
    T_std: float
    known: str


def _prepare():
    from stages.report_step9 import prepare
    dataset, events, data, laws, switching = prepare()
    keep = slice(None, None, STRIDE)
    days = dataset.days[keep]
    D = drag_shape(data.a[keep], data.speed[keep])
    B = thrust_shape(data.a[keep], data.speed[keep])
    s_det = events[keep].astype(float)
    throttle_true = dataset.thrust_ms2[keep] / THRUST_ACCEL_MS2
    a_true = dataset.mean_sma_km[keep] / DU
    speed_true = np.linalg.norm(dataset.v[:, keep], axis=0)
    rate_true = -SMOOTH_CD * drag_shape(a_true, speed_true) + THRUST_ACCEL * throttle_true * thrust_shape(a_true, speed_true)
    return dict(dataset=dataset, days=days, y=data.a_dot[keep], D=D, B=B, s=s_det, theta=throttle_true,
                coast=data.coast[keep], burn=data.burn[keep], rate_true=rate_true, switching=switching,
                train=days < TRAIN_DAYS, test=days >= TRAIN_DAYS)


def learn_scenarios(p):
    """The four learned models (SINDy point estimates with BINDy standard deviations)."""
    from .mean_element import calibrate
    c = calibrate()
    tr = p["train"]
    learned_lower, learned_upper = p["switching"].lower * DU, p["switching"].upper * DU
    out = []
    # A: everything at once, all samples, detected thruster state
    X = np.column_stack([-p["D"], p["s"] * p["B"]])[tr]
    cd, T = _ls(X, p["y"][tr])
    (_, _), (cd_sd, T_sd) = _ard(X, p["y"][tr])
    out.append(Learned(SCENARIOS[0], cd, T, learned_lower, learned_upper, cd_sd, T_sd, "nothing"))
    # B: natural dynamics known
    z = p["y"] + SMOOTH_CD * p["D"]
    X = (p["s"] * p["B"])[:, None][tr]
    (T,) = _ls(X, z[tr])
    _, (T_sd,) = _ard(X, z[tr])
    out.append(Learned(SCENARIOS[1], SMOOTH_CD, T, learned_lower, learned_upper, 0.0, T_sd, "c_d (natural law)"))
    # C: thrust known (true level and true throttle profile); edges known
    z = p["y"] - THRUST_ACCEL * p["theta"] * p["B"]
    X = (-p["D"])[:, None][tr]
    (cd,) = _ls(X, z[tr])
    _, (cd_sd,) = _ard(X, z[tr])
    out.append(Learned(SCENARIOS[2], cd, THRUST_ACCEL, c["lower"], c["upper"], cd_sd, 0.0, "T, throttle, edges"))
    # D: separately, on clean arcs: natural law from coasts, then thrust law from burns
    coast, burn = tr & p["coast"], tr & p["burn"]
    (cd,) = _ls((-p["D"])[coast][:, None], p["y"][coast])
    _, (cd_sd,) = _ard((-p["D"])[coast][:, None], p["y"][coast])
    z = p["y"] + cd * p["D"]
    (T,) = _ls(p["B"][burn][:, None], z[burn])
    _, (T_sd,) = _ard(p["B"][burn][:, None], z[burn])
    out.append(Learned(SCENARIOS[3], cd, T, learned_lower, learned_upper, cd_sd, T_sd, "nothing"))
    return out


def score(p, model):
    """Errors of one learned model against the true model."""
    from .mean_element import calibrate
    c = calibrate()
    te = p["test"]
    thrust_input = p["theta"] if model.name.startswith("C") else p["s"]
    predicted = -model.cd * p["D"] + model.T * thrust_input * p["B"]
    rms = float(np.sqrt(np.mean((predicted[te] - p["rate_true"][te]) ** 2)) * TO_M_PER_HOUR)
    coast_rms = float(np.sqrt(np.mean(((predicted - p["rate_true"])[te & (p["s"] == 0)]) ** 2)) * TO_M_PER_HOUR)
    return dict(scenario=model.name, known=model.known, cd_ratio=model.cd / SMOOTH_CD, T_ratio=model.T / THRUST_ACCEL,
                cd_std_pct=100 * model.cd_std / SMOOTH_CD, T_std_pct=100 * model.T_std / THRUST_ACCEL,
                lower_error_m=1000 * (model.lower_km - c["lower"]), upper_error_m=1000 * (model.upper_km - c["upper"]),
                rate_rms_m_per_hour=rms, coast_rate_rms_m_per_hour=coast_rms, **forecast_errors(p, model))


def forecast_errors(p, model):
    """
    Free-running onsets from the end of the last TRAIN burn: coast at the learned drag rate, burn at the
    learned net thrust rate, between the learned edges (rates at the band centre; the band is 500 m, so
    the shapes D and B are constant to a few ppm). Mean |onset error| [min] within each horizon.
    """
    from .derivative_detection import runs_of_ones
    dataset = p["dataset"]
    burns = runs_of_ones(dataset.t, dataset.true_on)
    onsets = np.array([s for s, _ in burns]) * TU / DAY
    ends = np.array([e for _, e in burns]) * TU / DAY
    # the shapes averaged over the TRAIN coast / burn samples (measured speeds: the orbit is not exactly
    # circular, and |v|^3 averaged over an orbit differs from the circular value by ~0.5%)
    D_bar = np.mean(p["D"][p["train"] & p["coast"]])
    B_bar = np.mean(p["B"][p["train"] & p["burn"]])
    coast_rate = model.cd * D_bar * DU / TU * DAY                                         # km/day, >0 = decay
    burn_rate = (model.T * B_bar - model.cd * D_bar) * DU / TU * DAY
    width = model.upper_km - model.lower_km
    period = width / coast_rate + width / burn_rate
    last_end = ends[ends < TRAIN_DAYS][-1]
    first = last_end + width / coast_rate
    actual = onsets[onsets > TRAIN_DAYS]
    predicted = first + period * np.arange(0, int(400 / period) + 2)
    errors = np.array([predicted[np.argmin(np.abs(predicted - x))] - x for x in actual]) * 1440
    lead = actual - TRAIN_DAYS
    return dict(period_days=float(period),
                forecast_error_min={f"{h}d": float(np.mean(np.abs(errors[lead <= h]))) for h in HORIZONS_DAYS})


def run_scenarios():
    p = _prepare()
    learned = learn_scenarios(p)
    rows = [score(p, m) for m in learned]
    truth_period = float(np.mean(np.diff([s for s, _ in __import__("deadband.derivative_detection", fromlist=["x"])
                                         .runs_of_ones(p["dataset"].t, p["dataset"].true_on)])) * TU / DAY)
    return p, learned, rows, truth_period


# ---------------------------------------------------------------------------
# Experiment E: the natural law without a deadband
# ---------------------------------------------------------------------------
NOISE_POSITION_M, NOISE_VELOCITY_MMS = 0.1, 0.5
SMOOTH_WINDOW = 61                           # 1-hour smoothing derivative of the 1-min mean SMA
H_KM = 60.0


def free_decay_data(source="drag only", seed=0):
    t, states = (drag_only_run() if source == "drag only" else natural_run())
    rng = np.random.default_rng(seed)
    r = states[0:3] + NOISE_POSITION_M / 1000 / DU * rng.standard_normal(states[0:3].shape)
    v = states[3:6] + NOISE_VELOCITY_MMS / 1e6 / (DU / TU) * rng.standard_normal(states[3:6].shape)
    a = mean_sma_km_series(np.vstack([r, v]), j2=J2) / DU
    h = t[1] - t[0]
    rate = savgol_filter(a, SMOOTH_WINDOW, 2, deriv=1, delta=h)
    a_true = mean_sma_km_series(states, j2=J2) / DU
    days = t * TU / DAY
    edge = SMOOTH_WINDOW
    keep = slice(edge, -edge)
    return dict(source=source, days=days[keep], a=a[keep], rate=rate[keep], a_true=a_true[keep],
                speed=np.linalg.norm(v, axis=0)[keep], t=t[keep])


def free_libraries(a0):
    """{name: function(a) -> (theta, names)} for da/dt = f(a) (a non-dimensional; offsets in km)."""
    x = lambda a: (a - a0) * DU                                                      # noqa: E731
    sq = lambda a: np.sqrt(a / a0)                                                   # noqa: E731
    D = lambda a: drag_shape(a, np.sqrt(MU / a)) / drag_shape(a0, np.sqrt(MU / a0))  # noqa: E731
    lib = {
        "1. constant": lambda a: (np.ones((a.size, 1)), ["1"]),
        "2. linear": lambda a: (np.column_stack([np.ones(a.size), x(a)]), ["1", "x"]),
        "3. quadratic": lambda a: (np.column_stack([np.ones(a.size), x(a), x(a) ** 2]), ["1", "x", "x^2"]),
        "4. sqrt(a) (true form)": lambda a: (sq(a)[:, None], ["sqrt(a/a0)"]),
        "5. physics shape D(a) (true form)": lambda a: (D(a)[:, None], ["D(a)/D(a0)"]),
        "6. exponential density": lambda a: (np.exp(-x(a) / H_KM)[:, None], ["exp(-x/H)"]),
        "7. power laws a^-1, a^0.5, a^2": lambda a: (np.column_stack([(a / a0) ** -1, sq(a), (a / a0) ** 2]),
                                                       ["(a/a0)^-1", "(a/a0)^0.5", "(a/a0)^2"]),
    }

    parts = dict(lib)                                   # everything defined so far (not the sink itself)

    def sink(a):
        cols, names = [], []
        for name, f in parts.items():
            th, nm = f(a)
            for k, n in enumerate(nm):
                if n not in names:
                    cols.append(th[:, k]); names.append(n)
        return np.column_stack(cols), names
    lib["8. kitchen sink"] = sink
    return lib


def free_decay_study(source="drag only"):
    d = free_decay_data(source)
    a0 = float(np.median(d["a"][d["days"] < TRAIN_DAYS]))
    train, test = d["days"] < TRAIN_DAYS, d["days"] >= TRAIN_DAYS
    true_rate = -SMOOTH_CD * 2 * np.sqrt(MU * d["a_true"])                         # the law the simulator uses
    rows = []
    for name, library in free_libraries(a0).items():
        theta, names = library(d["a"])
        xi = stlsq(theta[train], d["rate"][train])
        pred = theta @ xi
        # free-running forecast of the SMA from day 40 with the learned law (Euler, 1 h steps)
        t0 = np.searchsorted(d["days"], TRAIN_DAYS)
        a_now, day, path_days, path_a = d["a_true"][t0], d["days"][t0], [], []
        while day < d["days"][-1]:
            th, _ = library(np.array([a_now]))
            a_now = a_now + float((th @ xi)[0]) * (3600 / TU)
            day += 1 / 24
            path_days.append(day); path_a.append(a_now)
        truth = np.interp(path_days, d["days"], d["a_true"])
        err_km = (np.array(path_a) - truth) * DU
        cd_hat = None
        if names == ["sqrt(a/a0)"]:
            cd_hat = -xi[0] / (2 * np.sqrt(MU * a0)) / SMOOTH_CD
        if names == ["D(a)/D(a0)"]:
            cd_hat = -xi[0] / drag_shape(a0, np.sqrt(MU / a0)) / SMOOTH_CD
        rows.append(dict(library=name, names=names, coefficients=xi.tolist(), active=int(np.sum(xi != 0)),
                         test_rms_m_per_hour=float(np.sqrt(np.mean((pred[test] - true_rate[test]) ** 2)) * TO_M_PER_HOUR),
                         sma_error_km_360d=float(err_km[-1]), sma_rms_error_km=float(np.sqrt(np.mean(err_km**2))),
                         cd_ratio=cd_hat, forecast_days=path_days[::24], forecast_err_km=err_km[::24].tolist()))
    return d, rows, true_rate

"""
The SINDy "equation zoo" (step 18): fit MANY candidate equations (libraries) to the satellite's behaviour
and compare them on held-out data, sparsity, information criteria and free-running forecasts.

Three datasets
--------------
    LEO, constant drag   the full-physics 400-day controlled run (6-s states, reference noise), mean SMA
                         and its rate, clusters from the step 9 GMM (report_step9.prepare), every 60 s
    LEO, variable drag   the space-weather scenario of step 10 (mean-element, measured every 10 min),
                         clusters from the detected burns, plus the observable space-weather proxy
    GEO, free drift      the longitude libration of step 17 (measured every 6 h), lambda'' from a smoothing
                         derivative; target law lambda'' = -A sin(2 (lambda - lambda_s))

For the LEO sets the target is da/dt [m/hour] with regressors x = a - a_ref [km], the thruster state s
(from the detected clusters), time t [days] and the proxy p. Candidate libraries range from the step 9
choice (one constant per cluster) through polynomial, physics-shaped, exponential-density, time-polynomial,
Fourier-in-time and space-weather libraries, to a "kitchen sink" and a deliberately WRONG library with no
cluster indicator. For GEO: polynomials of several orders, first / second / mixed Fourier harmonics.

Each library is fitted on TRAIN by STLSQ with a SIGNIFICANCE threshold (a term stays if |coefficient| >= 3
standard errors; classic STLSQ, which drops terms small relative to the largest, prunes the small coast
drag next to the huge rare burn term; both are compared in the step 18 figures) and scored on TEST:
    normalised RMSE   RMS(test residual) / RMS(test target), overall and on the COAST rows only (the
                      burn rows are ~250 x larger and would hide every difference in the drag model)
    active terms      number of non-zero coefficients
    BIC               n ln(RSS / n) + k ln n  (train; lower = better trade-off of fit and size)
    forecast error    LEO: mean |burn-onset error| over 120 free-running days with the learned law and
                      the learned band edges; GEO: RMS longitude error over 35 free-drift days.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from .mean_element import DAY_S, SCALE_HEIGHT_KM
from .sindy import stlsq, stlsq_significance

TO_M_PER_HOUR = 1000 * 3600


@dataclass
class ZooData:
    name: str
    target: np.ndarray          # (N,) the derivative to explain
    x: np.ndarray               # (N,) state: SMA offset [km] (LEO) or longitude [deg] (GEO)
    s: np.ndarray               # (N,) cluster indicator (LEO) or zeros
    t_days: np.ndarray
    p: np.ndarray               # (N,) space-weather proxy (ones when not available)
    train: np.ndarray
    test: np.ndarray
    unit: str
    extra: dict


# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------
def leo_libraries(has_proxy):
    """{name: function(x, s, t, p) -> (theta, term names)} for the LEO datasets."""
    def cols(*pairs):
        return np.column_stack([c for c, _ in pairs]), [n for _, n in pairs]

    c = lambda s: 1 - s                                               # noqa: E731  (coast indicator)
    libs = {
        "1. constant per cluster (step 9)": lambda x, s, t, p: cols((c(s), "coast"), (s, "burn")),
        "2. + linear in a": lambda x, s, t, p: cols((c(s), "coast"), (s, "burn"), (c(s) * x, "coast x"),
                                                    (s * x, "burn x")),
        "3. cubic in a per cluster": lambda x, s, t, p: cols(*[(c(s) * x**k, f"coast x^{k}") for k in range(4)],
                                                             *[(s * x**k, f"burn x^{k}") for k in range(4)]),
        "4. exponential density": lambda x, s, t, p: cols((c(s) * np.exp(-x / SCALE_HEIGHT_KM), "coast e^(-x/H)"),
                                                          (s, "burn"), (s * np.exp(-x / SCALE_HEIGHT_KM), "burn e^(-x/H)")),
        "5. time polynomial (secular)": lambda x, s, t, p: cols((c(s), "coast"), (s, "burn"), (c(s) * t / 100, "coast t"),
                                                                (c(s) * (t / 100) ** 2, "coast t^2")),
        "6. Fourier in time (27 d, 1 yr)": lambda x, s, t, p: cols(
            (c(s), "coast"), (s, "burn"),
            (c(s) * np.sin(2 * np.pi * t / 27), "coast sin 27d"), (c(s) * np.cos(2 * np.pi * t / 27), "coast cos 27d"),
            (c(s) * np.sin(2 * np.pi * t / 365.25), "coast sin 1yr"), (c(s) * np.cos(2 * np.pi * t / 365.25), "coast cos 1yr")),
        "9. no cluster indicator (wrong)": lambda x, s, t, p: cols((np.ones_like(x), "1"), (x, "x"), (x**2, "x^2")),
    }
    if has_proxy:
        libs["7. space-weather proxy"] = lambda x, s, t, p: cols((c(s) * p, "coast p"), (s, "burn"))
        libs["8. proxy x density"] = lambda x, s, t, p: cols((c(s) * p * np.exp(-x / SCALE_HEIGHT_KM), "coast p e^(-x/H)"),
                                                            (s, "burn"), (s * p, "burn p"))

    parts = dict(libs)                                                 # everything defined so far

    def sink(x, s, t, p):
        thetas, names = [], []
        for name, f in parts.items():
            if "wrong" in name:
                continue
            th, nm = f(x, s, t, p)
            for k, n in enumerate(nm):
                if n not in names:
                    thetas.append(th[:, k]); names.append(n)
        return np.column_stack(thetas), names
    libs["10. kitchen sink (all of the above)"] = sink
    return dict(sorted(libs.items(), key=lambda kv: int(kv[0].split(".")[0])))


def geo_libraries():
    def sink(x, s, t, p):
        a, na = poly(5)(x, s, t, p)
        b, nb = fourier([1, 2, 3, 4])(x, s, t, p)
        return np.concatenate([a, b], axis=1), na + nb

    def poly(order):
        return lambda x, s, t, p: (np.column_stack([(x - 70) ** k for k in range(order + 1)]),
                                   [f"(l-70)^{k}" for k in range(order + 1)])

    def fourier(harmonics, constant=False):
        def f(x, s, t, p):
            cols, names = ([np.ones_like(x)], ["1"]) if constant else ([], [])
            for h in harmonics:
                cols += [np.sin(h * np.radians(x)), np.cos(h * np.radians(x))]
                names += [f"sin {h}l", f"cos {h}l"]
            return np.column_stack(cols), names
        return f

    return {"1. constant": poly(0), "2. linear": poly(1), "3. cubic": poly(3), "4. quintic": poly(5),
            "5. Fourier 1st harmonic": fourier([1]), "6. Fourier 2nd harmonic (true form)": fourier([2]),
            "7. Fourier 1st + 2nd": fourier([1, 2]), "8. Fourier 2nd + constant": fourier([2], constant=True),
            "9. Fourier 1st-4th": fourier([1, 2, 3, 4]), "10. kitchen sink (quintic + Fourier 1-4)": sink}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def leo_constant_data():
    """The step 9 data: full-physics controlled run, reference noise, GMM clusters, every 60 s."""
    from stages.report_step9 import prepare
    from .constants import DU, TU
    dataset, events, data, laws, switching = prepare()
    every = slice(None, None, 10)
    keep = (data.coast | data.burn)[every]
    t_days = (data.t_s / DAY_S)[every][keep]
    x = ((data.a - laws.a_ref) * DU)[every][keep]
    s = data.burn[every][keep].astype(float)
    target = (data.a_dot * DU / TU)[every][keep] * TO_M_PER_HOUR
    train, test = t_days < 40, (t_days >= 40) & (t_days < 100)
    return ZooData("LEO, constant drag (full physics)", target, x, s, t_days, np.ones_like(x), train, test,
                   "m/hour", dict(lower=switching.lower * DU - laws.a_ref * DU, upper=switching.upper * DU - laws.a_ref * DU,
                                  onsets_days=None, a_ref_km=laws.a_ref * DU))


def leo_variable_data():
    """The step 10 scenario (variable drag): measured SMA every 10 min, its smoothed rate, detected clusters."""
    from .forecasting import scenario
    from .mean_element import calibrate
    sc = scenario()
    L, t, a = sc.learned, sc.t, sc.a
    h = t[1] - t[0]
    rate = savgol_filter(a, 13, 2, deriv=1, delta=h)
    near = np.zeros(t.size, dtype=bool)
    for on, off in zip(L.onsets, L.ends):
        near |= (t > on - 3600) & (t < off + 3600)                      # burns and their smoothed edges
    a_ref = calibrate()["a_ref"]
    p_all = np.interp(t, sc.weather.t, sc.weather.proxy)
    # coast rows: the smoothed rate away from the burns; burn rows: one per detected burn, its mean rate
    # (a 23-min burn is only ~2 samples at a 10-min cadence, too short for a smoothing derivative)
    mid = 0.5 * (L.onsets + L.ends)
    from .forecasting import SCENARIO_SIGMA_KM
    from .mean_element import detect_burns
    _, _, a_on, a_off = detect_burns(t, a, SCENARIO_SIGMA_KM)
    burn_rate = (a_off - a_on) / (L.ends - L.onsets)
    t_rows = np.concatenate([t[~near], mid])
    x_rows = np.concatenate([a[~near] - a_ref, 0.5 * (a_on + a_off) - a_ref])
    s_rows = np.concatenate([np.zeros((~near).sum()), np.ones(mid.size)])
    y_rows = np.concatenate([rate[~near], burn_rate]) * TO_M_PER_HOUR
    p_rows = np.concatenate([p_all[~near], np.interp(mid, sc.weather.t, sc.weather.proxy)])
    order = np.argsort(t_rows)
    t_days = t_rows[order] / DAY_S
    return ZooData("LEO, variable drag (space weather)", y_rows[order], x_rows[order], s_rows[order], t_days,
                   p_rows[order], t_days < 40, (t_days >= 40) & (t_days < 160), "m/hour",
                   dict(lower=L.lower - a_ref, upper=L.upper - a_ref, a_ref_km=a_ref, scenario=sc))


def geo_data():
    from . import geo
    run = geo.simulate()
    t, lam, _ = geo.measure(run)
    free = t < geo.FREE_DAYS - 2
    acc, smooth = geo.second_derivative(t[free], lam[free])
    edge = 25
    t_f, x = t[free][edge:-edge], smooth[edge:-edge]
    target = acc[edge:-edge]
    train, test = t_f < 80, t_f >= 80
    return ZooData("GEO longitude, free drift", target, x, np.zeros_like(x), t_f, np.ones_like(x), train, test,
                   "deg/day^2", dict(run=run, t=t[free], lam=lam[free]))


# ---------------------------------------------------------------------------
# Fitting and scoring
# ---------------------------------------------------------------------------
def fit_library(data, name, library, threshold=None, method="significance"):
    """method: "significance" (drop terms with |coef| < 3 standard errors) or "relative" (classic STLSQ)."""
    theta, names = library(data.x, data.s, data.t_days, data.p)
    if method == "relative":
        xi = stlsq(theta[data.train], data.target[data.train], threshold=threshold)
    else:
        xi = stlsq_significance(theta[data.train], data.target[data.train])
    resid_train = data.target[data.train] - theta[data.train] @ xi
    resid_test = data.target[data.test] - theta[data.test] @ xi
    n, k = int(data.train.sum()), int(np.sum(xi != 0))
    rss = float(np.sum(resid_train**2))
    coast_test = data.test & (data.s == 0)
    resid_all = data.target - theta @ xi
    coast_nrmse = float(np.sqrt(np.mean(resid_all[coast_test] ** 2)) / np.sqrt(np.mean(data.target[coast_test] ** 2)))         if coast_test.any() else float("nan")
    return dict(library=name, names=names, coefficients=xi, active=k, n_terms=len(names), coast_test_nrmse=coast_nrmse,
                train_nrmse=float(np.sqrt(np.mean(resid_train**2)) / np.sqrt(np.mean(data.target[data.train] ** 2))),
                test_nrmse=float(np.sqrt(np.mean(resid_test**2)) / np.sqrt(np.mean(data.target[data.test] ** 2))),
                bic=float(n * np.log(rss / n) + k * np.log(n)), library_fn=library)


def leo_forecast_error(data, fit, days=120.0, start_day=40.0, dt=300.0):
    """Free-run the learned law (hysteresis at the learned edges) and compare burn onsets with the truth."""
    lower, upper = data.extra["lower"], data.extra["upper"]
    sc = data.extra.get("scenario")
    if sc is not None:
        true_onsets = sc.run.onsets / DAY_S
        weather_t, proxy = sc.weather.t / DAY_S, sc.weather.proxy
    else:
        from .derivative_detection import runs_of_ones
        from .long_run import controlled_run
        from .constants import TU
        run = controlled_run()
        true_onsets = np.array([s for s, _ in runs_of_ones(run.t * TU, run.true_on)]) / DAY_S
        weather_t, proxy = np.array([0.0, 1e4]), np.array([1.0, 1.0])
    library = fit["library_fn"]
    xi = fit["coefficients"]
    # start from the state just after the last true burn before start_day (coasting at the upper edge)
    previous = true_onsets[true_onsets < start_day]
    t = previous[-1] + 0.02 if previous.size else start_day
    x, s, onsets = upper, 0, []
    while t < start_day + days:
        if s == 0 and x <= lower:
            s = 1
            onsets.append(t)
        elif s == 1 and x >= upper:
            s = 0
        p = np.interp(t, weather_t, proxy)
        theta, _ = library(np.array([x]), np.array([float(s)]), np.array([t]), np.array([p]))
        rate_km_per_day = float((theta @ xi)[0]) * 24 / 1000
        if s == 0 and rate_km_per_day >= 0:
            return float("nan")                      # the law never brings the SMA down: it cannot forecast
        x += rate_km_per_day * dt / DAY_S
        if s == 1 and x > upper:                     # a step that would cross an edge stops ON the edge
            x = upper
        elif s == 0 and x < lower:
            x = lower
        t += dt / DAY_S
    onsets = np.array(onsets)
    actual = true_onsets[(true_onsets > start_day) & (true_onsets < start_day + days)]
    if not onsets.size:
        return float("nan")
    return float(np.mean([np.min(np.abs(onsets - o)) for o in actual]) * 1440)   # minutes


def geo_forecast_error(data, fit, start_day=80.0, days=35.0, dt=0.01):
    """Free-run lambda'' = law(lambda) from the measured state at start_day; RMS longitude error [deg]."""
    t_all, lam_all = data.extra["t"], data.extra["lam"]
    i0 = np.searchsorted(t_all, start_day)
    window = slice(max(0, i0 - 20), i0 + 1)
    c = np.polyfit(t_all[window] - t_all[i0], lam_all[window], 2)
    x, v = c[2], c[1]
    library, xi = fit["library_fn"], fit["coefficients"]
    ts, xs = [t_all[i0]], [x]
    t = t_all[i0]
    while t < start_day + days:
        theta, _ = library(np.array([x]), np.zeros(1), np.zeros(1), np.ones(1))
        v += float((theta @ xi)[0]) * dt
        x += v * dt
        t += dt
        ts.append(t); xs.append(x)
    truth = np.interp(ts, data.extra["run"].t, data.extra["run"].lam)
    return float(np.sqrt(np.mean((np.array(xs) - truth) ** 2)))


def run_zoo():
    """{dataset name: list of fit rows (with forecast errors and, for GEO, recovered A and lambda_s)}."""
    from .geo import recover
    results = {}
    for data, libs in ((leo_constant_data(), leo_libraries(False)), (leo_variable_data(), leo_libraries(True)),
                       (geo_data(), geo_libraries())):
        rows = []
        for name, library in libs.items():
            fit = fit_library(data, name, library)
            classic = fit_library(data, name, library, method="relative")
            fit["classic_active"], fit["classic_test_nrmse"] = classic["active"], classic["test_nrmse"]
            fit["classic_names_kept"] = [n for n, c in zip(classic["names"], classic["coefficients"]) if c != 0]
            classic["library_fn"] = library
            if data.name.startswith("GEO"):
                fit["forecast"] = geo_forecast_error(data, fit)
                fit["classic_forecast"] = geo_forecast_error(data, classic)
                fit["forecast_unit"] = "RMS longitude error [deg]"
                fit["recovered"] = recover(list(fit["coefficients"]), fit["names"])
            else:
                fit["forecast"] = leo_forecast_error(data, fit)
                fit["classic_forecast"] = leo_forecast_error(data, classic)
                fit["forecast_unit"] = "mean |burn-onset error| [min]"
            rows.append(fit)
        results[data.name] = (data, rows)
    return results

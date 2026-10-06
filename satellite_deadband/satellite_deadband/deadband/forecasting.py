"""
Forecasting future burns from noisy tracking data, when drag VARIES (space weather), and calibrated
burn-time intervals by split conformal prediction.

The learned control law (steps 8-9) is: burn when the mean SMA reaches the lower edge, until the
upper edge; coast at a drag-set rate in between. A forecast integrates that law forward. The four
forecasters differ only in what they assume about future drag:

    periodic baseline      next burns every <mean TRAIN interval> after the last burn (no physics)
    law, constant drag     the law with the TRAIN-average coast rate
    law, latest coast      the law with the coast rate measured on the most recent coast arc
    law + space weather    the law with drag = k_hat x space-weather proxy; the future proxy comes from
                           27-day solar-rotation persistence (proxy(t) = proxy(t - 27 d))
    law + perfect forecast the law with the TRUE future proxy (upper bound: a perfect space-weather forecast)

When drag is constant all of them are nearly identical; when it varies, the interval between burns
varies (coasts shorten in storms) but the EDGES do not, which is why a learned law can beat a period.

Forecasts are event-driven and fast: with the drag-integral G(t) = int k_hat w_hat dt on the 60-s grid,
a coast from a0 at t0 ends where G(t) - G(t0) = a0 - lower (one searchsorted call).

Conformal intervals (`conformal`): the absolute onset errors of forecasts made from CALIBRATION origins
give, per lead-time bucket, the (1 - alpha) quantile q; the interval for a new forecast is +-q, and
on new data it covers the true onset with probability ~1 - alpha if the errors are exchangeable.
"""
from dataclasses import dataclass

import numpy as np

from .long_run import HORIZONS_DAYS, TRAIN_DAYS
from .mean_element import DAY_S, DT_S, SOLAR_ROTATION_DAYS, calibrate, detect_burns

FORECASTERS = ("periodic baseline", "law, constant drag", "law, latest coast", "law + space weather",
               "law + perfect forecast")
ORIGIN_STEP_DAYS = 5.0
MAX_LEAD_DAYS = max(HORIZONS_DAYS)
LEAD_EDGES_DAYS = (0,) + tuple(HORIZONS_DAYS)
CALIBRATION_END_DAYS = 200.0


@dataclass
class Learned:
    """What is learned from the measured data on TRAIN (days 0-40)."""
    lower: float
    upper: float
    k_mean: float                # mean coast rate [km/s]
    k_proxy: float               # coast rate per unit proxy [km/s]
    T_net: float                 # mean burn rate [km/s]
    period_s: float
    onsets: np.ndarray           # detected onsets (all data)
    ends: np.ndarray
    t: np.ndarray
    a: np.ndarray


def _coast_arcs(t, a, onsets, ends, before=None):
    """(start, end, slope [km/s]) of every measured coast arc (between a burn end and the next onset)."""
    arcs = []
    for end, nxt in zip(ends[:-1], onsets[1:]):
        if before is not None and nxt > before:
            break
        keep = (t > end + 1800) & (t < nxt - 1800)
        if keep.sum() > 10:
            arcs.append((end, nxt, np.polyfit(t[keep], a[keep], 1)[0]))
    return arcs


def learn(t, a_meas, sigma_km, weather, train_days=TRAIN_DAYS):
    onsets, ends, a_on, a_off = detect_burns(t, a_meas, sigma_km)
    train = onsets < train_days * DAY_S
    arcs = _coast_arcs(t, a_meas, onsets, ends, before=train_days * DAY_S)
    slopes = np.array([-s for _, _, s in arcs])
    proxy = np.array([np.mean(weather.proxy[(weather.t > s) & (weather.t < e)]) for s, e, _ in arcs])
    k_proxy = float(np.sum(slopes * proxy) / np.sum(proxy**2))          # least squares through the origin
    durations = (ends - onsets)[train]
    return Learned(lower=float(np.median(a_on[train])), upper=float(np.median(a_off[train])),
                   k_mean=float(np.mean(slopes)), k_proxy=k_proxy,
                   T_net=float(np.median((a_off - a_on)[train] / durations)),
                   period_s=float(np.mean(np.diff(onsets[train]))), onsets=onsets, ends=ends, t=t, a=a_meas)


# ---------------------------------------------------------------------------
# One forecast
# ---------------------------------------------------------------------------
def law_onsets(t0, a0, drag_grid, grid_t, lower, upper, T_net, t_max):
    """
    Burn onsets after t0 predicted by the hysteresis law, starting COASTING at SMA a0.
    drag_grid: predicted coast rate [km/s, > 0] on grid_t. Burns last (upper - lower) / T_net.
    """
    G = np.concatenate([[0.0], np.cumsum(drag_grid[:-1] * DT_S)])       # integral of the coast rate
    onsets, t, a = [], t0, a0
    burn_s = (upper - lower) / T_net
    while t < t_max:
        g0 = np.interp(t, grid_t, G)
        t_on = np.interp(g0 + (a - lower), G, grid_t)                     # coast down to the lower edge
        if t_on >= grid_t[-1] or t_on > t_max:
            break
        onsets.append(t_on)
        t, a = t_on + burn_s, upper
    return np.array(onsets)


def current_state(L, t0):
    """Measured SMA at t0 (linear fit of the last 6 h of coast) or None if a burn is in progress."""
    last_on = L.onsets[L.onsets <= t0]
    last_end = L.ends[L.ends <= t0]
    if last_on.size and (not last_end.size or last_end[-1] < last_on[-1]):
        return None                                                         # mid-burn
    start = max(t0 - 6 * 3600, last_end[-1] + 600 if last_end.size else 0)
    keep = (L.t >= start) & (L.t <= t0)
    if keep.sum() < 5:
        return None
    return float(np.polyval(np.polyfit(L.t[keep], L.a[keep], 1), t0))


def forecast(name, L, weather, t0):
    """Predicted onsets after t0 (up to MAX_LEAD_DAYS) by one forecaster, using data up to t0 only."""
    t_max = t0 + MAX_LEAD_DAYS * DAY_S
    if name == "periodic baseline":
        last = L.onsets[L.onsets <= t0]
        base = last[-1] if last.size else L.onsets[0]
        n = np.arange(1, int((t_max - base) / L.period_s) + 2)
        out = base + n * L.period_s
        return out[(out > t0) & (out <= t_max)]
    a0 = current_state(L, t0)
    if a0 is None:
        return None
    grid_t = weather.t
    if name == "law, constant drag":
        drag = np.full(grid_t.size, L.k_mean)
    elif name == "law, latest coast":
        arcs = _coast_arcs(L.t, L.a, L.onsets, L.ends, before=t0)
        drag = np.full(grid_t.size, -arcs[-1][2] if arcs else L.k_mean)
    elif name == "law + space weather":
        i0 = int(np.searchsorted(grid_t, t0, side="right"))                # first unknown sample
        lag = int(SOLAR_ROTATION_DAYS * DAY_S / DT_S)
        proxy = weather.proxy.copy()
        last_rotation = proxy[max(0, i0 - lag):i0]                         # the last 27 known days ...
        proxy[i0:] = np.resize(last_rotation, proxy.size - i0)             # ... repeated (27-day persistence)
        drag = L.k_proxy * proxy
    else:                                                                   # perfect space-weather forecast
        drag = L.k_proxy * weather.proxy
    return law_onsets(t0, a0, drag, grid_t, L.lower, L.upper, L.T_net, t_max)


# ---------------------------------------------------------------------------
# Rolling-origin evaluation
# ---------------------------------------------------------------------------
def rolling_errors(L, weather, true_onsets, first_day=TRAIN_DAYS, last_day=None, forecasters=FORECASTERS):
    """
    Forecast from an origin every ORIGIN_STEP_DAYS; for every TRUE onset within MAX_LEAD_DAYS of the origin,
    the error = nearest predicted onset - true onset. Rows: (forecaster, origin day, lead days, error s).
    """
    last_day = last_day or weather.t[-1] / DAY_S - 1
    rows = []
    for origin in np.arange(first_day, last_day, ORIGIN_STEP_DAYS):
        t0 = origin * DAY_S
        upcoming = true_onsets[(true_onsets > t0) & (true_onsets <= t0 + MAX_LEAD_DAYS * DAY_S)]
        for name in forecasters:
            predicted = forecast(name, L, weather, t0)
            if predicted is None or not predicted.size:
                continue
            for onset in upcoming:
                error = predicted[np.argmin(np.abs(predicted - onset))] - onset
                rows.append((name, origin, (onset - t0) / DAY_S, error))
    return rows


def horizon_table(rows, forecasters=FORECASTERS):
    """{forecaster: [mean |error| in minutes over all leads <= H for H in HORIZONS_DAYS]}."""
    table = {}
    for name in forecasters:
        mine = np.array([(lead, err) for n, _, lead, err in rows if n == name])
        table[name] = [float(np.mean(np.abs(mine[mine[:, 0] <= h, 1])) / 60) if mine.size else np.nan
                       for h in HORIZONS_DAYS]
    return table


# ---------------------------------------------------------------------------
# Split conformal prediction intervals
# ---------------------------------------------------------------------------
def conformal(rows, name, alpha=0.1, calibration_end=CALIBRATION_END_DAYS):
    """
    Per lead bucket: the conformal half-width q [min] from origins < calibration_end, and its coverage
    and the mean |error| on origins >= calibration_end. Returns a list of dict rows, one per bucket.
    """
    mine = np.array([(origin, lead, err) for n, origin, lead, err in rows if n == name])
    out = []
    for lo, hi in zip(LEAD_EDGES_DAYS[:-1], LEAD_EDGES_DAYS[1:]):
        bucket = mine[(mine[:, 1] > lo) & (mine[:, 1] <= hi)]
        cal = np.abs(bucket[bucket[:, 0] < calibration_end, 2])
        test = np.abs(bucket[bucket[:, 0] >= calibration_end, 2])
        if cal.size < 5 or test.size == 0:
            continue
        level = min(1.0, np.ceil((cal.size + 1) * (1 - alpha)) / cal.size)
        q = float(np.quantile(cal, level))
        out.append(dict(lead_lo=lo, lead_hi=hi, q_min=q / 60, coverage=float(np.mean(test <= q)),
                        n_cal=int(cal.size), n_test=int(test.size)))
    return out


# ---------------------------------------------------------------------------
# The shared scenario of steps 10, 12, 13 and 16
# ---------------------------------------------------------------------------
SCENARIO_DAYS = 400
SCENARIO_SIGMA_KM = 0.02           # measured mean-SMA noise (~ a good orbit determination)
SCENARIO_EVERY_S = 600.0           # one SMA estimate every 10 min


@dataclass
class Scenario:
    weather: object
    run: object
    t: np.ndarray
    a: np.ndarray
    learned: Learned
    rows: list


_SCENARIOS = {}


def scenario(constant_drag=False, seed=3):
    """The 400-day mean-element run (variable or constant drag), its measurements and rolling forecasts (cached)."""
    from .mean_element import measure, simulate, space_weather
    key = (constant_drag, seed)
    if key not in _SCENARIOS:
        weather = space_weather(SCENARIO_DAYS, seed=seed, constant=constant_drag)
        run = simulate(weather)
        t, a = measure(run, SCENARIO_EVERY_S, SCENARIO_SIGMA_KM, seed=seed)
        learned = learn(t, a, SCENARIO_SIGMA_KM, weather)
        _SCENARIOS[key] = Scenario(weather, run, t, a, learned, rolling_errors(learned, weather, run.onsets))
    return _SCENARIOS[key]

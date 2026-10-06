"""
What manoeuvre prediction buys for close-approach screening (step 12) and for sensor tasking and track
association (step 13), using the variable-drag scenario of step 10 (forecasting.scenario).

From SMA error to position error
--------------------------------
An SMA error da changes the mean motion by dn = -(3/2) (n / a) da, so the along-track position error
grows as the integral of that rate:
    ds(t) = -(3/2) n * int_0^t (a_predicted - a_true) dt          [km, n = sqrt(mu / a^3)]
A 500 m station-keeping burn that the catalogue does not know about therefore makes the predicted
position drift by (3/2) n x 0.5 km = 3 km per hour: that is Figure 1 of the brief.

Screening (step 12)
-------------------
For each prediction method and lead time (1-14 days) the along-track errors over many forecast origins
give its error distribution. Monte Carlo encounters at TCA (true miss in an encounter plane) are then
screened: the operator's estimate = truth + that method's error; collision probability Pc for a small
hard body (radius R): Pc ~ pi R^2 x N(miss_est; 0, Sigma) with Sigma = catalogue + method variance.
Alert if Pc > PC_ALERT. A conjunction is DANGEROUS if the true miss < DANGER_KM. Scored: dangerous
conjunctions missed, and false alerts.

Tasking and association (step 13)
---------------------------------
Each day offers PASSES_PER_DAY observation opportunities at random times; the budget is B per day.
After an UNOBSERVED burn, the catalogue (which predicts drag only) drifts at 3 km/h until the next
observation; if the error exceeds the association gate the new track cannot be linked (custody lost).
A manoeuvre-aware catalogue includes the predicted burn, so its error is only (3/2) n da x timing error.
"""
import numpy as np

from .constants import EARTH_MU_KM3_S2
from .forecasting import FORECASTERS, forecast, law_onsets
from .mean_element import DAY_S, DT_S

LEADS_DAYS = (1, 3, 7, 14)
METHODS = ("drag only (no manoeuvres)",) + FORECASTERS
CATALOGUE_SIGMA_KM = 0.05          # position uncertainty of a fresh catalogue state
HARD_BODY_KM = 0.02                # combined hard-body radius
PC_ALERT = 1e-4
DANGER_KM = 0.2
N_ENCOUNTERS = 4000
PASSES_PER_DAY = 6
GATE_KM = 20.0                     # association gate on the along-track error
BUDGETS = (1, 2, 4)
STRATEGIES = ("random", "evenly spaced", "manoeuvre-aware")


def mean_motion(a_km):
    return np.sqrt(EARTH_MU_KM3_S2 / a_km**3)


def predicted_path(method, L, weather, t0, t_end):
    """Predicted mean SMA on the weather grid between t0 and t_end (None if no forecast can be made)."""
    from .forecasting import current_state
    grid = weather.t
    i0, i1 = np.searchsorted(grid, [t0, t_end])
    t = grid[i0:i1]
    a0 = current_state(L, t0)
    if a0 is None:
        return None, None
    if method == "drag only (no manoeuvres)":
        return t, a0 - L.k_mean * (t - t0)
    if method == "periodic baseline":
        onsets = forecast(method, L, weather, t0)
        drag = np.full(grid.size, L.k_mean)
    else:
        onsets = forecast(method, L, weather, t0)
        drag = _drag_grid(method, L, weather, t0)
    if onsets is None:
        return None, None
    G = np.concatenate([[0.0], np.cumsum(drag[:-1] * DT_S)])
    a = np.empty(t.size)
    burn_s = (L.upper - L.lower) / L.T_net
    seg_start, seg_a = t0, a0                    # piecewise: coast (drag integral), then a linear burn
    events = [(on, on + burn_s) for on in onsets if on < t_end]
    for k, ti in enumerate(t):
        burning = next(((on, off) for on, off in events if on <= ti < off), None)
        if burning:
            a[k] = L.lower + L.T_net * (ti - burning[0])
            continue
        done = [off for on, off in events if off <= ti]
        if done:
            seg_start, seg_a = done[-1], L.upper
        else:
            seg_start, seg_a = t0, a0
        a[k] = seg_a - (np.interp(ti, grid, G) - np.interp(seg_start, grid, G))
    return t, a


def _drag_grid(method, L, weather, t0):
    """The future coast rate each law forecaster assumes (same choices as forecasting.forecast)."""
    from .forecasting import _coast_arcs
    from .mean_element import SOLAR_ROTATION_DAYS
    if method in ("law, constant drag", "periodic baseline"):
        return np.full(weather.t.size, L.k_mean)
    if method == "law, latest coast":
        arcs = _coast_arcs(L.t, L.a, L.onsets, L.ends, before=t0)
        return np.full(weather.t.size, -arcs[-1][2] if arcs else L.k_mean)
    if method == "law + space weather":
        i0 = int(np.searchsorted(weather.t, t0, side="right"))
        lag = int(SOLAR_ROTATION_DAYS * DAY_S / DT_S)
        proxy = weather.proxy.copy()
        proxy[i0:] = np.resize(proxy[max(0, i0 - lag):i0], proxy.size - i0)
        return L.k_proxy * proxy
    return L.k_proxy * weather.proxy


def along_track_errors(sc, stride_s=600.0, origin_step_days=5.0, first_day=40.0):
    """{method: {lead_days: array of along-track errors [km] over all origins}}."""
    out = {m: {lead: [] for lead in LEADS_DAYS} for m in METHODS}
    weather, run, L = sc.weather, sc.run, sc.learned
    last = weather.t[-1] / DAY_S - max(LEADS_DAYS) - 1
    for origin in np.arange(first_day, last, origin_step_days):
        t0 = origin * DAY_S
        t_end = t0 + max(LEADS_DAYS) * DAY_S
        for method in METHODS:
            t, a_hat = predicted_path(method, L, weather, t0, t_end)
            if t is None:
                continue
            a_true = np.interp(t, run.t, run.a)
            n = mean_motion(a_true)
            ds = -1.5 * np.cumsum(n * (a_hat - a_true)) * DT_S
            for lead in LEADS_DAYS:
                k = min(int(lead * DAY_S / DT_S), ds.size - 1)
                out[method][lead].append(ds[k])
    return {m: {lead: np.array(v) for lead, v in leads.items()} for m, leads in out.items()}


# ---------------------------------------------------------------------------
# Close-approach screening
# ---------------------------------------------------------------------------
def collision_probability(x, y, sigma_x, sigma_y, radius=HARD_BODY_KM):
    """Pc of a small hard body: pi R^2 x the 2-D Gaussian density of the estimated miss at the origin."""
    density = np.exp(-0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2)) / (2 * np.pi * sigma_x * sigma_y)
    return np.minimum(np.pi * radius**2 * density, 1.0)


def screening(errors, seed=0):
    """
    Rows per (method, lead, covariance) with missed dangerous conjunctions and false alerts. Two covariance
    choices: 'honest' (catalogue + the method's own RMS error) and 'catalogue only' (today's practice:
    the catalogue covariance, which knows nothing about manoeuvres).
    """
    rng = np.random.default_rng(seed)
    x_true = rng.uniform(-3, 3, N_ENCOUNTERS)
    y_true = rng.uniform(-0.5, 0.5, N_ENCOUNTERS)
    dangerous = np.hypot(x_true, y_true) < DANGER_KM
    rows = []
    for method, leads in errors.items():
        for lead, e in leads.items():
            if e.size == 0:
                continue
            draw = rng.choice(e, N_ENCOUNTERS)
            x_est = x_true + draw + CATALOGUE_SIGMA_KM * rng.standard_normal(N_ENCOUNTERS)
            y_est = y_true + CATALOGUE_SIGMA_KM * rng.standard_normal(N_ENCOUNTERS)
            rms = float(np.sqrt(np.mean(e**2)))
            for cov, sx in (("honest", np.hypot(CATALOGUE_SIGMA_KM, rms)), ("catalogue only", CATALOGUE_SIGMA_KM)):
                alert = collision_probability(x_est, y_est, sx, CATALOGUE_SIGMA_KM) > PC_ALERT
                rows.append(dict(method=method, lead=lead, covariance=cov, rms_km=rms,
                                 missed=float(np.mean(~alert[dangerous])),
                                 false_alerts_per_1000=float(1000 * np.mean(alert & ~dangerous)),
                                 n_dangerous=int(dangerous.sum())))
    return rows


# ---------------------------------------------------------------------------
# Sensor tasking and track association
# ---------------------------------------------------------------------------
def _schedule(strategy, budget, opportunities, predicted_ends, rng):
    """Chosen observation times for one day (opportunities sorted)."""
    if strategy == "random":
        return np.sort(rng.choice(opportunities, budget, replace=False))
    targets = list(np.linspace(opportunities[0], opportunities[-1], budget + 2)[1:-1])
    if strategy == "manoeuvre-aware":
        for end in predicted_ends:
            after = opportunities[opportunities >= end]
            if after.size:
                targets = [after[0]] + targets[:budget - 1]
    chosen = sorted({opportunities[np.argmin(np.abs(opportunities - x))] for x in targets})
    for extra in opportunities:                       # top up if two targets picked the same pass
        if len(chosen) >= budget:
            break
        if extra not in chosen:
            chosen.append(extra)
    return np.sort(chosen)[:budget]


def tasking(sc, seed=0):
    """Rows per (strategy, budget): custody loss, latency and association with / without a manoeuvre model."""
    rng = np.random.default_rng(seed)
    run, L = sc.run, sc.learned
    first, last = 40 * DAY_S, run.t[-1] - DAY_S
    burns = [(on, off) for on, off in zip(run.onsets, run.ends) if first < on < last]
    burn_da = L.upper - L.lower
    drift_kmps = 1.5 * mean_motion(L.upper) * burn_da          # along-track drift after an unknown burn
    # predicted burn ends and their timing errors: forecasts from the latest 5-day origin before each burn
    errors_by_onset, best_lead = {}, {}
    for name, origin, lead, err in sc.rows:               # keep the forecast with the SHORTEST lead
        key = round(origin * DAY_S + lead * DAY_S)
        if name == "law + space weather" and lead < best_lead.get(key, np.inf):
            errors_by_onset[key], best_lead[key] = err, lead
    predicted = {on: on + errors_by_onset.get(round(on), 0.0) for on, _ in burns}
    burn_s = (L.upper - L.lower) / L.T_net
    rows = []
    days = np.arange(first / DAY_S, last / DAY_S)
    passes = {d: np.sort(rng.uniform(d * DAY_S, (d + 1) * DAY_S, PASSES_PER_DAY)) for d in days}
    for strategy in STRATEGIES:
        for budget in BUDGETS:
            obs = []
            for d in days:
                ends = [p + burn_s for p in predicted.values() if d * DAY_S <= p + burn_s < (d + 1) * DAY_S]
                obs.extend(_schedule(strategy, budget, passes[d], ends, rng))
            obs = np.array(obs)
            lost, lost_aware, latency = [], [], []
            for on, off in burns:
                after = obs[obs >= off]
                if not after.size:
                    continue
                gap = after[0] - 0.5 * (on + off)
                latency.append((after[0] - off) / 3600)
                lost.append(drift_kmps * gap > GATE_KM)
                # the manoeuvre-aware catalogue is wrong only between the true and the predicted burn
                timing_error = abs(predicted[on] - on)
                lost_aware.append(drift_kmps * min(timing_error, gap) > GATE_KM)
            rows.append(dict(strategy=strategy, budget=budget, custody_lost=float(np.mean(lost)),
                             custody_lost_with_manoeuvre_model=float(np.mean(lost_aware)),
                             mean_latency_h=float(np.mean(latency)), median_latency_h=float(np.median(latency))))
    return rows

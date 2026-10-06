"""
Fleet-level learning (step 15): 40 satellites of one constellation shell, simulated together (mean-element
model, vectorised over satellites), all feeling the SAME space weather but with their own band
(+-5 km), drag (x0.7 ... x1.3, attitude / area differences) and thrust (+-10%).

Three pattern-of-life changes are planted:
    satellite 7    its band moves up 1 km at day 200          (new operational altitude)
    satellite 12   its thrust halves at day 250                (degraded thruster: longer burns)
    satellite 20   its band narrows to 200 m at day 300       (tighter station keeping)

What the fleet view adds
------------------------
1. Change detection: for every satellite, rolling 30-day windows give its learned law (lower edge, band
   width, burn duration, coast rate). Each new window is compared with the satellite's OWN first windows
   (robust z-score); the coast rate is first divided by the fleet median in the same window, which
   removes the common space-weather signal, so only satellite-specific changes remain.
2. Pooled estimation (empirical Bayes): with only a few days of data a single satellite's drag estimate
   is noisy; shrinking it towards the fleet mean, weighted by the two variances, gives a better estimate.
"""
import numpy as np

from .mean_element import DAY_S, SCALE_HEIGHT_KM, calibrate, space_weather

N_SATS = 40
DAYS = 400
DT = 120.0
SIGMA_KM = 0.02
EVERY_S = 600.0
WINDOW_DAYS = 30
STEP_DAYS = 10
BASELINE_WINDOWS = 6
Z_FLAG = 6.0
SPAN = 3                       # burn detection on 30-min rises (a weak thruster still shows)
ANOMALIES = {7: ("band moved up 1 km", 200.0), 12: ("thrust halved", 250.0), 20: ("band narrowed to 200 m", 300.0)}
POOL_DAYS = (2, 4, 8, 15, 30)


def simulate_fleet(seed=0):
    """(t [s], true SMA (S, N), measured SMA (S, M) every EVERY_S, burn on/off (S, N), parameters)."""
    c = calibrate()
    rng = np.random.default_rng(seed)
    weather = space_weather(DAYS, seed=seed + 20)
    t = np.arange(0, DAYS * DAY_S, DT)
    w = np.interp(t, weather.t, weather.w)
    centre = c["a_ref"] + rng.uniform(-5, 5, N_SATS)
    width = np.full(N_SATS, c["upper"] - c["lower"])
    k = c["k"] * rng.uniform(0.7, 1.3, N_SATS)
    T = c["T"] * rng.uniform(0.9, 1.1, N_SATS)
    a = centre + width / 2 - 0.05
    s = np.zeros(N_SATS, dtype=bool)
    A = np.empty((N_SATS, t.size))
    S = np.zeros((N_SATS, t.size), dtype=bool)
    for i, ti in enumerate(t):
        day = ti / DAY_S
        ctr, wid, thrust = centre.copy(), width.copy(), T.copy()
        if day >= 200: ctr[7] += 1.0
        if day >= 250: thrust[12] *= 0.5
        if day >= 300: wid[20] = 0.2
        lower, upper = ctr - wid / 2, ctr + wid / 2
        s = np.where(~s & (a <= lower), True, np.where(s & (a >= upper), False, s))
        a = a + (-k * w[i] * np.exp(-(a - c["a_ref"]) / SCALE_HEIGHT_KM) + thrust * s) * DT
        a = np.where(s, np.minimum(a, upper), np.maximum(a, lower))
        A[:, i], S[:, i] = a, s
    stride = int(EVERY_S / DT)
    measured = A[:, ::stride] + SIGMA_KM * rng.standard_normal(A[:, ::stride].shape)
    return dict(t=t, a=A, s=S, t_meas=t[::stride], a_meas=measured, k=k, T=T, centre=centre, weather=weather)


def window_laws(fleet):
    """laws[sat][window] = (window start day, lower edge, band width, burn minutes, coast rate) from measurements."""
    from .mean_element import detect_burns
    t, A = fleet["t_meas"], fleet["a_meas"]
    starts = np.arange(0, DAYS - WINDOW_DAYS + 1, STEP_DAYS)
    out = np.full((N_SATS, starts.size, 4), np.nan)
    for sat in range(N_SATS):
        onsets, ends, a_on, a_off = detect_burns(t, A[sat], SIGMA_KM, span=SPAN)
        for w, d0 in enumerate(starts):
            mine = (onsets >= d0 * DAY_S) & (onsets < (d0 + WINDOW_DAYS) * DAY_S)
            if mine.sum() < 2:
                continue
            slopes = []
            for end, nxt in zip(ends[mine][:-1], onsets[mine][1:]):
                keep = (t > end + 1800) & (t < nxt - 1800)
                if keep.sum() > 20:
                    slopes.append(np.polyfit(t[keep], A[sat, keep], 1)[0])
            out[sat, w] = (np.median(a_on[mine]), np.median(a_off[mine] - a_on[mine]),
                           np.median((ends - onsets)[mine]) / 60, -np.median(slopes) if slopes else np.nan)
    return starts, out


def change_scores(starts, laws):
    """Robust z-score of every window against the satellite's own first BASELINE_WINDOWS windows."""
    params = laws.copy()
    fleet_coast = np.nanmedian(params[:, :, 3], axis=0)
    params[:, :, 3] = params[:, :, 3] / fleet_coast          # remove the common space-weather signal
    base = params[:, :BASELINE_WINDOWS]
    median = np.nanmedian(base, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(base - median), axis=1, keepdims=True)
    floor = np.array([0.01, 0.01, 2.0, 0.05])[None, None, :]            # km, km, min, ratio: measurement limits
    return (params - median) / np.maximum(mad, floor)


def detections(starts, z):
    """{sat: first window end day where any |z| > Z_FLAG} for every satellite that is flagged."""
    out = {}
    for sat in range(z.shape[0]):
        flagged = np.flatnonzero(np.nan_to_num(np.nanmax(np.abs(z[sat]), axis=1), nan=0.0) > Z_FLAG)
        flagged = flagged[flagged >= BASELINE_WINDOWS]
        if flagged.size:
            out[sat] = float(starts[flagged[0]] + WINDOW_DAYS)
    return out


def pooled_drag(fleet, days_list=POOL_DAYS, first_day=40.0):
    """
    RMS relative error of each satellite's drag coefficient k_i (coast rate per unit space-weather proxy,
    fitted on its coast arcs: k_i = sum(rate x proxy) / sum(proxy^2)) from `days` of data, individual vs
    empirical-Bayes shrinkage towards the fleet mean (weights: the between-satellite variance tau^2 and
    each estimate's own variance, from the scatter of its arcs).
    """
    from .mean_element import detect_burns
    t, A, w = fleet["t_meas"], fleet["a_meas"], fleet["weather"]
    truth = fleet["k"] * np.exp(-(fleet["centre"] - calibrate()["a_ref"]) / SCALE_HEIGHT_KM)
    detected = [detect_burns(t, A[sat], SIGMA_KM, span=SPAN) for sat in range(N_SATS)]
    rows = []
    for days in days_list:
        t0, t1 = first_day * DAY_S, (first_day + days) * DAY_S
        estimates, variances = np.full(N_SATS, np.nan), np.full(N_SATS, np.nan)
        for sat in range(N_SATS):
            onsets, ends, _, _ = detected[sat]
            rates, proxies = [], []
            bounds = np.sort(np.concatenate([[t0], ends[(ends > t0) & (ends < t1)]]))
            for b0 in bounds:
                nxt = onsets[onsets > b0 + 1800]
                b1 = min(nxt[0] if nxt.size else t1, t1)
                keep = (t > b0 + 1800) & (t < b1 - 1800)
                if keep.sum() > 10:
                    rates.append(-np.polyfit(t[keep], A[sat, keep], 1)[0])
                    proxies.append(np.mean(np.interp(t[keep], w.t, w.proxy)))
            if rates:
                rates, proxies = np.array(rates), np.array(proxies)
                k_hat = np.sum(rates * proxies) / np.sum(proxies**2)
                spread = np.var(rates / proxies) if rates.size > 1 else (0.3 * k_hat) ** 2
                estimates[sat], variances[sat] = k_hat, spread / rates.size
        ok = np.isfinite(estimates)
        mu = np.mean(estimates[ok])
        tau2 = max(np.var(estimates[ok]) - np.mean(variances[ok]), 1e-30)
        shrunk = (tau2 * estimates + variances * mu) / (tau2 + variances)
        rows.append(dict(days=days, individual_rms_pct=float(100 * np.sqrt(np.mean((estimates[ok] / truth[ok] - 1) ** 2))),
                         pooled_rms_pct=float(100 * np.sqrt(np.mean((shrunk[ok] / truth[ok] - 1) ** 2))),
                         n_satellites=int(ok.sum())))
    return rows

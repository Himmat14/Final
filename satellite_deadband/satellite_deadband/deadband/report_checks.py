"""
Back-of-the-envelope checks (step 20): every headline number of the report recomputed from a closed-form
formula and compared with what the simulation / analysis measured. A check passes when the two agree to
the stated tolerance; the reason for any residual difference is given with it.

`checks(results)` takes the results dictionary written by main.py (outputs/results.json), so the checks use
exactly the numbers in the report.
"""
import numpy as np

from .constants import (DU, EARTH_EQUATORIAL_RADIUS_KM, EARTH_MU_KM3_S2, INCLINATION, J2, SMOOTH_A_LOWER_KM,
                        SMOOTH_A_UPPER_KM, SMOOTH_CD, THRUST_ACCEL_MS2)

DAY = 86400.0


def _row(name, formula, predicted, measured, unit, tolerance, note="", bound=None):
    """bound=None: the two should agree to `tolerance`; "upper" / "lower": the formula bounds the measurement."""
    rel = abs(predicted - measured) / abs(measured) if measured else abs(predicted)
    if bound == "upper":
        passed = measured <= predicted * (1 + tolerance)
    elif bound == "lower":
        passed = measured >= predicted * (1 - tolerance)
    else:
        passed = rel <= tolerance
    return dict(check=name, formula=formula, predicted=float(predicted), measured=float(measured), unit=unit,
                relative_difference=float(rel), tolerance=tolerance, passed=bool(passed), note=note, bound=bound)


def checks(results):
    from .mean_element import calibrate
    c = calibrate()
    a = c["a_ref"]                                         # mean SMA of the controlled run [km]
    mu = EARTH_MU_KM3_S2
    n = np.sqrt(mu / a ** 3)                               # mean motion [rad/s]
    v = np.sqrt(mu / a)                                    # circular speed [km/s]
    period_s = 2 * np.pi / n
    Re = EARTH_EQUATORIAL_RADIUS_KM
    width = c["upper"] - c["lower"]
    k = c["k"]                                             # measured coast decay [km/s]
    deadband = results["deadband"]
    smooth = results["smooth_deadband"]
    rows = []

    # 1. drag decay of a circular orbit under a = -c_d |v| v:  da/dt = 2 a^2 v a_t / mu = -2 c_d sqrt(mu a)
    cd_km = SMOOTH_CD / DU
    decay = 2 * cd_km * np.sqrt(mu * a)
    rows.append(_row("coast decay rate", r"$\dot a = -2c_d\sqrt{\mu a}$", decay * 3.6e6, k * 3.6e6, "m/h", 0.005,
                     "measured = median slope of the mean SMA between burns"))

    # 2. nodal regression  dOmega/dt = -3/2 n J2 (Re/a)^2 cos i
    rel = results["report_step1"]["j2_raan_check"]["relative_difference_vs_theory"]
    raan = -1.5 * n * J2 * (Re / a) ** 2 * np.cos(INCLINATION) * DAY * 180 / np.pi
    rows.append(_row("J2 nodal regression", r"$\dot\Omega=-\tfrac32 nJ_2(R_e/a)^2\cos i$", raan, raan * (1 + rel),
                     "deg/day", 0.002, "measured on the 400-day natural run (step 1)"))

    # 3. short-period J2 swing of the osculating SMA: peak-to-peak 3 J2 Re^2 / a sin^2 i
    swing = 3 * J2 * Re ** 2 / a * np.sin(INCLINATION) ** 2
    rows.append(_row("osculating SMA swing (J2)", r"$3J_2R_e^2\sin^2 i/a$", swing, smooth["osculating_sma_swing_km"],
                     "km", 0.02, "why the controller must use the MEAN SMA: the swing is 25x the 0.5 km band"))

    # 4. orbital period and the J2 Nyquist limit (signature at 2 x orbital frequency -> sample faster than T/4)
    rows.append(_row("J2 Nyquist sampling limit", r"$T_{\rm orb}/4$", period_s / 4 / 60,
                     results["spectral_nyquist_limit_min"], "min", 0.01))

    # 5. delta-v per burn: Delta a = 2 a Delta v / v  ->  Delta v = v Delta a / (2 a) (+ drag made up during the burn)
    burn_s = c["burn_s"]
    dv = v * width / (2 * a) * 1e3 + (k * burn_s) * v / (2 * a) * 1e3
    rows.append(_row("delta-v per burn", r"$\Delta v = v\,\Delta a/(2a)$", dv, deadband["delta_v_mean_ms"], "m/s", 0.01,
                     "Delta a = effective band + the decay during the burn"))

    # 6. SMA climb rate at full thrust: da/dt = 2 a T / v - k
    climb = (2 * a * THRUST_ACCEL_MS2 * 1e-3 / v - k) * 3.6e6
    rows.append(_row("SMA climb rate in a burn", r"$\dot a = 2aT/v - k$", climb, (c["T"] - k) * 3.6e6, "m/h", 0.01,
                     "measured = SMA gained / burn duration (includes the 30-s throttle ramps)"))

    # 7. burn duration = delta-v / thrust
    rows.append(_row("burn duration", r"$t_b = \Delta v / T$", dv / THRUST_ACCEL_MS2, burn_s, "s", 0.01))

    # 8. deadband period = band / decay rate + burn
    period = width / k + burn_s
    rows.append(_row("burn period", r"$P = w/k + t_b$", period / DAY, deadband["mean_interval_days"], "days", 0.001))

    # 9. nominal band: the effective band differs from the commanded 500 m by the smooth switch (5 m)
    rows.append(_row("band width", r"$a_{\rm up}-a_{\rm low}$", (SMOOTH_A_UPPER_KM - SMOOTH_A_LOWER_KM) * 1e3,
                     width * 1e3, "m", 0.01, "commanded vs effective band"))

    # 10. delta-v budget: one burn per period for 400 days
    rows.append(_row("yearly delta-v budget", r"$N\,\Delta v$", dv * 400 / (period / DAY), deadband["delta_v_total_ms"],
                     "m/s", 0.02, "102 burns in 400 days"))

    # 11. along-track drift of an unknown manoeuvre: Delta n = -3/2 n Delta a / a  ->  a Delta n t = 3/2 n Delta a t
    drift_week = 1.5 * n * width * 7 * DAY
    rms_week = results["report_step12"]["rms_along_track_km"]["drag only (no manoeuvres)"]["7d"]
    rows.append(_row("missed-burn along-track drift (7 d)", r"$\tfrac32 n\,\Delta a\, t$", drift_week, rms_week, "km",
                     0.0, bound="upper", note="brief: 'hundreds of km a week'. The measured RMS is over random "
                                              "phases in the cycle, so it is below the worst case of a full band"))

    # 12. forecast error from a period error: the n-th onset is off by n e P -> mean |error| over H = H |e| / 2
    true_p = results["report_step19"]["true_period_days"]
    for sc in results["report_step19"]["scenarios"][2:]:
        e = abs(sc["period_days"] / true_p - 1)
        rows.append(_row(f"forecast error, scenario {sc['scenario'][0]} (360 d)", r"$\tfrac12 H|\Delta P/P|$",
                         0.5 * 360 * e * 1440, sc["forecast_error_min"]["360d"], "min", 0.05))

    # 13. stacking N burns averages the noise down as 1/sqrt(N) until the bias floor
    nv = dict((int(k_), val) for k_, val in results["report_step8"]["noise_vs_bursts"])
    floor = nv[max(nv)]
    noise1 = np.sqrt(max(nv[1] ** 2 - floor ** 2, 0))
    noise4 = np.sqrt(max(nv[4] ** 2 - floor ** 2, 0))
    rows.append(_row("stacking noise (1 -> 4 burns)", r"$\sigma/\sqrt N$", noise1 / 2, noise4, "m/s$^2$", 0.25,
                     "after removing the bias floor (the derivative's own smoothing error)"))

    # 14. finite-difference noise: sigma(a) = sqrt(6) sigma_r / h^2 -> the c_d error grows linearly with the noise
    fd = results["report_step3_noise"]["median_error_percent"]["white"]["30d"]["cd"]
    rows.append(_row("FD c_d error linear in noise (1 -> 10 m)", r"$\propto\sigma_r$", 10 * fd["1m"], fd["10m"], r"\%",
                     0.05, "white noise, 30 days of 1-min data"))

    # 15. conformal guarantee: coverage >= 1 - alpha on exchangeable data
    cov = [r_["coverage"] for r_ in results["report_step16"]["tables"]["law + space weather"]] \
        if "law + space weather" in results["report_step16"]["tables"] else []
    if cov:
        rows.append(_row("conformal coverage", r"$\ge 1-\alpha$", 0.9, float(np.mean(cov)), "", 0.0, bound="lower",
                         note="mean coverage over lead-time bins; the guarantee is marginal, not per bin"))
    return rows


def to_latex(rows):
    lines = [r"\begin{tabular}{lllrrr}", r"\toprule",
             r"Check & Formula & Unit & Formula & Measured & Diff.\\", r"\midrule"]
    for r in rows:
        mark = r"\checkmark" if r["passed"] else r"$\times$"
        name = r["check"].replace("c_d", "$c_d$").replace("->", r"$\to$")
        lines.append(f"{name} & {r['formula']} & {r['unit']} & {_fmt(r['predicted'])} & {_fmt(r['measured'])} & "
                     f"{_diff(r)} {mark}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _diff(r):
    if r["bound"] == "upper":
        return r"bound $\le$"
    if r["bound"] == "lower":
        return r"bound $\ge$"
    return f"{100 * r['relative_difference']:.2f}\\,\\%"


def _fmt(x):
    if x == 0:
        return "0"
    if abs(x) >= 1000 or abs(x) < 1e-2:
        mant, exp = f"{x:.3e}".split("e")
        return f"${mant}\\times10^{{{int(exp)}}}$"
    return f"{x:.4g}"

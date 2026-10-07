"""
Step 22: the whole chain on a controlled run with EVERY force (J2 + drag + Moon + Sun + SRP).

At 550 km the Moon (8.8e-7 m/s^2), the Sun (4.0e-7) and SRP (1.1e-7) are as large as the drag (8.1e-7), so
they belong in the analysis. The controller still keeps the J2 mean SMA in its band (an operator does not need
them to station-keep), but they now sit in the unmodelled acceleration next to the drag, and they move the
mean SMA periodically. Three analyst set-ups are compared:

    "J2 + drag"                     the run of steps 5-20 (no third bodies anywhere)
    "all forces, J2 model"          truth has Moon / Sun / SRP; the analyst subtracts only gravity + J2
    "all forces, catalogue model"   the analyst also subtracts a Moon / Sun / SRP model (catalogue force model)

For each: the GMM detection (clean and at the reference noise), the coast cloud, the learned c_d, T and edges,
and the forecast error at every horizon, plus the error budget for the all-forces case.
"""
import numpy as np

from .classifiers import detection_dataset
from .constants import ACCEL_UNIT_MS2, SMOOTH_CD, THRUST_ACCEL, TU
from .derivative_detection import runs_of_ones
from .pipeline import (all_forecasts, catalogue_accel_ms2, error_budget, learn_laws, prepare, stage_metrics,
                       true_law)

CASES = (("J2 + drag", "j2_drag", False), ("all forces, J2 model", "full", False),
         ("all forces, catalogue model", "full", True))
DAY = 86400.0


def force_series(days=2.0):
    """Magnitude and along-track part of drag, Moon, Sun and SRP along the controlled all-forces orbit."""
    from .perturbations import MOON_DISTANCE, MOON_RATE, MU_MOON, MU_SUN, SRP_ACCEL, SUN_DISTANCE, SUN_RATE
    ds = detection_dataset("full")
    keep = np.flatnonzero(ds.days < days)[::5]
    t, r, v = ds.t[keep], ds.r[:, keep], ds.v[:, keep]
    vhat = v / np.linalg.norm(v, axis=0)
    out = {"drag": -SMOOTH_CD * np.linalg.norm(v, axis=0) * v}
    for name, rate, distance, mu in (("Moon", MOON_RATE, MOON_DISTANCE, MU_MOON), ("Sun", SUN_RATE, SUN_DISTANCE, MU_SUN)):
        body = distance * np.array([np.cos(rate * t), np.sin(rate * t), np.zeros_like(t)])
        to_body = body - r
        out[name] = mu * (to_body / np.linalg.norm(to_body, axis=0) ** 3 - body / distance ** 3)
        if name == "Sun":
            out["SRP"] = SRP_ACCEL * (-to_body) / np.linalg.norm(to_body, axis=0)
    series = {k: dict(magnitude=np.linalg.norm(a, axis=0) * ACCEL_UNIT_MS2,
                      along=np.sum(a * vhat, axis=0) * ACCEL_UNIT_MS2) for k, a in out.items()}
    return dict(hours=(t - t[0]) * TU / 3600, series=series)


def sma_signature(days=8.0):
    """Mean SMA minus its straight-line coast trend, on the first coast arcs of both runs [m]."""
    out = {}
    for source in ("j2_drag", "full"):
        ds = detection_dataset(source)
        burns = runs_of_ones(ds.t, ds.true_on)
        end, start = burns[0][1], burns[1][0]
        arc = np.flatnonzero((ds.t > end + 600 / TU) & (ds.t < start - 600 / TU))[::10]
        a = ds.mean_sma_km[arc]
        hours = (ds.t[arc] - ds.t[arc[0]]) * TU / 3600
        trend = np.polyval(np.polyfit(hours, a, 1), hours)
        out[source] = dict(hours=hours, residual_m=(a - trend) * 1000, slope_m_per_h=float(np.polyfit(hours, a, 1)[0] * 1000))
    return out


def cases():
    rows = []
    for label, source, catalogue in CASES:
        clean = prepare(1, "gmm", noise_scale=0.0, source=source, catalogue=catalogue)
        p = prepare(1, "gmm", source=source, catalogue=catalogue)
        laws, shapes = learn_laws(p)
        forecasts = all_forecasts(p, laws, shapes)
        metrics = {name: stage_metrics(p, law) for name, law in laws.items()}
        truth = true_law(source)
        rows.append(dict(label=label, source=source, catalogue=catalogue, f1_clean=clean.f1, f1_noisy=p.f1,
                         metrics=metrics, forecasts={m: (e, per) for m, (e, per) in forecasts.items()},
                         true_period_days=float(np.mean(np.diff([s for s, _ in runs_of_ones(p.dataset.t, p.dataset.true_on)])) * TU / DAY),
                         true_edges_km=(truth.lower_km, truth.upper_km)))
        del clean, p
    return rows


def coast_features(source="full", catalogue=False, noise_scale=0.0, n=20000):
    """x1, x2 of TRAIN coast samples (for the coast-cloud comparison figure)."""
    from .classifiers import REFERENCE_NOISE, add_noise
    from .pipeline import _features
    ds = detection_dataset(source)
    r, v = add_noise(ds, REFERENCE_NOISE[0] * noise_scale, REFERENCE_NOISE[1] * noise_scale, 1)
    X, _, _ = _features(ds, r, v, ds.t if catalogue else None)
    rows = np.flatnonzero(ds.train & (ds.true_on == 0))
    pick = np.random.default_rng(0).choice(rows, n, replace=False)
    return X[pick]


def budget_full(catalogue=False):
    rows, _, _, _, effective = error_budget("full", catalogue)
    return rows, effective

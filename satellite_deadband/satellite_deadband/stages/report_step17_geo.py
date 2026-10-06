"""
Report step 17: GEO station keeping (deadband/geo.py): east-west and north-south boxes with impulsive
chemical burns, an electric-propulsion (continuous, low-thrust) variant, learning the longitude law with
SINDy, detecting the burns from 6-hourly tracking and forecasting the next east-west burns.

Figures (outputs/report/step17_geo/)
    step17_longitude        free drift (days 0-120) then station keeping in a 0.1 deg box at 70 deg E
    step17_law              lambda'' measured on the free drift vs the learned law, and the recovered A and lambda_s
    step17_box              inside the box: chemical (impulses) vs electric propulsion (continuous)
    step17_detection        drift-rate jumps from noisy longitude: detected vs true impulses; inclination sawtooth
    step17_forecast         predicted vs true east-west burn times from a learned local acceleration
    step17_summary
"""
import numpy as np
import matplotlib.pyplot as plt

from deadband import geo as G
from deadband.sindy import stlsq
from .common import AMBER, GREEN, GREY, NAVY, RUST
from .report_common import panel_label, report_style, save, step_dir


def learn_law(t, lam):
    free = t < G.FREE_DAYS - 2
    acc, smooth = G.second_derivative(t[free], lam[free])
    edge = 25
    x, y = smooth[edge:-edge], acc[edge:-edge]
    theta = np.column_stack([np.sin(2 * np.radians(x)), np.cos(2 * np.radians(x))])
    xi = stlsq(theta, y)
    A, lam_s = G.recover(list(xi), ["sin 2l", "cos 2l"])
    return dict(x=x, y=y, t=t[free][edge:-edge], xi=xi, A=A, lambda_s=lam_s)


def box_forecast(t, lam, detected, start_day=170.0):
    """Learn the local acceleration and box edges from data before start_day, forecast the next impulses."""
    before = (t > G.FREE_DAYS + 2) & (t < start_day)
    # one parabola per coast segment between detected impulses; the local acceleration = 2 x its curvature
    bounds = np.concatenate([[G.FREE_DAYS + 2], detected[detected < start_day], [start_day]])
    curvatures, weights = [], []
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        seg = (t > b0 + 1.0) & (t < b1 - 1.0)
        if seg.sum() > 10:
            curvatures.append(2 * np.polyfit(t[seg] - t[seg].mean(), lam[seg], 2)[0])
            weights.append(seg.sum() ** 5)                       # curvature variance ~ 1 / n^5
    accel = float(np.average(curvatures, weights=weights)) if curvatures else np.nan
    # the learned edges: the extremes the longitude reaches inside the box
    lo = np.percentile(lam[before], 0.5)
    hi = np.percentile(lam[before], 99.5)
    last = detected[detected < start_day]
    i0 = np.searchsorted(t, start_day)
    window = slice(i0 - 16, i0 + 1)
    c = np.polyfit(t[window] - t[i0], lam[window], 2)
    predicted = G.forecast_ew(t[i0], c[2], c[1], accel if np.isfinite(accel) else 1e-4, lo, hi, G.DAYS)
    return predicted, accel, (lo, hi), last


def _draw_longitude(ax, run, t, lam):
    ax.plot(t, lam, ".", ms=1, color=GREY, label="measured (6-h, 0.0005 deg noise)")
    ax.plot(run.t, run.lam, color=NAVY, lw=0.8, label="truth")
    ax.axvline(G.FREE_DAYS, color=RUST, ls="--", label="relocation into the box")
    ax.set(xlabel="time [days]", ylabel="mean longitude [deg E]", title="Free drift towards 75 deg E, then station keeping at 70 deg E")
    ax.legend(fontsize=7)


def _draw_law(ax, law):
    order = np.argsort(law["x"])
    ax.plot(law["x"], law["y"] * 1e3, ".", ms=2, color=GREY, label="lambda'' from noisy longitude (smoothed)")
    ax.plot(law["x"][order], G.natural_accel(law["x"][order]) * 1e3, color=NAVY, lw=2, label="true law")
    theta = np.column_stack([np.sin(2 * np.radians(law["x"][order])), np.cos(2 * np.radians(law["x"][order]))])
    ax.plot(law["x"][order], theta @ law["xi"] * 1e3, "--", color=RUST, lw=2,
            label=f"SINDy: A = {law['A']:.5f} (true {G.A_TRUE}), lambda_s = {law['lambda_s']:.2f} (true {G.LAMBDA_S})")
    ax.set(xlabel="longitude [deg E]", ylabel="longitude acceleration [1e-3 deg/day$^2$]",
           title="The triaxiality law learned from the free drift")
    ax.legend(fontsize=7)


def _draw_box(ax, chem, ep):
    for run, color, label in ((chem, NAVY, "chemical, impulsive"), (ep, GREEN, "electric propulsion, continuous")):
        box = run.t > G.FREE_DAYS + 1
        ax.plot(run.t[box], run.lam[box], color=color, lw=0.8, label=f"{label} ({len(run.ew_burns)} burn starts)")
    for edge in (G.LAMBDA_BOX - G.BOX_HALF, G.LAMBDA_BOX + G.BOX_HALF):
        ax.axhline(edge, color=RUST, ls="--", lw=0.8)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set(xlabel="time [days]", ylabel="mean longitude [deg E]", title="Inside the +-0.05 deg box")
    ax.legend(fontsize=7)


def _draw_detection(ax, t, lam, inc, run, detected, jumps):
    box = t > G.FREE_DAYS + 1
    ax.plot(t[box], jumps[box] if jumps.size == t.size else np.interp(t[box], t, jumps), color=GREY, lw=0.8,
            label="drift-rate jump (8-day fits)")
    for d in detected:
        ax.axvline(d, color=RUST, ls="--", lw=1)
    ax.plot(run.ew_burns, np.zeros(run.ew_burns.size), "^", ms=9, color=NAVY, label="true impulses")
    twin = ax.twinx()
    twin.plot(t, inc, ".", ms=1, color=AMBER)
    twin.set_ylabel("inclination [deg] (N/S sawtooth)", color=AMBER)
    twin.grid(False)
    ax.set(xlabel="time [days]", ylabel="drift-rate jump [deg/day]", title="E/W impulses found from 6-hourly longitudes (red dashed)")
    ax.legend(fontsize=7, loc="lower left")


def _draw_forecast(ax, predicted, run, accel, edges):
    true = run.ew_burns[run.ew_burns > 170]
    ax.plot(true, np.ones(true.size), "o", ms=10, color=NAVY, label="true impulses")
    ax.plot(predicted, np.zeros(predicted.size), "s", ms=8, color=RUST,
            label=f"forecast from day 170 (learned accel {accel * 1e4:.2f}e-4 deg/day$^2$, "
                  f"edges {edges[0]:.3f} / {edges[1]:.3f})")
    ax.set(yticks=[0, 1], yticklabels=["forecast", "truth"], ylim=(-0.5, 1.5), xlabel="time [days]", ylabel="",
           title="East-west burns forecast 230 days ahead")
    ax.set_ylabel("burn times")
    ax.legend(fontsize=7, loc="center left")


def run(sim, out_dir):
    folder = step_dir(out_dir, "step17_geo")
    chem, ep = G.simulate("chemical"), G.simulate("electric")
    t, lam, inc = G.measure(chem)
    law = learn_law(t, lam)
    box = t > G.FREE_DAYS + 1
    detected, jumps_box = G.detect_ew_burns(t[box], lam[box])
    jumps = np.zeros(t.size); jumps[box] = jumps_box
    predicted, accel, edges, _ = box_forecast(t, lam, detected)
    true_after = chem.ew_burns[chem.ew_burns > 170]
    errors = [float(np.min(np.abs(predicted - x))) for x in true_after] if predicted.size else []
    with report_style():
        fig, ax = plt.subplots(figsize=(11, 4.5)); _draw_longitude(ax, chem, t, lam); save(fig, folder, "step17_longitude.png")
        fig, ax = plt.subplots(figsize=(9, 5)); _draw_law(ax, law); save(fig, folder, "step17_law.png")
        fig, ax = plt.subplots(figsize=(11, 4.5)); _draw_box(ax, chem, ep); save(fig, folder, "step17_box.png")
        fig, ax = plt.subplots(figsize=(11, 4.5)); _draw_detection(ax, t, lam, inc, chem, detected, jumps)
        save(fig, folder, "step17_detection.png")
        fig, ax = plt.subplots(figsize=(11, 3.5)); _draw_forecast(ax, predicted, chem, accel, edges)
        save(fig, folder, "step17_forecast.png")
        fig, axes = plt.subplots(2, 2, figsize=(17, 10))
        _draw_longitude(axes[0, 0], chem, t, lam); _draw_law(axes[0, 1], law); _draw_box(axes[1, 0], chem, ep)
        _draw_detection(axes[1, 1], t, lam, inc, chem, detected, jumps)
        for ax, letter in zip(axes.ravel(), "abcd"):
            panel_label(ax, letter)
        save(fig, folder, "step17_summary.png", suptitle="Step 17: GEO station keeping (E/W and N/S, chemical and electric)")
    return dict(report_step17=dict(
        A_learned=law["A"], lambda_s_learned=law["lambda_s"], A_true=G.A_TRUE, lambda_s_true=G.LAMBDA_S,
        ew_true=chem.ew_burns.tolist(), ew_detected=detected.tolist(), ns_burns=chem.ns_burns.tolist(),
        ep_switch_ons=len(ep.ew_burns), ep_longitude_range=[float(ep.lam[ep.t > 122].min()), float(ep.lam[ep.t > 122].max())],
        forecast_errors_days=errors))

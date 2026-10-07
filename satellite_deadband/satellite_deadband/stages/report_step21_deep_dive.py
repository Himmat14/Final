"""
Report step 21: a closer look at every modelling choice of the chain.
Library: deadband/gmm_physics.py, deadband/deep_dive.py, deadband/pca_coast.py.

Figures (outputs/report/step21_deep_dive/), all single panels:
    step21_subtraction_cascade      sizes of every term of dv/dt and of each numerical error (log scale)
    step21_numerical_error          clean-data error of the unmodelled acceleration vs time, against drag / noise / eps
    step21_basis_raw                the unmodelled acceleration in (radial, along-track) components: burns dwarf coast
    step21_basis_features           the same samples in (x1, x2): log magnitude and along-track, with the wedge |x2| <= e^x1
    step21_hatbox                   x2 / (e^x1 / 1e-4) for coast samples: uniform on [-1, 1] (hat-box theorem)
    step21_coast_tail               coast x1: histogram with Poisson error bars, Gaussian and log-chi fits (fat left tail)
    step21_mixture_fits             x1 of all samples with GMM K = 4, log-chi mixture and Student-t mixture
    step21_information_criteria     BIC / AIC vs K, with the two 2-component alternatives
    step21_sg_response              frequency response of the SG derivative vs finite differences and the ideal one
    step21_sg_noise_gain            white-noise gain of the SG derivative vs window and order
    step21_switches                 the smooth switch on(a), off(a) and the throttle theta(s) vs the ideal piecewise law
    step21_thrust_models            stacked burns: truth, GP law and the sigmoid (piecewise-smooth) law
    step21_thrust_residuals         the residuals of the two thrust laws against the truth
    step21_hysteresis_field         the controller as a 2-state ODE: (a, s) vector field and the true trajectory with arrows
    step21_hysteresis_rate          the (a, da/dt) loop with direction arrows, coloured by the GMM decision
    step21_sawtooth                 the mean-SMA sawtooth over 20 days, true and measured
    step21_pca_variance             PCA of the coasting state: variance per mode
    step21_pca_forecast             position error of linear / PCA / physics SINDy models of the coast
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from deadband import deep_dive as D
from deadband import gmm_physics as G
from deadband import pca_coast as PC
from deadband.classifiers import REFERENCE_NOISE, add_noise, build_features, detection_dataset
from deadband.constants import SMOOTH_A_LOWER_KM, SMOOTH_A_UPPER_KM, THRUST_ACCEL_MS2
from .common import AMBER, GREEN, GREY, NAVY, PURPLE, RUST, SKY_BLUE, SLATE
from .report_common import report_style, save, step_dir

SIZE = (8.5, 5.0)
BURN_CMAP = LinearSegmentedColormap.from_list("coast_to_burn", [NAVY, PURPLE, RUST])


# ---------------------------------------------------------------------------
# Subtraction and numerical error
# ---------------------------------------------------------------------------
def _draw_cascade(ax, est):
    order = ["gravity", "J2", "thrust", "derivative noise at the reference noise", "drag (the smallest real signal)",
             "total numerical error (clean data)", "SG truncation (exact Kepler orbit)",
             "solver tolerance 1e-11 through the SG filter", "round-off of the SG derivative (eps |v| ||c||)",
             "machine epsilon x |gravity| (subtraction round-off)"]
    colors = [SLATE, SLATE, RUST, GREY, NAVY, PURPLE, PURPLE, PURPLE, PURPLE, PURPLE]
    y = np.arange(len(order))[::-1]
    values = [est[k] for k in order]
    ax.barh(y, values, color=colors)
    for yi, val in zip(y, values):
        ax.text(val * 1.6, yi, f"{val:.1e}", va="center", fontsize=8)
    ax.set(xscale="log", xlim=(1e-16, 1e3), yticks=y, xlabel="acceleration [m/s$^2$] (log scale)",
           ylabel="term", title="The subtraction: dv/dt - gravity - J2 leaves drag + thrust + errors")
    ax.set_yticklabels(order, fontsize=8)


def _draw_numerical(ax, s):
    keep = slice(s["window"], -s["window"])
    ax.semilogy(s["hours"][keep], s["error_norm"][keep], color=PURPLE, lw=0.6, label="clean-data error of the unmodelled acceleration")
    est = s["estimates"]
    for key, color, ls in (("drag (the smallest real signal)", NAVY, "-"),
                           ("derivative noise at the reference noise", GREY, "--"),
                           ("SG truncation (exact Kepler orbit)", GREEN, "-."),
                           ("machine epsilon x |gravity| (subtraction round-off)", RUST, ":")):
        ax.axhline(est[key], color=color, ls=ls, lw=1.5, label=key)
    on = s["on"].astype(bool)
    if on.any():
        ax.axvspan(s["hours"][on][0], s["hours"][on][-1], color=AMBER, alpha=0.2, label="burn")
    ax.set(xlabel="time [hours]", ylabel="|error| [m/s$^2$]", ylim=(1e-16, 1e-4),
           title="Noise-free data: the numerical error is 80x below drag and 10$^7$x above machine epsilon")
    ax.legend(fontsize=7.5, loc="upper right")


# ---------------------------------------------------------------------------
# The x1, x2 basis
# ---------------------------------------------------------------------------
def _basis_samples(seed=1, n=6000):
    ds = detection_dataset()
    keep = np.flatnonzero(ds.days < 10)
    r, v = add_noise(ds, REFERENCE_NOISE[0], REFERENCE_NOISE[1], seed)
    f = build_features(r[:, keep], v[:, keep], ds.step_s)
    rr, vv = ds.r[:, keep], ds.v[:, keep]
    rhat = rr / np.linalg.norm(rr, axis=0)
    radial = np.sum(f.accel_from_v * rhat, axis=0)
    on = ds.true_on[keep].astype(bool)
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.choice(np.flatnonzero(~on), n, replace=False), np.flatnonzero(on)])
    return dict(radial=radial[pick], along=f.along_track[pick], X=f.matrix[pick], on=on[pick])


def _draw_basis_raw(ax, b):
    ax.scatter(b["radial"][~b["on"]] / 1e-4, b["along"][~b["on"]] / 1e-4, s=3, color=SKY_BLUE, label="coast")
    ax.scatter(b["radial"][b["on"]] / 1e-4, b["along"][b["on"]] / 1e-4, s=5, color=AMBER, label="burn")
    ax.set(xlabel="radial component [1e-4 m/s$^2$]", ylabel="along-track component [1e-4 m/s$^2$]",
           title="Raw basis (radial, along-track): the coast collapses to a dot at the origin")
    ax.legend(fontsize=8)


def _draw_basis_features(ax, b):
    X = b["X"]
    ax.scatter(X[~b["on"], 0], X[~b["on"], 1], s=3, color=SKY_BLUE, label="coast")
    ax.scatter(X[b["on"], 0], X[b["on"], 1], s=5, color=AMBER, label="burn")
    grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
    ax.plot(grid, np.exp(grid) / 1e-4, color=SLATE, ls="--", lw=1, label="|x2| = exp(x1) / 1e-4 (all along-track)")
    ax.plot(grid, -np.exp(grid) / 1e-4, color=SLATE, ls="--", lw=1)
    ax.set(xlabel="x1 = log |a_u|", ylabel="x2 = along-track / 1e-4 m/s$^2$", ylim=(-0.6, 2.6),
           title="Feature basis (x1, x2): four decades separate the clusters; the coast fills a wedge")
    ax.legend(fontsize=8, loc="upper left")


def _draw_hatbox(ax, X, truth):
    u = X[~truth, 1] / (np.exp(X[~truth, 0]) / 1e-4)
    ax.hist(u, bins=60, density=True, color=SKY_BLUE, label="coast samples")
    ax.axhline(0.5, color=RUST, ls="--", lw=1.5, label="uniform on [-1, 1] (isotropic noise)")
    ax.set(xlabel="x2 / (exp(x1) / 1e-4) = cosine of the angle to the velocity", ylabel="density",
           title="Hat-box theorem: the along-track share of isotropic noise is uniform")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Tails and mixtures
# ---------------------------------------------------------------------------
def _draw_tail(ax, ts):
    c, d, n = ts["centres"], ts["density"], ts["counts"]
    err = np.sqrt(np.maximum(n, 1)) / (ts["n"] * ts["width"])
    ok = n > 0
    ax.errorbar(c[ok], d[ok], yerr=err[ok], fmt="o", ms=3, color=SKY_BLUE, ecolor=GREY, elinewidth=0.8,
                label="coast samples (+-1 Poisson sd per bin)")
    grid = np.linspace(c.min() - 0.3, c.max() + 0.3, 400)
    mu, sd = ts["gauss"]
    ax.plot(grid, np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)), color=GREY, ls="--", lw=1.5,
            label=f"single Gaussian (log-lik {ts['loglik_gauss']:.0f})")
    ax.plot(grid, np.exp(G.log_chi_logpdf(grid, ts["k"], ts["log_sigma"])), color=RUST, lw=2,
            label=f"log-chi, k = {ts['k']:.2f} (log-lik {ts['loglik_chi']:.0f})")
    ref = ts["mode"] - 2.0
    left = grid < ts["mode"] - 0.5
    ax.plot(grid[left], np.exp(G.log_chi_logpdf(ref, ts["k"], ts["log_sigma"])) * np.exp(3 * (grid[left] - ref)),
            color=NAVY, ls=":", lw=1.5, label=f"left-tail law e$^{{3 x_1}}$ (measured slope {ts['left_tail_slope']:.2f})")
    ax.set(yscale="log", ylim=(1e-4, 5), xlabel="x1 = log |unmodelled acceleration|", ylabel="probability density",
           title="Coast feature: a fat exponential left tail and a thin right tail (log-chi)")
    ax.legend(fontsize=7.5, loc="lower center")


def _draw_mixture_fits(ax, comp):
    X = comp["X_train"]
    bins = np.linspace(X[:, 0].min(), X[:, 0].max(), 160)
    counts, _ = np.histogram(X[:, 0], bins)
    width = bins[1] - bins[0]
    centres = 0.5 * (bins[1:] + bins[:-1])
    ax.bar(centres, counts / (len(X) * width), width, color=SKY_BLUE, alpha=0.6, label="all TRAIN samples")
    grid = np.linspace(bins[0] - 0.3, bins[-1] + 0.3, 600)
    gmm = comp["models"]["GMM, K = 4"]
    dens = sum(w * np.exp(-0.5 * (grid - m[0]) ** 2 / c[0][0]) / np.sqrt(2 * np.pi * c[0][0])
               for w, m, c in zip(gmm.weights_, gmm.means_, gmm.covariances_))
    ax.plot(grid, dens, color=SLATE, lw=1.5, label="GMM, K = 4 (23 parameters)")
    lc = comp["models"]["log-chi coast + Gaussian burn"]
    k, log_sigma = lc.coast["params"][:2]
    mb, cb = lc.burn["mean"], lc.burn["cov"]
    dens_lc = ((1 - lc.weight_burn) * np.exp(G.log_chi_logpdf(grid, k, log_sigma))
               + lc.weight_burn * np.exp(-0.5 * (grid - mb[0]) ** 2 / cb[0, 0]) / np.sqrt(2 * np.pi * cb[0, 0]))
    ax.plot(grid, dens_lc, color=RUST, lw=2, ls="--", label="log-chi coast + Gaussian burn (10 parameters)")
    tm = comp["models"]["Student-t coast + Gaussian burn"]
    from scipy import stats
    scale = np.sqrt(tm.coast["cov"][0, 0])
    dens_t = ((1 - tm.weight_burn) * stats.t.pdf(grid, tm.coast["df"], tm.coast["mean"][0], scale)
              + tm.weight_burn * np.exp(-0.5 * (grid - tm.burn["mean"][0]) ** 2 / tm.burn["cov"][0, 0])
              / np.sqrt(2 * np.pi * tm.burn["cov"][0, 0]))
    ax.plot(grid, dens_t, color=GREEN, lw=1.5, ls="-.", label=f"Student-t coast (df {tm.coast['df']:.1f}) + Gaussian burn")
    ax.set(yscale="log", ylim=(1e-4, 5), xlabel="x1 = log |unmodelled acceleration|", ylabel="probability density",
           title="Mixtures along x1: fat-tailed coast + thin-tailed burn")
    ax.legend(fontsize=7.5, loc="upper right")


def _draw_ic(ax, rows):
    gmm = [r for r in rows if r["model"].startswith("GMM")]
    k = [r["k"] for r in gmm]
    base = min(r["bic"] for r in rows)
    ax.plot(k, [r["bic"] - base for r in gmm], "o-", color=NAVY, label="GMM: BIC - best BIC")
    ax.plot(k, [r["aic"] - min(x["aic"] for x in rows) for r in gmm], "s--", color=GREY, label="GMM: AIC - best AIC")
    for r, color, ls in zip([r for r in rows if not r["model"].startswith("GMM")], (RUST, GREEN), (":", "-.")):
        ax.axhline(r["bic"] - base, color=color, ls=ls, lw=1.8, label=f"{r['model']}: BIC ({r['n_params']} parameters)")
    ax.axvline(4, color=GREY, ls=":", lw=1)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(-1, None)
    ax.set(xlabel="number of Gaussian components K", ylabel="criterion above the best model (0 = best)",
           title="BIC = -2 ln L + p ln n: components are added until the penalty outweighs the fit")
    ax.legend(fontsize=7.5)


# ---------------------------------------------------------------------------
# Savitzky-Golay
# ---------------------------------------------------------------------------
def _draw_sg_response(ax, sg):
    f = sg["freqs"]
    styles = {"ideal differentiator": ("black", ":", 2.0)}
    palette = [GREY, SLATE, PURPLE, SKY_BLUE, NAVY, RUST]
    for (name, curve), color in zip(sg["curves"].items(), [None] + palette):
        c, ls, lw = styles.get(name, (color, "-" if "Savitzky" in name else "--", 1.5))
        ax.loglog(f * 3600, np.maximum(curve, 1e-12), color=c, ls=ls, lw=lw, label=name)
    orbit = 1 / (95.53 * 60)
    for x, label in ((orbit * 3600, "orbit"), (2 * orbit * 3600, "J2 (2/orbit)"), (sg["cutoff_hz"] * 3600, "tuned SG cut-off")):
        ax.axvline(x, color=GREY, lw=0.8)
        ax.text(x * 1.08, 1e-4, label, rotation=90, fontsize=8, color=GREY)
    ax.set(xlabel="frequency [cycles per hour]", ylabel="|H(f)| [1/s]", ylim=(1e-5, 1),
           title="Derivative filters: SG follows the ideal i 2 pi f, then rolls off the noise")
    ax.legend(fontsize=7.5, loc="upper left")


def _draw_sg_gain(ax, sg):
    for (order, gains), color in zip(sg["gains"].items(), (NAVY, GREEN, RUST, PURPLE)):
        ax.loglog(np.array(sg["windows"]) * sg["step_s"], gains, "o-", color=color, label=f"order {order}")
    ax.axvline(sg["tuned"][0] * sg["step_s"], color=GREY, ls=":", label=f"tuned: {sg['tuned'][0]} samples, order {sg['tuned'][1]}")
    ax.set(xlabel="window length [s]", ylabel="noise gain sqrt(sum c$_j^2$) [1/s]",
           title="White noise passed by the SG derivative: falls as window$^{-3/2}$")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Thrust model
# ---------------------------------------------------------------------------
def _draw_switches(ax):
    from deadband.constants import SWITCH_SMA_WIDTH_M
    a = np.linspace(-30, 30, 600)
    sw = lambda x: 0.5 * (1 + np.tanh(x / SWITCH_SMA_WIDTH_M))
    ax.plot(a, sw(-a), color=NAVY, lw=2, label="on(a) = sigma(a_L - a), w = 5 m (smooth)")
    ax.plot(a, (a < 0).astype(float), color=NAVY, ls=":", lw=1.5, label="ideal: switch on below a_L (Heaviside)")
    ax.plot(a, sw(a), color=RUST, lw=2, label="off(a) = sigma(a - a_U) (relative to a_U)")
    ax.plot(a, (a > 0).astype(float), color=RUST, ls=":", lw=1.5, label="ideal: switch off above a_U")
    ax.set(xlabel="mean SMA minus the band edge [m]", ylabel="switch value",
           title="The piecewise deadband law and its tanh (sigmoid) smoothing")
    ax.legend(fontsize=7.5, loc="center right")


def _draw_throttle(ax):
    from deadband.smooth_controller import throttle
    s = np.linspace(0, 1, 500)
    ax.plot(s, throttle(s), color=RUST, lw=2, label="throttle theta(s) = sigma((s - 0.9) / 0.02)")
    ax.plot(s, s * (1 - s) * (2 * s - 1), color=NAVY, lw=1.5, ls="--", label="latch(s) = s(1 - s)(2s - 1)")
    ax.axvline(0.5, color=GREY, ls=":", lw=1)
    ax.set(xlabel="thruster state s", ylabel="value", title="Throttle and latch: the engine opens only once s has committed")
    ax.legend(fontsize=8)


def _draw_thrust_models(ax, m):
    t = m["offsets_s"] / 60
    ax.plot(t, m["stack_mean"] * 1e4, color=GREY, lw=0.8, label=f"stacked mean of {m['bursts']} noisy burns")
    ax.plot(t, m["truth"] * 1e4, color="black", lw=2, label="true thrust profile")
    ax.plot(t, m["gp_mean"] * 1e4, color=NAVY, lw=1.5, label=f"GP law (RMS error {m['rms_gp'] * 1e6:.1f}e-6)")
    ax.fill_between(t, (m["gp_mean"] - 2 * m["gp_std"]) * 1e4, (m["gp_mean"] + 2 * m["gp_std"]) * 1e4, color=NAVY, alpha=0.15)
    ax.plot(t, m["sigmoid"] * 1e4, color=RUST, ls="--", lw=1.5,
            label=f"sigmoid law, 6 parameters (RMS error {m['rms_sigmoid'] * 1e6:.1f}e-6)")
    ax.set(xlabel="minutes from the burn start", ylabel="along-track acceleration [1e-4 m/s$^2$]",
           title="Thrust law from repetition: non-parametric GP vs parametric sigmoid")
    ax.legend(fontsize=7.5, loc="center")


def _draw_thrust_residuals(ax, m):
    t = m["offsets_s"] / 60
    ax.plot(t, (m["gp_mean"] - m["truth"]) * 1e6, color=NAVY, lw=1.2, label="GP - truth")
    ax.plot(t, (m["sigmoid"] - m["truth"]) * 1e6, color=RUST, lw=1.2, ls="--", label="sigmoid - truth")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="minutes from the burn start", ylabel="error [1e-6 m/s$^2$]",
           title="Residuals: both laws err only on the ramps (derivative smoothing)")
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Hysteresis, sawtooth, PCA
# ---------------------------------------------------------------------------
def _arrows(ax, x, y, color, n=10):
    idx = np.linspace(0, len(x) - 2, n).astype(int)
    for i in idx:
        ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=14))


def _draw_field(ax, h):
    a = np.linspace(SMOOTH_A_LOWER_KM - 0.05, SMOOTH_A_UPPER_KM + 0.05, 40)
    s = np.linspace(0, 1, 30)
    A, S = np.meshgrid(a, s)
    a_dot, s_dot = D.reduced_field(A, S)
    # directions only: the two rates differ by orders of magnitude, so each is normalised by its range
    u = a_dot / 1316.0
    w = s_dot / 60.0
    ax.streamplot(A, S, u, w, color=GREY, density=1.1, linewidth=0.7, arrowsize=0.8)
    coast, burn = h["on"] == 0, h["on"] == 1
    ax.plot(h["a"], h["s"], color=SLATE, lw=0.8, alpha=0.6)
    ax.scatter(h["a"][coast][::50], h["s"][coast][::50], s=6, color=NAVY, label="coasting (s = 0, a decreasing)")
    ax.scatter(h["a"][burn][::5], h["s"][burn][::5], s=6, color=RUST, label="burning (s = 1, a increasing)")
    for x, label in ((SMOOTH_A_LOWER_KM, "a_L: switch on"), (SMOOTH_A_UPPER_KM, "a_U: switch off")):
        ax.axvline(x, color=PURPLE, ls=":", lw=1.2)
        ax.text(x, 0.5, label, rotation=90, fontsize=8, color=PURPLE, ha="right")
    ax.annotate("", xy=(6922.0, 0.03), xytext=(6922.3, 0.03), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.5))
    ax.annotate("", xy=(6922.3, 0.97), xytext=(6922.0, 0.97), arrowprops=dict(arrowstyle="-|>", color=RUST, lw=2.5))
    ax.annotate("", xy=(SMOOTH_A_LOWER_KM - 0.01, 0.85), xytext=(SMOOTH_A_LOWER_KM - 0.01, 0.15),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=2))
    ax.annotate("", xy=(SMOOTH_A_UPPER_KM + 0.01, 0.15), xytext=(SMOOTH_A_UPPER_KM + 0.01, 0.85),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=2))
    ax.ticklabel_format(axis="x", useOffset=False)
    ax.set(xlabel="mean SMA a [km]", ylabel="thruster state s", ylim=(-0.05, 1.05),
           title="State space (a, s): vector field of the controller ODE and the true cycle")
    ax.legend(fontsize=8, loc="center", bbox_to_anchor=(0.5, 0.3), framealpha=0.95)


def _draw_rate_loop(ax, h, prob):
    points = ax.scatter(h["a"][::10], h["rate"][::10], c=prob[::10], cmap=BURN_CMAP, s=5, vmin=0, vmax=1)
    plt.colorbar(points, ax=ax, label="GMM P(burn | x)")
    coast = np.flatnonzero(h["on"] == 0)
    burn = np.flatnonzero(h["on"] == 1)
    _arrows(ax, h["a"][coast][::400], h["rate"][coast][::400], NAVY, n=6)
    _arrows(ax, h["a"][burn][::20], h["rate"][burn][::20], RUST, n=5)
    ax.ticklabel_format(axis="x", useOffset=False)
    ax.set(xlabel="mean SMA a [km]", ylabel="da/dt [m/hour]",
           title="Hysteresis loop (a, da/dt) and its direction of travel")
    for x, y0, y1, color in ((SMOOTH_A_LOWER_KM - 0.004, 0, 1250, SLATE), (SMOOTH_A_UPPER_KM + 0.004, 1250, 0, SLATE)):
        ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
    ax.text(SMOOTH_A_LOWER_KM + 0.01, 600, "switch on at a_L", fontsize=8, color=SLATE)
    ax.text(SMOOTH_A_UPPER_KM - 0.11, 600, "switch off at a_U", fontsize=8, color=SLATE)
    ax.text(6922.1, 60, "coast: -5.3 m/h", fontsize=8, color=NAVY)
    ax.text(6922.1, 1350, "burn: +1300 m/h", fontsize=8, color=RUST)


def _draw_sawtooth(ax, sw):
    ax.plot(sw["measured_days"], sw["measured"], ".", ms=2, color=GREY, label="measured mean SMA (1 m / 5 mm/s noise, every 10 min)")
    ax.plot(sw["days"], sw["a"], color=NAVY, lw=1.5, label="true mean SMA")
    for i, (s, e) in enumerate(sw["burns"]):
        ax.axvspan(s, e, color=RUST, alpha=0.4, label="burn (22.8 min)" if i == 0 else None)
    ax.axhline(SMOOTH_A_LOWER_KM, color=RUST, ls="--", lw=1, label="band edges")
    ax.axhline(SMOOTH_A_UPPER_KM, color=RUST, ls="--", lw=1)
    if len(sw["burns"]) > 1:
        ax.annotate("", xy=(sw["burns"][1][0], SMOOTH_A_UPPER_KM + 0.03), xytext=(sw["burns"][0][0], SMOOTH_A_UPPER_KM + 0.03),
                    arrowprops=dict(arrowstyle="<->", color=SLATE))
        ax.text(0.5 * (sw["burns"][0][0] + sw["burns"][1][0]), SMOOTH_A_UPPER_KM + 0.04, "period 3.905 d", ha="center", fontsize=8)
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set(xlabel="time [days]", ylabel="mean SMA [km]", ylim=(SMOOTH_A_LOWER_KM - 0.08, SMOOTH_A_UPPER_KM + 0.08),
           title="The deadband sawtooth: slow drag decay, fast burn back to the upper edge")
    ax.legend(fontsize=7.5, loc="lower left")


def _draw_pca_variance(ax, fit):
    ex = fit["pca"]["explained"]
    ax.bar(np.arange(1, len(ex) + 1), np.maximum(ex, 1e-12), color=NAVY)
    for i, e in enumerate(ex):
        ax.text(i + 1, max(e, 1e-12) * 1.5, f"{e:.1e}", ha="center", fontsize=8)
    ax.set(yscale="log", ylim=(1e-12, 3), xlabel="PCA mode", ylabel="share of variance",
           title="PCA of the coasting state (r, v): one oscillator pair holds 99.9%")


def _draw_pca_forecast(ax, fc):
    styles = {"Cartesian, linear": (GREY, "--"), "PCA, linear, 6 modes": (NAVY, "-"), "PCA, linear, 4 modes": (SKY_BLUE, "-."),
              "PCA, quadratic, 4 modes": (PURPLE, ":"), "Cartesian, physics library": (RUST, "-")}
    orbit = 96                                                  # samples per orbit (1-min data)
    for name, f in fc.items():
        c, ls = styles[name]
        err = np.maximum(f["position_km"], 1e-4)
        n = len(err) // orbit * orbit
        envelope = err[:n].reshape(-1, orbit).max(axis=1)        # worst error within each orbit (no wiggles)
        ax.semilogy(f["days"][:n:orbit], envelope, color=c, ls=ls, lw=2, marker="o", ms=3, markevery=15, label=name)
    ax.set(xlabel="days after the end of TRAIN (2 days)", ylabel="largest position error in each orbit [km]",
           title="Coasting dynamics learned in different frames: free-running forecast error")
    ax.legend(fontsize=8)


def run(sim, out_dir):
    folder = step_dir(out_dir, "step21_deep_dive")
    budget = G.subtraction_budget()
    basis = _basis_samples()
    comp = G.compare_models()
    tails = G.tail_study(comp["X_train"], comp["truth_train"])
    sg = D.sg_study()
    thrust = D.thrust_models()
    hyst = D.hysteresis_data()
    saw = D.sawtooth_data()
    pca_fit = PC.fit_models(noise=True)
    pca_fc = PC.forecast(pca_fit)
    gmm4 = comp["models"]["GMM, K = 4"]
    # GMM P(burn) along the hysteresis cycle (noisy features of those days)
    ds = detection_dataset()
    keep = (ds.days >= hyst["days"][0]) & (ds.days <= hyst["days"][-1])
    r, v = add_noise(ds, REFERENCE_NOISE[0], REFERENCE_NOISE[1], 1)
    Xh = build_features(r[:, keep], v[:, keep], ds.step_s).matrix
    from deadband.classifiers import mixture_burn_components
    prob = gmm4.predict_proba(Xh)[:, mixture_burn_components(gmm4)].sum(axis=1)

    with report_style():
        for name, draw, args in (
                ("step21_subtraction_cascade.png", _draw_cascade, (budget["estimates"],)),
                ("step21_numerical_error.png", _draw_numerical, (budget,)),
                ("step21_basis_raw.png", _draw_basis_raw, (basis,)),
                ("step21_basis_features.png", _draw_basis_features, (basis,)),
                ("step21_hatbox.png", _draw_hatbox, (comp["X_train"], comp["truth_train"])),
                ("step21_coast_tail.png", _draw_tail, (tails,)),
                ("step21_mixture_fits.png", _draw_mixture_fits, (comp,)),
                ("step21_information_criteria.png", _draw_ic, (comp["rows"],)),
                ("step21_sg_response.png", _draw_sg_response, (sg,)),
                ("step21_sg_noise_gain.png", _draw_sg_gain, (sg,)),
                ("step21_switches.png", _draw_switches, ()),
                ("step21_throttle.png", _draw_throttle, ()),
                ("step21_thrust_models.png", _draw_thrust_models, (thrust,)),
                ("step21_thrust_residuals.png", _draw_thrust_residuals, (thrust,)),
                ("step21_hysteresis_field.png", _draw_field, (hyst,)),
                ("step21_hysteresis_rate.png", _draw_rate_loop, (hyst, prob)),
                ("step21_sawtooth.png", _draw_sawtooth, (saw,)),
                ("step21_pca_variance.png", _draw_pca_variance, (pca_fit,)),
                ("step21_pca_forecast.png", _draw_pca_forecast, (pca_fc,))):
            fig, ax = plt.subplots(figsize=SIZE)
            draw(ax, *args)
            save(fig, folder, name)

    clean = lambda d: {k: (float(v) if isinstance(v, (np.floating, float, int)) else v) for k, v in d.items()
                       if not isinstance(v, np.ndarray)}
    return dict(report_step21=dict(
        subtraction=budget["estimates"], tails=clean(tails),
        mixtures=comp["rows"],
        log_chi_mixture=dict(params=comp["models"]["log-chi coast + Gaussian burn"].coast["params"].tolist(),
                             weight_burn=comp["models"]["log-chi coast + Gaussian burn"].weight_burn),
        t_mixture_df=comp["models"]["Student-t coast + Gaussian burn"].coast["df"],
        sg=dict(cutoff_hz=sg["cutoff_hz"], cutoff_period_min=1 / sg["cutoff_hz"] / 60, tuned=sg["tuned"],
                gains={str(k): v for k, v in sg["gains"].items()}, windows=sg["windows"]),
        thrust=dict(rms_gp=thrust["rms_gp"], rms_sigmoid=thrust["rms_sigmoid"], rms_stack=thrust["rms_stack"],
                    gp=thrust["gp"], sigmoid={k: v for k, v in thrust["sigmoid_params"].items() if k != "std"},
                    sigmoid_std=thrust["sigmoid_params"]["std"],
                    sigmoid_true={k: v for k, v in thrust["sigmoid_true_params"].items() if k != "std"}),
        pca=dict(explained=pca_fit["pca"]["explained"].tolist(), rows=PC.summary(pca_fit, pca_fc),
                 bindy_active=pca_fit["bindy_active"],
                 eigen_6=[str(np.round(e, 5)) for e in pca_fit["models"]["PCA, linear, 6 modes"]["eigen"]],
                 physics_xi=pca_fit["models"]["Cartesian, physics library"]["xi"].tolist()),
    ))

"""
Can a PCA frame help SINDy / BINDy learn the COASTING dynamics (step 21)?

Data: the drag-only run (J2 + drag, no thruster, 1-min samples). TRAIN = days 0-2; the learned models are
then run freely from day 2 for 10 days and mapped back to (r, v).

Frames and libraries
--------------------
    Cartesian, linear          x_dot = A x + b            x = (r, v), 6 x 7 dense coefficients
    PCA, linear (6 or 4 modes) z_dot = A_z z              z = W^T (x - mean) / scale, sparse by STLSQ
    PCA, quadratic (4 modes)   z_dot = [z, z_i z_j] xi
    Cartesian, physics         r_dot = v,  v_dot = xi_1 r/|r|^3 + xi_2 a_J2-shape + xi_3 (-|v| v)

A linear change of frame cannot change what a LINEAR least-squares model can fit (A_z = W^T A W): PCA helps
only by (i) truncating to the dominant modes and (ii) making the coefficient matrix sparse (an orbit is two
coupled oscillator pairs). The coasting dynamics, however, are NONLINEAR: 1/r^2 gravity, the J2 term at twice
the orbital frequency, and the slow drag decay. A linear model in any frame captures the rotation only for a
while; the physics library (or the mean-element frame of steps 9 and 19) captures the decay itself.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.signal import savgol_filter

from .constants import ACCEL_UNIT_MS2, DU, J2, MU, TU, VU
from .derivative_detection import known_accel
from .long_run import drag_only_run
from .sindy import stlsq

DAY = 86400.0
TRAIN_DAYS = 2.0
FORECAST_DAYS = 10.0
SG_WINDOW, SG_ORDER = 11, 5


def _data(noise=True, seed=0):
    t, states = drag_only_run()
    days = t * TU / DAY
    keep = days < TRAIN_DAYS + FORECAST_DAYS
    t, x = t[keep], states[:, keep].copy()
    if noise:
        rng = np.random.default_rng(seed)
        x[:3] += 0.1e-3 / DU * rng.standard_normal(x[:3].shape)
        x[3:] += 0.5e-6 / VU * rng.standard_normal(x[3:].shape)
    h = t[1] - t[0]
    x_dot = savgol_filter(x, SG_WINDOW, SG_ORDER, deriv=1, delta=h, axis=1)
    return t, x, x_dot, days[keep], states[:, keep]


def _pca(x):
    mean = x.mean(axis=1, keepdims=True)
    scale = x.std(axis=1, keepdims=True)
    u, s, _ = np.linalg.svd((x - mean) / scale, full_matrices=False)
    return mean, scale, u, s ** 2 / np.sum(s ** 2)


def _quadratic(z):
    k = z.shape[0]
    cols = [z[i] for i in range(k)] + [z[i] * z[j] for i in range(k) for j in range(i, k)]
    return np.array(cols)


def fit_models(noise=True, threshold=0.05):
    t, x, x_dot, days, truth = _data(noise)
    train = days < TRAIN_DAYS
    xt, xdt = x[:, train], x_dot[:, train]
    mean, scale, W, explained = _pca(xt)
    models = {}

    # Cartesian linear with a constant: [x; 1]
    theta = np.vstack([xt, np.ones(xt.shape[1])]).T
    coef = np.linalg.lstsq(theta, xdt.T, rcond=None)[0]                     # (7, 6)
    models["Cartesian, linear"] = dict(kind="cart_linear", A=coef[:6].T, b=coef[6], terms=int(np.sum(np.abs(coef) > 0)))

    # PCA frames
    z = W.T @ ((xt - mean) / scale)
    zd = W.T @ (xdt / scale)
    for k in (6, 4):
        Z, ZD = z[:k], zd[:k]
        xi = np.array([stlsq(Z.T, ZD[i], threshold) for i in range(k)])        # (k, k)
        models[f"PCA, linear, {k} modes"] = dict(kind="pca_linear", A=xi, k=k, terms=int(np.sum(xi != 0)),
                                                eigen=np.linalg.eigvals(xi))
    Q = _quadratic(z[:4])
    xi = np.array([stlsq(Q.T, zd[i], threshold) for i in range(4)])
    models["PCA, quadratic, 4 modes"] = dict(kind="pca_quad", xi=xi, k=4, terms=int(np.sum(xi != 0)))

    # Cartesian physics library for v_dot (r_dot = v)
    r, v = xt[:3], xt[3:]
    radius = np.linalg.norm(r, axis=0)
    shapes = [r / radius ** 3, known_accel(r) - known_accel(r, j2=0.0), -np.linalg.norm(v, axis=0) * v]
    theta = np.column_stack([s.ravel() for s in shapes])
    xi = np.linalg.lstsq(theta, xdt[3:].ravel(), rcond=None)[0]
    models["Cartesian, physics library"] = dict(kind="physics", xi=xi, terms=3,
                                                mu_ratio=float(-xi[0] / MU), j2_ratio=float(xi[1]),
                                                cd_ratio=None)
    # BINDy (ARD) on the 6-mode PCA frame: posterior sd of every coefficient
    from sklearn.linear_model import ARDRegression
    sds = []
    for i in range(6):
        ard = ARDRegression(fit_intercept=False).fit(z.T, zd[i])
        sds.append(np.sqrt(np.diag(ard.sigma_)) if ard.sigma_.size else np.zeros(0))
    pca = dict(mean=mean, scale=scale, W=W, explained=explained)
    return dict(models=models, pca=pca, t=t, x=x, days=days, truth=truth, train=train,
                bindy_active=[int(np.sum(np.abs(ARDRegression(fit_intercept=False).fit(z.T, zd[i]).coef_) > 1e-6 * np.abs(zd[i]).max()))
                              for i in range(6)])


def _rhs(model, pca):
    mean, scale, W = pca["mean"][:, 0], pca["scale"][:, 0], pca["W"]
    kind = model["kind"]
    if kind == "cart_linear":
        return lambda t, x: model["A"] @ x + model["b"]
    if kind == "physics":
        xi = model["xi"]

        def f(t, x):
            r, v = x[:3], x[3:]
            accel = (xi[0] * r / np.linalg.norm(r) ** 3
                     + xi[1] * (known_accel(r[:, None])[:, 0] - known_accel(r[:, None], j2=0.0)[:, 0])
                     - xi[2] * np.linalg.norm(v) * v)
            return np.concatenate([v, accel])
        return f
    k = model["k"]

    def f(t, x):
        z = W[:, :k].T @ ((x - mean) / scale)
        zd = model["A"] @ z if kind == "pca_linear" else model["xi"] @ _quadratic(z[:, None])[:, 0]
        return scale * (W[:, :k] @ zd)
    return f


def forecast(fit):
    """Run every model freely from the end of TRAIN; position error [km] and mean-SMA error [m] vs time."""
    from .smooth_controller import mean_sma_km_series
    t, days, truth = fit["t"], fit["days"], fit["truth"]
    i0 = int(np.argmax(~fit["train"]))
    x0 = fit["x"][:, i0]
    t_eval = t[i0:]
    true_sma = mean_sma_km_series(truth[:, i0:])
    out = {}
    for name, model in fit["models"].items():
        try:
            sol = solve_ivp(_rhs(model, fit["pca"]), (t_eval[0], t_eval[-1]), x0, t_eval=t_eval, method="DOP853",
                            rtol=1e-10, atol=1e-12)
            pred = sol.y
            n = pred.shape[1]
            err = np.linalg.norm(pred[:3] - truth[:3, i0:i0 + n], axis=0) * DU
            sma_err = (mean_sma_km_series(pred) - true_sma[:n]) * 1000
        except Exception:
            err = sma_err = np.full(1, np.nan)
            n = 1
        out[name] = dict(days=days[i0:i0 + n] - days[i0], position_km=err, sma_m=sma_err)
    return out


def summary(fit, fc):
    rows = []
    for name, model in fit["models"].items():
        f = fc[name]
        at = lambda d: float(np.interp(d, f["days"], f["position_km"])) if f["days"][-1] >= d else np.nan
        rows.append(dict(model=name, terms=model["terms"], pos_1orbit_km=at(95.5 / 1440), pos_1d_km=at(1.0),
                         pos_10d_km=at(FORECAST_DAYS - 0.01),
                         sma_10d_m=float(f["sma_m"][-1]) if f["days"][-1] >= FORECAST_DAYS - 0.01 else np.nan))
    return rows

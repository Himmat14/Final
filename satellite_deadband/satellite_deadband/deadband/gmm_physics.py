"""
Why the GMM's coast cloud looks the way it does, and better-shaped mixtures (step 21).

The features (classifiers.Features.matrix)
------------------------------------------
    x1 = ln |a_u|                         a_u = unmodelled acceleration (3-vector) [m/s^2]
    x2 = (a_u . v_hat) / 1e-4             its along-track component, in units of 1e-4 m/s^2

While coasting, a_u is mostly derivative noise n (plus the small drag). If n is isotropic Gaussian with
standard deviation sigma per axis, |n| / sigma follows a chi distribution with k = 3 degrees of freedom, and
x1 = ln |n| has the density ("log-chi")

    f(x1) = 2 / (2^(k/2) Gamma(k/2)) * exp( k (x1 - ln sigma) - exp(2 (x1 - ln sigma)) / 2 ).

Its LEFT tail falls only like exp(k x1) (a straight line on a log-density plot: a fat, exponential tail
towards tiny accelerations, because a 3-vector is rarely close to zero but can be), while its RIGHT tail falls
like exp(-exp(2 x1)) (far thinner than a Gaussian). A Gaussian mixture needs several components to imitate
this skewed shape, which is why the BIC keeps falling as components are added.

Given the magnitude, the along-track component of an isotropic vector is UNIFORM on [-|n|, |n|]
(Archimedes' hat-box theorem): x2 is confined to the wedge |x2| <= exp(x1) / 1e-4 with variance
exp(2 x1) / 3 / 1e-8. The coast model below uses x2 | x1 ~ N(mu2, c exp(2 x1) / 1e-8), with c = 1/3 for
isotropic noise and mu2 the drag (a small negative along-track offset).

A burn is the opposite: a nearly constant, forward-pointing acceleration, so its features form a narrow,
THIN-tailed cluster; a Gaussian is appropriate.

Models compared (all fitted by EM on the same 20 000 TRAIN samples):
    GMM K = 1..7               sklearn, 6K - 1 parameters
    log-chi coast + Gaussian burn          10 parameters (the physics-shaped mixture)
    Student-t coast + Gaussian burn        12 parameters (a generic fat-tailed coast)
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats
from sklearn.mixture import GaussianMixture

from .classifiers import REFERENCE_NOISE, add_noise, build_features, detection_dataset, mixture_burn_components
from .derivative_detection import event_scores, group_into_events, runs_of_ones
from .settings import tuned

FIT_SAMPLES = 20_000
SCALE_X2 = 1e-4          # x2 is in units of 1e-4 m/s^2


# ---------------------------------------------------------------------------
# Densities
# ---------------------------------------------------------------------------
def log_chi_logpdf(x, k, log_sigma):
    """log density of x = ln r when r / sigma ~ chi_k."""
    z = x - log_sigma
    return np.log(2.0) - (k / 2) * np.log(2.0) - special.gammaln(k / 2) + k * z - np.exp(2 * z) / 2


def coast_logpdf(X, params):
    """Physics-shaped coast density: x1 ~ log-chi(k, sigma); x2 | x1 ~ N(mu2, c exp(2 x1) / SCALE_X2^2)."""
    k, log_sigma, mu2, log_c = params
    sd2 = np.sqrt(np.exp(log_c)) * np.exp(X[:, 0]) / SCALE_X2
    return log_chi_logpdf(X[:, 0], k, log_sigma) + stats.norm.logpdf(X[:, 1], mu2, sd2)


def gauss_logpdf(X, mean, cov):
    return stats.multivariate_normal.logpdf(X, mean, cov, allow_singular=True)


def t_logpdf(X, mean, cov, df):
    return stats.multivariate_t.logpdf(X, mean, cov, df=df)


# ---------------------------------------------------------------------------
# Two-component mixtures fitted by EM
# ---------------------------------------------------------------------------
@dataclass
class Mixture:
    name: str
    coast: dict
    burn: dict
    weight_burn: float
    n_params: int
    history: list = field(default_factory=list)

    def component_logpdf(self, X):
        if self.coast["kind"] == "log-chi":
            lc = coast_logpdf(X, self.coast["params"])
        else:
            lc = t_logpdf(X, self.coast["mean"], self.coast["cov"], self.coast["df"])
        lb = gauss_logpdf(X, self.burn["mean"], self.burn["cov"])
        return np.log(1 - self.weight_burn) + lc, np.log(self.weight_burn) + lb

    def loglik(self, X):
        lc, lb = self.component_logpdf(X)
        return np.logaddexp(lc, lb)

    def burn_probability(self, X):
        lc, lb = self.component_logpdf(X)
        return np.exp(lb - np.logaddexp(lc, lb))


def _initial_split(X):
    """Start EM from the step 6 rule: burn = far above the coast in x1 (3 coast sd)."""
    model = GaussianMixture(n_components=tuned("gmm_components"), n_init=1, random_state=0).fit(X)
    burn = mixture_burn_components(model)[model.predict(X)]
    return burn.astype(float)


def fit_log_chi_mixture(X, iterations=25):
    r = _initial_split(X)
    params = np.array([3.0, np.median(X[:, 0]) - 0.5 * np.log(3.0), 0.0, np.log(1 / 3)])
    history = []
    for _ in range(iterations):
        w = r.mean()
        mb = np.sum(r[:, None] * X, axis=0) / r.sum()
        cb = np.cov(X.T, aweights=r + 1e-12) + 1e-9 * np.eye(2)
        weights = 1 - r

        def nll(p):
            if p[0] <= 0.2:
                return 1e12
            return -np.sum(weights * coast_logpdf(X, p))
        params = optimize.minimize(nll, params, method="Nelder-Mead",
                                   options=dict(xatol=1e-5, fatol=1e-3, maxiter=400)).x
        model = Mixture("log-chi coast + Gaussian burn", dict(kind="log-chi", params=params),
                        dict(mean=mb, cov=cb), w, n_params=10)
        lc, lb = model.component_logpdf(X)
        r = np.exp(lb - np.logaddexp(lc, lb))
        history.append(float(np.sum(model.loglik(X))))
    model.history = history
    return model


def fit_t_mixture(X, iterations=40):
    """Student-t coast + Gaussian burn by expectation-conditional-maximisation (latent t scale weights)."""
    r = _initial_split(X)
    coast = X[r < 0.5]
    mean, cov, df = coast.mean(axis=0), np.cov(coast.T), 5.0
    history = []
    for _ in range(iterations):
        w = r.mean()
        mb = np.sum(r[:, None] * X, axis=0) / r.sum()
        cb = np.cov(X.T, aweights=r + 1e-12) + 1e-9 * np.eye(2)
        gamma = 1 - r
        diff = X - mean
        delta = np.einsum("ni,ij,nj->n", diff, np.linalg.inv(cov), diff)
        u = (df + 2) / (df + delta)                                      # E[scale weight | x]
        mean = np.sum((gamma * u)[:, None] * X, axis=0) / np.sum(gamma * u)
        diff = X - mean
        cov = (gamma * u * diff.T) @ diff / gamma.sum() + 1e-12 * np.eye(2)
        df = optimize.minimize_scalar(lambda nu: -np.sum(gamma * t_logpdf(X, mean, cov, nu)),
                                      bounds=(0.5, 200), method="bounded").x
        model = Mixture("Student-t coast + Gaussian burn", dict(kind="t", mean=mean, cov=cov, df=float(df)),
                        dict(mean=mb, cov=cb), w, n_params=12)
        lc, lb = model.component_logpdf(X)
        r = np.exp(lb - np.logaddexp(lc, lb))
        history.append(float(np.sum(model.loglik(X))))
    model.history = history
    return model


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------
def _events_f1(dataset, flags):
    events = group_into_events(flags.astype(int), min_samples=10)
    test = dataset.test
    return event_scores(dataset.t[test], events[test], dataset.true_on[test], dataset.step_s, 60.0)


def compare_models(seed=1, max_k=7, source="j2_drag"):
    """Fit every model on TRAIN, score it on TRAIN (BIC, AIC) and on held-out TEST samples and events."""
    ds = detection_dataset(source)
    r, v = add_noise(ds, REFERENCE_NOISE[0], REFERENCE_NOISE[1], seed)
    X = build_features(r, v, ds.step_s).matrix
    rng = np.random.default_rng(0)
    train_rows = rng.choice(np.flatnonzero(ds.train), FIT_SAMPLES, replace=False)
    test_rows = rng.choice(np.flatnonzero(ds.test), FIT_SAMPLES, replace=False)
    Xtr, Xte = X[train_rows], X[test_rows]
    n = len(Xtr)
    rows, fitted = [], {}
    for k in range(1, max_k + 1):
        gmm = GaussianMixture(n_components=k, n_init=3, random_state=0).fit(Xtr)
        ll = float(gmm.score(Xtr) * n)
        p = 6 * k - 1
        flags = mixture_burn_components(gmm)[gmm.predict(X)] if k > 1 else np.zeros(len(X), bool)
        rows.append(dict(model=f"GMM, K = {k}", k=k, n_params=p, loglik=ll, bic=-2 * ll + p * np.log(n),
                         aic=-2 * ll + 2 * p, test_loglik_per_sample=float(gmm.score(Xte)),
                         event_f1=float(_events_f1(ds, flags)["f1"]) if k > 1 else 0.0,
                         false_samples=int((flags[ds.test] & (ds.true_on[ds.test] == 0)).sum())))
        fitted[f"GMM, K = {k}"] = gmm
    for fit in (fit_log_chi_mixture, fit_t_mixture):
        model = fit(Xtr)
        ll = float(np.sum(model.loglik(Xtr)))
        flags = model.burn_probability(X) > 0.5
        rows.append(dict(model=model.name, k=2, n_params=model.n_params, loglik=ll,
                         bic=-2 * ll + model.n_params * np.log(n), aic=-2 * ll + 2 * model.n_params,
                         test_loglik_per_sample=float(np.mean(model.loglik(Xte))),
                         event_f1=float(_events_f1(ds, flags)["f1"]),
                         false_samples=int((flags[ds.test] & (ds.true_on[ds.test] == 0)).sum())))
        fitted[model.name] = model
    truth = ds.true_on[train_rows].astype(bool)
    return dict(rows=rows, models=fitted, X_train=Xtr, truth_train=truth)


def tail_study(X, truth):
    """Left- and right-tail behaviour of the TRUE coast samples in x1, against the log-chi theory."""
    x = X[~truth, 0]
    counts, edges = np.histogram(x, bins=120)
    centres = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]
    density = counts / (len(x) * width)
    mode = centres[np.argmax(counts)]
    left = (centres < mode - 0.8) & (counts >= 5)
    slope = float(np.polyfit(centres[left], np.log(density[left]), 1)[0]) if left.sum() > 3 else np.nan
    # maximum-likelihood log-chi fit (k free) and a single Gaussian, on the coast alone
    nll = lambda p: -np.sum(log_chi_logpdf(x, p[0], p[1])) if p[0] > 0.2 else 1e12
    k, log_sigma = optimize.minimize(nll, [3.0, np.median(x)], method="Nelder-Mead").x
    mu, sd = x.mean(), x.std()
    ll_chi = float(np.sum(log_chi_logpdf(x, k, log_sigma)))
    ll_gauss = float(np.sum(stats.norm.logpdf(x, mu, sd)))
    return dict(centres=centres, density=density, counts=counts, width=width, mode=float(mode),
                left_tail_slope=slope, k=float(k), sigma=float(np.exp(log_sigma)), log_sigma=float(log_sigma),
                gauss=(float(mu), float(sd)), loglik_chi=ll_chi, loglik_gauss=ll_gauss, n=int(len(x)),
                skewness=float(stats.skew(x)), excess_kurtosis=float(stats.kurtosis(x)))


# ---------------------------------------------------------------------------
# The subtraction: how much of the "unmodelled" acceleration is numerical?
# ---------------------------------------------------------------------------
def subtraction_budget(days=2.0, source="j2_drag"):
    """
    Clean (noise-free) data: measured dv/dt = gravity + J2 + drag + thrust, and the classifier subtracts
    gravity + J2. Whatever is left beyond the true drag + thrust is NUMERICAL error: the Savitzky-Golay
    truncation, floating-point round-off and the integrator / dense-output interpolation error. Each is
    estimated separately and compared with machine epsilon.
    """
    from scipy.signal import savgol_coeffs
    from .constants import ACCEL_UNIT_MS2, DU, MU, SECONDS, SMOOTH_CD, THRUST_ACCEL, TU, VU
    from .derivative_detection import known_accel
    from .smooth_controller import throttle
    from .long_run import controlled_full_run, controlled_run
    run = controlled_run() if source == "j2_drag" else controlled_full_run()
    ds = detection_dataset(source)
    keep = ds.days < days
    t, r, v = ds.t[keep], ds.r[:, keep], ds.v[:, keep]
    h = ds.step_s * SECONDS
    window, order = tuned("sg_window"), tuned("sg_order")
    feats = build_features(r, v, ds.step_s)
    speed = np.linalg.norm(v, axis=0)
    drag = -SMOOTH_CD * speed * v
    thrust = THRUST_ACCEL * throttle(run.s[keep]) * v / speed
    true_u = (drag + thrust) * ACCEL_UNIT_MS2
    error = feats.accel_from_v - true_u
    if source == "full":
        from .pipeline import catalogue_accel_ms2
        error = error - catalogue_accel_ms2(t, r)
    gravity = np.linalg.norm(known_accel(r, j2=0.0), axis=0) * ACCEL_UNIT_MS2
    j2 = np.linalg.norm(known_accel(r) - known_accel(r, j2=0.0), axis=0) * ACCEL_UNIT_MS2
    # SG truncation alone: the same filter on an exact two-body circular orbit (closed form, no solver)
    a_km = float(np.median(np.linalg.norm(r, axis=0)))
    n = np.sqrt(MU / a_km ** 3)
    tt = t - t[0]
    rk = a_km * np.array([np.cos(n * tt), np.sin(n * tt), 0 * tt])
    vk = a_km * n * np.array([-np.sin(n * tt), np.cos(n * tt), 0 * tt])
    from scipy.signal import savgol_filter
    dv = savgol_filter(vk, window, order, deriv=1, delta=h, axis=1)
    exact = -MU * rk / a_km ** 3
    trunc = np.linalg.norm(dv - exact, axis=0)[window:-window] * ACCEL_UNIT_MS2
    coeffs = savgol_coeffs(window, order, deriv=1, delta=h)
    eps = np.finfo(float).eps
    v_ms = float(np.median(speed)) * VU * 1000
    estimates = {
        "machine epsilon x |gravity| (subtraction round-off)": eps * float(np.median(gravity)),
        "round-off of the SG derivative (eps |v| ||c||)": eps * float(np.median(speed)) * np.linalg.norm(coeffs) * ACCEL_UNIT_MS2,
        "SG truncation (exact Kepler orbit)": float(np.sqrt(np.mean(trunc ** 2))),
        "solver tolerance 1e-11 through the SG filter": 1e-11 * float(np.median(speed)) * np.linalg.norm(coeffs) * ACCEL_UNIT_MS2,
        "total numerical error (clean data)": float(np.sqrt(np.mean(np.linalg.norm(error[:, window:-window], axis=0) ** 2))),
        "drag (the smallest real signal)": float(np.median(np.linalg.norm(drag, axis=0))) * ACCEL_UNIT_MS2,
        "J2": float(np.median(j2)),
        "gravity": float(np.median(gravity)),
        "thrust": THRUST_ACCEL * ACCEL_UNIT_MS2,
    }
    noisy_r, noisy_v = add_noise(ds, REFERENCE_NOISE[0], REFERENCE_NOISE[1], 1)
    noisy = build_features(noisy_r[:, keep], noisy_v[:, keep], ds.step_s).accel_from_v - true_u
    estimates["derivative noise at the reference noise"] = float(np.sqrt(np.mean(np.linalg.norm(noisy[:, window:-window], axis=0) ** 2)))
    hours = (t - t[0]) * TU / 3600
    return dict(estimates=estimates, hours=hours, error_norm=np.linalg.norm(error, axis=0), window=window,
                v_ms=v_ms, solver_steps_hours=None,
                along_error=np.sum(error * v, axis=0) / speed, x1_clean=feats.matrix[:, 0],
                on=ds.true_on[keep])

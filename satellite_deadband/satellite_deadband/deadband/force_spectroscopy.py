"""
"Force spectroscopy": read WHICH perturbations are in a trajectory, and in what mixture, from the
spectrum of its unmodelled acceleration, the way infrared spectroscopy reads the functional groups
of an organic molecule from its absorption bands.

The analogy
-----------
    IR spectroscopy                          here
    ---------------------------------------  ---------------------------------------------------------
    absorbance vs wavenumber                 |FFT| of the unmodelled acceleration vs frequency, in
                                             cycles per orbit, for each RTN component
    reference spectrum of a pure compound    spectrum of one force on its own (from the model)
    characteristic band (C=O at 1715 cm-1)   a "diagnostic band": a frequency band where ONE force
                                             carries >= 90% of the power (drag: along-track DC line;
                                             J2: radial 2 cycles/orbit; SRP: 1 cycle/orbit ...)
    band AREA -> concentration (Beer-Lambert) band power of the mixture / band power of the reference
                                             = (strength of that force)^2

Each force acts on the satellite LINEARLY (a = sum of forces), so the complex spectra add exactly.
Powers |X|^2 only add when two forces do not share a frequency bin.

The composition of a mixture
----------------------------
The "force budget" bar chart of step 1 shows each force's mean |acceleration| along the orbit.
Parseval's theorem says the summed POWER of a spectrum equals the mean SQUARE of the signal, so:
    * power shares  (sum |X_k|^2 / total)  give mean-SQUARE shares: J2 (99.98% of the budget) becomes
      99.999999% and every small force vanishes. Power percentages do NOT match the bar chart;
    * amplitude shares (sqrt of each force's summed power) give RMS shares, and RMS ~ mean |a| for
      these near-constant-size forces. Those DO match the bar chart.

Three ways to recover the mixture from one measured spectrum
------------------------------------------------------------
1. band_reading     IR-style: read each force's strength off its diagnostic band (integrated over
                    +-1% of the band frequency, so a slightly shifted line still falls inside). The
                    library can come from a DIFFERENT stretch of data, like a reference library.
2. peak_fit         phase-aware: least squares on the COMPLEX spectrum at the strongest library
                    bins. Exact, but needs the reference spectra of the SAME data window (the model
                    evaluated along the measured orbit), like a regression.
3. time_domain_fit  the step 2 joint regression on every sample (the reference answer).
"""
import time

import numpy as np

from .constants import ACCEL_UNIT_MS2, J2, PERIOD, SMOOTH_CD
from .perturbations import MU_MOON, MU_SUN, SRP_ACCEL
from .regression import least_squares, perturbation_shapes

FORCES = ("Drag", "J2", "Moon", "Sun", "SRP")
SHAPE_KEYS = {"Drag": "cd", "J2": "J2", "Moon": "mu_moon", "Sun": "mu_sun", "SRP": "SRP (P*CR*A/m)"}
TRUE_STRENGTH = {"Drag": SMOOTH_CD, "J2": J2, "Moon": MU_MOON, "Sun": MU_SUN, "SRP": SRP_ACCEL}
COMPONENTS = ("radial", "along-track", "cross-track")
BAND_FRACTION = 0.01      # a band is +-1% of its centre frequency (at least +-2 bins)
CANDIDATE_LINES = 30      # strongest lines of each force considered as its diagnostic band
PURE_BAND = 0.9           # a band is "diagnostic" if one force has >= 90% of its power
PEAK_BINS = 40            # strongest bins of each force used by the phase-aware peak fit


# ---------------------------------------------------------------------------
# Signals and spectra
# ---------------------------------------------------------------------------
def to_rtn(vectors, r, v):
    """(3, N) vectors -> (radial, along-track, cross-track) components in the satellite's frame."""
    radial = r / np.linalg.norm(r, axis=0)
    normal = np.cross(r.T, v.T).T
    normal = normal / np.linalg.norm(normal, axis=0)
    along = np.cross(normal.T, radial.T).T
    return np.array([np.sum(vectors * radial, axis=0), np.sum(vectors * along, axis=0),
                     np.sum(vectors * normal, axis=0)])


def force_signals(t, r, v, scales=None):
    """{force: (3, N) RTN acceleration [m/s^2]} at the true strengths, each multiplied by scales[force]."""
    shapes = perturbation_shapes(t, r, v)
    scales = scales or {}
    return {name: scales.get(name, 1.0) * TRUE_STRENGTH[name] * to_rtn(shapes[SHAPE_KEYS[name]], r, v) * ACCEL_UNIT_MS2
            for name in FORCES}


def spectrum(signal):
    """
    One-sided complex spectrum of a (3, N) signal, Hann window, DC kept (drag lives there).
    Every bin except DC stands for a +f and a -f frequency, so it is multiplied by sqrt(2):
    then sum |X|^2 counts all the power once (Parseval) and DC is not double-weighted.
    """
    window = np.hanning(signal.shape[1])
    X = np.fft.rfft(signal * window, axis=1)
    X[:, 1:] *= np.sqrt(2)
    return X


def cycles_per_orbit(n_samples, step):
    """Frequency of each bin in cycles per orbit (the 'wavenumber' axis of the force spectrum)."""
    return np.fft.rfftfreq(n_samples, d=step) * PERIOD


# ---------------------------------------------------------------------------
# Composition: shares of the force budget
# ---------------------------------------------------------------------------
def _shares(values):
    total = sum(values.values())
    return {name: 100 * value / total for name, value in values.items()}


def budget_shares(signals):
    """Step 1 bar chart: mean |acceleration| of each force, as % of the sum."""
    return _shares({name: float(np.mean(np.linalg.norm(a, axis=0))) for name, a in signals.items()})


def power_shares(spectra):
    """Summed spectral power of each force, as %: this is a mean-SQUARE share (does not match the budget)."""
    return _shares({name: float(np.sum(np.abs(X) ** 2)) for name, X in spectra.items()})


def amplitude_shares(spectra):
    """Square root of each force's summed power (its RMS, by Parseval), as %: this matches the budget."""
    return _shares({name: float(np.sqrt(np.sum(np.abs(X) ** 2))) for name, X in spectra.items()})


def shares_from_strengths(strengths, library_signals):
    """Composition implied by recovered strengths: strength x the library force's mean |a|, as %."""
    return _shares({name: strengths[name] * float(np.mean(np.linalg.norm(library_signals[name], axis=0)))
                    for name in FORCES})


# ---------------------------------------------------------------------------
# 1. IR-style band reading
# ---------------------------------------------------------------------------
def _band(centre, n_bins):
    half = max(2, int(BAND_FRACTION * centre))
    return slice(max(0, centre - half), min(n_bins, centre + half + 1))


def diagnostic_bands(library):
    """
    For each force: (component, band, purity) of its best band. Among its CANDIDATE_LINES strongest
    lines, prefer the strongest band where it owns >= PURE_BAND of the power; if it has none,
    take the purest band it has (and report that purity, so a poor band is visible).
    """
    power = {name: np.abs(X) ** 2 for name, X in library.items()}
    total = sum(power.values())
    n_bins = total.shape[1]
    bands = {}
    for name, p in power.items():
        best = None
        for component in range(3):
            for centre in np.argsort(p[component])[-CANDIDATE_LINES:]:
                band = _band(centre, n_bins)
                purity = p[component, band].sum() / total[component, band].sum()
                score = (1, p[component, band].sum()) if purity >= PURE_BAND else (0, purity)
                if best is None or score > best[0]:
                    best = (score, component, band, purity)
        bands[name] = dict(component=best[1], band=best[2], purity=float(best[3]))
    return bands


def band_reading(mixture_spectrum, library, bands=None):
    """Strength of each force relative to the library: sqrt(mixture band power / library band power)."""
    bands = bands or diagnostic_bands(library)
    strengths = {}
    for name, info in bands.items():
        c, band = info["component"], info["band"]
        strengths[name] = float(np.sqrt(np.sum(np.abs(mixture_spectrum[c, band]) ** 2) /
                                        np.sum(np.abs(library[name][c, band]) ** 2)))
    return strengths


# ---------------------------------------------------------------------------
# 2. Phase-aware peak fit and 3. time-domain fit
# ---------------------------------------------------------------------------
def peak_fit(mixture_spectrum, library):
    """Least squares on the complex spectrum at the PEAK_BINS strongest bins of every force."""
    bins = set()
    for X in library.values():
        bins.update(np.argsort(np.abs(X).ravel())[-PEAK_BINS:])
    bins = np.array(sorted(bins))
    columns = [np.concatenate([X.ravel()[bins].real, X.ravel()[bins].imag]) for X in library.values()]
    target = np.concatenate([mixture_spectrum.ravel()[bins].real, mixture_spectrum.ravel()[bins].imag])
    coefficients, _ = least_squares(columns, target)
    return dict(zip(library, (float(c) for c in coefficients)))


def time_domain_fit(mixture_signal, library_signals):
    """Least squares on every sample: mixture(t) = sum_k strength_k * force_k(t)."""
    coefficients, _ = least_squares(list(library_signals.values()), mixture_signal)
    return dict(zip(library_signals, (float(c) for c in coefficients)))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def timed(function, *args, repeats=3):
    """(result, best wall time [s] of `repeats` runs)."""
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(*args)
        best = min(best, time.perf_counter() - start)
    return result, best

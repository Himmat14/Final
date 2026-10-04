"""Checks on the derivative methods, the realistic noise, force spectroscopy and the J2 potential."""
import unittest

import numpy as np

from deadband.constants import DU, EARTH_EQUATORIAL_RADIUS_KM, J2, MU, from_days
from deadband.derivative_detection import differentiate
from deadband.force_spectroscopy import (FORCES, amplitude_shares, band_reading, budget_shares, force_signals,
                                         power_shares, spectrum)
from deadband.noise_models import realistic_noise, white_noise
from deadband.physics import initial_state, j2_accel, propagate


class DerivativeMethodTests(unittest.TestCase):
    def test_higher_order_central_differences_are_more_accurate(self):
        h = 0.05
        x = np.arange(0, 20, h)
        signal = np.tile(np.sin(x), (3, 1))
        errors = []
        for method in ("central 2nd order", "central 4th order", "central 6th order", "central 8th order"):
            estimate = differentiate(signal, h, method, 1)[0, 10:-10]
            errors.append(np.max(np.abs(estimate - np.cos(x[10:-10]))))
        self.assertTrue(all(a > 10 * b for a, b in zip(errors, errors[1:])), errors)

    def test_savitzky_golay_second_derivative(self):
        h = 0.05
        x = np.arange(0, 20, h)
        estimate = differentiate(np.tile(np.sin(x), (3, 1)), h, "Savitzky-Golay", 2)[0, 20:-20]
        np.testing.assert_allclose(estimate, -np.sin(x[20:-20]), atol=1e-4)


class NoiseModelTests(unittest.TestCase):
    def test_white_and_realistic_noise_have_the_requested_rms(self):
        t = np.linspace(0, 50, 20000)
        for noise in (white_noise(t, 2.0, 0), realistic_noise(t, 60.0, 2.0, 0)):
            rms_m = np.sqrt(np.mean(noise**2, axis=1)) * DU * 1000
            np.testing.assert_allclose(rms_m, 2.0, rtol=1e-9)


class SpectroscopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = np.linspace(0, from_days(2), 2880)
        states = propagate(1e-3, cls.t)
        cls.signals = force_signals(cls.t, states[0:3], states[3:6])
        cls.spectra = {name: spectrum(a) for name, a in cls.signals.items()}

    def test_sqrt_power_matches_the_budget_and_power_does_not(self):
        budget, amplitude, power = budget_shares(self.signals), amplitude_shares(self.spectra), power_shares(self.spectra)
        for name in ("Drag", "Moon", "SRP"):
            self.assertAlmostEqual(amplitude[name] / budget[name], 1.0, delta=0.15)
            self.assertLess(power[name], 1e-3 * budget[name])

    def test_band_reading_of_a_scaled_mixture(self):
        scales = dict(zip(FORCES, (2.0, 1.0, 0.5, 1.0, 3.0)))
        mixture = spectrum(sum(scales[n] * self.signals[n] for n in FORCES))
        strengths = band_reading(mixture, self.spectra)
        for name in ("Drag", "J2"):
            self.assertAlmostEqual(strengths[name] / scales[name], 1.0, delta=0.02)


class J2PotentialTest(unittest.TestCase):
    def test_j2_force_is_minus_the_gradient_of_the_j2_potential(self):
        """The mean SMA adds U = mu J2 R^2 (3 sin^2 lat - 1) / (2 r^3); the force must be -grad U."""
        earth_radius = EARTH_EQUATORIAL_RADIUS_KM / DU

        def potential(r):
            radius = np.linalg.norm(r)
            return MU * J2 * earth_radius**2 / (2 * radius**3) * (3 * (r[2] / radius) ** 2 - 1)

        r = initial_state()[0:3] + np.array([0.0, 0.3, 0.5])
        step = 1e-6
        gradient = np.array([(potential(r + step * e) - potential(r - step * e)) / (2 * step) for e in np.eye(3)])
        np.testing.assert_allclose(j2_accel(r), -gradient, rtol=1e-6)



class TuningTests(unittest.TestCase):
    def test_bias_squared_plus_variance_is_the_mse(self):
        from deadband.tuning import _decompose
        rng = np.random.default_rng(0)
        truth = rng.standard_normal(50)
        estimates = truth + 0.3 + 0.5 * rng.standard_normal((200, 50))
        bias_sq, variance, mse = _decompose(estimates, truth)
        self.assertAlmostEqual(mse, np.mean((estimates - truth) ** 2), places=10)
        self.assertAlmostEqual(bias_sq, 0.09, delta=0.01)
        self.assertAlmostEqual(variance, 0.25, delta=0.02)

    def test_every_setting_has_a_default(self):
        from deadband.settings import DEFAULTS, tuned
        for name in DEFAULTS:
            self.assertIsNotNone(tuned(name))


if __name__ == "__main__":
    unittest.main()

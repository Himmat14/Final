"""Fast sanity checks on the report-step library modules.  Run with:  python -m unittest discover tests"""
import unittest

import numpy as np

from deadband.burn_folding import autocorrelation, dominant_period, stack_bursts
from deadband.constants import ACCEL_UNIT_MS2, CD_TRUE, J2, MU, from_days, from_hours, from_minutes
from deadband.kalman import run_ekf
from deadband.perturbations import force_accelerations
from deadband.physics import initial_state, propagate
from deadband.regression import add_position_noise, fit_cd, fit_cd_energy, fit_mu
from deadband.sindy import hysteresis_states, run_hybrid, stlsq


class RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = from_minutes(5)
        cls.t = np.arange(0, from_days(1), cls.h)
        cls.positions = propagate(CD_TRUE, cls.t)[0:3]

    def test_mu_and_cd_from_clean_positions(self):
        self.assertAlmostEqual(fit_mu(self.positions, self.h) / MU, 1.0, places=4)
        self.assertAlmostEqual(fit_cd(self.positions, self.h) / CD_TRUE, 1.0, places=2)

    def test_energy_method_beats_fd_under_noise(self):
        noisy = add_position_noise(self.positions, 1.0, seed=0)
        fd_error = abs(fit_cd(noisy, self.h) / CD_TRUE - 1)
        energy_error = abs(fit_cd_energy(self.t, noisy, self.h)[0] / CD_TRUE - 1)
        self.assertLess(energy_error, fd_error)


class PerturbationTests(unittest.TestCase):
    def test_force_ordering_at_550_km(self):
        state = initial_state()
        sizes = {k: np.linalg.norm(a) * ACCEL_UNIT_MS2 for k, a in force_accelerations(0.0, state[:3], state[3:]).items()}
        self.assertGreater(sizes["Gravity"], sizes["J2"])
        self.assertGreater(sizes["J2"], sizes["Drag"])
        self.assertGreater(sizes["Drag"], sizes["Moon"])


class KalmanTests(unittest.TestCase):
    def test_ekf_recovers_cd(self):
        t = np.arange(0, from_hours(12), from_minutes(5))
        measured = add_position_noise(propagate(CD_TRUE, t, j2=J2)[0:3], 0.1, seed=1)
        run = run_ekf(t, measured, initial_state(), 1.5 * CD_TRUE, 0.1)
        self.assertAlmostEqual(run.estimates[6, -1] / CD_TRUE, 1.0, delta=0.02)


class FoldingTests(unittest.TestCase):
    def test_autocorrelation_finds_the_period_and_stacking_averages(self):
        step, period = 6.0, 600                                   # a pulse every 600 samples
        t = np.arange(20_000) * step
        pulses = ((np.arange(20_000) % period) < 30).astype(float)
        noisy = pulses + 0.5 * np.random.default_rng(0).standard_normal(t.size)
        acf = autocorrelation(noisy, 2000)
        self.assertAlmostEqual(dominant_period(acf, step, min_lag_s=100 * step), period * step, delta=step)
        stack = stack_bursts(t / 13713.441, noisy, pulses.astype(int), step, before_s=60, after_s=240)
        self.assertLess(np.std(stack.mean[:10]), np.std(stack.windows[0][:10]))


class SindyTests(unittest.TestCase):
    def test_stlsq_keeps_real_terms_and_drops_distractors(self):
        rng = np.random.default_rng(0)
        theta = rng.standard_normal((2000, 5))
        target = 3 * theta[:, 0] - 2 * theta[:, 3] + 0.01 * rng.standard_normal(2000)
        xi = stlsq(theta, target)
        np.testing.assert_allclose(xi[[0, 3]], [3, -2], rtol=1e-2)
        self.assertTrue(np.all(xi[[1, 2, 4]] == 0))

    def test_hysteresis_and_hybrid_switch_at_the_edges(self):
        states = hysteresis_states(np.array([5, 1, 3, 5, 9, 5, 1]), lower=2, upper=8)
        np.testing.assert_array_equal(states, [0, 1, 1, 1, 0, 0, 1])
        a, s = run_hybrid(5.0, 0, 400, 0.1, lambda a, s: 1.0 if s else -1.0, lower=2.0, upper=8.0)
        self.assertGreaterEqual(a.min(), 2.0)
        self.assertLessEqual(a.max(), 8.0)
        self.assertGreater(np.sum(np.diff(s) == 1), 1)


if __name__ == "__main__":
    unittest.main()

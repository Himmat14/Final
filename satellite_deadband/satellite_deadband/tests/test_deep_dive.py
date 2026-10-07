"""Checks on step 21-22 additions: log-chi density, hat-box coast model, sigmoid thrust law, controller field, tables."""
import unittest

import numpy as np
from scipy import integrate

from deadband import deep_dive as D
from deadband import gmm_physics as G
from deadband.report_tables import _table


class LogChiTests(unittest.TestCase):
    def test_density_integrates_to_one(self):
        for k in (1.0, 2.0, 3.0, 4.5):
            total, _ = integrate.quad(lambda x: np.exp(G.log_chi_logpdf(x, k, -12.0)), -40, 0)
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_log_of_a_3d_gaussian_norm_is_log_chi_3(self):
        rng = np.random.default_rng(0)
        x = np.log(np.linalg.norm(2e-6 * rng.standard_normal((200_000, 3)), axis=1))
        counts, edges = np.histogram(x, bins=60, density=True)
        centres = 0.5 * (edges[1:] + edges[:-1])
        model = np.exp(G.log_chi_logpdf(centres, 3, np.log(2e-6)))
        busy = counts > 0.05
        self.assertLess(np.max(np.abs(counts[busy] / model[busy] - 1)), 0.05)

    def test_hat_box_along_track_share_is_uniform(self):
        rng = np.random.default_rng(1)
        n = rng.standard_normal((100_000, 3))
        cos = n[:, 0] / np.linalg.norm(n, axis=1)
        self.assertAlmostEqual(np.var(cos), 1 / 3, places=2)
        hist, _ = np.histogram(cos, bins=10, range=(-1, 1), density=True)
        self.assertLess(np.max(np.abs(hist - 0.5)), 0.03)


class ThrustModelTests(unittest.TestCase):
    def test_sigmoid_fit_recovers_its_parameters(self):
        t = np.linspace(-200, 1600, 2000)
        y = D.sigmoid_burn(t, 2e-4, 5.0, 1370.0, 10.0, 12.0, 0.0)
        fit = D.fit_sigmoid_burn(t, y + 1e-7 * np.random.default_rng(0).standard_normal(t.size))
        self.assertAlmostEqual(fit["plateau"] / 2e-4, 1, places=3)
        self.assertAlmostEqual(fit["duration_s"], 1365.0, delta=1.0)
        self.assertAlmostEqual(fit["rise_10_90_s"], 2 * np.log(9) * 10.0, delta=1.0)


class ControllerFieldTests(unittest.TestCase):
    def test_reduced_field_directions(self):
        a_dot, s_dot = D.reduced_field(np.array([6921.85, 6922.15, 6922.45]), np.array([0.0, 0.0, 1.0]))
        self.assertLess(a_dot[0], 0)                  # coasting decays
        self.assertGreater(s_dot[0], 0)               # below the lower edge the thruster switches on
        self.assertAlmostEqual(s_dot[1], 0.0)         # inside the band with s = 0 nothing happens (memory)
        self.assertLess(s_dot[2], 0)                  # above the upper edge it switches off


class TableTests(unittest.TestCase):
    def test_every_horizon_is_a_column(self):
        tex = _table([("law", {"15d": 1.234, "30d": 12.0, "60d": 30, "120d": 60, "240d": 120, "360d": 180}, None)])
        self.assertIn("360\\,d", tex)
        self.assertIn("law & 1.2 & 12 & 30 & 60 & 120 & 180", tex)


if __name__ == "__main__":
    unittest.main()

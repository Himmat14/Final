"""Sanity checks on the physics and controller.  Run with:  python -m unittest discover tests"""
import unittest

import numpy as np

from deadband.constants import A_LOWER_KM, A_UPPER_KM, MU, ORBITAL_PERIOD_S, PERIOD, VU
from deadband.controller import simulate_deadband


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = simulate_deadband(PERIOD * 8)

    def test_unit_system_matches_slides(self):
        self.assertAlmostEqual(MU, 289.873, places=3)
        self.assertAlmostEqual(ORBITAL_PERIOD_S / 60, 95.5, delta=0.05)

    def test_burn_count_and_size(self):
        self.assertEqual(self.sim.n_burns, 24)                                   # Week 4, slide 10
        self.assertAlmostEqual(np.mean(self.sim.raise_dv) * VU * 1000, 0.6305, delta=1e-3)
        self.assertAlmostEqual(np.mean(self.sim.trim_dv) * VU * 1000, -0.0822, delta=1e-3)

    def test_orbit_stays_inside_the_band(self):
        self.assertGreaterEqual(self.sim.sma_km.min(), A_LOWER_KM - 1e-3)
        self.assertLessEqual(self.sim.sma_km.max(), A_UPPER_KM + 0.2)            # raise overshoots the upper edge a little

    def test_every_burn_is_labelled(self):
        self.assertEqual(self.sim.burn_labels.sum(), 3 * self.sim.n_burns)       # pre-burn, post-raise, post-trim samples


if __name__ == "__main__":
    unittest.main()

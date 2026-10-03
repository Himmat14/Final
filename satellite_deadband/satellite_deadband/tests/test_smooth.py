"""Sanity checks on the Week 5 smooth controller and derivative detector.  Run with:  python -m unittest discover tests"""
import unittest

import numpy as np

from deadband.constants import J2, PERIOD, SAMPLE_STEP_S, SMOOTH_A_LOWER_KM, SMOOTH_A_UPPER_KM, from_hours
from deadband.derivative_detection import build_derivative_evidence, fit_derivative_gmm, group_into_events
from deadband.physics import j2_accel, propagate
from deadband.smooth_controller import (
    default_runs, j2_potential, mean_sma, rising_switch, simulate_smooth_deadband, throttle,
)


class SmoothControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controlled, cls.drag_only = default_runs()

    def test_j2_potential_matches_j2_accel(self):
        r, h = np.array([0.7, 0.5, 0.6]), 1e-6
        gradient = np.array([(j2_potential(r + h * e) - j2_potential(r - h * e)) / (2 * h) for e in np.eye(3)])
        np.testing.assert_allclose(-gradient, j2_accel(r), rtol=1e-6)

    def test_mean_sma_is_constant_under_j2_alone(self):
        t = np.linspace(0, 3 * PERIOD, 600)
        states = propagate(0.0, t, j2=J2)
        sma = np.array([mean_sma(states[0:3, i], states[3:6, i]) for i in range(len(t))])
        self.assertLess(np.ptp(sma) / np.mean(sma), 1e-8)

    def test_switches_are_smooth_steps(self):
        self.assertAlmostEqual(rising_switch(0.0, 1.0), 0.5)
        self.assertLess(throttle(0.0), 1e-15)            # thruster shut when s = 0
        self.assertGreater(throttle(1.0), 1 - 1e-4)      # fully open (99.995%) when s = 1

    def test_two_burns_in_eight_days_and_orbit_stays_in_band(self):
        self.assertEqual(len(self.controlled.burn_intervals(from_hours(0.01))), 2)
        self.assertGreater(self.controlled.mean_sma_km.min(), SMOOTH_A_LOWER_KM - 0.01)
        self.assertLess(self.controlled.mean_sma_km.max(), SMOOTH_A_UPPER_KM + 0.01)
        self.assertLess(self.drag_only.mean_sma_km[-1], SMOOTH_A_LOWER_KM - 0.5)   # drag-only leaves the band

    def test_osculating_trigger_chatters(self):
        run = simulate_smooth_deadband(from_hours(6), use_mean_sma=False)
        self.assertGreater(len(run.burn_intervals(from_hours(0.01))), 3)

    def test_every_gmm_space_flags_both_burns_and_no_false_events(self):
        evidence = build_derivative_evidence(self.controlled, SAMPLE_STEP_S)
        for space in ("rdot", "vdot", "both"):
            scores = fit_derivative_gmm(evidence, space).event_scores
            self.assertEqual((scores["n_found"], scores["n_false_events"]), (2, 0), space)

    def test_single_points_never_become_events(self):
        flags = np.zeros(100, dtype=int)
        flags[[5, 40, 41]] = 1               # isolated points
        flags[60:80] = 1
        flags[70] = 0                        # a short gap inside a real run
        events = group_into_events(flags)
        self.assertEqual(events[:50].sum(), 0)
        self.assertTrue(events[60:80].all())


if __name__ == "__main__":
    unittest.main()

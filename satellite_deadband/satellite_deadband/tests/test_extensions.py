"""Checks on the extension steps 10-18 (mean-element model, forecasting, conformal, screening, tracking passes,
regimes, fleet, GEO and the SINDy zoo). Run with:  python -m unittest discover tests"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from deadband import conjunction as C, fleet as Fl, forecasting as F, geo as G, mean_element as M, observation as O
from deadband import regimes as R, sindy_zoo as Z
from deadband.sindy import stlsq


class MeanElementTests(unittest.TestCase):
    def test_model_reproduces_the_full_controlled_run(self):
        """Calibrated on the 6-s run: the same burn period (<1%) and burn length (<2%)."""
        c = M.calibrate()
        run = M.simulate(M.space_weather(120, constant=True))
        self.assertAlmostEqual(np.mean(np.diff(run.onsets)) / M.DAY_S / c["period_days"], 1.0, delta=0.01)
        self.assertAlmostEqual(np.median(run.ends - run.onsets) / c["burn_s"], 1.0, delta=0.02)

    def test_burn_detection_finds_every_burn_and_its_edges(self):
        run = M.simulate(M.space_weather(60, constant=True))
        t, a = M.measure(run, sigma_km=0.02)
        onsets, ends, a_on, a_off = M.detect_burns(t, a, 0.02)
        self.assertEqual(onsets.size, run.onsets.size)
        self.assertLess(np.max(np.abs(onsets - run.onsets)), 600)            # within one sample
        self.assertAlmostEqual(np.median(a_on), M.calibrate()["lower"], delta=0.01)


class ForecastTests(unittest.TestCase):
    def test_law_with_perfect_drag_knowledge_reproduces_clean_onsets(self):
        weather = M.space_weather(60, seed=1)
        run = M.simulate(weather)
        c = M.calibrate()
        i0 = np.searchsorted(run.t, run.ends[2]) + 5                         # coasting just after a burn
        drag = c["k"] * weather.w * np.exp(-(run.a[i0] - c["a_ref"]) / M.SCALE_HEIGHT_KM)
        predicted = F.law_onsets(run.t[i0], run.a[i0], drag, weather.t, c["lower"], c["upper"], c["T"] - c["k"], run.t[-1])
        true = run.onsets[run.onsets > run.t[i0]]
        n = min(predicted.size, true.size, 5)
        self.assertLess(np.max(np.abs(predicted[:n] - true[:n])), 4 * 3600)   # hours over ~20 days

    def test_conformal_intervals_cover_exchangeable_errors(self):
        rng = np.random.default_rng(0)
        rows = [("m", origin, lead, rng.normal(0, 600 * (1 + lead / 60)))
                for origin in np.arange(40, 360, 1.0) for lead in rng.uniform(0, 240, 20)]
        for row in F.conformal(rows, "m", alpha=0.1):
            self.assertAlmostEqual(row["coverage"], 0.9, delta=0.06)


class ScreeningTests(unittest.TestCase):
    def test_collision_probability_matches_monte_carlo(self):
        rng = np.random.default_rng(0)
        x0, y0, sx, sy, R_ = 0.1, -0.05, 0.3, 0.2, 0.02
        samples = np.column_stack([x0 + sx * rng.standard_normal(2_000_000), y0 + sy * rng.standard_normal(2_000_000)])
        mc = np.mean(np.hypot(*samples.T) < R_)
        self.assertAlmostEqual(C.collision_probability(x0, y0, sx, sy, R_) / mc, 1.0, delta=0.1)

    def test_unknown_burn_drifts_three_km_per_hour(self):
        drift = 1.5 * C.mean_motion(6922.4) * 0.5 * 3600
        self.assertAlmostEqual(drift, 2.96, delta=0.05)


class ObservationTests(unittest.TestCase):
    def test_pass_orbit_fit_recovers_the_sma(self):
        t, sma, truth = O.pass_sma(0.01, 4, days=3)
        self.assertLess(np.sqrt(np.mean((sma - truth) ** 2)), 0.05)           # < 50 m from 10 m noise

    def test_oem_reader_round_trip(self):
        text = ["CCSDS_OEM_VERS = 2.0", "META_START", "OBJECT_NAME = TEST", "META_STOP"]
        r0 = np.array([6922.0, 0.0, 0.0])
        for k in range(5):
            text.append(f"2026-01-01T00:0{k}:00.000 {r0[0]:.3f} {60.0 * k:.3f} 0.000 0.000000 7.500000 0.000000")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.oem"
            path.write_text("\n".join(text))
            t, r, v = O.read_oem(path)
        np.testing.assert_allclose(t, np.arange(5) * 60.0)
        self.assertEqual(r.shape, (3, 5))
        self.assertAlmostEqual(v[1, 0], 7.5)


class RegimeFleetGeoTests(unittest.TestCase):
    def test_regimes_are_told_apart_and_the_band_change_is_found(self):
        rr = R.simulate_regimes()
        an = R.analyse(rr)
        m = R.confusion(an.rule_labels, an.true_labels)
        self.assertGreaterEqual(np.trace(m) / m.sum(), 0.97)
        self.assertTrue(any(abs(cp - R.BAND_SHIFT_DAY) < 2 for cp in an.changepoints))

    def test_fleet_flags_exactly_the_changed_satellites(self):
        fleet = Fl.simulate_fleet()
        starts, laws = Fl.window_laws(fleet)
        flags = Fl.detections(starts, Fl.change_scores(starts, laws))
        self.assertEqual(set(flags), set(Fl.ANOMALIES))
        for sat, (_, day) in Fl.ANOMALIES.items():
            self.assertGreaterEqual(flags[sat], day)

    def test_geo_law_is_recovered_from_the_free_drift(self):
        run = G.simulate()
        t, lam, _ = G.measure(run)
        free = t < G.FREE_DAYS - 2
        acc, smooth = G.second_derivative(t[free], lam[free])
        x, y = smooth[25:-25], acc[25:-25]
        xi = stlsq(np.column_stack([np.sin(2 * np.radians(x)), np.cos(2 * np.radians(x))]), y)
        A, lam_s = G.recover(list(xi), ["sin 2l", "cos 2l"])
        self.assertAlmostEqual(A / G.A_TRUE, 1.0, delta=0.05)
        self.assertAlmostEqual(lam_s, G.LAMBDA_S, delta=0.5)

    def test_geo_impulses_are_detected(self):
        run = G.simulate()
        t, lam, _ = G.measure(run)
        box = t > G.FREE_DAYS + 1
        detected, _ = G.detect_ew_burns(t[box], lam[box])
        for burn in run.ew_burns[run.ew_burns < t[-1] - 10]:
            self.assertLess(np.min(np.abs(detected - burn)), 0.5)


class ZooTests(unittest.TestCase):
    def test_the_true_library_wins_on_synthetic_data(self):
        """Data from da/dt = -5 p (coast) + 1300 (burn): the proxy library must beat the constant one."""
        rng = np.random.default_rng(0)
        n = 4000
        s = (rng.uniform(size=n) < 0.05).astype(float)
        p = 1 + 0.5 * np.sin(np.linspace(0, 20, n))
        target = -5 * p * (1 - s) + 1300 * s + 0.5 * rng.standard_normal(n)
        data = Z.ZooData("synthetic", target, np.zeros(n), s, np.linspace(0, 100, n), p,
                         np.arange(n) < 2000, np.arange(n) >= 2000, "m/hour", {})
        libs = Z.leo_libraries(True)
        proxy = Z.fit_library(data, "7", libs["7. space-weather proxy"])
        constant = Z.fit_library(data, "1", libs["1. constant per cluster (step 9)"])
        self.assertLess(proxy["bic"], constant["bic"])
        self.assertAlmostEqual(proxy["coefficients"][0], -5.0, delta=0.05)
        # classic (relative) STLSQ drops the small, well-measured coast term next to the rare huge burn term
        classic = Z.fit_library(data, "7", libs["7. space-weather proxy"], method="relative")
        self.assertEqual(classic["coefficients"][0], 0.0)


class SplitLearningTests(unittest.TestCase):
    def test_free_decay_recovers_the_drag_law(self):
        """No deadband: the sqrt(a) / physics-shape library recovers c_d and forecasts the SMA to metres."""
        from deadband import sindy_split as S
        _, rows, _ = S.free_decay_study("drag only")
        true_form = next(r for r in rows if r["library"].startswith("5."))
        self.assertAlmostEqual(true_form["cd_ratio"], 1.0, delta=0.005)
        self.assertLess(abs(true_form["sma_error_km_360d"]), 0.05)

    def test_mean_element_period_matches_the_full_run(self):
        c = M.calibrate()
        run = M.simulate(M.space_weather(80, constant=True))
        self.assertAlmostEqual(np.mean(np.diff(run.onsets)) / M.DAY_S / c["period_days"], 1.0, delta=0.002)


if __name__ == "__main__":
    unittest.main()

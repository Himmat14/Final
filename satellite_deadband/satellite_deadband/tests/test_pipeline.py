"""Checks on the end-to-end chain (step 20): the law forecast, the error budget's sanity row and the closed-form checks."""
import json
import unittest
from pathlib import Path

import numpy as np

from deadband import pipeline as P
from deadband.report_checks import _row, to_latex

RESULTS = Path(__file__).resolve().parents[1] / "outputs" / "results.json"


class LawForecastTests(unittest.TestCase):
    def test_period_is_coast_plus_burn_time(self):
        law = P.Law("x", cd=1.0, T=3.0, lower_km=0.0, upper_km=1.0)
        shapes = (1.0, 1.0)
        scale = P.DU / P.TU * P.DAY                     # rate per unit c_d D_bar, in km/day
        onsets, period = P.law_forecast(law, shapes, start_day=0.0)
        self.assertAlmostEqual(period, 1 / scale + 1 / (2 * scale))
        self.assertAlmostEqual(onsets[0], 1 / scale)
        self.assertTrue(np.allclose(np.diff(onsets), period))

    def test_a_law_that_cannot_climb_gives_no_forecast(self):
        onsets, period = P.law_forecast(P.Law("x", 1.0, 0.5, 0.0, 1.0), (1.0, 1.0), 0.0)
        self.assertIsNone(onsets)
        self.assertTrue(np.isnan(period))


class CheckRowTests(unittest.TestCase):
    def test_equality_and_bounds(self):
        self.assertTrue(_row("a", "f", 1.0, 1.004, "", 0.005)["passed"])
        self.assertFalse(_row("a", "f", 1.0, 1.1, "", 0.005)["passed"])
        self.assertTrue(_row("b", "f", 10.0, 8.0, "", 0.0, bound="upper")["passed"])
        self.assertFalse(_row("b", "f", 10.0, 12.0, "", 0.0, bound="upper")["passed"])
        self.assertTrue(_row("c", "f", 0.9, 0.95, "", 0.0, bound="lower")["passed"])

    def test_latex_table_has_one_line_per_check(self):
        rows = [_row("a", "f", 1.0, 1.0, "m", 0.01), _row("b", "g", 2.0, 1.0, "s", 0.0, bound="upper")]
        tex = to_latex(rows)
        self.assertEqual(tex.count(r"\\"), 3)          # header + 2 rows
        self.assertIn("bound", tex)

    @unittest.skipUnless(RESULTS.exists(), "needs a full run (outputs/results.json)")
    def test_every_headline_number_passes_its_check(self):
        results = json.loads(RESULTS.read_text())
        if "report_step20" not in results:
            self.skipTest("results.json predates step 20")
        failed = [r["check"] for r in results["report_step20"]["checks"] if not r["passed"]]
        self.assertEqual(failed, [])

    @unittest.skipUnless(RESULTS.exists(), "needs a full run (outputs/results.json)")
    def test_error_budget_sanity_row_is_near_zero(self):
        results = json.loads(RESULTS.read_text())
        if "report_step20" not in results:
            self.skipTest("results.json predates step 20")
        rows = {r["variant"]: r for r in results["report_step20"]["error_budget"]}
        self.assertLess(rows["all stages true"]["errors"]["360d"], 5.0)
        self.assertLess(rows["pipeline as learned"]["errors"]["360d"], rows["physical c_d, T (model-form floor)"]["errors"]["360d"] * 2)


if __name__ == "__main__":
    unittest.main()

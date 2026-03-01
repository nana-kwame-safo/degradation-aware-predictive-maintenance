from __future__ import annotations

import unittest

import numpy as np

from src.evaluation.metrics import (
    error_asymmetry_metrics,
    stratified_metrics_by_rul_bands,
    unit_level_metrics,
)


class TestBaselineEvaluationMetrics(unittest.TestCase):
    def test_stratified_metrics_include_expected_bands(self) -> None:
        y_true = [5, 12, 33, 45, 70, 90]
        y_pred = [6, 10, 35, 40, 68, 88]
        bands = [(0, 20), (21, 50), (51, 125)]

        rows = stratified_metrics_by_rul_bands(y_true, y_pred, bands=bands)
        labels = [str(row["band"]) for row in rows]

        self.assertEqual(labels, ["0-20", "21-50", "51-125"])
        for row in rows:
            self.assertIn("MAE", row)
            self.assertIn("RMSE", row)
            self.assertIn("n", row)

    def test_asymmetry_outputs_have_required_fields(self) -> None:
        y_true = [3, 8, 12, 25, 60]
        y_pred = [9, 7, 20, 20, 58]

        asym = error_asymmetry_metrics(
            y_true, y_pred, eol_band=(0, 20), severe_threshold=10.0
        )
        required = {
            "bias_overall",
            "bias_eol",
            "pct_overestimation_eol",
            "pct_severe_overestimation_eol",
            "n_total",
            "n_eol",
            "eol_low",
            "eol_high",
            "severe_overestimate_threshold",
        }
        self.assertSetEqual(set(asym.keys()), required)
        self.assertEqual(asym["eol_low"], 0)
        self.assertEqual(asym["eol_high"], 20)
        self.assertEqual(asym["n_total"], 5)

    def test_unit_level_metrics_one_row_per_unit_and_non_negative(self) -> None:
        y_true = np.asarray([10, 9, 6, 3, 15, 14], dtype=float)
        y_pred = np.asarray([12, 8, 5, 2, 10, 16], dtype=float)
        unit_ids = np.asarray([1, 1, 2, 2, 3, 3], dtype=int)

        unit_df = unit_level_metrics(y_true=y_true, y_pred=y_pred, unit_ids=unit_ids)

        self.assertEqual(unit_df.shape[0], 3)
        self.assertTrue((unit_df["MAE"] >= 0).all())
        self.assertTrue((unit_df["RMSE"] >= 0).all())


if __name__ == "__main__":
    unittest.main()

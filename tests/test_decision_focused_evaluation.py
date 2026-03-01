from __future__ import annotations

import unittest

from src.evaluation.reliability_analysis import (
    alert_threshold_metrics,
    alert_threshold_sweep,
    weighted_error_cost,
)


class TestDecisionFocusedEvaluation(unittest.TestCase):
    def test_alert_threshold_metrics_confusion_fields(self) -> None:
        y_true = [5, 15, 35, 50]
        y_pred = [10, 25, 10, 60]

        out = alert_threshold_metrics(y_true=y_true, y_pred=y_pred, threshold=20)

        self.assertEqual(out["tp"], 1)
        self.assertEqual(out["fp"], 1)
        self.assertEqual(out["fn"], 1)
        self.assertEqual(out["tn"], 1)
        self.assertAlmostEqual(float(out["precision"]), 0.5, places=9)
        self.assertAlmostEqual(float(out["recall"]), 0.5, places=9)
        self.assertAlmostEqual(float(out["false_alarm_rate"]), 0.5, places=9)
        self.assertAlmostEqual(float(out["miss_rate"]), 0.5, places=9)
        self.assertAlmostEqual(float(out["mean_true_rul_at_alert"]), 20.0, places=9)
        self.assertAlmostEqual(float(out["median_true_rul_at_alert"]), 20.0, places=9)

    def test_alert_threshold_sweep_sorted_and_deduplicated(self) -> None:
        y_true = [5, 15, 35, 50]
        y_pred = [10, 25, 10, 60]

        rows = alert_threshold_sweep(
            y_true=y_true,
            y_pred=y_pred,
            thresholds=[30, 10, 30],
        )
        thresholds = [int(row["threshold"]) for row in rows]

        self.assertEqual(thresholds, [10, 30])

    def test_weighted_error_cost_includes_severe_late_component(self) -> None:
        y_true = [10, 10, 10]
        y_pred = [5, 20, 30]  # errors: -5, +10, +20

        out = weighted_error_cost(
            y_true=y_true,
            y_pred=y_pred,
            early_weight=1.0,
            late_weight=2.0,
            severe_late_threshold=10.0,
            severe_late_multiplier=2.0,
            eol_band=(0, 20),
        )

        self.assertAlmostEqual(float(out["early_cycles_sum"]), 5.0, places=9)
        self.assertAlmostEqual(float(out["late_cycles_sum"]), 30.0, places=9)
        self.assertAlmostEqual(float(out["severe_late_cycles_sum"]), 10.0, places=9)
        self.assertAlmostEqual(float(out["cost_sum"]), 85.0, places=9)
        self.assertAlmostEqual(float(out["cost_mean"]), 85.0 / 3.0, places=9)
        self.assertAlmostEqual(float(out["cost_mean_eol"]), 85.0 / 3.0, places=9)

    def test_weighted_error_cost_rejects_negative_weights(self) -> None:
        with self.assertRaises(ValueError):
            weighted_error_cost(y_true=[1], y_pred=[1], early_weight=-1.0)


if __name__ == "__main__":
    unittest.main()

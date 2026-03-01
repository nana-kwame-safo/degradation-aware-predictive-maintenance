from __future__ import annotations

import unittest

import pandas as pd

from src.evaluation.policy import first_trigger, policy_metrics


class TestPolicyEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.meta_df = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2, 2, 2],
                "cycle_end": [30, 31, 32, 30, 31, 32],
                "true_rul": [10.0, 9.0, 8.0, 12.0, 11.0, 10.0],
            }
        )
        self.y_true = [10.0, 9.0, 8.0, 12.0, 11.0, 10.0]
        self.y_pred = [25.0, 19.0, 7.0, 30.0, 25.0, 22.0]

    def test_first_trigger_has_required_columns(self) -> None:
        out = first_trigger(meta_df=self.meta_df, y_pred=self.y_pred, threshold=20)
        required = {
            "unit_id",
            "triggered",
            "trigger_cycle_end",
            "pred_rul_at_trigger",
            "true_rul_at_trigger",
        }
        self.assertSetEqual(set(out.columns), required)
        self.assertEqual(out.shape[0], 2)

    def test_policy_metrics_output_columns(self) -> None:
        unit_df, summary = policy_metrics(
            meta_df=self.meta_df,
            y_true=self.y_true,
            y_pred=self.y_pred,
            threshold=20,
            eol_critical=5,
            false_alarm_margin=10,
        )

        expected_cols = [
            "unit_id",
            "triggered",
            "trigger_cycle_end",
            "pred_rul_at_trigger",
            "true_rul_at_trigger",
            "lead_time",
            "is_false_alarm",
            "is_late_trigger",
            "missed_trigger",
        ]
        self.assertEqual(unit_df.columns.tolist(), expected_cols)
        self.assertIn("trigger_rate", summary)

    def test_lead_time_non_negative_when_triggered(self) -> None:
        unit_df, _ = policy_metrics(
            meta_df=self.meta_df,
            y_true=self.y_true,
            y_pred=self.y_pred,
            threshold=20,
        )
        triggered = unit_df[unit_df["triggered"]]
        self.assertTrue((triggered["lead_time"] >= 0.0).all())

    def test_trigger_rate_is_in_unit_interval(self) -> None:
        _, summary = policy_metrics(
            meta_df=self.meta_df,
            y_true=self.y_true,
            y_pred=self.y_pred,
            threshold=20,
        )
        trigger_rate = float(summary["trigger_rate"])
        self.assertGreaterEqual(trigger_rate, 0.0)
        self.assertLessEqual(trigger_rate, 1.0)

    def test_metadata_row_count_must_match_targets(self) -> None:
        bad_meta = self.meta_df.iloc[:-1].copy()
        with self.assertRaises(ValueError):
            policy_metrics(
                meta_df=bad_meta,
                y_true=self.y_true,
                y_pred=self.y_pred,
                threshold=20,
            )


if __name__ == "__main__":
    unittest.main()

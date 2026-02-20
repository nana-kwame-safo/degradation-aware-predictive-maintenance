from __future__ import annotations

import unittest

from src.config import Config
from src.data.data_loader import CMAPSSPaths, cmapss_all_columns, load_cmapss_subset
from src.data.preprocessing import make_windows, unit_train_val_split


class TestMilestone1Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = Config()
        raw_dir = cls.cfg.cmapss_raw_dir
        if not raw_dir.exists():
            raise unittest.SkipTest(f"CMAPSS data directory not found: {raw_dir}")

        cls.train_df, cls.test_df = load_cmapss_subset(
            paths=CMAPSSPaths(root_dir=raw_dir),
            subset=cls.cfg.cmapss_subset,
            rul_cap=cls.cfg.clip_rul,
        )
        cls.sensor_cols = [c for c in cls.train_df.columns if c.startswith("sensor_")]

    def test_loader_returns_expected_columns(self) -> None:
        expected = set(cmapss_all_columns() + ["rul"])
        self.assertSetEqual(set(self.train_df.columns), expected)
        self.assertSetEqual(set(self.test_df.columns), expected)

    def test_rul_exists_and_is_non_negative(self) -> None:
        self.assertIn("rul", self.train_df.columns)
        self.assertIn("rul", self.test_df.columns)
        self.assertTrue((self.train_df["rul"] >= 0).all())
        self.assertTrue((self.test_df["rul"] >= 0).all())

    def test_unit_split_is_disjoint(self) -> None:
        train_split, val_split = unit_train_val_split(
            df=self.train_df,
            val_fraction=self.cfg.val_fraction,
            seed=self.cfg.random_state,
        )

        train_units = set(train_split["unit_id"].unique())
        val_units = set(val_split["unit_id"].unique())
        self.assertSetEqual(train_units.intersection(val_units), set())

    def test_window_generation_shapes(self) -> None:
        train_split, _ = unit_train_val_split(
            df=self.train_df,
            val_fraction=self.cfg.val_fraction,
            seed=self.cfg.random_state,
        )

        x_arr, y_arr = make_windows(
            df=train_split,
            feature_cols=self.sensor_cols,
            window=self.cfg.window,
            step=self.cfg.step,
        )

        self.assertEqual(x_arr.ndim, 3)
        self.assertEqual(y_arr.ndim, 1)
        self.assertGreater(x_arr.shape[0], 0)
        self.assertEqual(x_arr.shape[0], y_arr.shape[0])
        self.assertEqual(x_arr.shape[1], self.cfg.window)
        self.assertEqual(x_arr.shape[2], len(self.sensor_cols))


if __name__ == "__main__":
    unittest.main()

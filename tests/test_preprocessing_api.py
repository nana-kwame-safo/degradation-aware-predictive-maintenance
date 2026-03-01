from __future__ import annotations

import unittest
import warnings

import numpy as np
import pandas as pd

from src.data.preprocessing import (
    build_tabular_baseline_features,
    generate_unit_windows,
    make_window_features,
    make_windows,
)


class TestPreprocessingCanonicalAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2, 2, 2],
                "cycle": [1, 2, 3, 1, 2, 3],
                "rul": [30.0, 29.0, 28.0, 20.0, 19.0, 18.0],
                "temp": [100.0, 101.0, 102.0, 95.0, 94.0, 93.0],
                "pressure": [10.0, 11.0, 12.0, 9.0, 8.0, 7.0],
            }
        )
        self.feature_cols = ["temp", "pressure"]

    def test_make_windows_return_meta_aligned(self) -> None:
        x, y, meta = make_windows(
            df=self.df,
            feature_cols=self.feature_cols,
            window=2,
            step=1,
            return_meta=True,
        )

        self.assertEqual(meta.columns.tolist(), ["unit_id", "cycle_end", "true_rul"])
        self.assertEqual(x.shape[0], y.shape[0])
        self.assertEqual(meta.shape[0], y.shape[0])
        self.assertTrue(np.allclose(meta["true_rul"].to_numpy(dtype=float), y))

    def test_make_window_features_names_use_feature_cols(self) -> None:
        x, _ = make_windows(
            df=self.df,
            feature_cols=self.feature_cols,
            window=2,
            step=1,
        )
        _, names = make_window_features(x, feature_cols=self.feature_cols)

        expected_prefix = [
            "temp_mean",
            "pressure_mean",
            "temp_std",
            "pressure_std",
        ]
        self.assertEqual(names[:4], expected_prefix)
        self.assertTrue(all(not name.startswith("sensor_") for name in names))

    def test_generate_unit_windows_is_wrapper(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            x_wrap, y_wrap, u_wrap, c_wrap = generate_unit_windows(
                df=self.df,
                feature_cols=self.feature_cols,
                window_size=2,
                stride=1,
            )
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

        x, y, meta = make_windows(
            df=self.df,
            feature_cols=self.feature_cols,
            window=2,
            step=1,
            return_meta=True,
        )
        self.assertTrue(np.allclose(x_wrap, x))
        self.assertTrue(np.allclose(y_wrap, y))
        self.assertTrue(np.array_equal(u_wrap, meta["unit_id"].to_numpy(dtype=int)))
        self.assertTrue(np.array_equal(c_wrap, meta["cycle_end"].to_numpy(dtype=int)))

    def test_build_tabular_baseline_features_uses_column_names(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            out = build_tabular_baseline_features(
                df=self.df,
                sensor_cols=self.feature_cols,
                window_size=2,
            )
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        self.assertIn("temp_mean", out.columns)
        self.assertIn("pressure_slope", out.columns)
        self.assertIn("unit_id", out.columns)
        self.assertIn("cycle", out.columns)
        self.assertIn("rul", out.columns)


if __name__ == "__main__":
    unittest.main()

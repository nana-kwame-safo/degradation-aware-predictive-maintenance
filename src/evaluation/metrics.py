"""
Evaluation metrics.

Keep evaluation in a dedicated module:
- encourages consistent reporting
- makes it easy to compare models fairly
"""

from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rul_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Standard regression metrics for RUL prediction.

    Parameters
    ----------
    y_true:
        True RUL values.
    y_pred:
        Predicted RUL values.

    Returns
    -------
    dict
        Dictionary containing MAE and RMSE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

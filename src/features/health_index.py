"""
Health Index utilities.

A Health Index (HI) is a scalar signal representing degradation over time.
This module provides simple, defensible quality checks that align with
reliability engineering expectations.
"""

from __future__ import annotations

import numpy as np


def monotonicity_score(signal) -> float:
    """
    Compute a monotonicity score for a degradation signal.

    We expect degradation to be mostly monotonic (often decreasing),
    so we measure the fraction of steps where the signal moves in the
    expected direction (diff <= 0).

    Parameters
    ----------
    signal:
        Sequence of HI values.

    Returns
    -------
    float
        Value in [0, 1]; higher means more monotonic.
    """
    s = np.asarray(signal, dtype=float)
    if s.size < 2:
        return 1.0  # Trivial monotonicity for very short sequences.

    diffs = np.diff(s)
    return float(np.mean(diffs <= 0))

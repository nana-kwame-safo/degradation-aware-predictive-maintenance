"""
Shared array validation helpers for evaluation and analysis modules.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def as_float_1d(x: Iterable[float], name: str) -> np.ndarray:
    """
    Convert an iterable into a non-empty 1D float numpy array.
    """
    arr = np.asarray(list(x), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D. Received shape: {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def validate_same_length(*arrays: np.ndarray) -> None:
    """
    Raise when inputs do not share the same first-dimension length.
    """
    lengths = {int(a.shape[0]) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(
            f"All inputs must have the same length. Received lengths: {sorted(lengths)}"
        )

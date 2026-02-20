"""
Environment check utility.

This is useful for:
- confirming baseline dependencies (NumPy/Pandas/sklearn)
- checking optional deep-learning dependencies (PyTorch, TensorFlow)
- quickly debugging setup issues
- demonstrating reproducibility to reviewers/recruiters
"""

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import sklearn


def main() -> None:
    """Print key environment information."""
    print("numpy:", np.__version__)
    print("pandas:", pd.__version__)
    print("sklearn:", sklearn.__version__)

    try:
        import torch  # type: ignore

        print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
    except Exception as exc:
        print("torch: not available |", exc)

    try:
        import tensorflow as tf  # type: ignore

        print("tf:", tf.__version__, "| gpus:", tf.config.list_physical_devices("GPU"))
    except Exception as exc:
        print("tf: not available |", exc)


if __name__ == "__main__":
    main()

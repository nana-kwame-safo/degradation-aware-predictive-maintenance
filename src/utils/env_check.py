"""
Environment check utility.

This is useful for:
- confirming installations (PyTorch, TensorFlow)
- quickly debugging setup issues
- demonstrating reproducibility to reviewers/recruiters
"""

from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import tensorflow as tf


def main() -> None:
    """Print key environment information."""
    print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
    print("tf:", tf.__version__, "| gpus:", tf.config.list_physical_devices("GPU"))


if __name__ == "__main__":
    main()

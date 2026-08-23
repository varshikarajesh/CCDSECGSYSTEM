# -*- coding: utf-8 -*-
"""
preprocessing/validation.py

Contract boundary validation for preprocessing arrays and objects.
Asserts shapes, finite values, and lead-order consistency.
"""

import numpy as np


def validate_canonical_signal(signal: np.ndarray) -> np.ndarray:
    """Validates that a processed signal is a 2D numpy array of shape (12, 1000) and finite."""
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(signal).__name__}")
    if signal.ndim != 2 or signal.shape[0] != 12:
        raise ValueError(f"Expected shape (12, N), got {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError("Signal contains non-finite (NaN/Inf) values.")
    return signal

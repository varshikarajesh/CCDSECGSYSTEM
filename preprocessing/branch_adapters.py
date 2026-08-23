# -*- coding: utf-8 -*-
"""
preprocessing/branch_adapters.py

Branch-specific input adapters:
1. prepare_retrieval_input: Applies historical retrieval preprocessing profile (normalization/scaling) -> (1, 12, 1000) float32 tensor.
"""

import torch
import numpy as np
from typing import Dict, Any
from preprocessing.preprocessing_config import RetrievalPrepConfig


def prepare_retrieval_input(
    canonical_result: Dict[str, Any],
    config: RetrievalPrepConfig = RetrievalPrepConfig()
) -> torch.Tensor:
    """
    Retrieval adapter: converts canonical signal result dict to (1, 12, 1000) tensor
    using RetrievalPrepConfig profile.
    """
    signal = canonical_result["signal"].copy()  # (12, 1000)

    # Per-channel z-score normalization for retrieval
    for i in range(12):
        std = np.std(signal[i])
        if std > 1e-6:
            signal[i] = (signal[i] - np.mean(signal[i])) / std

    # Clipping if enabled
    if config.clipping_std is not None:
        signal = np.clip(signal, -config.clipping_std, config.clipping_std)

    tensor = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0)  # (1, 12, 1000)
    return tensor



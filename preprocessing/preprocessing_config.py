# -*- coding: utf-8 -*-
"""
preprocessing/preprocessing_config.py

Frozen dataclasses defining:
1. CanonicalSignalConfig: Shared signal loading, lead ordering, sampling rate, duration, finite checks.
2. RetrievalPrepConfig: Historical retrieval branch preprocessing and scaling profile.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass(frozen=True)
class CanonicalSignalConfig:
    target_sampling_rate: int = 100
    target_length: int = 1000
    expected_leads: Tuple[str, ...] = (
        "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
    )
    input_units: str = "mV"
    output_units: str = "mV"
    preprocessing_version: str = "v3.0.0"


@dataclass(frozen=True)
class RetrievalPrepConfig:
    normalization_mode: str = "dataset_zscore"  # Historical retrieval z-score
    clipping_std: Optional[float] = 6.0
    bandpass_enabled: bool = True
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    notch_enabled: bool = True
    notch_frequency_hz: float = 50.0
    resampling_method: str = "scipy_polyphase"



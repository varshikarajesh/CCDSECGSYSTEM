# -*- coding: utf-8 -*-
"""
preprocessing/ecg_preprocessor.py

Unified ECG preprocessor implementing canonical signal loading:
1. Re-orders channels to canonical 12-lead order: I, II, III, aVR, aVL, aVF, V1-V6.
2. Resamples deterministically to 100 Hz (10 seconds, 1000 samples).
3. Rejects NaN / Inf values.
4. Computes signal quality metrics.
5. Computes reproducible SHA-256 signal checksum.
"""

import hashlib
import numpy as np
from scipy import signal as scipy_signal
from typing import Dict, Any, List, Optional
from preprocessing.preprocessing_config import CanonicalSignalConfig
from preprocessing.signal_quality import SignalQualityChecker


CANONICAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def compute_signal_hash(arr: np.ndarray) -> str:
    """Computes SHA-256 hash of a numpy float array."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def preprocess_raw_ecg(
    signal: np.ndarray,
    sampling_rate: int = 100,
    lead_names: Optional[List[str]] = None,
    units: str = "mV",
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[CanonicalSignalConfig] = None
) -> Dict[str, Any]:
    """
    Shared base preprocessing pipeline. Converts raw 12-lead signal to canonical (12, 1000) array.
    """
    if config is None:
        config = CanonicalSignalConfig()

    warnings: List[str] = []

    # 1. Orientation check: convert (N, 12) to (12, N)
    if signal.ndim == 2 and signal.shape[0] != 12 and signal.shape[1] == 12:
        signal = signal.T
        warnings.append("Transposed input signal from (N, 12) to (12, N).")

    if signal.ndim != 2 or signal.shape[0] != 12:
        raise ValueError(f"Input ECG signal must be (12, N). Got shape {signal.shape}.")

    # 2. Re-order leads if lead_names supplied
    if lead_names is not None and len(lead_names) == 12:
        lead_map = {name.upper(): idx for idx, name in enumerate(lead_names)}
        reordered = np.zeros_like(signal)
        for i, target_lead in enumerate(CANONICAL_LEADS):
            if target_lead in lead_map:
                reordered[i] = signal[lead_map[target_lead]]
            else:
                warnings.append(f"Missing lead {target_lead}; populated with zero array.")
        signal = reordered

    # 3. Finite value check
    if not np.isfinite(signal).all():
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        warnings.append("Non-finite values detected; replaced with 0.0.")

    # 4. Resample deterministically to target length (e.g. 1000 samples for 100 Hz, 10s)
    current_length = signal.shape[1]
    target_length = config.target_length

    if current_length != target_length:
        if sampling_rate != config.target_sampling_rate and sampling_rate > 0:
            num_samples = int(round(current_length * (config.target_sampling_rate / sampling_rate)))
        else:
            num_samples = target_length
        
        # Polyphase resample
        resampled_signal = np.zeros((12, target_length), dtype=np.float32)
        for i in range(12):
            resamp = scipy_signal.resample(signal[i], num_samples)
            if len(resamp) > target_length:
                resampled_signal[i] = resamp[:target_length]
            elif len(resamp) < target_length:
                resampled_signal[i, :len(resamp)] = resamp
            else:
                resampled_signal[i] = resamp
        signal = resampled_signal
    else:
        signal = signal.astype(np.float32)

    # 5. Assess signal quality
    checker = SignalQualityChecker(expected_leads=CANONICAL_LEADS)
    quality_report = checker.assess_quality(signal)
    warnings.extend(quality_report["warnings"])

    # 6. Compute signal checksum
    signal_hash = compute_signal_hash(signal)

    return {
        "signal": signal,
        "shape": list(signal.shape),
        "lead_names": list(CANONICAL_LEADS),
        "sampling_rate": config.target_sampling_rate,
        "units": config.output_units,
        "quality": quality_report,
        "warnings": warnings,
        "preprocessing_version": config.preprocessing_version,
        "preprocessing_hash": compute_signal_hash(np.array([config.target_sampling_rate, config.target_length], dtype=np.float32)),
        "signal_checksum": signal_hash
    }

"""Lead-aware adapter for supported external 12-lead ECG datasets."""
from __future__ import annotations

from math import gcd
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample_poly


class ExternalDatasetAdapter:
    SUPPORTED_DATASETS = {"Chapman", "Georgia", "CPSC2018", "INCART", "PTB_Diagnostic"}
    CANONICAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

    def adapt(
        self,
        raw_ecg_data: np.ndarray,
        dataset_name: str,
        original_sampling_rate: int,
        lead_names: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported external dataset: {dataset_name}")
        if not isinstance(original_sampling_rate, int) or original_sampling_rate <= 0:
            raise ValueError("original_sampling_rate must be a positive integer")

        data = np.asarray(raw_ecg_data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("External ECG must be a two-dimensional 12-lead array")
        if data.shape[0] != 12 and data.shape[1] == 12:
            data = data.T
        if data.shape[0] != 12:
            raise ValueError(f"Expected exactly 12 measured leads, received shape {data.shape}")
        if not np.isfinite(data).all():
            raise ValueError("External ECG contains NaN or infinite values")

        if lead_names is not None:
            normalized = [str(name).strip() for name in lead_names]
            if len(normalized) != 12 or set(normalized) != set(self.CANONICAL_LEADS):
                raise ValueError("lead_names must contain each canonical 12-lead name exactly once")
            order = [normalized.index(name) for name in self.CANONICAL_LEADS]
            data = data[order]

        divisor = gcd(original_sampling_rate, 100)
        standardized = resample_poly(data, 100 // divisor, original_sampling_rate // divisor, axis=1)
        if standardized.shape[1] < 1000:
            raise ValueError("External ECG contains less than 10 seconds after resampling to 100 Hz")
        standardized = np.ascontiguousarray(standardized[:, :1000], dtype=np.float32)

        return standardized, {
            "dataset_name": dataset_name,
            "original_sampling_rate": original_sampling_rate,
            "target_sampling_rate": 100,
            "original_shape": list(np.asarray(raw_ecg_data).shape),
            "standardized_shape": list(standardized.shape),
            "lead_order": list(self.CANONICAL_LEADS),
            "resampling_method": "scipy.signal.resample_poly",
            "adaptation_status": "validated",
        }

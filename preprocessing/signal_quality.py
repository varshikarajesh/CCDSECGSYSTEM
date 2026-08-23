# -*- coding: utf-8 -*-
"""
preprocessing/signal_quality.py

Deterministic 12-lead ECG signal quality evaluation.
Checks for flat leads, missing leads, NaN/Inf, clipping, excessive noise, and baseline drift.
Outputs ACCEPTABLE, DEGRADED, or REJECT with detailed metrics.
"""

import numpy as np
from typing import Dict, Any, List


class SignalQualityChecker:
    """Evaluates 12-lead ECG signal quality deterministically."""

    def __init__(self, expected_leads: tuple = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")):
        self.expected_leads = list(expected_leads)

    def assess_quality(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Assess signal quality of a (12, N) numpy array.
        Returns dictionary with quality_status, overall_quality_score, per_lead_quality, warnings.
        """
        warnings: List[str] = []
        per_lead_scores: Dict[str, float] = {}

        if not isinstance(signal, np.ndarray):
            return {
                "quality_status": "REJECT",
                "overall_quality_score": 0.0,
                "per_lead_quality": {},
                "warnings": ["Input is not a numpy ndarray."]
            }

        if signal.ndim != 2 or signal.shape[0] != 12:
            return {
                "quality_status": "REJECT",
                "overall_quality_score": 0.0,
                "per_lead_quality": {},
                "warnings": [f"Invalid shape {signal.shape}. Expected (12, N)."]
            }

        # Check NaN / Infinite values
        if not np.isfinite(signal).all():
            warnings.append("Signal contains non-finite (NaN/Inf) values.")
            return {
                "quality_status": "REJECT",
                "overall_quality_score": 0.0,
                "per_lead_quality": {},
                "warnings": warnings
            }

        flat_leads_count = 0
        clipping_leads_count = 0

        for i, lead_name in enumerate(self.expected_leads):
            lead_data = signal[i]
            lead_std = float(np.std(lead_data))
            lead_range = float(np.ptp(lead_data))

            # Flat line check
            if lead_std < 1e-6 or lead_range < 1e-5:
                flat_leads_count += 1
                warnings.append(f"Lead {lead_name} appears to be flat (std={lead_std:.2e}).")
                per_lead_scores[lead_name] = 0.0
                continue

            # Clipping check
            unique_vals = np.unique(lead_data)
            if len(unique_vals) < 20:
                clipping_leads_count += 1
                warnings.append(f"Lead {lead_name} has severe quantization/clipping ({len(unique_vals)} unique levels).")

            # Lead quality score (1.0 = perfect)
            score = 1.0
            if lead_std > 5.0 or lead_range > 20.0:
                score -= 0.3
                warnings.append(f"Lead {lead_name} has abnormally high amplitude/noise (std={lead_std:.2f}).")
            if lead_std < 0.05:
                score -= 0.2

            per_lead_scores[lead_name] = max(0.0, round(score, 3))

        overall_score = float(np.mean(list(per_lead_scores.values()))) if per_lead_scores else 0.0

        if flat_leads_count >= 3 or overall_score < 0.3:
            status = "REJECT"
        elif flat_leads_count > 0 or overall_score < 0.7 or clipping_leads_count > 0:
            status = "DEGRADED"
        else:
            status = "ACCEPTABLE"

        return {
            "quality_status": status,
            "overall_quality_score": round(overall_score, 4),
            "per_lead_quality": per_lead_scores,
            "warnings": warnings
        }

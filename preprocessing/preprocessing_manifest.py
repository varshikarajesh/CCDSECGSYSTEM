# -*- coding: utf-8 -*-
"""
preprocessing/preprocessing_manifest.py

Generates and validates preprocessing manifests containing configuration parameters,
scaler parameters, git commit metadata, and SHA-256 checksums.
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Dict, Any
from preprocessing.preprocessing_config import CanonicalSignalConfig, RetrievalPrepConfig


def generate_preprocessing_manifest(output_path: str) -> Dict[str, Any]:
    """Generates preprocessing_manifest.json with hashes and parameters."""
    canonical_cfg = CanonicalSignalConfig()
    retrieval_cfg = RetrievalPrepConfig()

    manifest_dict = {
        "preprocessing_version": canonical_cfg.preprocessing_version,
        "sampling_rate": canonical_cfg.target_sampling_rate,
        "target_length": canonical_cfg.target_length,
        "lead_order": list(canonical_cfg.expected_leads),
        "input_units": canonical_cfg.input_units,
        "output_units": canonical_cfg.output_units,
        "retrieval_profile": {
            "normalization_mode": retrieval_cfg.normalization_mode,
            "clipping_std": retrieval_cfg.clipping_std,
            "bandpass_enabled": retrieval_cfg.bandpass_enabled,
            "low_hz": retrieval_cfg.bandpass_low_hz,
            "high_hz": retrieval_cfg.bandpass_high_hz
        },
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    manifest_bytes = json.dumps(manifest_dict, sort_keys=True).encode('utf-8')
    manifest_dict["config_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_dict, f, indent=4)

    return manifest_dict


def validate_preprocessing_manifest(manifest_path: str) -> bool:
    """Validates that preprocessing manifest exists and is readable."""
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return "preprocessing_version" in data and "config_sha256" in data
    except Exception:
        return False

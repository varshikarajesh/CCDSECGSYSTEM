"""Coarse-to-fine ECG recording window selection.

The selector is diagnostic-model agnostic.  It consumes frozen V7 embeddings,
signal quality and measured ECG descriptors, then identifies windows that merit
the expensive classifier/retrieval/bridge path.  The experimental INCART model
may contribute localization probabilities, but never labels or diagnoses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample_poly

from backend.bridge.ecg_statistics import extract_ecg_statistics
from preprocessing.signal_quality import SignalQualityChecker


MODES = {"10s": 10, "2min": 120, "5min": 300}


@dataclass(frozen=True)
class RecordingSelectionConfig:
    target_rate_hz: int = 100
    window_seconds: int = 10
    coarse_stride_seconds: int = 10
    refine_offset_seconds: int = 5
    robust_sigma: float = 3.0
    quality_gate: float = 0.60
    selection_threshold: float = 0.55
    maximum_auto_episodes: int = 8
    stable_representatives: int = 2


def canonical_recording(
    signal: Any,
    sampling_rate_hz: int,
    lead_names: Optional[Sequence[str]] = None,
    target_rate_hz: int = 100,
) -> np.ndarray:
    """Validate orientation/leads and resample without shortening the recording."""
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"ECG must be a 2-D array, received {x.shape}")
    if x.shape[0] != 12 and x.shape[1] == 12:
        x = x.T
    if x.shape[0] != 12:
        raise ValueError(f"ECG must contain exactly 12 leads, received {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Recording contains NaN or infinite values")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")

    canonical = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    if lead_names is not None:
        if len(lead_names) != 12:
            raise ValueError("lead_names must contain exactly 12 names")
        lookup = {str(name).upper(): index for index, name in enumerate(lead_names)}
        missing = [name for name in canonical if name.upper() not in lookup]
        if missing:
            raise ValueError(f"Missing canonical leads: {missing}")
        x = np.stack([x[lookup[name.upper()]] for name in canonical])

    if sampling_rate_hz != target_rate_hz:
        x = resample_poly(x, target_rate_hz, sampling_rate_hz, axis=1).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def validate_mode_duration(signal: np.ndarray, mode: str, rate_hz: int = 100, tolerance_seconds: float = 1.0) -> None:
    if mode not in MODES:
        raise ValueError(f"Unsupported recording mode {mode!r}; expected one of {sorted(MODES)}")
    observed = signal.shape[1] / float(rate_hz)
    expected = MODES[mode]
    if abs(observed - expected) > tolerance_seconds:
        raise ValueError(f"Mode {mode} expects approximately {expected}s, received {observed:.2f}s")


def window_at(signal: np.ndarray, start_seconds: float, rate_hz: int = 100, window_seconds: int = 10) -> np.ndarray:
    start = int(round(start_seconds * rate_hz))
    stop = start + window_seconds * rate_hz
    if start < 0 or stop > signal.shape[1]:
        raise ValueError(f"Window {start_seconds:.1f}-{start_seconds + window_seconds:.1f}s is outside recording")
    return np.ascontiguousarray(signal[:, start:stop], dtype=np.float32)


def coarse_starts(duration_seconds: float, window_seconds: int = 10, stride_seconds: int = 10) -> List[float]:
    if duration_seconds < window_seconds:
        return []
    return [float(v) for v in np.arange(0.0, duration_seconds - window_seconds + 1e-6, stride_seconds)]


def robust_upper_threshold(values: np.ndarray, sigma: float = 3.0) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("inf")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median + sigma * max(1.4826 * mad, 1e-6)


def _unit_interval(values: np.ndarray, threshold: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(threshold) or threshold <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / threshold, 0.0, 1.0).astype(np.float32)


def _normalized_mean(vectors: np.ndarray) -> np.ndarray:
    mean = np.asarray(vectors, dtype=np.float64).mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero stable-state embedding")
    return (mean / norm).astype(np.float32)


def _morphology_deviation(statistics: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Robust deviation of measured descriptors; unavailable fields contribute nothing."""
    paths = (
        ("beat_detection", "mean_heart_rate_bpm"),
        ("time_domain_hrv", "mean_rr_ms"),
        ("time_domain_hrv", "rmssd_ms"),
        ("morphology", "qrs_duration_median_ms_estimate"),
        ("morphology", "lead_ii_r_amplitude_median"),
    )
    feature_scores = []
    for group, key in paths:
        vals = np.array([
            np.nan if row.get(group, {}).get(key) is None else float(row[group][key])
            for row in statistics
        ], dtype=np.float64)
        finite = np.isfinite(vals)
        score = np.zeros(len(vals), dtype=np.float64)
        if finite.sum() >= 3:
            med = np.median(vals[finite])
            mad = max(1.4826 * np.median(np.abs(vals[finite] - med)), 1e-6)
            score[finite] = np.clip(np.abs(vals[finite] - med) / (3.0 * mad), 0.0, 1.0)
        feature_scores.append(score)
    return np.max(np.stack(feature_scores), axis=0).astype(np.float32) if feature_scores else np.zeros(len(statistics), np.float32)


def score_embedding_states(
    embeddings: Any,
    statistics: Sequence[Dict[str, Any]],
    quality_scores: Sequence[float],
    temporal_probabilities: Optional[Sequence[float]] = None,
    config: RecordingSelectionConfig = RecordingSelectionConfig(),
) -> Dict[str, Any]:
    """Score coarse windows without assigning a diagnosis."""
    z = np.asarray(embeddings, dtype=np.float32)
    if z.ndim != 2 or z.shape[1] != 128:
        raise ValueError(f"Expected [windows,128] embeddings, received {z.shape}")
    if len(statistics) != len(z) or len(quality_scores) != len(z):
        raise ValueError("Window statistics/quality must align with embeddings")
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("Window embedding contains a zero vector")
    z = z / norms

    transition = np.zeros(len(z), dtype=np.float32)
    if len(z) > 1:
        transition[1:] = np.clip(1.0 - np.sum(z[1:] * z[:-1], axis=1), 0.0, 2.0)
    transition_threshold = robust_upper_threshold(transition[1:] if len(z) > 1 else transition, config.robust_sigma)

    provisional = np.argsort(transition)[:max(1, int(np.ceil(len(z) * 0.60)))]
    stable_centroid = _normalized_mean(z[provisional])
    centroid_distance = np.clip(1.0 - z @ stable_centroid, 0.0, 2.0).astype(np.float32)
    centroid_threshold = robust_upper_threshold(centroid_distance, config.robust_sigma)
    # One robust refinement keeps persistent altered states out of the baseline.
    stable_mask = (transition <= transition_threshold) & (centroid_distance <= centroid_threshold)
    if stable_mask.any():
        stable_centroid = _normalized_mean(z[stable_mask])
        centroid_distance = np.clip(1.0 - z @ stable_centroid, 0.0, 2.0).astype(np.float32)

    temporal = np.zeros(len(z), dtype=np.float32) if temporal_probabilities is None else np.clip(np.asarray(temporal_probabilities, dtype=np.float32), 0.0, 1.0)
    if temporal.shape != (len(z),):
        raise ValueError("temporal_probabilities must contain one value per window")
    morphology = _morphology_deviation(statistics)
    combined = (
        0.40 * temporal
        + 0.30 * _unit_interval(transition, transition_threshold)
        + 0.20 * _unit_interval(centroid_distance, centroid_threshold)
        + 0.10 * morphology
    ).astype(np.float32)

    quality = np.asarray(quality_scores, dtype=np.float32)
    artifact = quality < config.quality_gate
    candidate = (combined >= config.selection_threshold) & ~artifact
    return {
        "transition_distance": transition,
        "centroid_distance": centroid_distance,
        "temporal_probability": temporal,
        "morphology_deviation": morphology,
        "combined_score": combined,
        "quality_score": quality,
        "artifact_mask": artifact,
        "candidate_mask": candidate,
        "stable_mask": ~candidate & ~artifact,
        "stable_centroid": stable_centroid,
        "thresholds": {
            "transition": float(transition_threshold),
            "centroid": float(centroid_threshold),
            "combined": float(config.selection_threshold),
            "quality": float(config.quality_gate),
        },
    }


def refinement_starts(
    coarse_window_starts: Sequence[float],
    candidate_mask: Sequence[bool],
    duration_seconds: float,
    config: RecordingSelectionConfig = RecordingSelectionConfig(),
) -> List[float]:
    """Add only half-stride windows adjacent to coarse candidates/transitions."""
    existing = {round(float(v), 6) for v in coarse_window_starts}
    refined = set()
    for index, selected in enumerate(candidate_mask):
        if not selected:
            continue
        center = float(coarse_window_starts[index])
        for start in (center - config.refine_offset_seconds, center + config.refine_offset_seconds):
            if start >= 0 and start + config.window_seconds <= duration_seconds and round(start, 6) not in existing:
                refined.add(round(start, 6))
    return sorted(refined)


def merge_selected_episodes(
    windows: Sequence[Dict[str, Any]],
    maximum_episodes: int = 8,
) -> List[Dict[str, Any]]:
    selected = sorted(
        [w for w in windows if w.get("selected") and w.get("status") != "artifact_or_low_quality"],
        key=lambda w: (float(w["start_seconds"]), float(w["end_seconds"])),
    )
    episodes: List[List[Dict[str, Any]]] = []
    for window in selected:
        if not episodes or float(window["start_seconds"]) > max(float(w["end_seconds"]) for w in episodes[-1]):
            episodes.append([window])
        else:
            episodes[-1].append(window)
    result = []
    for members in episodes:
        representative = max(members, key=lambda w: float(w.get("combined_score", 0.0)))
        sources = sorted({str(w.get("selection_source", "automatic")) for w in members})
        result.append({
            "episode_index": len(result),
            "start_seconds": min(float(w["start_seconds"]) for w in members),
            "end_seconds": max(float(w["end_seconds"]) for w in members),
            "window_indices": [int(w["window_index"]) for w in members],
            "representative_window_index": int(representative["window_index"]),
            "peak_score": float(representative.get("combined_score", 0.0)),
            "selection_sources": sources,
        })
    return sorted(result, key=lambda e: e["peak_score"], reverse=True)[:maximum_episodes]


def apply_manual_selection(
    windows: List[Dict[str, Any]],
    manual_window_indices: Optional[Iterable[int]] = None,
) -> None:
    requested = {int(v) for v in (manual_window_indices or [])}
    valid = {int(w["window_index"]) for w in windows}
    missing = requested - valid
    if missing:
        raise ValueError(f"Manual window indices are unavailable: {sorted(missing)}")
    for window in windows:
        if int(window["window_index"]) in requested:
            window["selected"] = True
            window["selection_source"] = "clinician"
            window["overrides_automatic_selector"] = not bool(window.get("automatic_selected"))


class ExperimentalHolterLocalizer:
    """Loads the INCART model only as non-diagnostic localization evidence."""

    def __init__(self, checkpoint_path: str, device: Any):
        import torch
        from long_recording_v2.model import MILTemporalClassifier

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        architecture = str(checkpoint.get("architecture", "transformer"))
        targets = tuple(checkpoint.get("targets", ()))
        if targets != ("any_rhythm_abnormality", "ventricular_abnormality"):
            raise RuntimeError(f"Unsupported Holter localization targets: {targets}")
        self.model = MILTemporalClassifier(architecture, events=2).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.device = device
        self.provenance = {
            "checkpoint": checkpoint_path,
            "status": "experimental_localization_only",
            "diagnostic_authority": False,
            "targets": list(targets),
            "validation_score": checkpoint.get("validation_score"),
        }

    def probabilities(self, embeddings: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        import torch

        with torch.inference_mode():
            tensor = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).unsqueeze(0).to(self.device)
            logits = self.model(tensor)["window_logits"][0]
            values = torch.sigmoid(logits).cpu().numpy()
        return values.max(axis=1).astype(np.float32), dict(self.provenance)


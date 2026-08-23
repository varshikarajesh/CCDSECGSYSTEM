"""Bounded temporary storage for processed ECG acquisition sessions."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class TemporaryRecordingStore:
    """Store canonical recordings briefly so review requests never need raw live buffers."""

    def __init__(self, root: Optional[Path] = None, ttl_seconds: int = 3600, maximum_sessions: int = 20):
        configured = os.environ.get("TRACE_RECORDING_TEMP_DIR")
        self.root = Path(root or configured or (Path(tempfile.gettempdir()) / "trace_ecg_recordings"))
        self.ttl_seconds = max(60, int(os.environ.get("TRACE_RECORDING_TTL_SECONDS", ttl_seconds)))
        self.maximum_sessions = max(1, int(os.environ.get("TRACE_RECORDING_MAX_SESSIONS", maximum_sessions)))
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, recording_id: str) -> tuple[Path, Path]:
        if not recording_id.startswith("rec_") or not recording_id[4:].isalnum():
            raise ValueError("Invalid recording identifier")
        return self.root / f"{recording_id}.npy", self.root / f"{recording_id}.json"

    def cleanup(self) -> None:
        now = time.time()
        metadata_files = sorted(self.root.glob("rec_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for position, metadata_path in enumerate(metadata_files):
            expired = now - metadata_path.stat().st_mtime > self.ttl_seconds
            over_limit = position >= self.maximum_sessions
            if expired or over_limit:
                recording_id = metadata_path.stem
                waveform_path, _ = self._paths(recording_id)
                waveform_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)

    def save(self, recording: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        self.cleanup()
        signal = np.ascontiguousarray(recording, dtype=np.float32)
        digest = hashlib.sha256(signal.tobytes()).hexdigest()
        recording_id = f"rec_{digest[:20]}"
        waveform_path, metadata_path = self._paths(recording_id)
        np.save(waveform_path, signal, allow_pickle=False)
        payload = {
            **metadata,
            "recording_id": recording_id,
            "waveform_sha256": digest,
            "shape": list(signal.shape),
            "dtype": "float32",
            "stored_at_unix": time.time(),
            "expires_after_seconds": self.ttl_seconds,
            "storage_policy": "temporary_local_processing_cache",
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def load(self, recording_id: str) -> tuple[np.ndarray, Dict[str, Any]]:
        self.cleanup()
        waveform_path, metadata_path = self._paths(recording_id)
        if not waveform_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Recording session is missing or expired")
        signal = np.load(waveform_path, allow_pickle=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(np.ascontiguousarray(signal, dtype=np.float32).tobytes()).hexdigest()
        if digest != metadata.get("waveform_sha256"):
            raise RuntimeError("Temporary recording integrity check failed")
        return np.asarray(signal, dtype=np.float32), metadata

    def window(self, recording_id: str, start_seconds: float, end_seconds: float) -> Dict[str, Any]:
        signal, metadata = self.load(recording_id)
        rate = int(metadata["sampling_rate_hz"])
        start = int(round(float(start_seconds) * rate))
        stop = int(round(float(end_seconds) * rate))
        if start < 0 or stop <= start or stop > signal.shape[1]:
            raise ValueError("Requested interval is outside the stored recording")
        return {
            "recording_id": recording_id,
            "start_seconds": float(start_seconds),
            "end_seconds": float(end_seconds),
            "sampling_rate_hz": rate,
            "lead_order": metadata.get("lead_order"),
            "values": signal[:, start:stop].tolist(),
        }

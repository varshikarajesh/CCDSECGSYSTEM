"""Safe deployment ECG loader for 12-lead NumPy, CSV/TXT, and WFDB files."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


CANONICAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
MODE_SECONDS = {"10s": 10, "2min": 120, "5min": 300}


@dataclass(frozen=True)
class LoadedECG:
    signal: np.ndarray
    sampling_rate_hz: int
    lead_names: tuple[str, ...]
    source_path: str
    mode: str
    duration_seconds: float


def _orient(signal: Any) -> np.ndarray:
    value = np.asarray(signal, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"ECG must be a 2-D array, received {value.shape}")
    if value.shape[0] != 12 and value.shape[1] == 12:
        value = value.T
    if value.shape[0] != 12:
        raise ValueError(f"ECG must contain exactly 12 leads, received {value.shape}")
    if value.shape[1] < 2:
        raise ValueError("ECG contains too few samples")
    if not np.isfinite(value).all():
        raise ValueError("ECG contains NaN or infinite values")
    return np.ascontiguousarray(value, dtype=np.float32)


def infer_mode(sample_count: int, sampling_rate_hz: int, tolerance_seconds: float = 1.0) -> str:
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    duration = sample_count / float(sampling_rate_hz)
    matches = [mode for mode, seconds in MODE_SECONDS.items() if abs(duration - seconds) <= tolerance_seconds]
    if len(matches) != 1:
        expected = ", ".join(f"{name}={seconds}s" for name, seconds in MODE_SECONDS.items())
        raise ValueError(f"Cannot infer recording mode from {duration:.2f}s; expected {expected}. Supply --mode explicitly.")
    return matches[0]


def _load_csv(path: Path) -> np.ndarray:
    # genfromtxt handles both headerless numeric matrices and common lead-name headers.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        first = next(csv.reader(handle), [])
    has_header = any(cell.strip().lower() in {name.lower() for name in CANONICAL_LEADS} for cell in first)
    data = np.genfromtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None,
                         skip_header=1 if has_header else 0, dtype=np.float32)
    return data


def _load_wfdb(path: Path) -> tuple[np.ndarray, Optional[int], Optional[Sequence[str]]]:
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("WFDB input requires the optional 'wfdb' package") from exc
    record_base = path.with_suffix("")
    record = wfdb.rdrecord(str(record_base))
    return np.asarray(record.p_signal, dtype=np.float32), int(round(record.fs)), tuple(record.sig_name)


def load_ecg(
    path: str | Path,
    sampling_rate_hz: Optional[int] = None,
    mode: Optional[str] = None,
    lead_names: Optional[Sequence[str]] = None,
) -> LoadedECG:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ECG input does not exist: {source}")

    suffix = source.suffix.lower()
    embedded_rate: Optional[int] = None
    embedded_leads: Optional[Sequence[str]] = None
    if suffix == ".npy":
        raw = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        package = np.load(source, allow_pickle=False)
        keys = list(package.files)
        signal_key = next((key for key in ("ecg", "signal", "waveform", "data") if key in package), None)
        if signal_key is None:
            if len(keys) != 1:
                raise ValueError(f"NPZ must contain ecg/signal/waveform/data; found {keys}")
            signal_key = keys[0]
        raw = package[signal_key]
        if "sampling_rate_hz" in package:
            embedded_rate = int(np.asarray(package["sampling_rate_hz"]).item())
        if "lead_names" in package:
            embedded_leads = tuple(str(x) for x in np.asarray(package["lead_names"]).tolist())
    elif suffix in {".csv", ".txt"}:
        raw = _load_csv(source)
    elif suffix in {".hea", ".dat"}:
        raw, embedded_rate, embedded_leads = _load_wfdb(source)
    else:
        raise ValueError("Supported ECG formats are .npy, .npz, .csv, .txt, .hea and .dat")

    signal = _orient(raw)
    rate = int(sampling_rate_hz or embedded_rate or 0)
    if rate <= 0:
        raise ValueError("Sampling rate is required (use --sampling-rate or include it in NPZ/WFDB metadata)")
    names = tuple(lead_names or embedded_leads or CANONICAL_LEADS)
    if len(names) != 12:
        raise ValueError("Exactly 12 lead names are required")
    selected_mode = mode or infer_mode(signal.shape[1], rate)
    if selected_mode not in MODE_SECONDS:
        raise ValueError(f"mode must be one of {sorted(MODE_SECONDS)}")
    duration = signal.shape[1] / float(rate)
    expected = MODE_SECONDS[selected_mode]
    if abs(duration - expected) > 1.0:
        raise ValueError(f"Mode {selected_mode} expects approximately {expected}s, received {duration:.2f}s")
    return LoadedECG(signal, rate, names, str(source), selected_mode, duration)

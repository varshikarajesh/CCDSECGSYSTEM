# -*- coding: utf-8 -*-
"""
streamlit_components/ecg_visualization.py

Plotly visualization helpers for 12-lead ECG waveforms.
Provides Stacked 12-lead plot and Clinical Grid 3x4 view with interactive zoom/pan.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None

CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def validate_and_orient_ecg(arr: np.ndarray) -> np.ndarray:
    """
    Validates ECG signal array shape and finiteness.
    Accepts (12, N) or (N, 12) orientation, converting (N, 12) to (12, N).
    Rejects 1D, >2D, NaN/Inf, or non-12-lead shapes.
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"ECG signal must be 2-dimensional, received shape {arr.shape}.")

    if not np.isfinite(arr).all():
        raise ValueError("ECG signal contains NaN or infinite values.")

    if arr.shape[0] == 12:
        return arr
    elif arr.shape[1] == 12:
        return arr.T
    else:
        raise ValueError(f"ECG signal must contain exactly 12 leads, received shape {arr.shape}.")


def compute_signal_stats(signal: np.ndarray, sampling_rate: int = 100) -> Dict[str, Any]:
    """Computes summary statistics for a (12, N) signal array."""
    sig = validate_and_orient_ecg(signal)
    num_leads, num_samples = sig.shape
    duration_sec = num_samples / max(1, sampling_rate)

    return {
        "num_leads": num_leads,
        "num_samples": num_samples,
        "sampling_rate_hz": sampling_rate,
        "duration_sec": round(duration_sec, 2),
        "min_val": round(float(np.min(sig)), 4),
        "max_val": round(float(np.max(sig)), 4),
        "mean_val": round(float(np.mean(sig)), 4),
        "std_val": round(float(np.std(sig)), 4),
    }


def plot_stacked_ecg(
    signal: np.ndarray,
    sampling_rate: int = 100,
    lead_names: Optional[List[str]] = None,
    title: str = "12-Lead ECG (Stacked Layout)",
    normalize: bool = False,
) -> go.Figure:
    """
    Renders stacked 12-lead ECG Plotly figure with shared time axis in seconds.
    Optional display-only normalization does NOT mutate the underlying signal array.
    """
    sig = validate_and_orient_ecg(signal)
    if normalize:
        # Display-only min-max scaling per lead
        norm_sig = np.zeros_like(sig)
        for i in range(12):
            min_v = np.min(sig[i])
            max_v = np.max(sig[i])
            rng = max_v - min_v
            if rng > 1e-6:
                norm_sig[i] = (sig[i] - min_v) / rng
            else:
                norm_sig[i] = sig[i]
        display_sig = norm_sig
    else:
        display_sig = sig

    leads = lead_names if (lead_names and len(lead_names) == 12) else CANONICAL_LEADS
    time_axis = np.arange(display_sig.shape[1]) / max(1, sampling_rate)

    fig = make_subplots(
        rows=12,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        subplot_titles=leads,
    )

    for i in range(12):
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=display_sig[i],
                mode="lines",
                name=leads[i],
                line=dict(color="#1f77b4", width=1.2),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=900,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Time (seconds)", row=12, col=1)

    return fig


def plot_clinical_grid_ecg(
    signal: np.ndarray,
    sampling_rate: int = 100,
    lead_names: Optional[List[str]] = None,
    title: str = "12-Lead ECG (Clinical Grid View)",
) -> go.Figure:
    """
    Renders 3x4 clinical grid view for 12-lead ECG using Plotly.
    Grid arrangement:
      Row 1: I, aVR, V1, V4
      Row 2: II, aVL, V2, V5
      Row 3: III, aVF, V3, V6
    """
    sig = validate_and_orient_ecg(signal)
    leads = lead_names if (lead_names and len(lead_names) == 12) else CANONICAL_LEADS
    time_axis = np.arange(sig.shape[1]) / max(1, sampling_rate)

    grid_order = [
        [0, 3, 6, 9],    # I, aVR, V1, V4
        [1, 4, 7, 10],   # II, aVL, V2, V5
        [2, 5, 8, 11]    # III, aVF, V3, V6
    ]

    subplot_titles = [leads[idx] for row in grid_order for idx in row]

    fig = make_subplots(
        rows=3,
        cols=4,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.04,
        subplot_titles=subplot_titles,
    )

    for r_idx in range(3):
        for c_idx in range(4):
            lead_i = grid_order[r_idx][c_idx]
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=sig[lead_i],
                    mode="lines",
                    name=leads[lead_i],
                    line=dict(color="#003366", width=1.2),
                ),
                row=r_idx + 1,
                col=c_idx + 1,
            )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=650,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
        template="plotly_white",
    )

    return fig

# -*- coding: utf-8 -*-
"""
streamlit_components/neighbor_visualization.py

Plotly visualization helpers for retrieved Top-K ECG neighbors.
Provides query vs. neighbor overlay, side-by-side comparison, similarity bar charts,
and family distribution charts.
"""

from typing import Dict, List, Optional, Any
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None

from streamlit_components.ecg_visualization import validate_and_orient_ecg, CANONICAL_LEADS


def plot_query_neighbor_overlay(
    query_signal: np.ndarray,
    neighbor_signal: np.ndarray,
    lead_idx: int = 0,
    lead_name: str = "Lead I",
    sampling_rate: int = 100,
) -> go.Figure:
    """
    Renders single-lead overlay comparison between query ECG and retrieved neighbor ECG.
    """
    q_sig = validate_and_orient_ecg(query_signal)
    n_sig = validate_and_orient_ecg(neighbor_signal)

    lead_i = min(max(0, lead_idx), 11)
    q_lead = q_sig[lead_i]
    n_lead = n_sig[lead_i]

    time_q = np.arange(len(q_lead)) / max(1, sampling_rate)
    time_n = np.arange(len(n_lead)) / max(1, sampling_rate)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_q, y=q_lead, mode="lines", name="Query Waveform",
        line=dict(color="#1f77b4", width=2.0)
    ))
    fig.add_trace(go.Scatter(
        x=time_n, y=n_lead, mode="lines", name="Retrieved Neighbor",
        line=dict(color="#ff7f0e", width=2.0, dash="dash")
    ))

    fig.update_layout(
        title=dict(text=f"Query vs. Neighbor Overlay ({lead_name})", x=0.5),
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white",
        legend=dict(x=0.8, y=0.95),
    )
    return fig


def plot_query_neighbor_side_by_side(
    query_signal: np.ndarray,
    neighbor_signal: np.ndarray,
    sampling_rate: int = 100,
) -> go.Figure:
    """
    Renders 12-lead side-by-side comparison between query ECG and retrieved neighbor ECG.
    """
    q_sig = validate_and_orient_ecg(query_signal)
    n_sig = validate_and_orient_ecg(neighbor_signal)

    time_q = np.arange(q_sig.shape[1]) / max(1, sampling_rate)
    time_n = np.arange(n_sig.shape[1]) / max(1, sampling_rate)

    fig = make_subplots(
        rows=12, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.015,
        column_titles=["Query Waveform", "Retrieved Neighbor"],
    )

    for i in range(12):
        lead_label = CANONICAL_LEADS[i]
        fig.add_trace(
            go.Scatter(x=time_q, y=q_sig[i], mode="lines", name=f"Q {lead_label}", line=dict(color="#1f77b4", width=1.0)),
            row=i + 1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_n, y=n_sig[i], mode="lines", name=f"N {lead_label}", line=dict(color="#ff7f0e", width=1.0)),
            row=i + 1, col=2
        )

    fig.update_layout(
        title=dict(text="12-Lead Side-by-Side Comparison", x=0.5),
        height=950,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
        template="plotly_white",
    )
    return fig


def plot_topk_similarity_chart(neighbors: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders horizontal bar chart of raw similarity and reranked scores across Top-K neighbors.
    """
    if not neighbors:
        fig = go.Figure()
        fig.update_layout(title="No Retrieved Neighbors Available")
        return fig

    ranks = [f"Rank #{n.get('raw_rank', idx + 1)} (ID: {n.get('ecg_id', '?')})" for idx, n in enumerate(neighbors)]
    raw_sims = [float(n.get("raw_similarity", n.get("similarity", 0.0))) for n in neighbors]
    reranked_scores = [float(n.get("reranked_score", n.get("score", 0.0))) for n in neighbors]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ranks, x=raw_sims, name="Raw Vector Similarity",
        orientation="h", marker_color="#2ca02c"
    ))
    fig.add_trace(go.Bar(
        y=ranks, x=reranked_scores, name="Reranked Score",
        orientation="h", marker_color="#1f77b4"
    ))

    fig.update_layout(
        title=dict(text="Top-K Neighbor Similarity & Reranked Scores", x=0.5),
        xaxis_title="Score / Similarity",
        barmode="group",
        height=max(300, len(neighbors) * 35),
        margin=dict(l=150, r=40, t=50, b=40),
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_topk_family_distribution(neighbors: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders bar chart showing distribution of diagnostic families among Top-K neighbors.
    """
    if not neighbors:
        fig = go.Figure()
        fig.update_layout(title="No Retrieved Neighbors Available")
        return fig

    family_counts: Dict[str, int] = {}
    for n in neighbors:
        fams = n.get("families") or n.get("diagnostic_class") or []
        if isinstance(fams, str):
            fams = [f.strip() for f in fams.split(",") if f.strip()]
        if not fams:
            fams = ["Unknown"]
        for f in fams:
            family_counts[f] = family_counts.get(f, 0) + 1

    labels = list(family_counts.keys())
    counts = list(family_counts.values())

    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker_color="#9467bd"
    ))

    fig.update_layout(
        title=dict(text="Diagnostic Family Distribution among Top-K Neighbors", x=0.5),
        xaxis_title="Diagnostic Family",
        yaxis_title="Count",
        height=320,
        margin=dict(l=40, r=40, t=50, b=40),
        template="plotly_white",
    )
    return fig

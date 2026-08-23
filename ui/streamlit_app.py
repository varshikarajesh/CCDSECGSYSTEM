# -*- coding: utf-8 -*-
"""
streamlit_app.py

Complete end-to-end Streamlit testing interface for the TRACE ECG Evidence & LLM Reasoning System.
Connects directly to the existing FastAPI backend (http://127.0.0.1:8000).
"""

import io
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_components.api_client import ApiClient
from streamlit_components.ecg_visualization import (
    plot_stacked_ecg,
    plot_clinical_grid_ecg,
    compute_signal_stats,
    validate_and_orient_ecg,
)
from streamlit_components.neighbor_visualization import (
    plot_query_neighbor_overlay,
    plot_query_neighbor_side_by_side,
    plot_topk_similarity_chart,
    plot_topk_family_distribution,
)
from streamlit_components.result_panels import (
    render_summary_panel,
    render_classifier_panel,
    render_retrieval_panel,
    render_recording_panel,
    render_explanation_panel,
    render_feedback_panel,
    render_provenance_panel,
)
from streamlit_components.session_state import init_session_state

st.set_page_config(
    page_title="TRACE ECG Testing Interface",
    page_icon="🫀",
    layout="wide",
)

init_session_state()

# Disclaimer Header
st.title("🫀 TRACE ECG Inference & Evidence Reasoning System")
st.caption("🔬 Research and Testing Interface | Clinician-in-the-Loop Decision Support Prototype")
st.warning("⚠️ **Research Interface Disclaimer:** This tool is designed for engineering evaluation and clinical decision-support testing only. It is not a certified diagnostic device.")

# Sidebar Configuration Controls
with st.sidebar:
    st.header("⚙️ Backend & Pipeline Settings")
    api_url = st.text_input("FastAPI Server URL", value="http://127.0.0.1:8000")
    timeout_sec = st.number_input(
        "Request Timeout (sec)",
        min_value=30,
        max_value=1200,
        value=600,
        help="The first real Gemma request can take several minutes while the model loads.",
    )
    api_client = ApiClient(base_url=api_url, timeout_seconds=timeout_sec)

    st.markdown("---")
    st.subheader("Inference Controls")
    sampling_rate_hz = st.selectbox("Sampling Rate (Hz)", options=[100, 500, 1000], index=0)
    recording_mode = st.selectbox(
        "Acquisition mode",
        options=["10s", "2min", "5min"],
        help="10s uses the standard complete pipeline. Longer modes use coarse-to-fine recording analysis.",
    )
    top_k = st.slider("Retrieval Top-K Depth", min_value=1, max_value=20, value=5)
    manual_windows_text = st.text_input(
        "Clinician-selected window indices",
        value="",
        help="For 2/5 minute analysis, enter comma-separated indices after reviewing the automatic timeline, then rerun.",
    )
    ecg_id_input = st.number_input("Optional ECG ID", value=378, step=1)
    case_id_input = st.text_input("Optional Case / Conversation ID", value="test_case_001")
    question_input = st.text_area(
        "Question for TRACE",
        value="What is the primary finding and diagnostic conclusion?",
        help="The question router selects the relevant read-only evidence and permanent TRACE system policy.",
    )

    st.markdown("---")
    st.subheader("Pipeline Features")
    include_retrieval = st.checkbox("Include FAISS Retrieval", value=True)
    include_knowledge = st.checkbox("Include Knowledge Base", value=True)
    include_explanation = st.checkbox("Generate Base-Gemma Explanation", value=True)

    st.markdown("---")
    st.subheader("Backend System Health")
    if st.button("Check API Connection"):
        try:
            health = api_client.get_health()
            st.session_state["api_health"] = health
            st.success(f"Status: {health.get('status', 'healthy')}")
        except Exception as exc:
            st.error(f"Backend Connection Failed: {exc}")

    health = st.session_state.get("api_health")
    if health:
        st.write(f"**LLM Mode:** `{health.get('configured_llm_mode', 'disabled')}`")
        st.write(f"**LLM Backend:** `{health.get('configured_llm_backend', 'llama_cpp')}`")
        st.write(f"**LLM Variant:** `{health.get('llm_model_variant', 'base')}`")
        st.write(f"**Adapter Supported:** `{health.get('adapter_supported', False)}`")

    run_inference_btn = st.button("🚀 Run Complete Inference", type="primary", width="stretch")

# Define Main Application Tabs
tab_input, tab_summary, tab_cls, tab_ret, tab_exp, tab_fb, tab_prov = st.tabs([
    "1. ECG Input",
    "2. Inference Summary",
    "3. Classifier & Family",
    "4. Top-K Retrieval",
    "5. Base-Gemma Explanation",
    "6. Feedback",
    "7. Technical Provenance",
])

# Tab 1: ECG Input & Visualization
with tab_input:
    st.header("Upload & Inspect 12-Lead ECG Waveform")

    col_up, col_sample = st.columns([3, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload 12-Lead ECG File (.npy or .csv)",
            type=["npy", "csv"],
            help="Must be 2D array of shape (12, N) or (N, 12) with 12 leads and finite numeric values."
        )

    with col_sample:
        st.markdown("**Test Packaged Sample:**")
        if st.button("Use Sample ECG (378.npy)"):
            # Load sample file bytes
            sample_path = "sample_data/deployment_test_ecg_378.npy"
            try:
                with open(sample_path, "rb") as f:
                    st.session_state["uploaded_file_bytes"] = f.read()
                    st.session_state["uploaded_filename"] = "deployment_test_ecg_378.npy"
                st.success("Loaded deployment_test_ecg_378.npy sample!")
            except Exception as exc:
                st.error(f"Failed to load sample: {exc}")

    if uploaded_file is not None:
        st.session_state["uploaded_file_bytes"] = uploaded_file.getvalue()
        st.session_state["uploaded_filename"] = uploaded_file.name

    file_bytes = st.session_state.get("uploaded_file_bytes")
    filename = st.session_state.get("uploaded_filename")

    if file_bytes and filename:
        st.success(f"Selected File: `{filename}` ({len(file_bytes)} bytes)")
        try:
            # Parse & orient array locally without executing inference logic
            if filename.endswith(".npy") or file_bytes.startswith(b"\x93NUMPY"):
                raw_arr = np.load(io.BytesIO(file_bytes), allow_pickle=False)
            else:
                text = file_bytes.decode("utf-8")
                try:
                    raw_arr = np.loadtxt(io.StringIO(text), delimiter=",", dtype=np.float32)
                except Exception:
                    raw_arr = np.loadtxt(io.StringIO(text), dtype=np.float32)

            oriented_sig = validate_and_orient_ecg(raw_arr)
            st.session_state["parsed_ecg_signal"] = oriented_sig

            stats = compute_signal_stats(oriented_sig, sampling_rate=sampling_rate_hz)

            st.subheader("Waveform Metadata & Signal Statistics")
            st.json(stats)

            st.subheader("Visualization Layout")
            view_mode = st.radio("Layout View", options=["Stacked 12-Lead View", "Clinical Grid View (3x4)"], horizontal=True)
            norm_display = st.checkbox("Normalize amplitude for display only (does not alter inference data)", value=False)

            if view_mode == "Stacked 12-Lead View":
                fig_stacked = plot_stacked_ecg(oriented_sig, sampling_rate=sampling_rate_hz, normalize=norm_display)
                st.plotly_chart(fig_stacked, width="stretch")
            else:
                fig_grid = plot_clinical_grid_ecg(oriented_sig, sampling_rate=sampling_rate_hz)
                st.plotly_chart(fig_grid, width="stretch")

        except Exception as exc:
            st.error(f"ECG Signal Parse/Validation Error: {exc}")

# Execution Handler
if run_inference_btn:
    if not file_bytes or not filename:
        st.error("Please upload or select an ECG file first in the 'ECG Input' tab!")
    else:
        with st.spinner("Executing ECG analysis..."):
            if recording_mode == "10s":
                success, payload, status_code = api_client.run_inference(
                    file_bytes=file_bytes,
                    filename=filename,
                    sampling_rate_hz=sampling_rate_hz,
                    top_k=top_k,
                    include_retrieval=include_retrieval,
                    include_knowledge=include_knowledge,
                    include_explanation=include_explanation,
                    ecg_id=ecg_id_input,
                    case_id=case_id_input,
                    question=question_input,
                )
            else:
                try:
                    manual_indices = [int(v.strip()) for v in manual_windows_text.split(",") if v.strip()]
                except ValueError:
                    st.error("Clinician-selected windows must be comma-separated integer indices.")
                    st.stop()
                success, payload, status_code = api_client.run_recording_inference(
                    file_bytes=file_bytes,
                    filename=filename,
                    recording_mode=recording_mode,
                    sampling_rate_hz=sampling_rate_hz,
                    top_k=top_k,
                    manual_window_indices=manual_indices,
                )

            if success:
                st.session_state["inference_result"] = payload
                st.success("Inference completed successfully!")

                # Capture Immutability Upstream Snapshot
                st.session_state["upstream_snapshot"] = {
                    "classifier": payload.get("classifier"),
                    "family_head": payload.get("family_head"),
                    "retrieval": payload.get("retrieval"),
                    "confidence": payload.get("confidence"),
                    "decision_status": payload.get("decision_status"),
                }
            else:
                st.error(f"Inference API Error ({status_code}): {payload.get('detail', 'Unknown error')}")

# Render Result Tabs
result = st.session_state.get("inference_result")

with tab_summary:
    if result:
        if result.get("result_type") == "recording":
            render_recording_panel(result, api_client)
        else:
            render_summary_panel(result)
    else:
        st.info("Upload an ECG waveform and click 'Run Complete Inference' to view results.")

with tab_cls:
    if result:
        render_classifier_panel(result)
    else:
        st.info("No inference results available yet.")

with tab_ret:
    if result:
        query_sig = st.session_state.get("parsed_ecg_signal")
        render_retrieval_panel(result, api_client, query_sig)
    else:
        st.info("No inference results available yet.")

with tab_exp:
    if result:
        if result.get("result_type") == "recording":
            st.info("Recording-level LLM narration is intentionally disabled until the structured recording bridge is independently validated. The bridge and per-window evidence remain available.")
        else:
            render_explanation_panel(
                result,
                api_client=api_client,
                file_bytes=file_bytes,
                filename=filename,
                case_id=case_id_input,
            )
    else:
        st.info("No inference results available yet.")

with tab_fb:
    if result:
        render_feedback_panel(api_client, result)
    else:
        st.info("Run inference first before submitting case feedback.")

with tab_prov:
    if result:
        render_provenance_panel(result)
    else:
        st.info("No inference results available yet.")

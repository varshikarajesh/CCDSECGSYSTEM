# -*- coding: utf-8 -*-
"""
streamlit_components/result_panels.py

UI rendering components for the TRACE Streamlit testing interface.
Renders summary, classifier, OOD, retrieval, explanation, feedback, and provenance panels.
"""

from typing import Dict, Any, Optional
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

import streamlit as st

from streamlit_components.api_client import ApiClient
from streamlit_components.ecg_visualization import plot_stacked_ecg
from streamlit_components.neighbor_visualization import (
    plot_query_neighbor_overlay,
    plot_query_neighbor_side_by_side,
    plot_topk_similarity_chart,
    plot_topk_family_distribution,
)


def render_summary_panel(result: Dict[str, Any]) -> None:
    """
    Renders main summary tab.
    Leads with evidence-based decision reasoning, primary candidate, model agreement, OOD status, and confidence.
    """
    st.header("Diagnosis & Evidence Reasoning Summary")

    decision_status = result.get("decision_status", "Unknown")
    reasoning = result.get("decision_reasoning", {})
    summary_text = reasoning.get("summary", "Decision reasoning payload unavailable.")
    reason_codes = reasoning.get("reason_codes", [])
    evidence_conflicts = reasoning.get("evidence_conflicts", [])

    # Prominent Decision Reasoning Narrative Header
    st.info(f"**Evidence Reasoning Summary:**\n\n{summary_text}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Structured Decision", decision_status)
    with col2:
        cand = result.get("primary_candidate", {})
        label = cand.get("label", "N/A")
        prob = cand.get("probability")
        prob_str = f"{int(round(prob * 100))}%" if prob is not None else "N/A"
        st.metric("Leading Candidate", f"{label} ({prob_str})")
    with col3:
        conf = result.get("confidence", {})
        score = conf.get("score")
        score_str = f"{score:.2f}" if score is not None else "N/A"
        st.metric("Diagnostic Confidence", score_str)

    # Retrieval Unverified Banner Warning
    ret = result.get("retrieval", {})
    ret_prov = result.get("retrieval_status", {}) or ret.get("retrieval_provenance", {})
    if ret_prov.get("verification_status") != "verified":
        st.warning(
            "⚠️ **Retrieval index provenance is unverified.** Neighbors are displayed for engineering review and are excluded from diagnostic confidence."
        )

    if reason_codes:
        st.subheader("Structured Reason Codes")
        st.write(", ".join([f"`{code}`" for code in reason_codes]))

    if evidence_conflicts:
        with st.expander("Identified Evidence Conflicts", expanded=False):
            for conflict in evidence_conflicts:
                if isinstance(conflict, dict):
                    c_type = conflict.get("type", "Conflict")
                    c_effect = conflict.get("effect", "")
                    st.write(f"- **{c_type}**: {c_effect}")
                else:
                    st.write(f"- {conflict}")


def render_classifier_panel(result: Dict[str, Any]) -> None:
    """
    Renders classifier and family head predictions with probability bar charts.
    """
    st.header("Classifier & Family Head Predictions")

    cls = result.get("classifier", {})
    fam = result.get("family_head", {})

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Classifier Predictions")
        top_preds = cls.get("top_predictions", [])
        if top_preds:
            labels = [p.get("label", "?") for p in top_preds[:5]]
            probs = [float(p.get("probability", 0.0)) for p in top_preds[:5]]

            if PLOTLY_AVAILABLE and go is not None:
                fig = go.Figure(go.Bar(
                    x=probs, y=labels, orientation="h",
                    marker_color="#1f77b4"
                ))
                fig.update_layout(
                    xaxis=dict(title="Probability", range=[0, 1]),
                    yaxis=dict(autorange="reversed"),
                    height=300,
                    margin=dict(l=100, r=20, t=30, b=40),
                    template="plotly_white",
                )
                st.plotly_chart(fig, width="stretch")
            else:
                for l, p in zip(labels, probs):
                    st.write(f"- **{l}**: {p:.2%}")

    with col2:
        st.subheader("Top Family Predictions")
        top_fams = fam.get("top_predictions", []) or fam.get("family_predictions", [])
        if top_fams:
            if isinstance(top_fams, dict):
                sorted_fams = sorted(top_fams.items(), key=lambda x: x[1], reverse=True)[:5]
                fam_labels = [k for k, v in sorted_fams]
                fam_probs = [float(v) for k, v in sorted_fams]
            else:
                fam_labels = [p.get("family", p.get("label", "?")) for p in top_fams[:5]]
                fam_probs = [float(p.get("probability", 0.0)) for p in top_fams[:5]]

            if PLOTLY_AVAILABLE and go is not None:
                fig2 = go.Figure(go.Bar(
                    x=fam_probs, y=fam_labels, orientation="h",
                    marker_color="#2ca02c"
                ))
                fig2.update_layout(
                    xaxis=dict(title="Probability", range=[0, 1]),
                    yaxis=dict(autorange="reversed"),
                    height=300,
                    margin=dict(l=100, r=20, t=30, b=40),
                    template="plotly_white",
                )
                st.plotly_chart(fig2, width="stretch")
            else:
                for fl, fp in zip(fam_labels, fam_probs):
                    st.write(f"- **{fl}**: {fp:.2%}")


def render_retrieval_panel(result: Dict[str, Any], api_client: ApiClient, query_signal: Optional[np.ndarray]) -> None:
    """
    Renders Top-K FAISS Retrieval neighbors.
    """
    st.header("Top-K FAISS Retrieval Neighbors")

    ret = result.get("retrieval", {})
    ret_prov = result.get("retrieval_status", {}) or ret.get("retrieval_provenance", {})

    # Mandatory Banner Warning if Retrieval Index is Unverified
    if ret_prov.get("verification_status") != "verified":
        st.warning(
            "⚠️ **Retrieval index provenance is unverified.** Neighbors are displayed for engineering review and are excluded from diagnostic confidence."
        )

    neighbors = ret.get("raw_neighbors", [])
    if not neighbors and "reranking" in result:
        neighbors = result["reranking"].get("top_5", [])

    if not neighbors:
        st.info("No retrieval neighbors returned or retrieval was disabled.")
        return

    st.subheader(f"Retrieved Top-{len(neighbors)} Neighbors")

    # Similarity & Family Charts
    col1, col2 = st.columns(2)
    with col1:
        fig_sim = plot_topk_similarity_chart(neighbors)
        if fig_sim is not None and PLOTLY_AVAILABLE:
            st.plotly_chart(fig_sim, width="stretch")
    with col2:
        fig_fam = plot_topk_family_distribution(neighbors)
        if fig_fam is not None and PLOTLY_AVAILABLE:
            st.plotly_chart(fig_fam, width="stretch")

    st.markdown("---")
    st.subheader("Neighbor Details & Waveform Comparison")

    selected_idx = st.selectbox(
        "Select Neighbor to Inspect",
        options=list(range(len(neighbors))),
        format_func=lambda idx: f"Rank #{neighbors[idx].get('raw_rank', idx+1)} - ECG ID {neighbors[idx].get('ecg_id')} (Sim: {neighbors[idx].get('raw_similarity', 0.0):.4f})"
    )

    n_data = neighbors[selected_idx]
    ecg_id = n_data.get("ecg_id")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Rank:** #{n_data.get('raw_rank', selected_idx + 1)}")
        st.write(f"**ECG ID:** `{ecg_id}`")
        st.write(f"**Patient ID:** `{n_data.get('patient_id', 'N/A')}`")
        st.write(f"**SCP Codes:** `{n_data.get('scp_codes', [])}`")
    with col_b:
        st.write(f"**Raw Vector Similarity:** `{n_data.get('raw_similarity', 0.0):.4f}`")
        st.write(f"**Reranked Score:** `{n_data.get('reranked_score', 0.0):.4f}`")
        st.write(f"**Diagnostic Families:** `{n_data.get('families', [])}`")

    # Waveform Loading for Selected Neighbor
    if ecg_id is not None:
        if st.button(f"Load Waveform for Neighbor #{ecg_id}"):
            with st.spinner(f"Fetching neighbor waveform for ECG ID {ecg_id}..."):
                success, waveform_resp = api_client.fetch_neighbor_waveform(int(ecg_id))
                if success and "values" in waveform_resp:
                    st.session_state["selected_neighbor_id"] = ecg_id
                    st.session_state["selected_neighbor_waveform"] = np.asarray(waveform_resp["values"], dtype=np.float32)
                    st.success(f"Successfully loaded waveform for Neighbor #{ecg_id}")
                else:
                    st.error(f"Could not load waveform for Neighbor #{ecg_id}: {waveform_resp.get('detail', 'Missing file')}")

    if st.session_state.get("selected_neighbor_id") == ecg_id and st.session_state.get("selected_neighbor_waveform") is not None:
        n_wave = st.session_state["selected_neighbor_waveform"]
        if query_signal is not None and PLOTLY_AVAILABLE:
            from streamlit_components.ecg_visualization import CANONICAL_LEADS
            tab_overlay, tab_side = st.tabs(["Lead Overlay View", "12-Lead Side-by-Side View"])
            with tab_overlay:
                lead_idx = st.selectbox(
                    "Select Lead for Overlay",
                    options=list(range(len(CANONICAL_LEADS))),
                    index=0,
                    format_func=lambda i: CANONICAL_LEADS[i],
                )
                fig_ov = plot_query_neighbor_overlay(query_signal, n_wave, lead_idx=lead_idx, lead_name=CANONICAL_LEADS[lead_idx])
                if fig_ov is not None:
                    st.plotly_chart(fig_ov, width="stretch")
            with tab_side:
                fig_sb = plot_query_neighbor_side_by_side(query_signal, n_wave)
                if fig_sb is not None:
                    st.plotly_chart(fig_sb, width="stretch")


def render_recording_panel(result: Dict[str, Any], api_client: Optional[ApiClient] = None) -> None:
    """Render recording-level selector state and clinician-review windows."""
    st.header("Recording-level analysis")
    selector = result.get("selector", {})
    bridge = result.get("recording_bridge", {})
    cols = st.columns(5)
    cols[0].metric("Mode", result.get("recording_mode", "N/A"))
    cols[1].metric("Duration", f"{result.get('duration_seconds', 0):.0f} s")
    cols[2].metric("Coarse windows", selector.get("coarse_window_count", 0))
    cols[3].metric("Refined windows", selector.get("refined_window_count", 0))
    cols[4].metric("Episodes", len(result.get("episodes", [])))

    st.write(f"**Recording bridge decision:** `{bridge.get('decision', 'indeterminate')}`")
    st.write(f"**Leading label:** `{bridge.get('primary_label', 'UNKNOWN')}`")
    st.caption("The experimental Holter model, when enabled, contributes localization only and has no diagnostic authority.")

    acquisition = result.get("acquisition", {})
    if acquisition:
        st.caption(
            f"Temporary acquisition: `{acquisition.get('recording_id')}` · "
            f"integrity `{str(acquisition.get('waveform_sha256', ''))[:12]}…` · "
            f"expires after {acquisition.get('expires_after_seconds', 0)} seconds"
        )

    statistics = result.get("recording_statistics", {})
    overall = statistics.get("overall", {})
    if overall:
        st.subheader("Complete-recording ECG statistics")
        beat = overall.get("beat_detection", {})
        hrv = overall.get("time_domain_hrv", {})
        morphology = overall.get("morphology", {})
        stat_cols = st.columns(6)
        stat_cols[0].metric("Mean HR", f"{beat.get('mean_heart_rate_bpm') or 0:.1f} bpm")
        stat_cols[1].metric("Mean RR", f"{hrv.get('mean_rr_ms') or 0:.1f} ms")
        stat_cols[2].metric("SDNN", f"{hrv.get('sdnn_ms') or 0:.1f} ms")
        stat_cols[3].metric("RMSSD", f"{hrv.get('rmssd_ms') or 0:.1f} ms")
        stat_cols[4].metric("QRS estimate", f"{morphology.get('qrs_duration_median_ms_estimate') or 0:.1f} ms")
        stat_cols[5].metric("R peaks", beat.get("r_peak_count", 0))
        for limitation in overall.get("limitations", []):
            st.caption(f"• {limitation}")

    windows = result.get("windows", [])
    if windows:
        table = [{
            "index": w.get("window_index"),
            "start_s": w.get("start_seconds"),
            "end_s": w.get("end_seconds"),
            "status": w.get("status"),
            "score": round(float(w.get("combined_score", 0.0)), 4),
            "embedding_change": round(float(w.get("transition_distance", 0.0)), 4),
            "quality": round(float(w.get("quality_score", 0.0)), 4),
            "selected_by": w.get("selection_source"),
        } for w in windows]
        st.subheader("Abnormal-window timeline")
        try:
            import plotly.graph_objects as go
            colors = {"candidate": "#d62728", "stable": "#2ca02c", "artifact_or_low_quality": "#7f7f7f"}
            fig = go.Figure()
            for row in windows:
                fig.add_trace(go.Bar(
                    x=[float(row["end_seconds"]) - float(row["start_seconds"])],
                    y=["Recording"],
                    base=[float(row["start_seconds"])],
                    orientation="h",
                    marker_color=colors.get(row.get("status"), "#1f77b4"),
                    name=f"Window {row['window_index']}",
                    text=f"#{row['window_index']} {row['status']} score={row['combined_score']:.3f}",
                    hoverinfo="text",
                    showlegend=False,
                ))
            fig.update_layout(barmode="overlay", xaxis_title="Seconds", height=220)
            st.plotly_chart(fig, width="stretch")
        except Exception:
            pass
        st.dataframe(table, width="stretch", hide_index=True)
        st.info("To force additional windows through classifier and retrieval, enter their indices in the sidebar and rerun recording analysis.")

    if result.get("episodes"):
        st.subheader("Merged abnormal episodes")
        st.json(result["episodes"])

    processed = result.get("processed_windows", [])
    if processed:
        st.subheader("Classifier and retrieval results for processed windows")
        for row in processed:
            with st.expander(
                f"Window {row['window_index']} · {row['start_seconds']:.0f}-{row['end_seconds']:.0f}s · {row['role']}"
            ):
                window_result = row.get("result", {})
                st.write("Classifier:", window_result.get("classifier", {}).get("top_predictions", []))
                st.write("Retrieval:", window_result.get("retrieval", {}).get("raw_neighbors", []))

        st.subheader("Inspect one processed window")
        selected = st.selectbox(
            "Choose abnormal, manual, or stable-reference window",
            processed,
            format_func=lambda row: (
                f"Window {row['window_index']} · {row['start_seconds']:.0f}-{row['end_seconds']:.0f}s · "
                f"{row['role']} · {row['selection_source']}"
            ),
        )
        selected_result = selected.get("result", {})
        if api_client and acquisition.get("recording_id"):
            ok, waveform = api_client.fetch_recording_window(
                acquisition["recording_id"], selected["start_seconds"], selected["end_seconds"]
            )
            if ok and waveform.get("values"):
                selected_signal = np.asarray(waveform["values"], dtype=np.float32)
                st.plotly_chart(
                    plot_stacked_ecg(
                        selected_signal,
                        sampling_rate=int(waveform.get("sampling_rate_hz", 100)),
                        title=f"Selected window {selected['window_index']}",
                    ),
                    width="stretch",
                )
            elif not ok:
                st.warning(waveform.get("detail", "The temporary recording window could not be loaded."))
        st.write("**Window classifier findings**", selected_result.get("classifier", {}).get("top_predictions", []))
        selected_neighbors = selected_result.get("retrieval", {}).get("raw_neighbors", [])
        st.write("**Window-specific V7 neighbors**", selected_neighbors)

    findings = result.get("recording_findings", [])
    if findings:
        st.subheader("Multi-label findings across selected episodes")
        st.dataframe(findings, width="stretch", hide_index=True)

    comparison = statistics.get("comparison", {})
    if comparison.get("status") == "available":
        st.subheader("Complete recording versus selected abnormal windows")
        st.json(comparison.get("metrics", {}))

    references = result.get("clinical_references", [])
    if references:
        st.subheader("Traceable clinical references")
        for reference in references:
            title = reference.get("title") or "Validated source"
            url = reference.get("url")
            label = f"[{title}]({url})" if url else title
            st.markdown(
                f"- {label}; {reference.get('organization_or_authors') or 'author unavailable'}; "
                f"{reference.get('year') or 'year unavailable'}; section `{reference.get('section')}`; "
                f"locator `{reference.get('page_or_locator')}`"
            )

    audit = bridge.get("function_audit", {})
    if audit:
        with st.expander("Bridge function audit", expanded=False):
            st.json(audit)
        validation = audit.get("validation", {})
        if validation.get("status") == "PASS":
            st.warning(
                "Bridge metrics shown in the audit apply to held-out 10-second PTB-XL ECGs only. "
                "They do not validate the new 2/5-minute recording decision path."
            )

    final_decision = result.get("final_diagnostic_decision", {})
    if final_decision:
        st.subheader("Structured final ECG decision")
        st.write(f"**Status:** `{final_decision.get('status')}`")
        st.write(f"**Primary finding:** `{final_decision.get('primary_label')}`")
        st.write("**Supported coexisting findings:**", final_decision.get("supported_labels", []))
        st.write("**Partially supported findings:**", final_decision.get("partially_supported_labels", []))
        st.warning(final_decision.get("scope", "Clinician review is required."))


def render_explanation_panel(
    result: Dict[str, Any],
    api_client: Optional[ApiClient] = None,
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    case_id: Optional[str] = None,
) -> None:
    """
    Renders Gemma 3 + ECG LoRA explanation, guardrail validation panel, response provenance, and live clinician Q&A chatbot.
    """
    st.header("Gemma 3 + ECG LoRA Explanation & Live Assistant")

    exp = result.get("explanation", {})
    prov = result.get("provenance", {}) or result.get("runtime_provenance", {})
    text = exp.get("text", "[No Explanation Generated]")
    answer_source = exp.get("answer_source", prov.get("answer_source", "N/A"))
    val_status = exp.get("guardrail_status", exp.get("validation", {}).get("status", "N/A"))
    intent_str = exp.get("intent", prov.get("intent", "N/A"))

    if exp.get("fallback_used") or answer_source == "deterministic_fallback":
        reason = exp.get("fallback_reason") or prov.get("failure_reason") or "Safety validation or system limits"
        st.warning(
            f"⚠️ **Gemma/LoRA did not produce the final answer.**\n\n"
            f"A deterministic safety fallback was shown because: {reason}"
        )

    st.markdown(f"### Clinician Narrative Explanation\n\n> {text}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Answer Source", answer_source)
    with col2:
        st.metric("Routed Intent", intent_str)
    with col3:
        st.metric("Guardrail Status", val_status)
    with col4:
        st.metric("Citations Count", len(exp.get("citations", [])))

    with st.expander("Response Provenance & Diagnostics", expanded=False):
        st.write(f"- **Answer Source:** `{answer_source}`")
        st.write(f"- **Routed Intent:** `{intent_str}` (Sub-Intent: `{exp.get('sub_intent', prov.get('sub_intent', 'N/A'))}`)")
        st.write(f"- **Patient Specific Query:** `{exp.get('patient_specific', prov.get('patient_specific', False))}`")
        st.write(f"- **Generation Attempted:** `{exp.get('generation_attempted', prov.get('generation_attempted', False))}`")
        st.write(f"- **LoRA Adapter Active:** `{prov.get('fine_tuned_adapter_used', False)}`")
        st.write(f"- **Repair Attempted:** `{exp.get('repair_attempted', prov.get('repair_attempted', False))}`")
        st.write(f"- **Repair Succeeded:** `{exp.get('repair_succeeded', prov.get('repair_succeeded', False))}`")
        st.write(f"- **Guardrail Status:** `{val_status}`")
        st.write(f"- **Fallback Used:** `{exp.get('fallback_used', prov.get('fallback_used', False))}`")
        st.write(f"- **Fallback Reason:** `{exp.get('fallback_reason', prov.get('failure_reason', 'N/A'))}`")
        st.write(f"- **Citation Status:** `{exp.get('citation_status', 'N/A')}`")
        st.write(f"- **Citations:** `{exp.get('citations', [])}`")
        st.write(f"- **Skills Activated:** `{exp.get('skills_activated', prov.get('skills_activated', []))}`")

    st.markdown("---")
    st.subheader("💬 Live Interactive Clinician Chatbot")
    st.caption("Ask follow-up questions about this ECG waveform, diagnostic findings, confidence drivers, or OOD conflicts.")

    if st.session_state.get("active_chat_case_id") != case_id or st.session_state.get("chat_messages") is None:
        st.session_state["active_chat_case_id"] = case_id
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": "You can ask about the structured ECG findings, confidence, OOD status, retrieval limitations or general educational topics. Answers cannot determine symptoms, coronary anatomy or procedures not established by the available evidence."
            }
        ]
        st.session_state["selected_neighbor"] = None
        st.session_state["loaded_neighbor_waveforms"] = {}
        st.session_state["latest_explanation"] = None
        st.session_state["citation_diagnostics"] = None
        st.session_state["fallback_diagnostics"] = None

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a follow-up question about this ECG case...")
    if user_query:
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Processing clinical response..."):
                if api_client and file_bytes and filename:
                    success, resp, status_code = api_client.run_inference(
                        file_bytes=file_bytes,
                        filename=filename,
                        question=user_query,
                        case_id=case_id or "chat_session",
                        include_retrieval=True,
                        include_explanation=True,
                    )
                    if success and "explanation" in resp:
                        ans_text = resp["explanation"].get("text", "No response generated.")
                    else:
                        ans_text = f"API Response ({status_code}): {resp.get('detail', 'Could not generate answer')}"
                else:
                    ans_text = "Backend API client or ECG file bytes unavailable. Please upload an ECG file and verify API connection."

                st.markdown(ans_text)
                st.session_state["chat_messages"].append({"role": "assistant", "content": ans_text})


def render_feedback_panel(api_client: ApiClient, result: Dict[str, Any]) -> None:
    """
    Renders clinician feedback submission panel calling POST /api/feedback.
    """
    st.header("Submit Clinician Feedback")
    st.info("Feedback is recorded for audit review and is not immediately used for retraining.")

    with st.form("clinician_feedback_form"):
        case_id = st.text_input("Case / ECG ID", value=str(result.get("metadata", {}).get("ecg_id", "12345")))
        reviewer_id = st.text_input("Reviewing Clinician ID", value="dr_smith")
        agreement = st.selectbox("Clinician Agreement", options=["Agreed", "Disagreed", "Uncertain"])
        corrected_label = st.text_input("Corrected Label (if disagreed)", value="")
        notes = st.text_area("Clinician Comments / Notes", value="Waveform reviewed.")

        submit_button = st.form_submit_button("Submit Feedback")

        if submit_button:
            payload = {
                "case_id": case_id,
                "reviewer_id": reviewer_id,
                "agreement_status": agreement,
                "corrected_label": corrected_label if agreement == "Disagreed" else None,
                "notes": notes,
                "new_status": "Approved" if agreement == "Agreed" else "Needs Clarification",
            }
            with st.spinner("Submitting feedback to backend..."):
                success, resp, status_code = api_client.submit_feedback(payload)
                if success:
                    st.success(f"Feedback successfully recorded! Reference ID: `{resp.get('feedback_id', 'N/A')}`")
                    st.json(resp)
                else:
                    st.error(f"Feedback submission failed ({status_code}): {resp.get('detail', 'Server error')}")


def render_provenance_panel(result: Dict[str, Any]) -> None:
    """
    Renders full technical runtime provenance payload.
    """
    st.header("Technical Runtime Provenance")
    prov = result.get("runtime_provenance", {})
    st.json(prov)

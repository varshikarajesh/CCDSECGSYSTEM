"""Deployment-only orchestrator for the authoritative TRACE V4/V7 pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


FINAL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FINAL_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment_config import (  # noqa: E402
    ACTIVE_MODELS_YAML_PATH,
    CLASSIFIER_CHECKPOINT_PATH,
    ENCODER_CHECKPOINT_PATH,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    KB_EMBEDDINGS_PATH,
    KB_ID_LIST_PATH,
    KB_PATH,
    CLINICAL_ONTOLOGY_PATH,
    CONDITION_CARDS_PATH,
    LABEL_TO_CONCEPT_MAP_PATH,
    ORDERED_SCP_LABELS_PATH,
    SCP_STATEMENTS_PATH,
    SCP_THRESHOLDS_PATH,
    TRACE_GGUF_MODEL,
)
from runtime.runtime_contracts import make_json_safe  # noqa: E402


ACTIVE_REQUIRED_ASSETS = {
    "model_registry": ACTIVE_MODELS_YAML_PATH,
    "classifier_checkpoint": CLASSIFIER_CHECKPOINT_PATH,
    "classifier_labels": ORDERED_SCP_LABELS_PATH,
    "classifier_thresholds": SCP_THRESHOLDS_PATH,
    "scp_statements": SCP_STATEMENTS_PATH,
    "v7_encoder": ENCODER_CHECKPOINT_PATH,
    "v7_faiss_index": FAISS_INDEX_PATH,
    "v7_faiss_metadata": FAISS_METADATA_PATH,
    "knowledge_database": KB_PATH,
    "knowledge_embeddings": KB_EMBEDDINGS_PATH,
    "knowledge_ids": KB_ID_LIST_PATH,
    "knowledge_source_registry": KB_PATH.parent / "source_registry.json",
    "knowledge_validation_report": KB_PATH.parent / "validation_report.json",
    "knowledge_statistics_rules": KB_PATH.parent / "ecg_statistics_rules.json",
    "knowledge_ontology": CLINICAL_ONTOLOGY_PATH,
    "knowledge_condition_cards": CONDITION_CARDS_PATH,
    "bridge_label_concept_mapping": LABEL_TO_CONCEPT_MAP_PATH,
}



# Helper functions for spelling-tolerant conversation, persistent multi-turn clinical context, and abbreviations
def init_chat_state(recording_id: str, rec_hash: str, dec_hash: str) -> dict:
    return {
        "recording_hash": rec_hash,
        "decision_hash": dec_hash,
        "active_recording_id": recording_id,
        "last_user_question": "",
        "last_question_type": "",
        "last_subject": "",
        "last_finding_labels": [],
        "last_retrieved_cases": [],
        "last_selected_windows": [],
        "last_citations": [],
        "last_tool_results": {},
        "conversation_summary": "",
        "turn_history": []
    }

def get_abbreviation_mapping() -> dict[str, str]:
    from deployment_config import CLINICAL_ONTOLOGY_PATH, CONDITION_CARDS_PATH
    import json
    mapping = {}
    try:
        if CLINICAL_ONTOLOGY_PATH.is_file():
            ont = json.loads(CLINICAL_ONTOLOGY_PATH.read_text(encoding="utf-8"))
            for k, v in ont.items():
                if isinstance(v, dict) and "description" in v:
                    mapping[k.upper()] = v["description"].strip()
    except Exception:
        pass
    try:
        if CONDITION_CARDS_PATH.is_file():
            cards = json.loads(CONDITION_CARDS_PATH.read_text(encoding="utf-8"))
            for k, v in cards.items():
                if isinstance(v, dict) and "name" in v:
                    mapping[k.upper()] = v["name"].strip()
    except Exception:
        pass
    overrides = {
        "1AVB": "first-degree atrioventricular block",
        "2AVB": "second-degree atrioventricular block",
        "3AVB": "third-degree atrioventricular block",
        "ISCAL": "ischemic changes in anterolateral leads",
    }
    for k, v in overrides.items():
        mapping[k] = v
    mapping["IRBBB"] = "incomplete right bundle branch block"
    return mapping

def expand_abbreviations_in_text(text: str, mapping: dict[str, str]) -> str:
    import re
    sorted_abbrevs = sorted(mapping.keys(), key=len, reverse=True)
    for abbrev in sorted_abbrevs:
        expansion = mapping[abbrev]
        pattern = re.compile(rf"\b{re.escape(abbrev)}\b", re.IGNORECASE)
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        already_expanded_pattern = re.compile(
            rf"{re.escape(expansion)}\s*(?:\()?{re.escape(abbrev)}(?:\))?", re.IGNORECASE
        )
        if already_expanded_pattern.search(text):
            continue
        first_match = matches[0]
        start, end = first_match.span()
        matched_str = first_match.group(0)
        replacement = f"{expansion} ({matched_str})"
        text = text[:start] + replacement + text[end:]
    return text

def apply_natural_names(text: str, question: str) -> str:
    import re
    q = question.lower()
    is_technical = any(term in q for term in ("version", "system status", "pipeline status", "audit", "metadata", "trace", "developer", "dump"))
    if is_technical:
        return text
    # Replace internal component names optionally including leading articles
    text = re.sub(r"\b(?:the\s+)?Bridge\s+V4\b", "the bridge", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:the\s+)?V7\s+FAISS\s+retrieval\b", "retrieval", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:the\s+)?FAISS\s+retrieval\b", "retrieval", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:the\s+)?PTB-XL\s+(?:hierarchical\s+)?(?:multi-label\s+)?classifier\b", "the classifier", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:the\s+)?PTB-XL\b", "the classifier", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:the\s+)?Holter\s+(?:temporal\s+)?classifier\b", "the rhythm model", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:the\s+)?Holter\s+localization\b", "the rhythm model", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:the\s+)?Holter\s+classifier\b", "the rhythm model", text, flags=re.IGNORECASE)
    
    # Internal engineering terminology cleanup
    text = re.sub(r"\bTRACE_KNOWLEDGE_GOVERNANCE\b", "clinical governance", text)
    text = re.sub(r"\bactivated\s+KB\b", "knowledge base", text, flags=re.IGNORECASE)
    text = re.sub(r"\bretrieval\s+fallback\b", "analysis fallback", text, flags=re.IGNORECASE)
    text = re.sub(r"\bchunk\s+selection\b", "evidence selection", text, flags=re.IGNORECASE)
    text = re.sub(r"\bBridge\s+V4\s+implementation\b", "bridge decision", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfamily-head\s+implementation\b", "rhythm analysis", text, flags=re.IGNORECASE)
    
    def capitalize_sentences(match):
        prefix = match.group(1)
        word = match.group(2)
        return prefix + word.capitalize()
    text = re.sub(r"(^|[\.\!\?\n]\s+)(the bridge|the classifier|the rhythm model|retrieval)\b", capitalize_sentences, text)
    return text

def format_sources(citations: list[dict]) -> str:
    if not citations:
        return ""
    lines = ["Sources:"]
    for i, c in enumerate(citations, 1):
        title = c.get("source_title") or c.get("title") or "Clinical Guideline"
        authors = c.get("organization_or_authors") or "Unknown Authors"
        year = c.get("date_or_version") or "unknown"
        section = c.get("section") or "unknown section"
        page = c.get("page") or "unknown page"
        doi_url = ""
        if c.get("doi"):
            doi_url = f"https://doi.org/{c['doi']}"
        elif c.get("url"):
            doi_url = c["url"]
        loc = f"; {doi_url}" if doi_url else ""
        lines.append(f"{i}. {title} — {authors}, {year}; {section}; {page}{loc}.")
    return "\n".join(lines)

def render_retrieval_fallback(retrieval_evidence: dict, ontology_mapping: dict[str, str]) -> str:
    matches = retrieval_evidence.get("result", {}).get("matches", []) if isinstance(retrieval_evidence, dict) else []
    if not matches and isinstance(retrieval_evidence, dict):
        matches = retrieval_evidence.get("matches", [])
    if not matches:
        return "No similar retrieved cases found in the database."
    lines = ["Similar retrieved cases (from retrieval, not based on the Bridge decision):"]
    for i, match in enumerate(matches, 1):
        ecg_id = match.get("retrieved_ecg_id") or match.get("faiss_row") or "Unknown ID"
        similarity = match.get("similarity")
        sim_str = f"{similarity:.4f}" if isinstance(similarity, (int, float)) else str(similarity)
        codes = match.get("reference_scp_codes") or []
        code_strs = []
        for c in codes:
            c_upper = c.upper()
            if c_upper in ontology_mapping:
                code_strs.append(f"{ontology_mapping[c_upper]} ({c_upper})")
            else:
                code_strs.append(c_upper)
        codes_str = ", ".join(code_strs) if code_strs else "None"
        win_idx = match.get("matched_from_query_window_index")
        start_sec = match.get("matched_from_query_start_seconds")
        line = f"{i}. ECG/Reference ID: {ecg_id} | Similarity: {sim_str} | SCP Labels: {codes_str}"
        if win_idx is not None:
            line += f" | Query Window Index: {win_idx}"
        if start_sec is not None:
            line += f" | Start Time: {start_sec}s"
        lines.append(line)
    lines.append("\nNote: Retrieved ECGs are similar examples for reference and comparison, not diagnostic proof.")
    return "\n".join(lines)

def render_evidence_fallback(state: dict, ontology_mapping: dict[str, str]) -> str:
    bridge = state.get("bridge", {})
    decision = state.get("final_diagnostic_decision", bridge)
    supported_labels = decision.get("supported_labels", [])
    ecg_evidence = ["A. ECG/model evidence:"]
    prob_lines = []
    findings = decision.get("recording_findings", [])
    for f in findings:
        label = f.get("label")
        prob = f.get("maximum_probability")
        if label and prob is not None:
            prob_lines.append(f"{label}: {prob:.4f}")
    if prob_lines:
        ecg_evidence.append(f"- Classifier probabilities: {', '.join(prob_lines)}")
    else:
        ecg_evidence.append("- Classifier probabilities: Not available")
    episodes = state.get("temporal_summary", {}).get("abnormal_windows", [])
    timing_str = ", ".join(f"window {ep.get('window_index')} ({ep.get('start_seconds', 0):.1f}s-{ep.get('end_seconds', 0):.1f}s)" for ep in episodes)
    if timing_str:
        ecg_evidence.append(f"- Recording/window timing: {timing_str}")
    else:
        ecg_evidence.append("- Recording/window timing: No abnormal episodes detected")
    stats = state.get("statistics", {})
    stat_lines = []
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            stat_lines.append(f"{k}: {v:.2f}")
    if stat_lines:
        ecg_evidence.append(f"- Rhythm/statistical measurements: {', '.join(stat_lines)}")
    else:
        ecg_evidence.append("- Rhythm/statistical measurements: Not available")
    retrieval = state.get("retrieval", {})
    matches = retrieval.get("matches", retrieval.get("raw_neighbors", []))
    sim_lines = []
    for m in matches[:3]:
        ecg_id = m.get("retrieved_ecg_id") or m.get("ecg_id") or m.get("faiss_row")
        sim = m.get("similarity")
        if ecg_id and sim is not None:
            sim_lines.append(f"{ecg_id} (sim: {sim:.4f})")
    if sim_lines:
        ecg_evidence.append(f"- Retrieval similarities: {', '.join(sim_lines)}")
    else:
        ecg_evidence.append("- Retrieval similarities: No retrieval examples available")
    limitations = decision.get("limitations", []) or bridge.get("limitations", [])
    if limitations:
        ecg_evidence.append(f"- Conflicts and limitations: {'; '.join(limitations)}")
    else:
        ecg_evidence.append("- Conflicts and limitations: None identified")
    lit_evidence = ["B. Literature/knowledge evidence:"]
    chunks = state.get("knowledge_chunks", [])
    criteria = []
    symptoms = []
    diff_diag = []
    risks = []
    mgmt = []
    for c in chunks:
        sec = str(c.get("section") or c.get("tags", {}).get("section", "")).lower()
        text = c.get("evidence_summary") or c.get("text", "")
        if "criteria" in sec or "diagnostic" in sec:
            criteria.append(text)
        elif "symptom" in sec:
            symptoms.append(text)
        elif "differential" in sec:
            diff_diag.append(text)
        elif "risk" in sec or "red" in sec:
            risks.append(text)
        elif "management" in sec or "treatment" in sec:
            mgmt.append(text)
    lit_evidence.append(f"- Diagnostic criteria: {'; '.join(criteria) if criteria else 'No validated clinical diagnostic criteria available in the active knowledge base.'}")
    lit_evidence.append(f"- Symptom associations: {'; '.join(symptoms) if symptoms else 'No validated clinical symptom associations available in the active knowledge base.'}")
    lit_evidence.append(f"- Differential diagnosis: {'; '.join(diff_diag) if diff_diag else 'No validated clinical differential diagnosis available in the active knowledge base.'}")
    lit_evidence.append(f"- Risks/red flags: {'; '.join(risks) if risks else 'No validated clinical risk or red-flag evidence available in the active knowledge base.'}")
    lit_evidence.append(f"- Management guidance: {'; '.join(mgmt) if mgmt else 'No validated clinical management guidance available in the active knowledge base.'}")
    return "\n".join(ecg_evidence) + "\n\n" + "\n".join(lit_evidence)

def render_next_steps_fallback(state: dict, ontology_mapping: dict[str, str]) -> str:
    bridge = state.get("bridge", {})
    decision = state.get("final_diagnostic_decision", bridge)
    supported_labels = decision.get("supported_labels", [])
    
    primary_label = decision.get("primary_label") or bridge.get("primary_label") or "UNKNOWN"
    primary_name = ontology_mapping.get(primary_label.upper(), primary_label)
    
    # Check if primary finding is normal
    is_normal = primary_label.upper() in ("NORM", "SR")
    
    lines = []
    if is_normal:
        lines.append(f"The ECG shows {primary_name}, which is a normal finding.")
        lines.append("\nClinical next steps and evaluation guidance:")
        lines.append("1. What the ECG supports: Normal conduction and rhythm without diagnostic abnormality.")
        lines.append("2. Management: No clinical work-up or investigation of an underlying cause is required for sinus rhythm itself.")
        lines.append("3. Verification: Fused decisions require clinician verification and correlation with the patient's presentation.")
    else:
        lines.append(f"Based on the ECG findings, the primary supported finding is {primary_name}.")
        
        secondary_labels = [l for l in supported_labels if l != primary_label]
        if secondary_labels:
            secondary_names = [ontology_mapping.get(l.upper(), l) for l in secondary_labels]
            lines.append(f"Secondary co-existing findings: {', '.join(secondary_names)}.")
            
        lines.append("\nClinical next steps and evaluation guidance:")
        lines.append(f"1. What the ECG supports: Waveform features compatible with {primary_name}.")
        lines.append("2. What requires confirmation: Fused decisions require clinician verification and correlation with original waveforms.")
        
        # Check for validated clinical guideline evidence in state
        criteria = []
        mgmt = []
        for c in state.get("knowledge_chunks", []):
            sec = str(c.get("section") or c.get("tags", {}).get("section", "")).lower()
            text = c.get("evidence_summary") or c.get("text", "")
            is_clinical = c.get("evidence_type") in ("clinical_practice_guideline", "textbook_chapter", "clinical_education_reference")
            if is_clinical:
                if "criteria" in sec or "diagnostic" in sec:
                    criteria.append(text)
                elif "management" in sec or "treatment" in sec or "guidance" in sec:
                    mgmt.append(text)
        
        if mgmt:
            # Use actual retrieved guideline evidence
            guideline_str = " ".join(mgmt)
            lines.append(f"3. Clinical guidelines: {guideline_str}")
        else:
            # Conservative next-step guidance without naming or implying a specific guideline
            lines.append("3. Recommended clinical next steps:")
            lines.append("   - Compare with prior ECG if available to evaluate temporal change.")
            lines.append("   - Correlate with symptoms and patient history.")
            lines.append("   - Review QRS morphology and measurements on the original waveform.")
            
        lines.append("\nImportant: Diagnostic decisions and management steps (such as hospitalization, medication adjustments, or interventions) cannot be determined from the ECG alone and require clinician review.")
    return "\n".join(lines)

def render_diagnostic_fallback(state: dict, ontology_mapping: dict[str, str]) -> str:
    bridge = state.get("bridge", {})
    decision = state.get("final_diagnostic_decision", bridge)
    supported_labels = decision.get("supported_labels", [])
    lines = []
    expanded_supported = []
    for label in supported_labels:
        label_upper = label.upper()
        if label_upper in ontology_mapping:
            expanded_supported.append(f"{ontology_mapping[label_upper]} ({label_upper})")
        else:
            expanded_supported.append(label_upper)
    if expanded_supported:
        lines.append(f"Supported findings: {', '.join(expanded_supported)}.")
    else:
        lines.append("Supported findings: None.")
    partial_lines = []
    uncertain_findings = decision.get("partially_supported_evidence", []) or bridge.get("uncertain_findings", [])
    for uf in uncertain_findings:
        label = uf.get("label")
        status = uf.get("status") or "partially supported"
        label_upper = label.upper() if label else ""
        if label_upper in ontology_mapping:
            partial_lines.append(f"{ontology_mapping[label_upper]} ({label_upper}) [{status}]")
        elif label_upper:
            partial_lines.append(f"{label_upper} [{status}]")
    if partial_lines:
        lines.append(f"Partially supported findings: {', '.join(partial_lines)}.")
    lines.append("\nSupporting Evidence:")
    findings = decision.get("recording_findings", [])
    for f in findings:
        label = f.get("label")
        prob = f.get("maximum_probability")
        prev = f.get("diagnostic_prevalence")
        windows = f.get("diagnostic_window_indices") or []
        label_upper = label.upper() if label else ""
        name = ontology_mapping.get(label_upper, label_upper)
        win_str = f" in windows {', '.join(map(str, windows))}" if windows else ""
        if prob is not None and prev is not None:
            lines.append(f"- {name} ({label_upper}) has a maximum probability of {prob:.4f} and prevalence of {prev:.4f}{win_str}.")
    conf = decision.get("confidence") or "moderate"
    if isinstance(conf, dict):
        conf_level = str(conf.get("confidence_level") or conf.get("final_fused_confidence") or "moderate")
    else:
        conf_level = str(conf)
    lines.append(f"\nFinal Fused Confidence: {conf_level.upper()}")
    limitations = decision.get("limitations", [])
    if limitations:
        lines.append(f"Reason for confidence rating: {'; '.join(limitations)}")
    else:
        lines.append("Confidence is based on classifier agreement and statistical measurements.")
    episodes = state.get("temporal_summary", {}).get("abnormal_windows", [])
    if episodes:
        episode_strs = [f"Window {ep.get('representative_window_index', ep.get('window_index', (ep.get('window_indices') or [0])[0]))} ({ep.get('start_seconds', 0):.1f}s to {ep.get('end_seconds', 0):.1f}s)" for ep in episodes]
        lines.append(f"\nAbnormal Episodes Timing: {', '.join(episode_strs)}.")
    contradictions = decision.get("contradictions", [])
    if contradictions:
        lines.append(f"Material conflicts: {', '.join(map(str, contradictions))}.")
    return "\n".join(lines)

def generate_deterministic_grounded_fallback(question: str, state: dict, ontology_mapping: dict[str, str]) -> str:
    import re
    q = question.lower()
    if any(term in q for term in ("retrieved", "similar case", "neighbor", "top case", "similar ecg", "retrievel")):
        return render_retrieval_fallback(state, ontology_mapping)
    if any(term in q for term in ("symptom", "pain", "dyspnea", "syncope", "presentation")):
        bridge = state.get("bridge", {})
        decision = state.get("final_diagnostic_decision", bridge)
        supported_labels = decision.get("supported_labels", [])
        findings_str = ", ".join(ontology_mapping.get(label.upper(), label.upper()) for label in supported_labels)
        lines = [
            "Patient symptoms were not provided.",
            f"In general, people with {findings_str or 'these ECG findings'} may experience a range of symptoms depending on their clinical status:",
        ]
        has_afib = any(l in ("AFIB", "AF", "AFLT") for l in supported_labels)
        has_mi = any("MI" in l or l in ("IMI", "AMI", "ASMI", "ALMI", "ILMI") for l in supported_labels)
        has_block = any("AVB" in l or l in ("1AVB", "2AVB", "3AVB", "LAFB", "LPFB", "IRBBB", "CRBBB", "CLBBB") for l in supported_labels)
        if has_afib:
            lines.append("- Symptoms: Palpitations, chest pain, fatigue, lightheadedness, or shortness of breath. The finding is commonly asymptomatic.")
            lines.append("- Urgency: Rapid ventricular response or hemodynamic compromise warrants urgent assessment.")
        elif has_mi:
            lines.append("- Symptoms: Substernal chest pain, pressure, radiating to left arm/jaw, diaphoresis, and dyspnea. Myocardial infarction can be asymptomatic, especially in elderly or diabetic patients.")
            lines.append("- Urgency: Acute chest pain or shortness of breath requires immediate emergency clinical evaluation.")
        elif has_block:
            lines.append("- Symptoms: Dizziness, syncope, fatigue, or exercise intolerance. Conduction blocks are frequently asymptomatic.")
            lines.append("- Urgency: High-degree block (such as third-degree AV block) with bradycardia or syncope warrants urgent pacing evaluation.")
        else:
            lines.append("- Symptoms: Findings are frequently asymptomatic. Associated symptoms depend on any underlying structural heart disease.")
            lines.append("- Urgency: Symptoms such as syncope, severe dyspnea, or chest pain require prompt clinical evaluation.")
        return "\n".join(lines)
    if any(term in q for term in ("abbreviation", "mean", "stand for", "expansion", "definition")):
        found_abbrev = None
        for abbrev in ontology_mapping:
            if abbrev.lower() in q:
                found_abbrev = abbrev
                break
        if found_abbrev:
            return f"{ontology_mapping[found_abbrev]} ({found_abbrev})"
        return "No specific ECG abbreviation was identified in the question."
    if any(term in q for term in ("evidence", "supports", "provenance")):
        return render_evidence_fallback(state, ontology_mapping)
    if any(term in q for term in ("next step", "management", "treatment", "stent", "hospital")):
        return render_next_steps_fallback(state, ontology_mapping)
    return render_diagnostic_fallback(state, ontology_mapping)

def resolve_contextual_question(q: str, chat_state: dict) -> tuple[str, str, str]:
    import re
    resolved_q = q
    q_lower = q.lower().strip()
    last_type = chat_state.get("last_question_type", "")
    last_sub = chat_state.get("last_subject", "")
    last_labels = chat_state.get("last_finding_labels", [])
    last_retrieved = chat_state.get("last_retrieved_cases", [])
    if q_lower in ("which segment", "which segment?", "which window", "which window?", "which window and start time?"):
        if last_type == "retrieval_explanation" or last_retrieved:
            resolved_q = "For each of the retrieved cases, which query window index and start time produced the match?"
            return resolved_q, "segment_followup", "retrieved_cases"
        elif last_labels:
            resolved_q = f"Which segment and window supports the finding {', '.join(last_labels)}?"
            return resolved_q, "segment_followup", "finding_labels"
    second_case_match = re.search(r"\b(?:second|2nd)\s+(?:one|case|match|neighbor|neighbour|ecg)\b", q_lower)
    if second_case_match and last_retrieved:
        if len(last_retrieved) >= 2:
            case = last_retrieved[1]
            case_id = case.get("retrieved_ecg_id") or case.get("ecg_id") or case.get("faiss_row") or "Unknown"
            case_win = case.get("matched_from_query_window_index")
            case_start = case.get("matched_from_query_start_seconds")
            case_codes = case.get("reference_scp_codes") or []
            case_codes_str = ", ".join(case_codes)
            if "segment" in q_lower or "window" in q_lower or "produced" in q_lower or "start time" in q_lower or "when" in q_lower:
                resolved_q = f"For the second retrieved case (ID: {case_id}), which query window index and start time produced the match? (Window: {case_win}, Start Time: {case_start}s)"
                return resolved_q, "retrieved_case_followup", f"case_{case_id}"
            elif "finding" in q_lower or "share" in q_lower or "labels" in q_lower or "codes" in q_lower or "what did" in q_lower:
                resolved_q = f"What findings/SCP labels did the second retrieved case (ID: {case_id}, labels: {case_codes_str}) share with the patient's ECG?"
                return resolved_q, "retrieved_case_followup", f"case_{case_id}"
    if ("that case" in q_lower or "this case" in q_lower or "it" in q_lower.split()) and last_sub.startswith("case_"):
        case_id = last_sub.split("case_", 1)[-1]
        case = next((c for c in last_retrieved if str(c.get("retrieved_ecg_id") or c.get("ecg_id") or c.get("faiss_row")) == case_id), None)
        if case:
            case_codes = case.get("reference_scp_codes") or []
            case_codes_str = ", ".join(case_codes)
            if "finding" in q_lower or "share" in q_lower or "labels" in q_lower or "what did" in q_lower:
                resolved_q = f"What findings/SCP labels did the retrieved case (ID: {case_id}, labels: {case_codes_str}) share with the patient's ECG?"
                return resolved_q, "retrieved_case_followup", last_sub
            elif "symptom" in q_lower:
                resolved_q = f"What symptoms are associated with the findings/labels ({case_codes_str}) of that retrieved case?"
                return resolved_q, "symptoms", f"labels_{case_codes_str}"
    has_pronoun = any(word in q_lower.split() for word in ("it", "its", "that", "this", "they", "them", "these", "those"))
    if (has_pronoun or "that finding" in q_lower) and last_labels:
        labels_str = ", ".join(last_labels)
        if "symptom" in q_lower:
            resolved_q = f"What symptoms are associated with {labels_str}?"
            return resolved_q, "symptoms", f"labels_{labels_str}"
        elif "evidence" in q_lower or "citation" in q_lower or "literature" in q_lower or "source" in q_lower:
            resolved_q = f"Show the validated clinical evidence and citations supporting {labels_str}."
            return resolved_q, "evidence", f"labels_{labels_str}"
        elif "uncertain" in q_lower or "confidence" in q_lower or "why" in q_lower:
            resolved_q = f"Why is the finding {labels_str} uncertain, and what limits its confidence?"
            return resolved_q, "confidence_reasoning", f"labels_{labels_str}"
    if q_lower in ("why", "why?", "based on which evidence?", "based on what evidence?", "why does that matter?", "why does that matter"):
        if last_type in ("treatment", "medication", "next_step", "investigations"):
            resolved_q = f"Why is that recommended management step or investigation ({last_sub}) indicated, and what clinical evidence supports it?"
            return resolved_q, last_type, last_sub
        elif last_type == "retrieved_case_followup":
            resolved_q = f"Why does the similarity or findings of that retrieved case ({last_sub}) matter clinically?"
            return resolved_q, last_type, last_sub
        elif last_labels:
            resolved_q = f"Why was the finding {', '.join(last_labels)} supported, and what is the underlying evidence?"
            return resolved_q, "diagnosis_reasoning", f"labels_{', '.join(last_labels)}"
    question_type = "diagnosis"
    subject = ""
    finding_labels = []
    ont_mapping = get_abbreviation_mapping()
    for abbrev, term in ont_mapping.items():
        if abbrev.lower() in q_lower or term.lower() in q_lower:
            finding_labels.append(abbrev)
    if finding_labels:
        subject = f"labels_{','.join(finding_labels)}"
    elif last_labels:
        finding_labels = last_labels
        subject = last_sub
    else:
        finding_labels = chat_state.get("current_case_labels", [])
        primary_lbl = chat_state.get("current_case_primary_label")
        if not finding_labels and primary_lbl:
            finding_labels = [primary_lbl]
        if finding_labels:
            subject = f"labels_{','.join(finding_labels)}"

    if finding_labels and q_lower:
        labels_str = ", ".join(finding_labels)
        if any(w in q_lower for w in ("symptom", "presentation", "feel", "sign")):
            resolved_q = f"What symptoms are associated with {labels_str}?"
            question_type = "symptoms"
        elif any(w in q_lower for w in ("cause", "etiology", "why does it happen", "risk factor")):
            resolved_q = f"What causes {labels_str}?"
            question_type = "differential_diagnosis"
        elif any(w in q_lower for w in ("dangerous", "prognosis", "risk", "complication")):
            resolved_q = f"Is {labels_str} dangerous, and what is its prognosis?"
            question_type = "risk_red_flags"
        elif any(w in q_lower for w in ("treatment", "therapy", "manage", "treat")):
            resolved_q = f"How is {labels_str} treated?"
            question_type = "next_step"
        elif any(w in q_lower for w in ("mean", "definition", "stand for", "abbreviation")):
            resolved_q = f"What does {labels_str} mean?"
            question_type = "evidence"
        else:
            if any(term in q_lower for term in ("retrieved", "similar case", "neighbor", "top case", "similar ecg", "retrievel")):
                question_type = "retrieval_explanation"
                subject = "retrieved_cases"
            elif any(term in q_lower for term in ("confidence", "uncertain", "why low", "reliability")):
                question_type = "confidence_reasoning"
    else:
        if any(term in q_lower for term in ("retrieved", "similar case", "neighbor", "top case", "similar ecg", "retrievel")):
            question_type = "retrieval_explanation"
            subject = "retrieved_cases"
        elif any(term in q_lower for term in ("symptom", "pain", "dyspnea", "syncope", "presentation")):
            question_type = "symptoms"
        elif any(term in q_lower for term in ("next step", "management", "treatment", "stent", "hospital", "test", "investigation")):
            question_type = "next_step"
        elif any(term in q_lower for term in ("evidence", "supports", "provenance", "citation", "source", "literature")):
            question_type = "evidence"
        elif any(term in q_lower for term in ("confidence", "uncertain", "why low", "reliability")):
            question_type = "confidence_reasoning"
            
    return resolved_q, question_type, subject or last_sub

def update_conversation_summary(chat_state: dict) -> None:
    history = chat_state.get("turn_history", [])
    if len(history) <= 4:
        return
    oldest_turn = history.pop(0)
    q = oldest_turn["question"]
    a = oldest_turn["answer"]
    summary_line = f"Previously asked: '{q}'. Answer summarized: '{a[:150]}...'"
    current_summary = chat_state.get("conversation_summary", "")
    if current_summary:
        chat_state["conversation_summary"] = current_summary + "\n" + summary_line
    else:
        chat_state["conversation_summary"] = summary_line
    if len(chat_state["conversation_summary"]) > 1000:
        chat_state["conversation_summary"] = "..." + chat_state["conversation_summary"][-900:]


def asset_status(*, include_llm: bool = False) -> dict[str, Any]:
    selected_assets = dict(ACTIVE_REQUIRED_ASSETS)
    if include_llm:
        selected_assets["gemma_base_gguf"] = TRACE_GGUF_MODEL
    assets = {name: {"path": str(path), "present": path.is_file()} for name, path in selected_assets.items()}
    return {"ready": all(item["present"] for item in assets.values()), "assets": assets}


class JetsonECGPipeline:
    """Load selected assets once and run 10-second, 2-minute, or 5-minute ECG inference."""

    def __init__(
        self,
        *,
        device: str = "auto",
        llm_mode: str = "real",
        llm_backend: str = "llama_cpp",
        enable_experimental_holter: bool = False,
    ) -> None:
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if device not in {"cpu", "cuda"} and not device.startswith("cuda:"):
            raise ValueError("device must be auto, cpu, cuda, or cuda:N")
        if llm_mode not in {"real", "disabled"}:
            raise ValueError("llm_mode must be real or disabled")
        if llm_mode == "real" and not device.startswith("cuda"):
            raise RuntimeError(
                "Real Gemma inference is GPU-only in this deployment. "
                "CUDA was not detected; CPU fallback is intentionally disabled."
            )

        os.environ["TRACE_DEVICE"] = device
        os.environ["TRACE_LLM_MODE"] = llm_mode
        os.environ["TRACE_LLM_BACKEND"] = llm_backend
        os.environ["TRACE_LLM_REQUIRE_GPU"] = "1"
        os.environ["TRACE_LLM_N_GPU_LAYERS"] = "-1"
        os.environ["TRACE_LLM_MAIN_GPU"] = "0"
        os.environ.setdefault("TRACE_THREADS", "2")
        os.environ["TRACE_REQUIRE_ADAPTER"] = "0"
        os.environ["TRACE_DISABLE_LORA"] = "1"
        os.environ["TRACE_ENABLE_EXPERIMENTAL_HOLTER"] = "1" if enable_experimental_holter else "0"

        state = asset_status(include_llm=llm_mode == "real")
        if not state["ready"]:
            missing = [f"{name}={item['path']}" for name, item in state["assets"].items() if not item["present"]]
            raise FileNotFoundError("Missing active deployment assets: " + "; ".join(missing))

        from backend.diagnosis_model import DiagnosisModel
        from backend.recording_analysis import RecordingAnalyzer

        self.device = device
        self.llm_mode = llm_mode
        self.model = DiagnosisModel()
        self.recording_analyzer = RecordingAnalyzer(self.model)
        # RecordingAnalyzer's per-window attribution uses the validated
        # classifier thresholds. Bridge V4 deliberately owns fusion thresholds
        # under ``thresholds`` and does not expose this classifier-specific map.
        # Attach it only to the recording adapter; Bridge V4 itself is unchanged.
        classifier_thresholds = json.loads(SCP_THRESHOLDS_PATH.read_text(encoding="utf-8"))
        if not isinstance(classifier_thresholds, dict) or not classifier_thresholds:
            raise RuntimeError("Validated SCP threshold manifest is empty or invalid")
        self.recording_analyzer.bridge.scp_thresholds = {
            str(label): float(value) for label, value in classifier_thresholds.items()
        }

    @staticmethod
    def _recording_id(signal: np.ndarray, sampling_rate_hz: int) -> str:
        digest = hashlib.sha256(np.ascontiguousarray(signal, dtype=np.float32).tobytes())
        digest.update(str(int(sampling_rate_hz)).encode("ascii"))
        return "ecg_" + digest.hexdigest()[:24]

    def run(
        self,
        signal: Any,
        *,
        sampling_rate_hz: int,
        mode: str,
        lead_names: Optional[Sequence[str]] = None,
        manual_window_indices: Optional[Iterable[int]] = None,
        top_k: int = 5,
        question: str = "What is the diagnostic conclusion and the evidence supporting it?",
        include_llm: bool = True,
        patient_id: Optional[str] = None,
        save_feedback_snapshot: bool = False,
    ) -> dict[str, Any]:
        if mode not in {"10s", "2min", "5min"}:
            raise ValueError("mode must be 10s, 2min, or 5min")
        if not 1 <= int(top_k) <= 20:
            raise ValueError("top_k must be between 1 and 20")
        ecg = np.asarray(signal, dtype=np.float32)
        recording_id = self._recording_id(ecg, sampling_rate_hz)
        started = time.perf_counter()

        if mode == "10s" and not manual_window_indices:
            result = self.model.predict(
                ecg,
                metadata={
                    "sampling_rate_hz": int(sampling_rate_hz),
                    "lead_names": list(lead_names) if lead_names else None,
                    "case_id": recording_id,
                    "patient_id": patient_id,
                },
                include_retrieval=True,
                include_knowledge=True,
                include_explanation=False,
                question=question,
                conversation_id=recording_id,
                llm_mode="disabled",
                top_k=int(top_k),
            )
            result["result_type"] = "window"
            result["recording_mode"] = mode
            if include_llm:
                result["explanation"] = self._explain_recording(result, question)
            else:
                result["explanation"] = {"status": "disabled", "text": "[Explanation disabled]"}
        else:
            # Long recordings use coarse-to-fine selection before expensive branches.
            result = self.recording_analyzer.analyze(
                ecg,
                int(sampling_rate_hz),
                mode,
                manual_window_indices,
                lead_names,
                int(top_k),
            )
            if include_llm:
                result["explanation"] = self._explain_recording(result, question)
            else:
                result["explanation"] = {"status": "disabled", "text": "[Explanation disabled]"}

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        result["deployment_runtime"] = {
            "entry_point": "runtime.jetson_runtime.JetsonECGPipeline",
            "device": self.device,
            "bridge_authority": "V4",
            "retriever": "V7",
            "llm_advisory_only": True,
            "llm_model_variant": "base",
            "elapsed_ms": elapsed_ms,
            "recording_id": recording_id,
        }
        result = make_json_safe(result)
        if save_feedback_snapshot:
            result["feedback_snapshot"] = self._save_snapshot(result, recording_id, patient_id, mode, elapsed_ms)
        return result

    def answer_question(
        self,
        result: dict[str, Any],
        question: str,
        conversation: Optional[Sequence[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """Answer another question without rerunning ECG inference or changing V4."""
        if self.llm_mode not in ("real", "disabled"):
            raise RuntimeError("Interactive questions require llm_mode='real' or 'disabled'.")

        q_strip = question.strip()
        if q_strip.lower() == "/clear":
            if hasattr(self, "chat_state"):
                self.chat_state = None
            return {
                "text": "Conversation context cleared.",
                "status": "cleared",
                "backend": "command_intercept",
                "bridge_decision_modified": False,
            }
        if q_strip.lower() == "/context":
            if not hasattr(self, "chat_state") or not self.chat_state:
                return {
                    "text": "No active conversation context.",
                    "status": "context",
                    "backend": "command_intercept",
                    "bridge_decision_modified": False,
                }
            context_focus = {
                "last_subject": self.chat_state.get("last_subject"),
                "last_finding_labels": self.chat_state.get("last_finding_labels"),
                "last_question_type": self.chat_state.get("last_question_type"),
                "turn_history_length": len(self.chat_state.get("turn_history", []))
            }
            return {
                "text": f"Current Conversational Focus: {json.dumps(context_focus, default=str)}",
                "status": "context",
                "backend": "command_intercept",
                "bridge_decision_modified": False,
            }
        if q_strip.lower() == "/decision":
            decision = result.get("final_diagnostic_decision", result.get("recording_bridge", result.get("bridge", {})))
            return {
                "text": json.dumps(decision, ensure_ascii=False, indent=2, default=str),
                "status": "decision",
                "backend": "command_intercept",
                "bridge_decision_modified": False,
            }

        return self._explain_recording(result, question, conversation=conversation)

    def _explain_recording(
        self,
        result: dict[str, Any],
        question: str,
        conversation: Optional[Sequence[dict[str, str]]] = None,
    ) -> dict[str, Any]:
        """Generate advisory prose from immutable Bridge V4 output; never modify it."""
        from prompt_builder.system_runtime import compile_question_tool_context, system_prompt_metadata
        from utils.llm import get_llm_backend
        from utils.question_router import normalize_spelling_typos
        import hashlib
        import json
        import re

        bridge = result.get("recording_bridge", result.get("bridge", {}))
        decision = result.get("final_diagnostic_decision", {})

        # Calculate recording hash & decision hash
        rec_id = result.get("deployment_runtime", {}).get("recording_id") or result.get("recording_id") or "unknown_ecg"
        rec_hash = hashlib.sha256(rec_id.encode("utf-8")).hexdigest()
        decision_str = json.dumps(decision, sort_keys=True, default=str)
        dec_hash = hashlib.sha256(decision_str.encode("utf-8")).hexdigest()

        # Initialize or clear context
        if (not hasattr(self, "chat_state") or 
            not self.chat_state or 
            self.chat_state.get("recording_hash") != rec_hash or 
            self.chat_state.get("decision_hash") != dec_hash):
            self.chat_state = init_chat_state(rec_id, rec_hash, dec_hash)
            # Store current case findings in chat_state for first-turn context resolution
            self.chat_state["current_case_labels"] = decision.get("supported_labels", []) or [bridge.get("primary_label")]
            self.chat_state["current_case_primary_label"] = decision.get("primary_label") or bridge.get("primary_label")

        # 1. Spelling-tolerant correction (stores original and normalized question)
        norm_res = normalize_spelling_typos(question)
        normalized_q = norm_res["normalized_question"]
        corrections = norm_res.get("corrections", [])
        conf = norm_res.get("correction_confidence", 1.0)

        # Fresh Session check with no active condition
        active_lbls = self.chat_state.get("current_case_labels", [])
        primary_lbl = self.chat_state.get("current_case_primary_label")
        if not active_lbls and primary_lbl:
            active_lbls = [primary_lbl]
        active_lbls = [l for l in active_lbls if l and l != "UNKNOWN"]
        
        is_generic_query = any(w in question.lower() for w in ("symptom", "cause", "dangerous", "prognosis", "treatment", "mean", "explain"))
        if not active_lbls and is_generic_query:
            return {
                "status": "deterministic_fallback",
                "text": "To provide accurate clinical information, the specific ECG condition or finding must first be known. Please specify a condition or load an ECG recording.",
                "backend": "fallback_assembler",
                "review_authority": "advisory_only",
                "bridge_decision_modified": False,
                "original_question": question,
                "normalized_question": normalized_q,
            }

        # Handle clarification if spelling normalization detected ambiguity
        if normalized_q.startswith("CLARIFICATION_REQUIRED:"):
            options = norm_res.get("clarification_options", [])
            clarification_text = f"I detected an ambiguous clinical term. Did you mean one of these: {', '.join(options)}?"
            return {
                "status": "clarification_request",
                "text": clarification_text,
                "backend": "typo_resolver",
                "review_authority": "advisory_only",
                "bridge_decision_modified": False,
                "original_question": question,
                "normalized_question": normalized_q,
            }

        # 2. Context resolution (pronouns, ellipsis, and context shifts)
        resolved_q, q_type, q_subj = resolve_contextual_question(normalized_q, self.chat_state)

        # Intercept ECG retrieval case queries (FAISS neighbors)
        is_retrieval_query = any(w in resolved_q.lower() for w in (
            "retrieved case", "retrieval case", "similar case", "similar ecg", "neighbor", "faiss", "top 5 retrieval", "top five retrieval"
        )) or (q_type == "retrieval_explanation")
        
        if is_retrieval_query:
            from prompt_builder.system_runtime import ReadOnlyECGTools
            tools_obj = ReadOnlyECGTools({
                "retrieval": result.get("retrieval", {}),
                "statistics": result.get("recording_statistics", {}),
                "windows": {str(item.get("window_index")): item for item in result.get("windows", [])}
            })
            retrieval_ev = tools_obj.get_retrieval_evidence(top_k=5)
            matches = retrieval_ev.get("matches", [])
            
            if not matches:
                rendered = "No similar retrieved cases found in the database."
            else:
                ont_map = get_abbreviation_mapping()
                lines = [f"Top {len(matches)} similar retrieved cases (from ECG FAISS neighbor retrieval):"]
                for i, match in enumerate(matches, 1):
                    ecg_id = match.get("retrieved_ecg_id") or match.get("faiss_row") or "Unknown"
                    similarity = match.get("similarity")
                    sim_str = f"{similarity:.4f}" if isinstance(similarity, (int, float)) else str(similarity)
                    codes = match.get("reference_scp_codes") or []
                    code_strs = [ont_map.get(c.upper(), c.upper()) + f" ({c.upper()})" if c.upper() in ont_map else c.upper() for c in codes]
                    codes_str = ", ".join(code_strs) if code_strs else "None"
                    
                    line = f"Rank {i}: Case/Row ID: {ecg_id} | Similarity: {sim_str} | Labels: {codes_str}"
                    win_idx = match.get("matched_from_query_window_index")
                    if win_idx is not None:
                        line += f" | Query Window Index: {win_idx}"
                    lines.append(line)
                rendered = "\n".join(lines)
            
            self.chat_state["last_user_question"] = question
            self.chat_state["last_normalized_question"] = normalized_q
            self.chat_state["last_preferred_sections"] = []
            self.chat_state["last_question_type"] = "retrieval_explanation"
            self.chat_state["last_subject"] = "retrieved_cases"
            self.chat_state["turn_history"].append({"question": question, "answer": rendered})
            update_conversation_summary(self.chat_state)
            
            return {
                "status": "deterministic_fallback",
                "text": rendered,
                "backend": "fallback_assembler",
                "review_authority": "advisory_only",
                "bridge_decision_modified": False,
                "original_question": question,
                "normalized_question": resolved_q,
            }

        # Intercept confidence queries
        is_confidence_query = any(w in resolved_q.lower() for w in (
            "confidence", "why low", "why is your confidence", "what is your confidence", "uncertainty in decision"
        )) or (q_type == "confidence_reasoning")
        
        if is_confidence_query:
            conf_data = bridge.get("confidence", {})
            fused_conf = conf_data.get("final_fused_confidence")
            if fused_conf is None:
                fused_conf = decision.get("confidence")
            
            if isinstance(fused_conf, (int, float)):
                pct_str = f"{int(fused_conf * 100)}%" if fused_conf <= 1.0 else f"{int(fused_conf)}%"
            else:
                pct_str = str(fused_conf) if fused_conf else "N/A"
                
            conf_level = conf_data.get("confidence_level") or decision.get("confidence_level") or "N/A"
            limitations = conf_data.get("limitations") or decision.get("limitations") or []
            drivers = conf_data.get("confidence_drivers") or decision.get("confidence_drivers") or []
            
            friendly_limitations = []
            for lim in limitations:
                lim_lower = lim.lower()
                if "independent family head strongly disagrees" in lim_lower:
                    friendly_limitations.append("The model's independent diagnostic signals are not fully consistent with one another, which limits confidence.")
                elif "retrieved examples" in lim_lower and "unverified" in lim_lower:
                    friendly_limitations.append("ECG retrieval matches could not be verified and were excluded from scoring.")
                else:
                    friendly_limitations.append(lim)
            
            ont_map = get_abbreviation_mapping()
            primary_lbl = decision.get("primary_label") or bridge.get("primary_label") or "UNKNOWN"
            primary_name = ont_map.get(primary_lbl.upper(), primary_lbl)
            if primary_name != primary_lbl:
                primary_disp = f"{primary_name} ({primary_lbl})"
            else:
                primary_disp = primary_lbl
                    
            lines = [
                f"Primary finding: {primary_disp}",
                f"Confidence: {pct_str} — {conf_level.title()}"
            ]
            
            if drivers:
                lines.append(f"Drivers: {', '.join(drivers)}")
            if friendly_limitations:
                lines.append(f"Limitations: {' '.join(friendly_limitations)}")
            else:
                lines.append("Limitations: None identified by the bridge rules.")
                
            rendered = "\n".join(lines)
            
            self.chat_state["last_user_question"] = question
            self.chat_state["last_normalized_question"] = normalized_q
            self.chat_state["last_preferred_sections"] = []
            self.chat_state["last_question_type"] = "confidence_reasoning"
            self.chat_state["last_subject"] = "confidence"
            self.chat_state["turn_history"].append({"question": question, "answer": rendered})
            update_conversation_summary(self.chat_state)
            
            return {
                "status": "deterministic_fallback",
                "text": rendered,
                "backend": "fallback_assembler",
                "review_authority": "advisory_only",
                "bridge_decision_modified": False,
                "original_question": question,
                "normalized_question": resolved_q,
            }

        # Retrieve knowledge using resolved query
        knowledge = result.get("knowledge", {})
        q_lower = resolved_q.lower()
        preferred_sections = []
        for terms, section in (
            (("symptom", "feel", "presentation"), "symptoms"),
            (("evidence", "literature", "source", "citation", "criteria"), "ecg_diagnostic_criteria"),
            (("differential", "underlying", "cause", "pathology"), "differential_diagnosis"),
            (("risk", "danger", "prognosis", "emergency", "red flag"), "risk_red_flags"),
            (("next step", "management", "treatment", "test", "hospital", "stent"), "management_guidance"),
        ):
            if any(term in q_lower for term in terms):
                preferred_sections.append(section)

        # Force fresh retrieval when normalized query or preferred sections change
        last_norm_q = self.chat_state.get("last_normalized_question", "")
        last_preferred = self.chat_state.get("last_preferred_sections", [])
        
        query_or_intent_changed = (normalized_q != last_norm_q) or (preferred_sections != last_preferred)

        question_knowledge = knowledge
        if query_or_intent_changed or not self.chat_state.get("last_citations"):
            primary_label = str(decision.get("primary_label") or "UNKNOWN")
            try:
                question_knowledge = self.model.knowledge_retriever.retrieve(
                    question=resolved_q,
                    bridge_result={
                        "primary_label": primary_label,
                        "decision": decision.get("status", "supported"),
                        "contradictions": decision.get("contradictions", []),
                    },
                    contradictions=decision.get("contradictions", []),
                    top_k=5,
                    preferred_sections=preferred_sections,
                )
            except Exception:
                question_knowledge = knowledge
        else:
            # Preserve cited evidence/chunks across follow-ups
            question_knowledge = {
                "all_chunks": self.chat_state.get("last_citations", []),
                "permitted_citations": self.chat_state.get("last_citations", [])
            }

        # Build turn history representation for Gemma dialog history
        dialogue = []
        for turn in self.chat_state["turn_history"][-4:]:
            dialogue.append({
                "question": str(turn.get("question", ""))[:1000],
                "answer": str(turn.get("answer", ""))[:2000],
            })

        state = {
            "recording": {
                "mode": result.get("recording_mode"),
                "duration_seconds": result.get("duration_seconds"),
            },
            "bridge": bridge,
            "signal_quality": result.get("signal_quality", {}),
            "statistics": result.get("recording_statistics", {}),
            "retrieval": result.get("retrieval", {}),
            "temporal_summary": {
                "abnormal_windows": result.get("episodes", []),
                "stable_reference_windows": result.get("stable_state", {}).get("representative_window_indices", []),
            },
            "windows": {str(item.get("window_index")): item for item in result.get("windows", [])},
            "knowledge_chunks": question_knowledge.get("all_chunks", knowledge.get("all_chunks", [])),
            "conversation_history": dialogue,
            "pipeline_status": {
                "window_selection": "complete", "classifier": "complete", "retrieval": "complete",
                "statistics": "complete", "bridge": "complete", "knowledge": "complete",
            },
            "versions": {"bridge": "V4", "retriever": "V7"},
            "experimental_holter": bool(result.get("selector", {}).get("holter_diagnostic_authority", False)),
        }

        # Compile question-resolved tool outputs
        tools = compile_question_tool_context(resolved_q, state)

        compact_decision = {
            key: decision.get(key) for key in (
                "status", "primary_label", "supported_labels",
                "partially_supported_labels", "selected_episode_count",
                "confidence", "contradictions", "requires_clinician_review", "scope",
            ) if decision.get(key) is not None
        }
        compact_decision["recording_findings"] = [
            {key: finding.get(key) for key in (
                "label", "family", "maximum_probability",
                "diagnostic_window_indices", "stable_reference_window_indices",
                "diagnostic_prevalence", "temporal_pattern",
            ) if finding.get(key) is not None}
            for finding in decision.get("recording_findings", [])[:8]
            if isinstance(finding, dict)
        ]
        partial_labels = set(compact_decision.get("partially_supported_labels", []))
        compact_decision["partially_supported_evidence"] = [
            {key: finding.get(key) for key in (
                "label", "family", "status", "fusion_score",
                "independent_sources", "evidence", "supporting_windows",
                "limitations", "explanation",
            ) if finding.get(key) is not None}
            for finding in bridge.get("uncertain_findings", [])
            if isinstance(finding, dict) and finding.get("label") in partial_labels
        ][:8]

        immutable = {
            "final_diagnostic_decision": compact_decision,
            "clinical_references": question_knowledge.get("permitted_citations", result.get("clinical_references", []))[:5],
            "question_knowledge": question_knowledge.get("all_chunks", [])[:5],
            "tool_results": tools,
        }

        clinical_literature_types = {
            "guideline", "textbook", "scientific_statement",
            "systematic_review", "clinical_review",
            "clinical_practice_guideline", "textbook_chapter", "clinical_education_reference"
        }
        clinical_chunks = [
            chunk for chunk in immutable["question_knowledge"]
            if str(chunk.get("evidence_type", "")).lower() in clinical_literature_types
        ]

        source_scope = {
            "activated_clinical_literature_available": bool(clinical_chunks),
            "activated_clinical_literature_citations": [
                chunk.get("citation_id") for chunk in clinical_chunks if chunk.get("citation_id")
            ],
            "ontology_is_label_definition_not_clinical_proof": True,
            "governance_is_safety_policy_not_clinical_literature": True,
        }

        label_context = []
        seen_labels = set()
        for finding in compact_decision.get("recording_findings", []) + compact_decision.get("partially_supported_evidence", []):
            label = finding.get("label") if isinstance(finding, dict) else None
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            label_context.append({
                "label": label,
                "family": finding.get("family"),
                "status": finding.get("status") or (
                    "supported" if label in compact_decision.get("supported_labels", []) else "partial"
                ),
            })

        anchor = {key: compact_decision.get(key) for key in (
            "status", "primary_label", "supported_labels", "partially_supported_labels",
            "confidence", "requires_clinician_review", "scope",
        ) if compact_decision.get(key) is not None}

        # Build LLM prompt using resolved question and including context details
        prompt = (
            "[AUTHORITATIVE BRIDGE V4 ANCHOR]\n"
            + json.dumps(anchor, ensure_ascii=False, separators=(",", ":"), default=str)[:2200]
            + "\n[/AUTHORITATIVE BRIDGE V4 ANCHOR]"
            + "\n[USER QUESTION]\n" + str(resolved_q)[:1000] + "\n[/USER QUESTION]"
            + "\n[LABEL AND SOURCE SCOPE]\n"
            + json.dumps({"labels": label_context, "knowledge_scope": source_scope}, ensure_ascii=False, separators=(",", ":"), default=str)[:2200]
            + "\n[/LABEL AND SOURCE SCOPE]"
            + "\n[QUESTION-RESOLVED READ-ONLY EVIDENCE]\n"
            + json.dumps(tools, ensure_ascii=False, separators=(",", ":"), default=str)[:6500]
            + "\n[/QUESTION-RESOLVED READ-ONLY EVIDENCE]"
            + "\n[VALIDATED KNOWLEDGE FOR THIS QUESTION]\n"
            + json.dumps({
                "chunks": immutable["question_knowledge"],
                "permitted_citations": immutable["clinical_references"],
            }, ensure_ascii=False, separators=(",", ":"), default=str)[:5000]
            + "\n[/VALIDATED KNOWLEDGE FOR THIS QUESTION]"
            + "\n[ADDITIONAL IMMUTABLE CASE EVIDENCE]\n"
            + json.dumps(compact_decision, ensure_ascii=False, separators=(",", ":"), default=str)[:5000]
            + "\n[/ADDITIONAL IMMUTABLE CASE EVIDENCE]"
            + "\n[NON_AUTHORITATIVE DIALOGUE HISTORY]\n"
            + json.dumps(dialogue, ensure_ascii=False, separators=(",", ":"), default=str)[:3000]
            + "\n[/NON_AUTHORITATIVE DIALOGUE HISTORY]"
            + "\nAnswer the user's actual question directly and immediately as the first sentence. "
            + "Do not repeat the complete diagnosis or the final interpretation unless it is necessary to answer the question. "
            + "Keep your response focused on what is asked."
        )

        # Debug logging to verify Gemma prompt grounding of selected chunks
        debug_chunks = [c.get("chunk_id") or c.get("id") for c in immutable["question_knowledge"]]
        debug_sources = [c.get("source_id") for c in immutable["question_knowledge"]]
        logger.debug(f"[TRACE DEBUG] Grounding check: Selected KB Chunks: {debug_chunks}, Sources: {debug_sources}")

        backend = get_llm_backend(mode=self.llm_mode)
        generated_text = ""
        fallback_used = False

        # Clean tags helper to sanitize model output
        def clean_tags(text: str) -> str:
            for closing_tag in (
                "[/AUTHORITATIVE BRIDGE V4 DECISION]",
                "[/AUTHORITATIVE BRIDGE V4 ANCHOR]",
                "[/LABEL AND SOURCE SCOPE]",
                "[/SUPPORTING READ-ONLY EVIDENCE]",
                "[/QUESTION-RESOLVED READ-ONLY EVIDENCE]",
                "[/VALIDATED KNOWLEDGE FOR THIS QUESTION]",
                "[/ADDITIONAL IMMUTABLE CASE EVIDENCE]",
                "[/NON_AUTHORITATIVE DIALOGUE HISTORY]",
                "</user_question>",
            ):
                if closing_tag in text:
                    text = text.rsplit(closing_tag, 1)[-1].strip()
            return text

        try:
            generated = backend.generate(prompt, generation_config={"temperature": 0.2, "max_output_tokens": 700})
            generated_text = clean_tags(generated.text.strip())
            if getattr(generated, "backend", None) == "disabled" or self.llm_mode == "disabled":
                generated_text = ""
        except Exception:
            generated_text = ""

        # Gemma empty-answer retry and fallback logic
        if not generated_text:
            smaller_prompt = (
                "[AUTHORITATIVE BRIDGE V4 ANCHOR]\n"
                + json.dumps(anchor, ensure_ascii=False, separators=(",", ":"), default=str)[:1000]
                + "\n[/AUTHORITATIVE BRIDGE V4 ANCHOR]"
                + "\n[USER QUESTION]\n" + str(resolved_q)[:1000] + "\n[/USER QUESTION]"
                + "\nAnswer the question directly based on the evidence."
            )
            try:
                generated = backend.generate(smaller_prompt, generation_config={"temperature": 0.1, "max_output_tokens": 500})
                generated_text = clean_tags(generated.text.strip())
                if getattr(generated, "backend", None) == "disabled" or self.llm_mode == "disabled":
                    generated_text = ""
            except Exception:
                generated_text = ""

            if not generated_text:
                # Still empty: produce deterministic grounded fallback response
                ont_map = get_abbreviation_mapping()
                generated_text = generate_deterministic_grounded_fallback(resolved_q, state, ont_map)
                fallback_used = True

        primary_label = str(compact_decision.get("primary_label") or "UNKNOWN")
        supported_status = str(compact_decision.get("status") or "").lower() in {
            "supported", "multi_label_supported", "probable", "partially_supported"
        }
        false_unknown = any(phrase in generated_text.lower() for phrase in (
            "primary finding is currently unknown", "primary finding is unknown",
            "decision is indeterminate", "diagnosis is unknown",
            "primary label and family are both unknown",
            "no diagnostic information is available",
        ))
        guardrail_fallback = supported_status and primary_label != "UNKNOWN" and false_unknown
        if guardrail_fallback:
            supported = ", ".join(compact_decision.get("supported_labels", [])) or primary_label
            partial = ", ".join(compact_decision.get("partially_supported_labels", []))
            generated_text = f"Bridge V4 supports {supported} as the ECG finding."
            if partial:
                generated_text += f" Partially supported findings requiring review are {partial}."
            generated_text += " This is ECG decision support, not a complete clinical diagnosis, and clinician review remains required."

        # Replace internal component names with natural component names in clinician-facing prose
        generated_text = apply_natural_names(generated_text, question)
        # Strip private prompt delimiters if a local backend echoes them.
        generated_text = re.sub(r"\[/?ANSWER\]", "", generated_text, flags=re.IGNORECASE).strip()

        # Defensive cleanup of exact legacy boilerplate disclaimers (just in case they are generated by the model)
        legacy_disclaimers = [
            "Knowledge note: the activated KB has no condition-specific clinical guideline or textbook passage for this topic.",
            "Any general symptom or clinical-context discussion above is conservative Gemma medical education, not a fact about this patient and not evidence used by Bridge V4.",
            "This reflects the current state of the pipeline and does not represent a definitive clinical diagnosis."
        ]
        for disc in legacy_disclaimers:
            if disc in generated_text:
                generated_text = generated_text.replace(disc, "").strip()
            elif disc.lower() in generated_text.lower():
                start = generated_text.lower().find(disc.lower())
                generated_text = (generated_text[:start] + generated_text[start+len(disc):]).strip()

        is_absent = not source_scope["activated_clinical_literature_available"]

        # Check if the user explicitly requested evidence/sources/reasoning
        evidence_requested = any(phrase in q_lower for phrase in (
            "/decision", "show evidence", "why", "show sources", "what supports this", "explain your reasoning", "explain reasoning", "give reasoning"
        ))

        # Remove any free-generated sources/references/bibliography blocks from the model's text
        ref_patterns = [
            r"\bSources\s*:\s*.*",
            r"\bReferences\s*:\s*.*",
            r"\bBibliography\s*:\s*.*",
            r"\bCitations\s*:\s*.*"
        ]
        for pat in ref_patterns:
            generated_text = re.sub(pat, "", generated_text, flags=re.IGNORECASE | re.DOTALL).strip()

        # If citations are present and allowed, and evidence/sources are requested, replace/add formatted citations
        # Only label-addressed ontology passages are safe as diagnostic support.
        # Generic semantic matches must not be cited as evidence for another label.
        decision_labels = {
            str(v).upper() for v in (
                [primary_label]
                + list(compact_decision.get("supported_labels", []) or [])
                + list(compact_decision.get("partially_supported_labels", []) or [])
            ) if v
        }
        citations = []
        seen_citations = set()
        for citation in question_knowledge.get("permitted_citations", []):
            if not isinstance(citation, dict):
                continue
            citation_id = str(citation.get("id") or citation.get("citation_id") or "")
            exact_ontology = any(citation_id.upper().startswith(f"SCP-{label}-") for label in decision_labels)
            if not exact_ontology:
                continue
            dedupe_key = citation_id or (citation.get("source_title"), citation.get("section"), citation.get("page"))
            if dedupe_key in seen_citations:
                continue
            seen_citations.add(dedupe_key)
            citations.append(citation)
        citations = citations[:5]
        if citations and not is_absent and evidence_requested:
            sources_match = re.search(r"\bSources\s*:\s*", generated_text, flags=re.IGNORECASE)
            if sources_match:
                start_idx = sources_match.start()
                sources_str = format_sources(citations)
                generated_text = generated_text[:start_idx] + sources_str
            else:
                if "sources:" not in generated_text.lower() and "source:" not in generated_text.lower():
                    generated_text += "\n\n" + format_sources(citations)

        # 4. Abbreviation expansion (first occurrence only)
        ont_map = get_abbreviation_mapping()
        generated_text = expand_abbreviations_in_text(generated_text, ont_map)

        # Clean up any trailing follow-up questions
        generated_text = re.sub(r"\b(?:do you want me to|would you like me to|do you want to|should we)\s+[^?]+\?\s*$", "", generated_text, flags=re.IGNORECASE).strip()

        # Remove trailing disclaimers/limitations blocks from the text if it is not a high-stakes question
        is_high_stakes = any(w in q_lower for w in ("diagnos", "treat", "manage", "stent", "hospital", "urgent", "danger", "risk", "next step"))
        if not is_high_stakes:
            disclaimer_patterns = [
                r"\bDisclaimer\s*:.*",
                r"\bLimitations\s*:.*",
                r"\bImportant Note\s*:.*"
            ]
            for pat in disclaimer_patterns:
                generated_text = re.sub(pat, "", generated_text, flags=re.IGNORECASE | re.DOTALL).strip()
            
            for prefix in ("response:", "answer:"):
                if generated_text.lower().startswith(prefix):
                    generated_text = generated_text[len(prefix):].strip()

        # Update persistent conversation state
        active_finding_labels = []
        for abbrev, term in ont_map.items():
            if abbrev.lower() in q_lower or term.lower() in q_lower:
                active_finding_labels.append(abbrev)
        if not active_finding_labels and self.chat_state.get("last_finding_labels"):
            active_finding_labels = self.chat_state.get("last_finding_labels")
        elif not active_finding_labels:
            active_finding_labels = compact_decision.get("supported_labels", []) or [primary_label]

        retrieved_cases = []
        retrieval_evidence = tools.get("calls", [])
        retrieval_call = next((call for call in retrieval_evidence if call.get("tool") == "get_retrieval_evidence"), None)
        if retrieval_call:
            retrieved_cases = retrieval_call.get("result", {}).get("matches", [])

        selected_windows = [
            ep.get("window_index") for ep in state.get("temporal_summary", {}).get("abnormal_windows", [])
        ]

        self.chat_state["last_user_question"] = question
        self.chat_state["last_normalized_question"] = normalized_q
        self.chat_state["last_preferred_sections"] = preferred_sections
        self.chat_state["last_question_type"] = q_type
        self.chat_state["last_subject"] = q_subj
        self.chat_state["last_finding_labels"] = active_finding_labels
        self.chat_state["last_retrieved_cases"] = retrieved_cases
        self.chat_state["last_selected_windows"] = selected_windows
        self.chat_state["last_citations"] = citations
        self.chat_state["last_tool_results"] = tools

        # Append turn to turn history
        self.chat_state["turn_history"].append({"question": question, "answer": generated_text})
        update_conversation_summary(self.chat_state)

        return {
            "status": "deterministic_fallback" if fallback_used else ("guardrail_fallback" if guardrail_fallback else ("generated" if not fallback_used and generated.backend != "disabled" else "disabled_fallback")),
            "text": generated_text,
            "backend": "fallback_assembler" if fallback_used else (generated.backend if not fallback_used else "disabled"),
            "adapter_used": False if fallback_used else (generated.adapter_used if not fallback_used else False),
            "generation_time_ms": 0.0 if fallback_used else (generated.generation_time_ms if not fallback_used else 0.0),
            "review_authority": "advisory_only",
            "bridge_decision_modified": False,
            "false_unknown_guard_triggered": guardrail_fallback,
            "system_prompt": system_prompt_metadata(),
            "original_question": question,
            "normalized_question": normalized_q,
            "corrections": corrections,
            "correction_confidence": conf,
            "fallback_used": fallback_used,
            "conversation_summary": self.chat_state.get("conversation_summary"),
            "provenance_limitations": {
                "activated_clinical_literature_available": source_scope["activated_clinical_literature_available"],
                "missing_validated_clinical_guidelines": not source_scope["activated_clinical_literature_available"],
            }
        }

    @staticmethod
    def _save_snapshot(
        result: dict[str, Any], recording_id: str, patient_id: Optional[str], mode: str, elapsed_ms: float
    ) -> dict[str, Any]:
        from clinician_feedback_system import FeedbackStore
        bridge = result.get("recording_bridge", result.get("bridge", {}))
        explanation = result.get("explanation", {})
        selected = [item.get("window_index") for item in result.get("windows", []) if item.get("selected")]
        return FeedbackStore().record_response({
            "recording_id": recording_id,
            "patient_id": patient_id,
            "recording_mode": mode,
            "duration_seconds": result.get("duration_seconds", 10.0),
            "selected_window_ids": selected,
            "pipeline_output": result,
            "bridge_decision": bridge,
            "llm_analysis": explanation,
            "llm_answer": explanation.get("text", "") if isinstance(explanation, dict) else "",
            "citations": result.get("clinical_references", []),
            "component_versions": {"bridge": "V4", "retriever": "V7"},
            "prompt_metadata": explanation.get("system_prompt", {}) if isinstance(explanation, dict) else {},
            "performance": {"elapsed_ms": elapsed_ms},
        })

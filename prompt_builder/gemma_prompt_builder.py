import json
import math
from typing import Any, Dict, List, Union

from runtime.runtime_contracts import PromptPackage, make_json_safe


def format_optional_score(value: Any) -> str:
    if value is None:
        return "UNAVAILABLE"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "UNAVAILABLE"
    if not math.isfinite(numeric):
        return "UNAVAILABLE"
    return f"{numeric:.2f}"


def format_optional_str(value: Any, default: str = "UNAVAILABLE") -> str:
    if value is None or value == "":
        return default
    return str(value)


def build_gemma_prompt(
    package_input: Union[PromptPackage, Dict[str, Any]],
    max_prompt_tokens: int = 3200,
    tokenizer_func: Any = None,
) -> str:
    """
    Constructs a compact, token-efficient Gemma 3 prompt with dynamic section budgeting.
    Target budget: <= max_prompt_tokens (default 3200 tokens).
    """
    if isinstance(package_input, PromptPackage):
        pkg = package_input.__dict__
    elif isinstance(package_input, dict):
        pkg = package_input
    else:
        pkg = {}

    # Initial Truncation Pass parameters
    max_chunks = 3
    chunk_char_limit = 250
    max_neighbors = 5
    max_preds = 5
    max_history_turns = 2  # max 4 messages
    include_ontology = True
    concise_skills = False

    for attempt in range(6):
        prompt = _assemble_compact_prompt(
            pkg=pkg,
            max_chunks=max_chunks,
            chunk_char_limit=chunk_char_limit,
            max_neighbors=max_neighbors,
            max_preds=max_preds,
            max_history_turns=max_history_turns,
            include_ontology=include_ontology,
            concise_skills=concise_skills,
        )

        # Estimate token count (1 word / 4 chars approx ratio if tokenizer_func not provided)
        if tokenizer_func is not None:
            token_count = len(tokenizer_func(prompt))
        else:
            token_count = len(prompt) // 3.5

        if token_count <= max_prompt_tokens:
            return prompt

        # Progressive truncation steps
        if attempt == 0:
            max_chunks = 2
            chunk_char_limit = 180
        elif attempt == 1:
            max_chunks = 1
            chunk_char_limit = 120
            max_neighbors = 3
        elif attempt == 2:
            max_chunks = 0
            max_preds = 3
            max_history_turns = 1
        elif attempt == 3:
            max_history_turns = 0
            concise_skills = True
        elif attempt == 4:
            include_ontology = False

    return prompt


def _assemble_compact_prompt(
    pkg: Dict[str, Any],
    max_chunks: int,
    chunk_char_limit: int,
    max_neighbors: int,
    max_preds: int,
    max_history_turns: int,
    include_ontology: bool,
    concise_skills: bool,
) -> str:
    question = pkg.get("question", "What is the primary finding?")
    conv = pkg.get("conversation_state", {})
    route = pkg.get("skill_route", {})
    classifier = pkg.get("classifier", {})
    family_head = pkg.get("family_head", {})
    raw_ret = pkg.get("raw_retrieval", {})
    reranked_ret = pkg.get("reranked_retrieval", {})
    sig_qual = pkg.get("signal_quality", {})
    contradictions = pkg.get("contradictions", [])
    conf = pkg.get("confidence", {})
    bridge = pkg.get("bridge", {})
    know = pkg.get("knowledge_chunks", [])
    permitted_citations = pkg.get("permitted_citations", [])
    ontology = pkg.get("ontology_context", {})
    condition_cards = pkg.get("condition_cards_context", {})
    patient_context = pkg.get("patient_context", {})
    ecg_statistics = pkg.get("ecg_statistics", {})
    temporal_summary = pkg.get("temporal_summary", {})
    comparison = pkg.get("ecg_comparison", {})

    # Top predictions (max_preds)
    top_pred = classifier.get("top_predictions", [])[:max_preds]
    top_pred_str = ", ".join([f"{item.get('label')}: {format_optional_score(item.get('probability'))}" for item in top_pred]) if top_pred else "None"

    # Reranked neighbors summary (max_neighbors)
    reranked_neighbors = reranked_ret.get("top_5", reranked_ret.get("reranked_neighbors", []))[:max_neighbors]
    compact_neighbors = [
        f"Rank {n.get('raw_rank', i+1)}: {n.get('scp_code', n.get('label', 'UNKNOWN'))} (sim={format_optional_score(n.get('similarity', n.get('raw_similarity')))})"
        for i, n in enumerate(reranked_neighbors)
    ]
    reranked_summary_str = "; ".join(compact_neighbors) if compact_neighbors else "No candidates available."

    # Conversation history (bounded to max_history_turns)
    history_msgs = list(conv.get("history", []))[-2 * max_history_turns :] if max_history_turns > 0 else []
    history_str = json.dumps(history_msgs) if history_msgs else "[]"

    # Authoritative Bridge conclusion fields
    decision_status = format_optional_str(bridge.get("decision") or bridge.get("decision_status"), "indeterminate")
    bridge_label = format_optional_str(bridge.get("primary_label"), "UNKNOWN")
    bridge_family = format_optional_str(bridge.get("primary_family"), "UNKNOWN")
    support_strength = format_optional_str(bridge.get("support_strength"), "UNAVAILABLE")

    classifier_label = format_optional_str(classifier.get("primary_label"), "UNKNOWN")
    family_head_label = format_optional_str(family_head.get("primary_family"), "UNKNOWN")
    quality_score = format_optional_score(sig_qual.get("overall_quality_score"))
    quality_status = format_optional_str(sig_qual.get("quality_status"), "UNAVAILABLE")
    conf_level = format_optional_str(conf.get("confidence_level"), "UNAVAILABLE")
    conf_score = format_optional_score(conf.get("final_fused_confidence") or conf.get("diagnostic_confidence"))

    # Compact, traceable Knowledge chunks. Provenance fields are never truncated.
    compact_chunks = []
    if max_chunks > 0:
        for c in know[:max_chunks]:
            cid = c.get("citation_id") or c.get("id") or c.get("chunk_id") or "KB"
            txt = str(c.get("evidence_summary") or c.get("text", ""))[:chunk_char_limit]
            compact_chunks.append({
                "citation_id": cid,
                "evidence_summary": txt,
                "full_text_available": bool(c.get("text")),
                "supported_labels": c.get("supported_labels", []),
                "source_id": c.get("source_id"),
                "source_title": c.get("source"),
                "section": c.get("section"),
                "page": c.get("page"),
                "year": c.get("date_or_version"),
                "doi": c.get("doi"),
                "url": c.get("url"),
                "document_sha256": c.get("document_sha256"),
                "evidence_type": c.get("evidence_type"),
                "evidence_strength": c.get("evidence_strength"),
                "validation_state": c.get("validation_state"),
            })

    # Permitted citation IDs only
    citation_manifest = [c for c in permitted_citations[:4] if isinstance(c, dict)]

    # Contradictions bounded to top 5
    compact_contradictions = contradictions[:5] if isinstance(contradictions, list) else []

    prompt = f"""[1. IMMUTABLE SAFETY & AUTHORITY RULES]
- Evidence Bridge V4 is authoritative. Never diagnose directly from raw ECGs.
- Never alter diagnostic labels, probability scores, or introduce unverified findings.
- Never strengthen an Unknown or uncertain result.
- Cite only permitted citation IDs from the provided manifest.
- Answer the user's question directly in natural prose using current evidence only when relevant.
- Distinguish general education from patient-specific conclusions.
- Do not invent patient symptoms, clinical history, troponin, imaging, or angiography.
- Do not alter upstream inference.
- When evidence cannot answer the question, explain what evidence is missing.
- Do not repeat the complete ECG uncertainty paragraph unless the question asks for the full inference explanation.
- Preserve every supported coexisting label; never force a single-label conclusion.
- Patient symptoms may come only from Patient Context, never from ECG inference.

<user_question>
{question}
</user_question>

[2. DETECTED QUESTION INTENT]
Intent: {route.get('intent', 'diagnosis')} (Sub-Intent: {route.get('sub_intent', 'N/A')})
Question Subjects: {json.dumps(route.get('question_subjects', []))}
Comparison Requested: {route.get('comparison_requested', False)}
Required Inputs: {json.dumps(route.get('required_inputs', []))}

[3. CONVERSATION CONTEXT]
History: {history_str}
Patient Context: {json.dumps(patient_context)}

<immutable_case_evidence>
Decision Status: {decision_status}
Primary Label: {bridge_label}
Primary Family: {bridge_family}
Support Strength: {support_strength}
Classifier Primary Label: {classifier_label}
Independent Family: {family_head_label}
Overall Quality Score: {quality_score}
Signal Quality Status: {quality_status}
Classifier Output: Primary={classifier_label}, TopPredictions=[{top_pred_str}]
Family Head Output: Primary={family_head_label} (prob={format_optional_score(family_head.get('primary_probability'))})
Signal Quality: Score={quality_score}, Status={quality_status}
Diagnostic Confidence: Level={conf_level}, Score={conf_score}
Authoritative Bridge Decision: Status={decision_status}, Label={bridge_label}, Family={bridge_family}, SupportStrength={support_strength}
Retrieval Reranked Candidates: {reranked_summary_str}
Contradictions: {json.dumps(compact_contradictions)}
ECG Statistics: {json.dumps(ecg_statistics)}
Temporal Summary: {json.dumps(temporal_summary)}
Paired ECG Comparison: {json.dumps(comparison)}
</immutable_case_evidence>

[4. RELEVANT ONTOLOGY]
Ontology: {json.dumps(ontology if include_ontology else {})}

[5. KNOWLEDGE CHUNKS & CITATIONS]
Chunks: {json.dumps(compact_chunks)}
Permitted Citation Manifest: {json.dumps(citation_manifest)}

[6. NATURAL-ANSWER REQUIREMENTS]
- Answer the question inside <user_question> directly.
- Preserve uncertainty and limitations.
- Include a Sources section only when permitted citations are referenced.
- When asked for source provenance, report the exact source, section, page/locator, DOI/URL, evidence strength, and hash from the manifest.
"""
    return prompt.strip()

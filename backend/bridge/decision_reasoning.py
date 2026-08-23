# -*- coding: utf-8 -*-
"""
backend/bridge/decision_reasoning.py

Deterministic decision conflict reasoning builder for TRACE ECG.
Constructs clinician-facing narrative reasoning and structured reason codes when evidence
is unsupported, conflicted, or unverified.
"""

from typing import Dict, Any, List, Optional


def build_decision_reasoning(
    bridge_result: Optional[Dict[str, Any]] = None,
    classifier_result: Optional[Dict[str, Any]] = None,
    family_result: Optional[Dict[str, Any]] = None,
    retrieval_result: Optional[Dict[str, Any]] = None,
    signal_quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bridge = bridge_result or {}
    cls_data = classifier_result or bridge.get("classifier", {})
    fam_data = family_result or bridge.get("family_head", {})
    ret_data = retrieval_result or bridge.get("retrieval_status", {})
    sq_data = signal_quality or bridge.get("signal_quality", {})

    reason_codes: List[str] = []
    evidence_conflicts: List[str] = []

    # 1. Candidate Label and Probability
    concept = cls_data.get("primary_label") or bridge.get("primary_label") or "ECG finding"
    raw_probability = cls_data.get("primary_probability", bridge.get("primary_probability"))
    if raw_probability is None and cls_data.get("top_predictions"):
        raw_probability = cls_data["top_predictions"][0].get("probability")

    if raw_probability is not None:
        try:
            cls_prob = float(raw_probability)
            cls_pct = int(round(cls_prob * 100))
            candidate_phrase = f"The classifier's leading candidate was {concept} at {cls_pct}%"
        except (TypeError, ValueError):
            candidate_phrase = f"The classifier's leading candidate was {concept}"
    else:
        candidate_phrase = f"The classifier's leading candidate was {concept}"

    # 2. Classifier Family vs Independent Family Head
    classifier_family = cls_data.get("mapped_family") or bridge.get("primary_family") or ""
    independent_family = bridge.get("independent_family_head") or fam_data.get("primary_family") or ""

    family_agree = bool(bridge.get("family_agreement", classifier_family == independent_family))

    if classifier_family and independent_family and not family_agree:
        reason_codes.append("CLASSIFIER_FAMILY_DISAGREEMENT")
        evidence_conflicts.append(f"Classifier family ({classifier_family}) disagrees with independent family head ({independent_family})")
        fam_phrase = f", while the independent family model favored {independent_family}, indicating disagreement between the diagnostic pathways"
    elif classifier_family and independent_family:
        fam_phrase = f", and the independent family model agreed on {independent_family}"
    elif independent_family:
        fam_phrase = f", with independent family head favoring {independent_family}"
    else:
        fam_phrase = ""

    # 3. Retrieval Neighbors & Provenance
    raw_neighbors = bridge.get("raw_neighbors", [])
    ret_provenance = bridge.get("retrieval_status", {})
    if not ret_provenance:
        ret_provenance = ret_data.get("retrieval_provenance", ret_data)

    ver_status = ret_provenance.get("verification_status", "unverified")

    if raw_neighbors:
        top_neighbor_families = [n.get("families", ["normal"])[0] for n in raw_neighbors[:3] if n.get("families")]
        if top_neighbor_families:
            neighbor_summary = f"{top_neighbor_families[0].lower()}-labelled"
        else:
            neighbor_summary = "normal-labelled"
    else:
        neighbor_summary = "normal-labelled"

    if ver_status != "verified":
        reason_codes.append("RETRIEVAL_UNVERIFIED")
        ret_phrase = f"Retrieved examples were predominantly {neighbor_summary}, but the retrieval index provenance is unverified, so those matches were excluded from diagnostic confidence."
    else:
        ret_agree = bridge.get("retrieval_support", False)
        if ret_agree:
            reason_codes.append("RETRIEVAL_SUPPORTIVE")
            ret_phrase = f"Verified retrieval neighbors supported {neighbor_summary} matches."
        else:
            reason_codes.append("RETRIEVAL_CONTRADICTORY")
            ret_phrase = f"Verified retrieval neighbors favored {neighbor_summary} matches, which conflicts with the classifier finding."

    # 4. Signal Quality
    overall_quality = float(sq_data.get("overall_quality_score", 1.0))
    sq_status = sq_data.get("quality_status", "ACCEPTABLE")
    if overall_quality < 0.60 or sq_status != "ACCEPTABLE":
        reason_codes.append("POOR_SIGNAL_QUALITY")
        evidence_conflicts.append(f"Signal quality degraded (score = {overall_quality:.2f})")
        sq_phrase = " Signal quality limitations were also noted."
    else:
        sq_phrase = ""

    # 5. Requirements for Stronger Conclusion
    req_list = bridge.get("requirements_for_stronger_conclusion", [])
    if not req_list:
        req_list = [
            "agreement across validated models",
            "verified similar ECG evidence",
            "relevant clinical history and clinician interpretation"
        ]
    req_str = "; ".join(req_list[:3])

    # 6. Dynamic Summary Construction
    summary = (
        f"{candidate_phrase}{fam_phrase}. "
        f"{ret_phrase}{sq_phrase} "
        f"A stronger conclusion requires {req_str}."
    )

    summary = summary.replace("..", ".").replace("  ", " ").strip()

    manual_review = bool(
        "CLASSIFIER_FAMILY_DISAGREEMENT" in reason_codes
        or "RETRIEVAL_UNVERIFIED" in reason_codes
        or "POOR_SIGNAL_QUALITY" in reason_codes
        or bridge.get("decision") == "Unknown"
    )

    return {
        "summary": summary,
        "reason_codes": reason_codes,
        "evidence_conflicts": evidence_conflicts,
        "requirements_for_stronger_conclusion": req_list,
        "manual_review_recommended": manual_review
    }

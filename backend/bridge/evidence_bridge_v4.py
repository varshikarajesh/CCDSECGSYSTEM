"""Evidence Bridge V4: multi-label, window-aware, citation-grounded fusion.

Inputs are read-only outputs from the PTB-XL classifier, V7 retrieval, optional
Holter temporal classifier, and ECG statistics evaluator.  The bridge never
changes upstream probabilities and contains no OOD pathway.
"""
from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from backend.bridge.statistics_knowledge import ECGStatisticsKnowledge
from backend.bridge.ecg_statistics import build_segment_statistics
from knowledge.provenance import citation_for, validated_chunks
from ptbxl_five_superclass.scp_family_mapping import SCP_TO_FAMILY_RAW


ROOT = Path(__file__).resolve().parents[2]
LABEL_CONCEPT_PATH = ROOT / "knowledge" / "label_to_concept_mapping.json"
KB_EMBEDDINGS_PATH = ROOT / "knowledge" / "embeddings_kb.npy"
KB_ID_LIST_PATH = ROOT / "knowledge" / "id_list.json"

RHYTHM_LABELS = {
    "SR", "AFIB", "AFLT", "STACH", "SBRAD", "SARRH", "SVTAC", "PSVT",
    "SVARR", "BIGU", "TRIGU", "PVC", "PAC", "PRC(S)", "1AVB", "2AVB", "3AVB"
}
HOLTER_ALIASES = {
    "AF": "AFIB", "ATRIAL_FIBRILLATION": "AFIB", "ATRIAL FLUTTER": "AFLT",
    "AFL": "AFLT", "SVT": "SVTAC", "SINUS_TACHYCARDIA": "STACH",
    "SINUS_BRADYCARDIA": "SBRAD", "PVC": "PVC", "PAC": "PAC",
    "NORMAL": "SR", "SINUS_RHYTHM": "SR"
}
KB_PATTERN_ALIASES = {
    "AFIB": ["AF", "AFIB"], "AFLT": ["AFLT", "AF"],
    "CRBBB": ["CRBBB", "RBBB"], "CLBBB": ["CLBBB", "LBBB"],
    "LNGQT": ["LNGQT", "LQTS"], "IMI": ["IMI"], "AMI": ["AMI"],
    "ASMI": ["ASMI", "AMI"], "ALMI": ["ALMI", "AMI"],
    "NORM": ["NORM"], "LVH": ["LVH"]
}


class BridgeInputError(ValueError):
    pass


class TraceableKnowledgeIndex:
    """Small exact-label/family index over hash-validated KB chunks."""

    def __init__(self, chunks: Optional[Sequence[Dict[str, Any]]] = None):
        self.chunks = list(validated_chunks() if chunks is None else chunks)
        self.embeddings = None
        self.embedding_row_by_id = {}
        self.label_concepts = {}
        if LABEL_CONCEPT_PATH.is_file():
            self.label_concepts = json.loads(LABEL_CONCEPT_PATH.read_text(encoding="utf-8"))
        try:
            ids = json.loads(KB_ID_LIST_PATH.read_text(encoding="utf-8"))
            embeddings = np.load(KB_EMBEDDINGS_PATH, allow_pickle=False)
            if embeddings.ndim == 2 and len(ids) == embeddings.shape[0]:
                self.embeddings = embeddings
                self.embedding_row_by_id = {str(chunk_id): i for i, chunk_id in enumerate(ids)}
        except Exception:
            self.embeddings = None
            self.embedding_row_by_id = {}

    @staticmethod
    def _query_embedding(text: str, dimensions: int = 384) -> np.ndarray:
        words = " ".join(str(text).lower().split()).split()
        tokens = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        vector = np.zeros(dimensions, dtype=np.float32)
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vector[value % dimensions] += 1.0 if (value >> 8) & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def for_label(self, label: str, family: str, limit: int = 2) -> List[Dict[str, Any]]:
        label = label.upper()
        aliases = {label, *(x.upper() for x in KB_PATTERN_ALIASES.get(label, []))}
        concept = str(self.label_concepts.get(label, "")).upper()
        query_vector = self._query_embedding(f"{label} {family} {concept}") if self.embeddings is not None else None
        scored: List[Tuple[int, str, Dict[str, Any]]] = []
        for chunk in self.chunks:
            tags = chunk.get("tags", {})
            pattern = str(tags.get("pattern", "")).upper()
            superclass = str(tags.get("superclass", "")).upper()
            evidence_section = str(tags.get("section", "")).lower()
            title = str(chunk.get("title", "")).upper()
            text_head = str(chunk.get("text", ""))[:1200].upper()
            score = 0
            if pattern in aliases:
                score += 100
            if label in title or label in text_head:
                score += 35
            if concept and (concept in title or concept in text_head):
                score += 20
            if family.upper() and superclass == family.upper():
                score += 8
            if evidence_section == "ecg_diagnostic_criteria":
                score += 90
            # A bridge decision must cite diagnostic/ECG evidence before management text.
            if "DIAGNOSTIC CRITERIA" in text_head:
                score += 90
            if "DEFINITION" in text_head or "SURFACE ELECTROCARDIOGRAM" in text_head:
                score += 55
            if any(term in text_head for term in ("ECG FINDING", "ECG FEATURE", "P WAVES", "RR INTERVAL")):
                score += 25
            if any(term in text_head for term in ("MANAGEMENT", "TREATMENT", "ANTICOAGULATION", "SHARED DECISION")):
                score -= 20
            semantic_score = 0.0
            row = self.embedding_row_by_id.get(str(chunk.get("id", "")))
            if query_vector is not None and row is not None:
                semantic_score = max(0.0, float(np.dot(query_vector, self.embeddings[row])))
                score += int(round(semantic_score * 30.0))
            if score:
                scored.append((score, str(chunk.get("id", "")), chunk, semantic_score))
        scored.sort(key=lambda item: (-item[0], item[1]))
        result = []
        seen_sources = set()
        for _, _, chunk, semantic_score in scored:
            citation = citation_for(chunk)
            source_key = (citation.get("title"), citation.get("section"))
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            result.append({
                "chunk_id": chunk.get("id"),
                "citation": citation,
                "evidence_summary": str(chunk.get("evidence_summary") or chunk.get("text", ""))[:320].strip(),
                "semantic_score": round(semantic_score, 4),
                "full_text_available": bool(chunk.get("text")),
            })
            if len(result) >= limit:
                break
        return result


class EvidenceBridgeV4:
    """Authoritative multi-label fusion for 10-second and long-recording modes."""

    VERSION = "4.0.0"
    DEFAULT_THRESHOLDS = {
        "classifier_candidate": 0.20,
        "classifier_positive": 0.30,
        "supported_score": 0.65,
        "probable_score": 0.50,
        "minimum_retrieval_similarity": 0.20,
        "minimum_independent_sources": 2,
        "max_citations_per_finding": 2
    }

    def __init__(
        self,
        thresholds: Optional[Mapping[str, float]] = None,
        statistics_knowledge: Optional[ECGStatisticsKnowledge] = None,
        knowledge_index: Optional[TraceableKnowledgeIndex] = None,
    ):
        self.thresholds = dict(self.DEFAULT_THRESHOLDS)
        self.thresholds.update(thresholds or {})
        self.statistics = statistics_knowledge or ECGStatisticsKnowledge()
        self.knowledge = knowledge_index or TraceableKnowledgeIndex()

    def process(
        self,
        classifier_output: Any,
        family_head_output: Optional[Mapping[str, Any]] = None,
        retrieval_output: Optional[Mapping[str, Any]] = None,
        model_embedding: Any = None,
        raw_ecg: Any = None,
        signal_quality_score: float = 1.0,
        sampling_rate_hz: int = 100,
        abnormal_segments: Optional[Sequence[Mapping[str, Any]]] = None,
        retrieval_query_plan: Optional[Mapping[str, Any]] = None,
        holter_output: Optional[Mapping[str, Any]] = None,
        selected_windows: Optional[Sequence[Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compatibility entry point used by the current API while V4 is authoritative.

        ``model_embedding`` and the independent family head are preserved for
        provenance but never alter classifier probabilities. The canonical V4
        fields remain in the returned object; legacy aliases exist only for
        downstream UI compatibility.
        """
        del model_embedding
        segments = list(abnormal_segments or [])
        measured = build_segment_statistics(raw_ecg, sampling_rate_hz, segments)
        window_stats = {}
        inferred_windows = []
        for index, item in enumerate(measured.get("abnormal_windows", [])):
            wid = str(item.get("window_index", index))
            window_stats[wid] = item
            inferred_windows.append({
                "window_id": wid,
                "start_seconds": item.get("start_seconds"),
                "end_seconds": item.get("end_seconds"),
                "role": "abnormal_candidate",
                "selected": True,
            })
        meta = {"mode": "10s" if not segments else "long_recording", **dict(metadata or {})}
        result = self.combine(
            classifier_output=classifier_output,
            retrieval_output=retrieval_output or {},
            holter_output=holter_output or {},
            statistics_output={"overall": measured.get("overall", {}), "windows": window_stats},
            selected_windows=list(selected_windows or inferred_windows),
            signal_quality={"overall_quality_score": signal_quality_score},
            metadata=meta,
        )
        return self._compatibility_projection(
            result, classifier_output, family_head_output or {}, retrieval_output or {},
            measured, retrieval_query_plan or {},
        )

    @staticmethod
    def _compatibility_projection(result, classifier, family, retrieval, measurements, query_plan):
        supported_details = list(result.get("supported_findings", []))
        uncertain_details = list(result.get("uncertain_findings", []))
        supported = [str(item.get("label")) for item in supported_details]
        partial = [str(item.get("label")) for item in uncertain_details]
        candidates = supported_details or uncertain_details
        primary = (candidates[0].get("label") if candidates else classifier.get("primary_label")) or "UNKNOWN"
        primary_family = (candidates[0].get("family") if candidates else classifier.get("mapped_family")) or "UNKNOWN"
        primary_score = float((candidates[0].get("fusion_score") if candidates else classifier.get("primary_probability", 0.0)) or 0.0)
        canonical_status = str(result.get("decision_status", "no_supported_finding"))
        decision = "supported" if supported else "uncertain" if partial else "Unknown"
        raw_neighbors = list(retrieval.get("reranked_neighbors") or retrieval.get("neighbors") or retrieval.get("raw_neighbors") or [])
        family_primary = family.get("primary_family", "UNKNOWN")
        family_agreement = family_primary in {"", "UNKNOWN", primary_family}
        family_probability = float(family.get("primary_probability", 0.0) or 0.0)
        material_family_conflict = (not family_agreement) and family_probability >= 0.75
        confidence_score = min(primary_score, 0.69) if material_family_conflict else primary_score
        confidence_limitations = []
        if material_family_conflict:
            confidence_limitations.append(
                "Independent family head strongly disagrees with the supported finding family; confidence is capped pending clinician review."
            )
        result.update({
            "canonical_decision_status": canonical_status,
            "decision": decision,
            "primary_label": str(primary),
            "primary_family": str(primary_family),
            "primary_probability": primary_score,
            "support_strength": "STRONG" if supported else "PARTIAL" if partial else "INSUFFICIENT",
            "finding_details": supported_details + uncertain_details + list(result.get("rejected_candidates", [])),
            "supported_findings": supported,
            "partially_supported_findings": partial,
            "classifier_results": list(classifier.get("top_predictions") or []),
            "family_results": list(family.get("top_predictions") or []),
            "independent_family_head": family_primary,
            "family_agreement": family_agreement,
            "contradictions": ([] if family_agreement else [{"type": "family_head_disagreement", "classifier_family": primary_family, "independent_family": family_primary}]),
            "raw_neighbors": raw_neighbors,
            "reranked_neighbors": raw_neighbors,
            "rerank_trace": [],
            "retrieval_status": {"verification_status": "verified" if raw_neighbors else "unavailable", "neighbor_count": len(raw_neighbors)},
            "retrieval_quality": {"neighbor_count": len(raw_neighbors), "label_support": result.get("retrieval_summary", {}).get("label_support", {})},
            "rare_case": {"state": "handled_by_multilabel_v4", "is_rare": False},
            "confidence": {
                "final_fused_confidence": confidence_score,
                "uncapped_fusion_score": primary_score,
                "confidence_level": "HIGH" if confidence_score >= 0.75 else "MODERATE" if confidence_score >= 0.50 else "LOW",
                "confidence_drivers": (candidates[0].get("independent_sources", []) if candidates else []),
                "limitations": confidence_limitations,
                "family_conflict_cap_applied": material_family_conflict,
            },
            "unknown_reasons": ([] if decision != "Unknown" else [{"condition": "independent_support", "passed": False}]),
            "requirements_for_stronger_conclusion": ([] if supported else ["Additional independent evidence and clinician review"]),
            "ecg_measurements": measurements,
            "retrieval_query_plan": dict(query_plan),
            "evidence_branches": {
                "classifier": {"available": bool(classifier.get("probabilities"))},
                "retrieval": {"available": bool(raw_neighbors)},
                "holter": {"available": bool(result.get("temporal_summary", {}).get("available"))},
                "statistics": {"available": bool(measurements)},
                "knowledge": {"available": bool(result.get("citations"))},
            },
            "ood": {"enabled": False, "ood_status": "not_used_v4", "input_source": "disabled_by_v4_policy"},
        })
        return result

    def combine(
        self,
        classifier_output: Any,
        retrieval_output: Optional[Mapping[str, Any]] = None,
        holter_output: Optional[Mapping[str, Any]] = None,
        statistics_output: Optional[Mapping[str, Any]] = None,
        selected_windows: Optional[Sequence[Mapping[str, Any]]] = None,
        signal_quality: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = dict(metadata or {})
        mode = str(meta.get("mode", "10s"))
        classifier = self._normalise_classifier(classifier_output)
        if not classifier["recording"] and not classifier["windows"]:
            raise BridgeInputError("Classifier output contains no usable label probabilities.")

        retrieval = self._normalise_retrieval(retrieval_output or {})
        holter = self._normalise_holter(holter_output or {})
        stats = self._normalise_statistics(statistics_output or {})
        window_catalog = self._window_catalog(selected_windows or [], classifier["window_metadata"], meta)
        selected = [wid for wid, item in window_catalog.items() if item.get("selected", True)]
        quality = self._quality(signal_quality or {})

        labels = set(classifier["recording"])
        labels.update(label for window in classifier["windows"].values() for label in window)
        labels.update(retrieval["label_support"])
        labels.update(holter["recording"])
        labels.update(e["labels"][0] for e in stats["overall"].get("supporting_evidence", []) if e.get("labels"))
        for evidence in stats["windows"].values():
            labels.update(label for e in evidence.get("supporting_evidence", []) for label in e.get("labels", []))

        candidates = sorted(label for label in labels if self._is_candidate(label, classifier, retrieval, holter, stats))
        findings = [self._fuse_label(label, classifier, retrieval, holter, stats, selected, quality) for label in candidates]
        findings = self._resolve_normal(findings)
        findings.sort(key=lambda item: (-item["fusion_score"], item["label"]))

        for finding in findings:
            finding["citations"] = self.knowledge.for_label(
                finding["label"], finding["family"], int(self.thresholds["max_citations_per_finding"])
            )
            if finding["status"] == "supported" and not finding["citations"]:
                finding["limitations"].append("No exact traceable KB citation was available for this label.")
            finding["time_intervals"] = self._time_intervals(finding["supporting_windows"], window_catalog)
            finding["explanation"] = self._finding_explanation(finding)

        supported = [f for f in findings if f["status"] == "supported"]
        uncertain = [f for f in findings if f["status"] in {"probable", "uncertain"}]
        red_flags = [f["label"] for f in findings if f["red_flag"]]
        decision_status = self._decision_status(supported, uncertain, quality)
        citations = self._deduplicate_citations(findings)
        normal_explanation = self._normal_explanation(findings, window_catalog, quality)
        abnormality_explanations = [
            {
                "label": finding["label"],
                "family": finding["family"],
                "status": finding["status"],
                "explanation": finding["explanation"],
                "appears_at": finding["time_intervals"],
                "evidence_summaries": [item.get("evidence_summary", "") for item in finding["citations"]],
                "citations": [item["citation"] for item in finding["citations"]],
            }
            for finding in findings if finding["label"] != "NORM" and finding["status"] in {"supported", "probable", "uncertain"}
        ]

        return {
            "bridge_version": self.VERSION,
            "architecture": "multi_label_window_aware_citation_grounded",
            "mode": mode,
            "decision_status": decision_status,
            "supported_findings": supported,
            "uncertain_findings": uncertain,
            "rejected_candidates": [f for f in findings if f["status"] == "not_supported"],
            "normal_explanation": normal_explanation,
            "abnormality_explanations": abnormality_explanations,
            "finding_to_window_attribution": {
                f["label"]: f["supporting_windows"] for f in findings if f["supporting_windows"]
            },
            "overall_vs_abnormal_statistics": {
                "overall": stats["overall"],
                "selected_windows": {wid: stats["windows"].get(wid, {}) for wid in selected},
                "comparisons": self._statistics_comparisons(stats, selected, window_catalog)
            },
            "temporal_summary": holter,
            "retrieval_summary": {
                "label_support": retrieval["label_support"],
                "neighbor_count": len(retrieval["neighbors"])
            },
            "signal_quality": quality,
            "red_flags": red_flags,
            "citations": citations,
            "limitations": self._global_limitations(mode, classifier, retrieval, holter, stats, quality),
            "decision_policy": {
                "llm_can_modify": False,
                "ood_used": False,
                "statistics_are_support_only": True,
                "independent_support_required": True
            },
            "provenance": {
                "raw_classifier_preserved": True,
                "raw_retrieval_preserved": True,
                "raw_holter_preserved": True,
                "knowledge_chunks_hash_validated": True,
                "thresholds": self.thresholds,
                "metadata": meta
            }
        }

    def _fuse_label(self, label, classifier, retrieval, holter, stats, selected, quality):
        family = SCP_TO_FAMILY_RAW.get(label, "Other")
        cls_recording = classifier["recording"].get(label, 0.0)
        window_scores = {wid: probs.get(label, 0.0) for wid, probs in classifier["windows"].items()}
        cls_window = max(window_scores.values(), default=0.0)
        cls_score = max(cls_recording, cls_window)
        ret_score = retrieval["label_support"].get(label, 0.0)
        holter_score = holter["recording"].get(label, 0.0) if label in RHYTHM_LABELS else 0.0
        stat_support, stat_contradiction, stat_red = self._statistics_for_label(label, stats)

        components = []
        if cls_score > 0: components.append(("classifier", cls_score, 0.50))
        if ret_score > 0: components.append(("retrieval", ret_score, 0.25))
        if label in RHYTHM_LABELS and holter_score > 0: components.append(("holter", holter_score, 0.15))
        if stat_support > 0: components.append(("statistics", stat_support, 0.15))
        denominator = sum(weight for _, _, weight in components) or 1.0
        fused = sum(score * weight for _, score, weight in components) / denominator
        fused = max(0.0, fused - 0.20 * stat_contradiction)
        if quality["status"] == "REJECTED": fused *= 0.40
        elif quality["status"] == "LIMITED": fused *= 0.80

        independent = {name for name, score, _ in components if score >= 0.25}
        model_anchor = cls_score >= self.thresholds["classifier_positive"] or ret_score >= 0.35
        if fused >= self.thresholds["supported_score"] and len(independent) >= self.thresholds["minimum_independent_sources"] and model_anchor:
            status = "supported"
        elif fused >= self.thresholds["probable_score"] and model_anchor:
            status = "probable"
        elif model_anchor or stat_support >= 0.65 or holter_score >= 0.60:
            status = "uncertain"
        else:
            status = "not_supported"

        windows = set()
        for wid, score in window_scores.items():
            if score >= self.thresholds["classifier_positive"]: windows.add(wid)
        windows.update(retrieval["label_windows"].get(label, []))
        windows.update(holter["label_windows"].get(label, []))
        windows.update(self._stat_windows(label, stats))
        if selected:
            windows = {wid for wid in windows if wid in selected} or windows

        limitations = []
        if len(independent) < self.thresholds["minimum_independent_sources"]:
            limitations.append("Fewer than two independent evidence sources support this finding.")
        if label in RHYTHM_LABELS and not holter["available"]:
            limitations.append("No validated Holter temporal output was available.")
        if stat_contradiction:
            limitations.append("ECG statistics contain contradictory evidence.")

        return {
            "label": label,
            "family": family,
            "status": status,
            "fusion_score": round(fused, 4),
            "independent_sources": sorted(independent),
            "evidence": {
                "classifier": round(cls_score, 4), "retrieval": round(ret_score, 4),
                "holter": round(holter_score, 4), "statistics": round(stat_support, 4),
                "statistics_contradiction": round(stat_contradiction, 4)
            },
            "supporting_windows": sorted(windows, key=str),
            "red_flag": bool(stat_red),
            "limitations": limitations,
            "citations": []
        }

    def _normalise_classifier(self, data: Any) -> Dict[str, Any]:
        recording, windows, window_metadata = {}, {}, {}
        if isinstance(data, list): recording = self._probability_map(data)
        elif isinstance(data, Mapping):
            recording = self._probability_map(data.get("probabilities") or data.get("calibrated_probabilities") or data.get("labels") or data.get("scp_predictions") or data)
            raw_windows = data.get("windows") or data.get("window_predictions") or []
            if isinstance(raw_windows, Mapping):
                windows = {str(k): self._probability_map(v) for k, v in raw_windows.items()}
                window_metadata = {str(k): self._extract_window_metadata(v, str(k)) for k, v in raw_windows.items() if isinstance(v, Mapping)}
            else:
                for index, item in enumerate(raw_windows):
                    if isinstance(item, Mapping):
                        wid = str(item.get("window_id", index))
                        windows[wid] = self._probability_map(item.get("probabilities") or item.get("predictions") or item)
                        window_metadata[wid] = self._extract_window_metadata(item, wid)
        return {"recording": recording, "windows": windows, "window_metadata": window_metadata}

    @staticmethod
    def _probability_map(data: Any) -> Dict[str, float]:
        result = {}
        if isinstance(data, Mapping):
            for key, value in data.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool): result[str(key).upper()] = float(value)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, Mapping):
                    label = item.get("label") or item.get("scp") or item.get("code")
                    value = item.get("probability", item.get("score"))
                    if label is not None and value is not None: result[str(label).upper()] = float(value)
        return result

    def _normalise_retrieval(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        neighbors = list(data.get("reranked_neighbors") or data.get("neighbors") or data.get("raw_neighbors") or [])
        support, denom, windows = defaultdict(float), 0.0, defaultdict(set)
        for neighbor in neighbors:
            sim = max(0.0, float(neighbor.get("similarity", neighbor.get("raw_similarity", neighbor.get("score", 0.0))) or 0.0))
            if sim < self.thresholds["minimum_retrieval_similarity"]: continue
            labels = neighbor.get("scp_codes") or neighbor.get("labels") or [neighbor.get("label") or neighbor.get("scp_code")]
            labels = [str(x).upper() for x in labels if x]
            denom += sim
            for label in labels:
                support[label] += sim
                if neighbor.get("query_window_id") is not None: windows[label].add(str(neighbor["query_window_id"]))
        if denom: support = defaultdict(float, {k: v / denom for k, v in support.items()})
        return {"neighbors": neighbors, "label_support": dict(support), "label_windows": {k: sorted(v) for k, v in windows.items()}}

    def _normalise_holter(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        recording = self._probability_map(data.get("probabilities") or data.get("recording_probabilities") or {})
        recording = {HOLTER_ALIASES.get(k, k): v for k, v in recording.items()}
        label_windows = defaultdict(set)
        windows = data.get("windows") or data.get("window_predictions") or []
        for index, window in enumerate(windows if isinstance(windows, list) else []):
            wid = str(window.get("window_id", index))
            for label, value in self._probability_map(window.get("probabilities") or window).items():
                mapped = HOLTER_ALIASES.get(label, label)
                if value >= self.thresholds["classifier_positive"]: label_windows[mapped].add(wid)
        return {"available": bool(recording or label_windows), "recording": recording, "label_windows": {k: sorted(v) for k, v in label_windows.items()}}

    def _normalise_statistics(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        def flatten_raw_stats(raw: Mapping[str, Any]) -> Dict[str, Any]:
            if not isinstance(raw, dict):
                return {}
            flat = {}
            bd = raw.get("beat_detection") or {}
            if isinstance(bd, dict):
                if bd.get("mean_heart_rate_bpm") is not None:
                    flat["heart_rate_bpm"] = bd["mean_heart_rate_bpm"]
                if bd.get("r_peak_count") is not None:
                    flat["beat_count"] = bd["r_peak_count"]
            hrv = raw.get("time_domain_hrv") or {}
            if isinstance(hrv, dict):
                for k, v in hrv.items():
                    if v is not None:
                        flat[k] = v
            morph = raw.get("morphology") or {}
            if isinstance(morph, dict):
                if morph.get("qrs_duration_median_ms_estimate") is not None:
                    flat["qrs_ms"] = morph["qrs_duration_median_ms_estimate"]
                if morph.get("pr_interval_ms") is not None:
                    flat["pr_ms"] = morph["pr_interval_ms"]
                if morph.get("qt_interval_ms") is not None:
                    flat["qt_ms"] = morph["qt_interval_ms"]
                if morph.get("qtc_bazett_ms") is not None:
                    flat["qtc_bazett_ms"] = morph["qtc_bazett_ms"]
                if morph.get("qtc_fridericia_ms") is not None:
                    flat["qtc_fridericia_ms"] = morph["qtc_fridericia_ms"]
            if raw.get("duration_seconds") is not None:
                flat["duration_seconds"] = raw["duration_seconds"]
            return flat

        overall_raw = data.get("overall") or data.get("recording") or data.get("measurements") or {}
        if "supporting_evidence" in overall_raw:
            overall = overall_raw
        else:
            flat_overall = flatten_raw_stats(overall_raw)
            overall = self.statistics.evaluate(flat_overall, "recording_overall")

        windows = {}
        raw_windows = data.get("windows") or data.get("window_statistics") or {}
        items = raw_windows.items() if isinstance(raw_windows, Mapping) else enumerate(raw_windows)
        for key, value in items:
            wid = str(value.get("window_id", key)) if isinstance(value, Mapping) else str(key)
            if isinstance(value, Mapping) and "supporting_evidence" in value:
                windows[wid] = value
            else:
                flat_val = flatten_raw_stats(value) if isinstance(value, Mapping) else {}
                windows[wid] = self.statistics.evaluate(flat_val, "abnormal_episode")
        return {"overall": overall, "windows": windows}

    @staticmethod
    def _extract_window_metadata(window: Mapping[str, Any], fallback_id: str) -> Dict[str, Any]:
        return {
            "window_id": str(window.get("window_id", fallback_id)),
            "start_seconds": window.get("start_seconds", window.get("start_sec")),
            "end_seconds": window.get("end_seconds", window.get("end_sec")),
            "role": window.get("role", window.get("window_role", "candidate")),
            "selected": bool(window.get("selected", True)),
        }

    def _window_catalog(self, windows, classifier_metadata, metadata):
        catalog = {str(k): dict(v) for k, v in classifier_metadata.items()}
        for index, window in enumerate(windows):
            if isinstance(window, Mapping):
                wid = str(window.get("window_id", index))
                catalog[wid] = {**catalog.get(wid, {}), **self._extract_window_metadata(window, wid)}
            else:
                wid = str(window)
                catalog.setdefault(wid, {"window_id": wid, "role": "candidate", "selected": True})
        duration = float(metadata.get("window_duration_seconds", 10.0))
        index_base = int(metadata.get("window_index_base", 0))
        for wid, item in catalog.items():
            if item.get("start_seconds") is None:
                digits = "".join(ch for ch in wid if ch.isdigit())
                if digits:
                    index = max(0, int(digits) - index_base)
                    item["start_seconds"] = index * duration
            if item.get("end_seconds") is None and item.get("start_seconds") is not None:
                item["end_seconds"] = float(item["start_seconds"]) + duration
        return catalog

    @staticmethod
    def _quality(data: Mapping[str, Any]) -> Dict[str, Any]:
        score = float(data.get("overall_quality_score", data.get("quality_score", data.get("score", 1.0))) or 0.0)
        status = "ACCEPTABLE" if score >= 0.70 else "LIMITED" if score >= 0.45 else "REJECTED"
        return {"status": status, "score": round(score, 4), "warnings": list(data.get("warnings", []))}

    def _is_candidate(self, label, classifier, retrieval, holter, stats):
        if classifier["recording"].get(label, 0) >= self.thresholds["classifier_candidate"]: return True
        if any(w.get(label, 0) >= self.thresholds["classifier_candidate"] for w in classifier["windows"].values()): return True
        if retrieval["label_support"].get(label, 0) >= 0.20: return True
        if holter["recording"].get(label, 0) >= 0.30: return True
        return self._statistics_for_label(label, stats)[0] >= 0.50

    @staticmethod
    def _statistics_for_label(label, stats):
        support, contradiction, red = 0.0, 0.0, False
        for evidence in [stats["overall"], *stats["windows"].values()]:
            for item in evidence.get("supporting_evidence", []):
                if label in item.get("labels", []):
                    support = max(support, float(item.get("weight", 0)))
                    red = red or bool(item.get("red_flag"))
            for item in evidence.get("contradicting_evidence", []):
                if label in item.get("labels", []): contradiction = max(contradiction, float(item.get("weight", 0)))
        return support, contradiction, red

    @staticmethod
    def _stat_windows(label, stats):
        return [wid for wid, evidence in stats["windows"].items() if any(label in item.get("labels", []) for item in evidence.get("supporting_evidence", []))]

    @staticmethod
    def _resolve_normal(findings):
        abnormal_supported = any(f["label"] != "NORM" and f["status"] == "supported" for f in findings)
        if abnormal_supported:
            for finding in findings:
                if finding["label"] == "NORM":
                    finding["status"] = "not_supported"
                    finding["limitations"].append("NORM suppressed because one or more abnormal findings are independently supported.")
        return findings

    @staticmethod
    def _decision_status(supported, uncertain, quality):
        if quality["status"] == "REJECTED": return "insufficient_signal_quality"
        if supported: return "multi_label_supported"
        if uncertain: return "uncertain_requires_review"
        return "no_supported_finding"

    @staticmethod
    def _deduplicate_citations(findings):
        output, seen = [], set()
        for finding in findings:
            for item in finding.get("citations", []):
                citation = item["citation"]
                key = citation.get("citation_id")
                if key not in seen:
                    seen.add(key)
                    output.append({**citation, "evidence_summary": item.get("evidence_summary", ""), "full_text_available": item.get("full_text_available", False)})
        return output

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60)
        remainder = seconds - minutes * 60
        return f"{minutes:02d}:{remainder:04.1f}"

    def _time_intervals(self, window_ids, catalog):
        intervals = []
        for wid in window_ids:
            item = catalog.get(str(wid), {})
            start, end = item.get("start_seconds"), item.get("end_seconds")
            intervals.append({
                "window_id": str(wid),
                "start_seconds": start,
                "end_seconds": end,
                "display": (
                    f"{self._format_time(start)}-{self._format_time(end)}"
                    if start is not None and end is not None else "time unavailable"
                ),
                "role": item.get("role", "candidate")
            })
        return intervals

    @staticmethod
    def _finding_explanation(finding):
        evidence = finding["evidence"]
        active = [
            f"classifier {evidence['classifier']:.2f}" if evidence["classifier"] else "",
            f"retrieval {evidence['retrieval']:.2f}" if evidence["retrieval"] else "",
            f"Holter temporal support {evidence['holter']:.2f}" if evidence["holter"] else "",
            f"measured ECG criteria {evidence['statistics']:.2f}" if evidence["statistics"] else "",
        ]
        active = [item for item in active if item]
        certainty = {
            "supported": "independently supported",
            "probable": "probable but not fully corroborated",
            "uncertain": "possible and requires clinician review",
            "not_supported": "not sufficiently supported"
        }[finding["status"]]
        return f"{finding['label']} is {certainty}. Evidence: " + (", ".join(active) if active else "no active evidence") + "."

    def _normal_explanation(self, findings, catalog, quality):
        norm = next((f for f in findings if f["label"] == "NORM"), None)
        stable_ids = [wid for wid, item in catalog.items() if str(item.get("role", "")).lower() in {"stable", "stable_reference", "reference", "normal_reference"}]
        abnormal = [f["label"] for f in findings if f["label"] != "NORM" and f["status"] == "supported"]
        if norm and norm["status"] == "supported" and not abnormal:
            status = "supported_normal_recording"
            explanation = "The classifier and independent evidence support NORM, with no independently supported abnormal label. This does not exclude conditions outside the measured ECG evidence."
        elif stable_ids:
            status = "stable_reference_regions_present"
            explanation = "These intervals were selected as stable reference regions. They are used as the patient's within-recording baseline and are not automatically declared clinically normal."
        elif abnormal:
            status = "no_normal_claim"
            explanation = "Abnormal findings are supported, so the bridge does not label the complete recording normal. Unflagged intervals are not assumed to be normal without positive reference evidence."
        else:
            status = "normality_not_established"
            explanation = "The available evidence does not establish either a supported normal recording or a supported abnormal diagnosis."
        return {
            "status": status,
            "explanation": explanation,
            "stable_reference_intervals": self._time_intervals(stable_ids, catalog),
            "signal_quality": quality,
            "normal_classifier_score": norm["evidence"]["classifier"] if norm else 0.0,
            "normal_retrieval_score": norm["evidence"]["retrieval"] if norm else 0.0,
            "suppressed_by_abnormal_findings": abnormal
        }

    @staticmethod
    def _statistics_comparisons(stats, selected, catalog):
        overall = stats["overall"].get("calculated_metrics", {})
        comparisons = []
        for wid in selected:
            window_values = stats["windows"].get(wid, {}).get("calculated_metrics", {})
            shared = sorted(set(overall).intersection(window_values))
            deltas = {
                metric: round(float(window_values[metric]) - float(overall[metric]), 6)
                for metric in shared
                if isinstance(overall[metric], (int, float)) and isinstance(window_values[metric], (int, float))
            }
            if deltas:
                comparisons.append({
                    "window_id": wid,
                    "role": catalog.get(wid, {}).get("role", "candidate"),
                    "overall_values": {metric: overall[metric] for metric in deltas},
                    "window_values": {metric: window_values[metric] for metric in deltas},
                    "window_minus_overall": deltas
                })
        return comparisons

    @staticmethod
    def _global_limitations(mode, classifier, retrieval, holter, stats, quality):
        items = []
        if mode in {"2min", "5min"} and not holter["available"]: items.append("Long-recording mode has no validated temporal-classifier output.")
        if not retrieval["neighbors"]: items.append("No V7 retrieval neighbours were available.")
        if not stats["overall"].get("calculated_metrics"): items.append("Complete-recording statistical measurements were unavailable.")
        if quality["status"] != "ACCEPTABLE": items.append("Signal quality limits diagnostic support.")
        items.append("The bridge provides ECG decision support and does not establish symptoms, coronary anatomy, or treatment need.")
        return items


def validate_bridge_result(bridge: Any) -> Dict[str, Any]:
    """Fail closed unless the active result is a structurally valid V4 decision."""
    if not isinstance(bridge, dict):
        raise ValueError("Bridge V4 result must be a dictionary")
    if str(bridge.get("bridge_version", "")).split(".")[0] != "4":
        raise ValueError("Non-V4 bridge result rejected by the active V4 validator")
    required = ("decision_status", "supported_findings", "uncertain_findings", "decision_policy", "provenance")
    missing = [key for key in required if key not in bridge]
    if missing:
        raise ValueError(f"Bridge V4 result missing required fields: {missing}")
    if bridge.get("decision_policy", {}).get("llm_can_modify") is not False:
        raise ValueError("Bridge V4 decision policy must prohibit LLM modification")
    validated = dict(bridge)
    validated.setdefault("decision", "Unknown")
    validated.setdefault("primary_label", "UNKNOWN")
    validated.setdefault("primary_family", "UNKNOWN")
    validated.setdefault("partially_supported_findings", [])
    validated.setdefault("confidence", {})
    validated.setdefault("contradictions", [])
    validated.setdefault("ood", {"enabled": False, "ood_status": "not_used_v4"})
    return validated

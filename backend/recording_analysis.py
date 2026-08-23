"""Recording-level orchestration over the validated 10-second TRACE pipeline."""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from backend.bridge.ecg_statistics import build_segment_statistics, extract_ecg_statistics
from backend.bridge.evidence_bridge_v4 import validate_bridge_result
from backend.recording_store import TemporaryRecordingStore
from long_recording_v2.inference import (
    ExperimentalHolterLocalizer,
    RecordingSelectionConfig,
    apply_manual_selection,
    canonical_recording,
    coarse_starts,
    merge_selected_episodes,
    refinement_starts,
    score_embedding_states,
    validate_mode_duration,
    window_at,
)
from preprocessing.ecg_preprocessor import preprocess_raw_ecg
from deployment_config import SCP_THRESHOLDS_PATH


class RecordingAnalyzer:
    """Select windows cheaply, then run expensive inference only where useful."""

    def __init__(self, diagnosis_model: Any, config: RecordingSelectionConfig = RecordingSelectionConfig()):
        self.diagnosis_model = diagnosis_model
        self.pipeline = diagnosis_model.pipeline
        self.bridge = diagnosis_model.bridge
        self.config = config
        raw_thresholds = json.loads(SCP_THRESHOLDS_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw_thresholds, dict) or not raw_thresholds:
            raise RuntimeError("Validated SCP threshold manifest is empty or invalid")
        self.scp_thresholds = {str(label): float(value) for label, value in raw_thresholds.items()}
        self.holter = self._load_optional_holter()
        self.recording_store = TemporaryRecordingStore()

    def _load_optional_holter(self) -> Optional[ExperimentalHolterLocalizer]:
        enabled = os.environ.get("TRACE_ENABLE_EXPERIMENTAL_HOLTER", "0").lower() in ("1", "true", "yes")
        if not enabled:
            return None
        configured = os.environ.get(
            "TRACE_HOLTER_CHECKPOINT",
            "outputs/long_recording_v2/incart_classifier_localizer/best_classifier.pt",
        )
        path = Path(configured)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        if not path.is_file():
            raise FileNotFoundError(f"Experimental Holter checkpoint missing: {path}")
        return ExperimentalHolterLocalizer(str(path), self.pipeline.device)

    def _encode_windows(self, recording: np.ndarray, starts: Sequence[float]) -> Dict[str, Any]:
        raw_windows = [window_at(recording, start, self.config.target_rate_hz, self.config.window_seconds) for start in starts]
        processed = []
        quality = []
        statistics = []
        for raw in raw_windows:
            prep = preprocess_raw_ecg(raw, sampling_rate=self.config.target_rate_hz)
            processed.append(np.asarray(prep["signal"], dtype=np.float32))
            quality.append(float(prep["quality"].get("overall_quality_score", 0.0)))
            statistics.append(extract_ecg_statistics(raw, self.config.target_rate_hz, scope="selection_window"))
        tensor = torch.from_numpy(np.stack(processed)).to(self.pipeline.device)
        with torch.inference_mode():
            embeddings = self.pipeline.retrieval_encoder(tensor).detach().cpu().numpy().astype(np.float32)
        return {"raw_windows": raw_windows, "processed_windows": processed, "embeddings": embeddings, "quality": quality, "statistics": statistics}

    @staticmethod
    def _top_predictions(probabilities: Dict[str, float], limit: int = 5, key: str = "label") -> List[Dict[str, Any]]:
        return [{key: name, "probability": float(value)} for name, value in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:limit]]

    def _aggregate_branch(self, heavy: Sequence[Dict[str, Any]], diagnostic_indices: set[int]) -> Dict[str, Any]:
        diagnostic = [row for row in heavy if row["window_index"] in diagnostic_indices]
        source = diagnostic or list(heavy)
        if not source:
            raise RuntimeError("No windows were available for recording aggregation")

        label_names = source[0]["result"]["classifier"]["probabilities"].keys()
        family_names = source[0]["result"]["family_head"]["probabilities"].keys()
        reducer = np.max if diagnostic else np.mean
        label_probs = {
            name: float(reducer([row["result"]["classifier"]["probabilities"][name] for row in source]))
            for name in label_names
        }
        family_probs = {
            name: float(reducer([row["result"]["family_head"]["probabilities"][name] for row in source]))
            for name in family_names
        }
        primary_label = max(label_probs, key=label_probs.get)
        primary_family = self.pipeline.family_mapper.get_family_name(primary_label)
        independent_family = max(family_probs, key=family_probs.get)
        classifier = {
            "probabilities": label_probs,
            "primary_label": primary_label,
            "primary_probability": label_probs[primary_label],
            "mapped_family": primary_family,
            "top_predictions": self._top_predictions(label_probs),
        }
        family = {
            "probabilities": family_probs,
            "primary_family": independent_family,
            "primary_probability": family_probs[independent_family],
            "top_predictions": self._top_predictions(family_probs, key="family"),
        }
        return {"classifier": classifier, "family_head": family, "source_window_count": len(source), "aggregation": "max" if diagnostic else "mean_stable"}

    @staticmethod
    def _merge_retrieval(heavy: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        neighbors = []
        verified = []
        for row in heavy:
            retrieval = row["result"].get("retrieval", {})
            provenance = retrieval.get("retrieval_provenance", {})
            verified.append(provenance.get("verification_status") == "verified")
            for item in retrieval.get("raw_neighbors", []):
                enriched = dict(item)
                enriched["query_window_index"] = row["window_index"]
                enriched["query_start_seconds"] = row["start_seconds"]
                neighbors.append(enriched)
        neighbors.sort(key=lambda x: float(x.get("raw_similarity", 0.0)), reverse=True)
        unique = []
        seen = set()
        for item in neighbors:
            # A clinical neighbor is one reference ECG. The same ECG retrieved
            # by several query windows must appear once, at its best similarity.
            key = item.get("ecg_id")
            if key not in seen:
                seen.add(key); unique.append(item)
        return {
            "raw_neighbors": unique[:20],
            "available": bool(unique),
            "retrieval_provenance": {
                "verification_status": "verified" if verified and all(verified) else "unverified",
                "reason": "Merged only from selected 10-second V7 queries.",
            },
        }

    def _recording_findings(self, heavy: Sequence[Dict[str, Any]], diagnostic_indices: set[int]) -> List[Dict[str, Any]]:
        """Retain which reviewed windows support each multi-label finding."""
        findings: Dict[str, Dict[str, Any]] = {}
        for row in heavy:
            probabilities = row["result"].get("classifier", {}).get("probabilities", {})
            for label, probability in probabilities.items():
                threshold = float(self.scp_thresholds.get(label, 0.5))
                if float(probability) < threshold:
                    continue
                item = findings.setdefault(label, {
                    "label": label,
                    "family": self.pipeline.family_mapper.get_family_name(label),
                    "maximum_probability": 0.0,
                    "supporting_window_indices": [],
                    "diagnostic_window_indices": [],
                    "stable_reference_window_indices": [],
                })
                item["maximum_probability"] = max(item["maximum_probability"], float(probability))
                item["supporting_window_indices"].append(int(row["window_index"]))
                target = "diagnostic_window_indices" if row["window_index"] in diagnostic_indices else "stable_reference_window_indices"
                item[target].append(int(row["window_index"]))
        total_diagnostic = max(1, len(diagnostic_indices))
        for item in findings.values():
            diagnostic_count = len(set(item["diagnostic_window_indices"]))
            item["diagnostic_prevalence"] = round(diagnostic_count / total_diagnostic, 4)
            item["temporal_pattern"] = (
                "persistent_across_selected_episodes" if diagnostic_count >= 2
                else "localized_to_one_selected_episode" if diagnostic_count == 1
                else "stable_reference_only"
            )
            for key in ("supporting_window_indices", "diagnostic_window_indices", "stable_reference_window_indices"):
                item[key] = sorted(set(item[key]))
        return sorted(findings.values(), key=lambda x: x["maximum_probability"], reverse=True)

    def _persistent_abnormal_baseline(
        self, heavy: Sequence[Dict[str, Any]], stable_reference: set[int]
    ) -> Dict[str, Any]:
        """Identify consistently abnormal sampled baselines when no transition is selected.

        This does not claim that every unobserved instant is abnormal.  It marks
        the reviewed stable-reference windows as persistent-baseline candidates
        only when the classifier is repeatedly strong and V7 retrieval supplies
        independent label support.
        """
        sampled = [row for row in heavy if int(row["window_index"]) in stable_reference]
        if len(sampled) < 2:
            return {"detected": False, "labels": [], "window_indices": [], "reason": "fewer_than_two_stable_samples"}
        classifier_counts: Dict[str, int] = {}
        retrieval_counts: Dict[str, int] = {}
        supporting_indices: Dict[str, set[int]] = {}
        for row in sampled:
            index = int(row["window_index"])
            probabilities = row["result"].get("classifier", {}).get("probabilities", {})
            for label, probability in probabilities.items():
                label = str(label).upper()
                threshold = max(0.50, float(self.scp_thresholds.get(label, 0.5)))
                if label != "NORM" and float(probability) >= threshold:
                    classifier_counts[label] = classifier_counts.get(label, 0) + 1
                    supporting_indices.setdefault(label, set()).add(index)
            seen_retrieval = set()
            for neighbor in row["result"].get("retrieval", {}).get("raw_neighbors", [])[:5]:
                if float(neighbor.get("raw_similarity", neighbor.get("similarity", 0.0)) or 0.0) < 0.20:
                    continue
                seen_retrieval.update(str(value).upper() for value in (neighbor.get("scp_codes") or neighbor.get("labels") or []) if value)
            for label in seen_retrieval:
                retrieval_counts[label] = retrieval_counts.get(label, 0) + 1

        required = max(2, int(np.ceil(0.75 * len(sampled))))
        labels = sorted(
            label for label, count in classifier_counts.items()
            if count >= required and retrieval_counts.get(label, 0) >= 1
        )
        indices = sorted(set().union(*(supporting_indices[label] for label in labels))) if labels else []
        return {
            "detected": bool(labels),
            "labels": labels,
            "window_indices": indices,
            "classifier_required_samples": required,
            "stable_samples_reviewed": len(sampled),
            "interpretation": (
                "Repeated abnormal evidence in stable sampled windows; this may represent a persistent baseline rather than a transient state change."
                if labels else "No persistent abnormal baseline met classifier-plus-retrieval requirements."
            ),
            "whole_recording_claim": False,
        }

    @staticmethod
    def _clinical_references(knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
        references = []
        seen = set()
        for citation in knowledge.get("permitted_citations", []):
            key = (citation.get("source_title") or citation.get("source"), citation.get("section"), citation.get("page"))
            if key in seen:
                continue
            seen.add(key)
            references.append({
                "title": citation.get("source_title") or citation.get("source") or citation.get("title"),
                "organization_or_authors": citation.get("organization_or_authors"),
                "year": citation.get("date_or_version"),
                "section": citation.get("section"),
                "page_or_locator": citation.get("page"),
                "doi": citation.get("doi"),
                "url": citation.get("url"),
                "document_sha256": citation.get("document_sha256"),
                "internal_citation_id": citation.get("citation_id"),
            })
        return references

    @staticmethod
    def _bridge_validation_summary() -> Dict[str, Any]:
        path = Path(__file__).resolve().parents[1] / "outputs" / "deployment_audits" / "v7_bridge_full_fold10" / "summary.json"
        if not path.is_file():
            return {"status": "unavailable", "scope": "10-second PTB-XL fold-10 only"}
        report = json.loads(path.read_text(encoding="utf-8"))
        return {
            "status": report.get("status"),
            "scope": "10-second PTB-XL fold-10 patient-held-out; not a validation of 2/5-minute recording decisions",
            "records": report.get("records"),
            "primary_label_accuracy": report.get("primary_label_accuracy"),
            "top3_diagnostic_recall": report.get("top3_diagnostic_recall"),
            "top5_diagnostic_recall": report.get("top5_diagnostic_recall"),
            "retrieval_exact_label_recall_at_5": report.get("retrieval_exact_label_recall_at_5"),
            "supported_finding_micro_precision": report.get("supported_finding_micro_precision"),
            "supported_finding_micro_recall": report.get("supported_finding_micro_recall"),
            "decision_coverage": report.get("decision_coverage"),
            "covered_primary_error_rate": report.get("covered_primary_error_rate"),
            "family_set_f1": report.get("family_set_f1_from_macro_pr"),
            "calibration_ece": (report.get("decision_calibration_all") or {}).get("ece"),
            "limitations": [
                "Recording-level temporal selection has not yet been evaluated on a patient-held-out long-recording test set.",
                "Bridge confidence is not a calibrated clinical probability.",
                "Rare-label and family-set performance remain weak for several labels.",
            ],
        }

    def analyze(
        self,
        signal: Any,
        sampling_rate_hz: int,
        mode: str,
        manual_window_indices: Optional[Iterable[int]] = None,
        lead_names: Optional[Sequence[str]] = None,
        top_k: int = 5,
        cache_lookup: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        recording = canonical_recording(signal, sampling_rate_hz, lead_names, self.config.target_rate_hz)
        validate_mode_duration(recording, mode, self.config.target_rate_hz)
        duration = recording.shape[1] / self.config.target_rate_hz
        acquisition = self.recording_store.save(recording, {
            "recording_mode": mode,
            "sampling_rate_hz": self.config.target_rate_hz,
            "duration_seconds": duration,
            "lead_order": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            "preprocessing_state": "canonical_resampled_recording_before_window_normalization",
        })

        starts = coarse_starts(duration, self.config.window_seconds, self.config.coarse_stride_seconds)
        first = self._encode_windows(recording, starts)
        temporal, temporal_provenance = (None, {"status": "disabled", "diagnostic_authority": False})
        if self.holter is not None:
            temporal, temporal_provenance = self.holter.probabilities(first["embeddings"])
        coarse_scores = score_embedding_states(first["embeddings"], first["statistics"], first["quality"], temporal, self.config)

        added = refinement_starts(starts, coarse_scores["candidate_mask"], duration, self.config)
        starts = sorted(set(starts + added))
        encoded = self._encode_windows(recording, starts)
        temporal = None
        if self.holter is not None:
            temporal, temporal_provenance = self.holter.probabilities(encoded["embeddings"])
        scores = score_embedding_states(encoded["embeddings"], encoded["statistics"], encoded["quality"], temporal, self.config)

        windows: List[Dict[str, Any]] = []
        for index, start in enumerate(starts):
            artifact = bool(scores["artifact_mask"][index])
            automatic = bool(scores["candidate_mask"][index])
            windows.append({
                "window_index": index,
                "start_seconds": float(start),
                "end_seconds": float(start + self.config.window_seconds),
                "coarse": bool(start % self.config.coarse_stride_seconds == 0),
                "refined": bool(start in added),
                "status": "artifact_or_low_quality" if artifact else ("candidate" if automatic else "stable"),
                "automatic_selected": automatic,
                "selected": automatic,
                "selection_source": "automatic" if automatic else "none",
                "overrides_automatic_selector": False,
                "combined_score": float(scores["combined_score"][index]),
                "transition_distance": float(scores["transition_distance"][index]),
                "centroid_distance": float(scores["centroid_distance"][index]),
                "temporal_probability": float(scores["temporal_probability"][index]),
                "morphology_deviation": float(scores["morphology_deviation"][index]),
                "quality_score": float(scores["quality_score"][index]),
            })
        apply_manual_selection(windows, manual_window_indices)
        episodes = merge_selected_episodes(windows, self.config.maximum_auto_episodes)

        representative = {int(ep["representative_window_index"]) for ep in episodes}
        manual = {int(w["window_index"]) for w in windows if w["selection_source"] == "clinician"}
        diagnostic_indices = representative | manual
        stable_indices = [i for i, w in enumerate(windows) if w["status"] == "stable"]
        stable_indices.sort(key=lambda i: float(scores["centroid_distance"][i]))
        stable_reference = set(stable_indices[:self.config.stable_representatives])
        heavy_indices = sorted(diagnostic_indices | stable_reference)

        heavy = []
        for index in heavy_indices:
            if cache_lookup and index in cache_lookup:
                result = cache_lookup[index]
            else:
                result = self.pipeline.run(
                    encoded["raw_windows"][index],
                    metadata={"sampling_rate_hz": self.config.target_rate_hz, "top_k": top_k},
                    include_retrieval=True,
                    top_k=top_k,
                )
            heavy.append({
                "window_index": index,
                "start_seconds": windows[index]["start_seconds"],
                "end_seconds": windows[index]["end_seconds"],
                "role": "diagnostic" if index in diagnostic_indices else "stable_reference",
                "selection_source": windows[index]["selection_source"],
                "result": result,
            })

        persistent_baseline = {"detected": False, "labels": [], "window_indices": []}
        if not diagnostic_indices:
            persistent_baseline = self._persistent_abnormal_baseline(heavy, stable_reference)
            if persistent_baseline["detected"]:
                persistent_indices = {int(value) for value in persistent_baseline["window_indices"]}
                diagnostic_indices.update(persistent_indices)
                stable_reference.difference_update(persistent_indices)
                for index in persistent_indices:
                    windows[index]["selected"] = True
                    windows[index]["status"] = "persistent_abnormal_baseline_candidate"
                    windows[index]["selection_source"] = "persistent_baseline_classifier_plus_retrieval"
                for row in heavy:
                    if int(row["window_index"]) in persistent_indices:
                        row["role"] = "persistent_abnormal_baseline"
                        row["selection_source"] = "persistent_baseline_classifier_plus_retrieval"
                stable_indices = [index for index in stable_indices if index not in persistent_indices]
                episodes = merge_selected_episodes(windows, self.config.maximum_auto_episodes)

        aggregate = self._aggregate_branch(heavy, diagnostic_indices)
        retrieval = self._merge_retrieval(heavy)
        recording_findings = self._recording_findings(heavy, diagnostic_indices)
        decision_embedding_indices = sorted(diagnostic_indices or stable_reference)
        decision_embedding = np.asarray(encoded["embeddings"])[decision_embedding_indices].mean(axis=0)
        decision_embedding /= max(float(np.linalg.norm(decision_embedding)), 1e-12)
        abnormal_segments = [
            {
                "ecg": encoded["raw_windows"][index],
                "window_index": index,
                "start_seconds": windows[index]["start_seconds"],
                "end_seconds": windows[index]["end_seconds"],
                "abnormal_probability": windows[index]["combined_score"],
            }
            for index in sorted(diagnostic_indices)
        ]
        overall_quality = float(np.mean([w["quality_score"] for w in windows])) if windows else 0.0
        bridge = self.bridge.process(
            classifier_output=aggregate["classifier"],
            family_head_output=aggregate["family_head"],
            retrieval_output=retrieval,
            model_embedding=decision_embedding,
            raw_ecg=recording,
            signal_quality_score=overall_quality,
            sampling_rate_hz=self.config.target_rate_hz,
            abnormal_segments=abnormal_segments,
            retrieval_query_plan={
                "strategy": "selected_abnormal_episode_representatives_plus_stable_reference",
                "diagnostic_window_indices": sorted(diagnostic_indices),
                "stable_reference_window_indices": sorted(stable_reference),
                "full_recording_embedding_not_indexed": True,
            },
            holter_output={"recording_probabilities": temporal} if isinstance(temporal, dict) else {},
            selected_windows=[{
                "window_id": str(index),
                "start_seconds": windows[index]["start_seconds"],
                "end_seconds": windows[index]["end_seconds"],
                "role": "diagnostic" if index in diagnostic_indices else "stable_reference",
                "selected": index in decision_embedding_indices,
            } for index in decision_embedding_indices],
            metadata={"mode": mode, "window_duration_seconds": self.config.window_seconds},
        )
        bridge = validate_bridge_result(bridge)
        knowledge = self.diagnosis_model.knowledge_retriever.retrieve(
            question="Summarize the supported recording-level findings and abnormal episodes.",
            bridge_result=bridge,
            classifier_result=aggregate["classifier"],
            family_result=aggregate["family_head"],
            contradictions=bridge.get("contradictions", []),
            top_k=6,
            preferred_sections=["ecg_diagnostic_criteria", "differential_diagnosis", "risk_red_flags"],
        )
        if not isinstance(knowledge, dict):
            knowledge = {"all_chunks": [], "permitted_citations": []}
        clinical_references = self._clinical_references(knowledge)
        bridge["recording_evidence"] = {
            "finding_count": len(recording_findings),
            "findings": recording_findings,
            "selected_episode_count": len(episodes),
            "diagnostic_window_indices": sorted(diagnostic_indices),
            "stable_reference_window_indices": sorted(stable_reference),
            "aggregation_policy": "per-label maximum across selected episodes; stable mean only when no episode is selected",
        }
        bridge["knowledge_evidence"] = {
            "status": "available" if clinical_references else "unavailable",
            "reference_count": len(clinical_references),
            "references": clinical_references,
            "unverified_sources_excluded": True,
        }
        bridge["function_audit"] = {
            "required_branches": ["statistics", "classifier", "family", "retrieval", "knowledge"],
            "available_branches": [
                name for name, available in {
                    "statistics": bool(abnormal_segments or recording.size),
                    "classifier": bool(aggregate["classifier"].get("probabilities")),
                    "family": bool(aggregate["family_head"].get("probabilities")),
                    "retrieval": bool(retrieval.get("raw_neighbors")),
                    "knowledge": bool(clinical_references),
                }.items() if available
            ],
            "selected_windows_reached_classifier": all("classifier" in row["result"] for row in heavy),
            "selected_windows_reached_retrieval": all("retrieval" in row["result"] for row in heavy),
            "multilabel_window_attribution_preserved": True,
            "experimental_holter_diagnostic_authority": False,
            "validation": self._bridge_validation_summary(),
        }
        final_decision = {
            "decision": bridge.get("decision", "indeterminate"),
            "status": bridge.get("decision", "indeterminate"),
            "primary_label": bridge.get("primary_label", "UNKNOWN"),
            "supported_labels": list(bridge.get("supported_findings", [])),
            "partially_supported_labels": list(bridge.get("partially_supported_findings", [])),
            "secondary_findings": list(bridge.get("partially_supported_findings", [])),
            "recording_findings": recording_findings,
            "selected_episode_count": len(episodes),
            "confidence": float(bridge.get("confidence", {}).get("final_fused_confidence", 0.0)),
            "confidence_level": bridge.get("confidence", {}).get("confidence_level", "MODERATE"),
            "confidence_reasons": bridge.get("confidence", {}).get("confidence_drivers", []),
            "contradictions": bridge.get("contradictions", []),
            "quality_constraints": bridge.get("confidence", {}).get("limitations", []),
            "clinical_references": clinical_references,
            "requires_clinician_review": True,
            "scope": "ECG decision-support finding; not a complete clinical diagnosis or treatment decision",
        }

        # Build clean Pydantic/contract-friendly entities
        recording_id = acquisition["recording_id"]
        recording_meta = {
            "mode": mode,
            "duration_seconds": duration,
            "sampling_rate_hz": self.config.target_rate_hz,
            "lead_count": int(recording.shape[0])
        }
        quality_payload = {
            "overall_quality_score": overall_quality,
            "quality_status": "ACCEPTABLE" if overall_quality >= self.config.quality_gate else "DEGRADED",
            "warnings": bridge.get("confidence", {}).get("limitations", [])
        }
        stable_reference_windows = [
            {
                "window_index": index,
                "start_seconds": windows[index]["start_seconds"],
                "end_seconds": windows[index]["end_seconds"]
            }
            for index in sorted(stable_reference)
        ]
        
        selected_windows_detailed = []
        for row in heavy:
            if row["role"] == "diagnostic" or row["role"] == "persistent_abnormal_baseline":
                idx = row["window_index"]
                win_data = windows[idx]
                res = row["result"]
                probs = res.get("classifier", {}).get("probabilities", {})
                primary_label = res.get("classifier", {}).get("primary_label", "UNKNOWN")
                confidence = float(probs.get(primary_label, 0.0)) if primary_label != "UNKNOWN" else 0.0
                selected_windows_detailed.append({
                    "window_index": idx,
                    "screening": {
                        "status": win_data["status"],
                        "combined_score": win_data["combined_score"],
                        "selected": win_data["selected"]
                    },
                    "diagnostic": {
                        "primary_label": primary_label,
                        "supported_labels": res.get("classifier", {}).get("top_predictions", []),
                        "probabilities": probs,
                        "confidence": confidence
                    },
                    "statistics": encoded["statistics"][idx] if idx < len(encoded["statistics"]) else {},
                    "retrieval": res.get("retrieval", {})
                })

        statistics_payload = {
            "whole_recording": build_segment_statistics(recording, self.config.target_rate_hz, abnormal_segments)
        }

        rhythm_payload = {
            "rhythm_label": "HOLTER_LOCALIZATION" if self.holter is not None else "NORMAL_RHYTHM",
            "rhythm_confidence": float(temporal.max()) if temporal is not None else 1.0,
            "availability": self.holter is not None,
            "experimental_status": True,
            "provenance": temporal_provenance
        }

        bridge_contract = {
            "primary_label": bridge.get("primary_label", "UNKNOWN"),
            "supported_labels": list(bridge.get("supported_findings", [])),
            "confidence": float(bridge.get("confidence", {}).get("final_fused_confidence", 0.0)),
            "confidence_reasons": bridge.get("confidence", {}).get("confidence_drivers", []),
            "contradictions": bridge.get("contradictions", [])
        }

        return {
            "status": "ok",
            "result_type": "recording",
            "recording_id": recording_id,
            "recording_mode": mode,
            "duration_seconds": duration,
            "sampling_rate_hz": self.config.target_rate_hz,
            "processed_shape": list(recording.shape),
            "acquisition": acquisition,
            "recording": recording_meta,
            "quality": quality_payload,
            "selector": {
                "strategy": "coarse_to_fine_v7_state_change",
                "coarse_window_count": len(first["embeddings"]),
                "refined_window_count": len(added),
                "total_window_count": len(windows),
                "thresholds": scores["thresholds"],
                "holter": temporal_provenance,
                "holter_diagnostic_authority": False,
                "persistent_abnormal_baseline": persistent_baseline,
            },
            "windows": windows,
            "episodes": episodes,
            "stable_state": {
                "window_indices": stable_indices,
                "representative_window_indices": sorted(stable_reference),
                "embedding": scores["stable_centroid"].tolist(),
            },
            "stable_reference_windows": stable_reference_windows,
            "selected_windows": selected_windows_detailed,
            "processed_windows": heavy,
            "aggregate": aggregate,
            "recording_findings": recording_findings,
            "recording_statistics": statistics_payload["whole_recording"],
            "statistics": statistics_payload,
            "recording_bridge": bridge,
            "knowledge": knowledge,
            "clinical_references": clinical_references,
            "clinical_evidence": clinical_references,
            "final_diagnostic_decision": final_decision,
            "decision_status": bridge.get("decision"),
            "classifier": aggregate["classifier"],
            "family_head": aggregate["family_head"],
            "retrieval": retrieval,
            "rhythm": rhythm_payload,
            "bridge": bridge_contract,
            "confidence": bridge.get("confidence", {}),
            "signal_quality": {
                "overall_quality_score": overall_quality,
                "quality_status": "ACCEPTABLE" if overall_quality >= self.config.quality_gate else "DEGRADED",
            },
            "ecg_measurements": bridge.get("ecg_measurements", {}),
            "guardrails": {
                "experimental_holter_used_for_localization_only": self.holter is not None,
                "experimental_holter_created_diagnosis": False,
                "faiss_queries_are_10_second_windows_only": True,
                "manual_windows_are_explicitly_marked": True,
            },
        }

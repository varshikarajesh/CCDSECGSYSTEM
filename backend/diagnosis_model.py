"""Public real-execution deployment interface for the complete evidence pipeline."""
from __future__ import annotations

import hashlib
import json
import os
import time
import logging
from typing import Any, Dict, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

from backend.bridge.evidence_bridge_v4 import EvidenceBridgeV4, validate_bridge_result
from backend.diagnosis.external_dataset_adapter import ExternalDatasetAdapter
from backend.inference_pipeline import InferencePipeline
from deployment_config import TRACE_LLM_BACKEND, TRACE_LLM_MODE
from runtime.runtime_contracts import ConversationState, PromptPackage, make_json_safe
from utils.knowledge_retriever import KnowledgeRetriever
from utils.llm import get_llm_backend
from utils.question_router import route_question
from utils.response_validator import ResponseValidator


def derive_case_id(waveform: np.ndarray, sampling_rate_hz: int) -> str:
    """
    Derives a deterministic, canonical case ID and fingerprint from the input waveform bytes (Part D).
    """
    canonical = np.ascontiguousarray(waveform, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(canonical.tobytes())
    digest.update(str(int(sampling_rate_hz)).encode("ascii"))
    return f"ecg_{digest.hexdigest()[:24]}"


class DiagnosisModel:
    def __init__(self, config_path: Optional[str] = None):
        self.pipeline = InferencePipeline(config_path)
        self.bridge = EvidenceBridgeV4()
        self.external_adapter = ExternalDatasetAdapter()
        self.knowledge_retriever = KnowledgeRetriever()
        self.sessions: Dict[str, ConversationState] = {}

    def predict(
        self,
        ecg: Any,
        metadata: Optional[Dict[str, Any]] = None,
        include_retrieval: bool = True,
        include_knowledge: bool = True,
        include_explanation: bool = True,
        question: Optional[str] = None,
        conversation_id: Optional[str] = None,
        llm_mode: Optional[str] = None,
        llm_backend: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        meta = dict(metadata or {})
        meta["top_k"] = top_k
        case_id = conversation_id or str(meta.get("ecg_id", meta.get("case_id", "standard_case")))
        user_question = question or "What is the primary finding and diagnostic conclusion?"

        selected_llm_mode = (
            llm_mode
            or os.environ.get("TRACE_LLM_MODE")
            or TRACE_LLM_MODE
        ).strip().lower()

        selected_llm_backend = (
            llm_backend
            or os.environ.get("TRACE_LLM_BACKEND")
            or TRACE_LLM_BACKEND
        ).strip().lower()

        if selected_llm_mode not in {"disabled", "real"}:
            selected_llm_mode = "disabled"
        if selected_llm_backend not in {"llama_cpp", "transformers_peft"}:
            selected_llm_backend = "llama_cpp"

        ecg_arr = np.asarray(ecg, dtype=np.float32).copy()
        ecg = ecg_arr
        adaptation = None
        dataset_name = meta.get("external_dataset") or meta.get("dataset_name")
        if dataset_name:
            ecg, adaptation = self.external_adapter.adapt(
                np.asarray(ecg),
                dataset_name=str(dataset_name),
                original_sampling_rate=int(meta.get("sampling_rate_hz", meta.get("sampling_rate", 0))),
                lead_names=meta.get("lead_names"),
            )
            meta["sampling_rate_hz"] = 100
            meta["external_adaptation"] = adaptation

        # Case ID derivation & isolation (Part D)
        sampling_rate = int(meta.get("sampling_rate_hz", meta.get("sampling_rate", 100)))
        waveform_fingerprint = derive_case_id(ecg, sampling_rate)

        if not case_id or str(case_id).strip() in ("standard_case", "chat_session", "", "None"):
            case_id = waveform_fingerprint
        else:
            case_id = str(case_id).strip()

        # 1. Deterministic ECG pipeline execution
        base = self.pipeline.run(ecg, metadata=meta, include_retrieval=include_retrieval, top_k=top_k)
        classifier = base["classifier"]
        family_head = base["family_head"]

        # Validate classifier source of truth (Part B4)
        from utils.evidence_fields import resolve_primary_probability, validate_classifier_evidence
        validate_classifier_evidence(classifier)

        quality = base.get("signal_quality", {})
        quality_score = float(quality.get("overall_quality_score", 0.0))

        # Handle include_retrieval=False safely
        retrieval_data = base["retrieval"]
        if not include_retrieval:
            retrieval_data = {
                "raw_neighbors": [],
                "query_embedding_checksum": "RETRIEVAL_DISABLED",
                "retrieval_disabled": True,
            }

        # 2. Authoritative Evidence Bridge V4 + fail-closed contract validation
        raw_bridge = self.bridge.process(
            classifier_output=classifier,
            family_head_output=family_head,
            retrieval_output=retrieval_data,
            model_embedding=np.asarray(base["model_embedding"], dtype=np.float32),
            raw_ecg=np.asarray(ecg, dtype=np.float32),
            signal_quality_score=quality_score,
            sampling_rate_hz=int(meta.get("sampling_rate_hz",meta.get("sampling_rate",100))),
        )
        bridge = validate_bridge_result(raw_bridge)


        from backend.bridge.decision_reasoning import build_decision_reasoning
        from utils.evidence_fields import resolve_primary_probability

        decision_reasoning_payload = build_decision_reasoning(
            bridge_result=bridge,
            classifier_result=classifier,
            family_result=family_head,
            retrieval_result=retrieval_data,
            signal_quality=quality,
        )

        prob_resolved = resolve_primary_probability(bridge, classifier)

        primary_candidate_payload = {
            "label": classifier.get("primary_label", "Unknown"),
            "probability": prob_resolved,
            "supported_as_final": bool(bridge.get("decision") != "Unknown"),
        }

        # 1.5 Deep-copy upstream snapshot for structural immutability check (Section 6)
        import copy
        upstream_snapshot = {
            "classifier": copy.deepcopy(classifier),
            "family_head": copy.deepcopy(family_head),
            "retrieval": copy.deepcopy(retrieval_data),
            "confidence": copy.deepcopy(bridge.get("confidence")),
            "decision": copy.deepcopy(bridge.get("decision")),
        }

        # Base structured result
        structured_result = {
            "status": "ok",
            "decision_status": bridge.get("decision", "Unknown"),
            "primary_candidate": primary_candidate_payload,
            "decision_reasoning": decision_reasoning_payload,
            "classifier": classifier,
            "classifier_results": bridge.get("classifier_results", []),
            "family_head": family_head,
            "family_results": bridge.get("family_results", []),
            "retrieval": retrieval_data,
            "retrieval_status": bridge.get("retrieval_status", {}),
            "reranking": {
                "top_5": bridge.get("reranked_neighbors", []),
                "trace": bridge.get("rerank_trace", []),
            },
            "retrieval_quality": bridge.get("retrieval_quality", {}),
            "rare_case": bridge.get("rare_case", {}),
            "bridge": bridge,
            "confidence": bridge.get("confidence", {}),
            "signal_quality": quality,
            "ecg_measurements":bridge.get("ecg_measurements",{}),
            "evidence_branches":bridge.get("evidence_branches",{}),
            "unknown_reasons": bridge.get("unknown_reasons", []),
            "requirements_for_stronger_conclusion": bridge.get("requirements_for_stronger_conclusion", []),
            "preprocessing": base.get("preprocessing", {}),
            "external_adaptation": adaptation,
            "metadata": base.get("metadata", {}),
        }

        # 3. Handle include_explanation=False
        if not include_explanation:
            explanation_payload = {
                "text": "[Explanation Disabled]",
                "status": "disabled",
                "explanation_disabled": True,
                "answer_source": "disabled",
                "intent": "none",
                "sub_intent": None,
                "patient_specific": False,
                "question_received": bool(user_question),
                "generation_attempted": False,
                "generation_succeeded": False,
                "repair_attempted": False,
                "repair_succeeded": False,
                "repair_backend": None,
                "repair_adapter_used": False,
                "repair_adapter_active": False,
                "fallback_used": False,
                "fallback_reason": "explanation_disabled",
                "initial_guardrail_violations": [],
                "repair_guardrail_violations": [],
                "guardrail_status": "skipped",
                "guardrail_violations": [],
                "citations": [],
                "citation_status": "unavailable",
                "citation_reason": "Explanation disabled",
                "backend": "disabled",
                "skills_activated": [],
            }
            runtime_provenance = {
                "real_classifier_checkpoint": True,
                "real_retrieval_checkpoint": True,
                "real_faiss_search": include_retrieval,
                "mock_neighbors_used": False,
                "ood_input": bridge.get("ood", {}).get("input_source"),
                "external_adapter_used": adaptation is not None,
                "llm_mode": selected_llm_mode,
                "requested_llm_backend": selected_llm_backend,
                "effective_llm_backend": "disabled",
                "base_model_used": False,
                "fine_tuned_adapter_used": False,
                "fallback_used": True,
                "explanation_disabled": True,
                "knowledge_available": False,
                "skills_activated": [],
                "citations_validated": True,
                "audit_id": f"audit_{hashlib.sha256((case_id + str(time.time())).encode('utf-8')).hexdigest()[:16]}",
            }
            res = dict(structured_result)
            res.update({
                "decision_status": bridge.get("decision", "Unknown"),
                "primary_candidate": primary_candidate_payload,
                "decision_reasoning": decision_reasoning_payload,
                "structured_result": structured_result,
                "explanation": explanation_payload,
                "provenance": runtime_provenance,
            })
            return make_json_safe(res)

        # 4. Intent Routing
        question_route_obj = route_question(user_question, self.sessions.get(case_id))
        intent = question_route_obj.intent

        # 5. Knowledge Retrieval (Honor include_knowledge flag)
        knowledge_res: Dict[str, Any] = {"all_chunks": [], "permitted_citations": []}
        if include_knowledge:
            ret_output = self.knowledge_retriever.retrieve(
                question=user_question,
                bridge_result=bridge,
                classifier_result=classifier,
                family_result=family_head,
                rare_case_result=bridge.get("rare_case"),
                contradictions=bridge.get("contradictions", []),
                top_k=6,
                preferred_sections=question_route_obj.preferred_sections,
            )
            if isinstance(ret_output, dict):
                knowledge_res = ret_output
        knowledge_branch=bridge.setdefault("evidence_branches",{}).setdefault("knowledge",{})
        knowledge_branch.update({"status":"available" if knowledge_res.get("all_chunks") else "unavailable","chunk_count":len(knowledge_res.get("all_chunks",[])),"permitted_citation_count":len(knowledge_res.get("permitted_citations",[])),"citation_ids":[c.get("citation_id") for c in knowledge_res.get("permitted_citations",[]) if isinstance(c,dict) and c.get("citation_id")]})

        # 6. Question routing only. Legacy Markdown skill injection is disabled.
        route_payload = dict(question_route_obj.__dict__)

        # 7. Case-Scoped Conversation Memory
        if case_id not in self.sessions:
            if len(self.sessions) > 50:
                self.sessions.clear()
            self.sessions[case_id] = ConversationState(case_id=case_id)

        conv_state = self.sessions[case_id]
        if conv_state.case_id != case_id:
            conv_state.reset_case(new_case_id=case_id)

        primary_label = bridge.get("primary_label") or "UNKNOWN"
        primary_family = bridge.get("primary_family") or "UNKNOWN"

        conv_state.bridge_conclusion = {"decision_status": bridge.get("decision"), "primary_label": primary_label, "primary_family": primary_family}
        conv_state.primary_findings = [primary_label] if primary_label != "UNKNOWN" else []
        conv_state.secondary_findings = bridge.get("supported_findings", [])
        conv_state.ood_state = bridge.get("ood", {})
        conv_state.signal_quality_limitations = quality.get("warnings", [])
        conv_state.contradictions = bridge.get("contradictions", [])
        conv_state.confidence_level = bridge.get("confidence", {}).get("confidence_level", "MODERATE")
        conv_state.confidence_drivers = bridge.get("confidence", {}).get("confidence_drivers", [])
        conv_state.rare_case_state = bridge.get("rare_case", {})
        conv_state.supplied_knowledge_chunks = knowledge_res.get("all_chunks", [])
        conv_state.permitted_citation_ids = [c["citation_id"] for c in knowledge_res.get("permitted_citations", []) if isinstance(c, dict) and "citation_id" in c]

        # 8. Prompt Construction
        prompt_pkg = PromptPackage(
            question=user_question,
            patient_context=dict(meta.get("patient_context") or {}),
            ecg_statistics=dict(bridge.get("ecg_measurements") or {}),
            temporal_summary=dict(meta.get("temporal_summary") or {}),
            ecg_comparison=dict(meta.get("ecg_comparison") or {}),
            conversation_state={
                "case_id": case_id,
                "history": list(conv_state.history),
            },
            skill_route=route_payload,
            skill_instructions=[],
            classifier=classifier,
            family_head=family_head,
            raw_retrieval=retrieval_data,
            reranked_retrieval={
                "top_5": bridge.get("reranked_neighbors", []),
                "trace": bridge.get("rerank_trace", []),
            },
            retrieval_quality=bridge.get("retrieval_quality", {}),
            signal_quality=quality,
            ood={},
            rare_case=bridge.get("rare_case", {}),
            contradictions=bridge.get("contradictions", []),
            confidence=bridge.get("confidence", {}),
            bridge=bridge,
            knowledge_chunks=knowledge_res.get("all_chunks", []),
            permitted_citations=knowledge_res.get("permitted_citations", []),
            ontology_context=self.knowledge_retriever.ontology,
            condition_cards_context=self.knowledge_retriever.condition_cards,
        )

        from prompt_builder.gemma_prompt_builder import build_gemma_prompt
        # V4 uses one permanent system policy. Legacy Markdown skills are not
        # injected because they contained obsolete V3/OOD and treatment rules.
        prompt_pkg.skill_instructions = []
        prompt_text = build_gemma_prompt(prompt_pkg, max_prompt_tokens=1900)
        # Deterministically gather only question-relevant read-only evidence.
        # Gemma receives tool results but cannot execute writes or alter Bridge.
        from prompt_builder.system_runtime import (
            compile_question_tool_context,
            system_prompt_metadata,
        )
        analytical_state = {
            "recording": {"case_id": case_id},
            "bridge": bridge,
            "signal_quality": quality,
            "statistics": bridge.get("ecg_measurements", {}),
            "retrieval": {
                "neighbors": bridge.get("reranked_neighbors", []) or retrieval_data.get("neighbors", []),
                "quality": bridge.get("retrieval_quality", {}),
            },
            "temporal_summary": bridge.get("temporal_summary", {}),
            "abnormal_windows": bridge.get("abnormal_windows", []),
            "stable_reference_windows": bridge.get("stable_reference_windows", []),
            "windows": bridge.get("windows", {}),
            "knowledge_chunks": knowledge_res.get("all_chunks", []),
            "pipeline_status": {
                "classifier": "complete",
                "retrieval": "complete" if include_retrieval else "not_requested",
                "bridge": "complete",
                "knowledge": "complete" if knowledge_res.get("all_chunks") else "no_chunks",
                "llm": selected_llm_mode,
            },
            "versions": {"bridge": bridge.get("bridge_version", "legacy"), "retriever": "V7"},
            "experimental_holter": bool(bridge.get("temporal_summary", {}).get("available", False)),
        }
        analytical_tool_context = compile_question_tool_context(user_question, analytical_state)
        prompt_text += "\n\n[READ_ONLY ANALYTICAL TOOL RESULTS]\n" + json.dumps(
            analytical_tool_context, ensure_ascii=False, separators=(",", ":"), default=str
        ) + "\n[/READ_ONLY ANALYTICAL TOOL RESULTS]"

        # 9. Configurable LLM Backend & Honest Provenance
        strict_lora = (os.environ.get("TRACE_REQUIRE_ADAPTER", "0") in ("1", "true"))
        try:
            backend = get_llm_backend(mode=selected_llm_mode, backend_type=selected_llm_backend, require_adapter=strict_lora)
        except Exception as exc:
            if strict_lora:
                failure_reason = f"adapter_load_error_{type(exc).__name__}: {str(exc)}"
                audit_id = f"audit_{hashlib.sha256((case_id + str(time.time())).encode('utf-8')).hexdigest()[:16]}"
                strict_provenance = {
                    "status": "error",
                    "real_classifier_checkpoint": True,
                    "real_retrieval_checkpoint": True,
                    "real_faiss_search": include_retrieval,
                    "requested_mode": selected_llm_mode,
                    "requested_backend": selected_llm_backend,
                    "effective_backend": selected_llm_backend,
                    "base_model_path": None,
                    "base_model_loaded": False,
                    "adapter_requested": True,
                    "adapter_load_attempted": True,
                    "adapter_loaded": False,
                    "fine_tuned_adapter_used": False,
                    "adapter_path": str(os.environ.get("TRACE_LORA_GGUF_PATH", "")),
                    "adapter_mode": "none",
                    "adapter_scale": float(os.environ.get("TRACE_LORA_SCALE", "1.0")),
                    "generation_attempted": False,
                    "generation_succeeded": False,
                    "model_used_for_final_answer": False,
                    "fallback_used": False,
                    "failure_reason": failure_reason,
                    "initialization_error": failure_reason,
                    "audit_id": audit_id,
                }
                raise RuntimeError(f"Strict LoRA execution failed: {failure_reason}. Provenance: {json.dumps(strict_provenance)}") from exc
            raise

        requested_backend = selected_llm_backend
        effective_backend = backend.__class__.__name__

        generation_attempted = False
        generation_succeeded = False
        repair_attempted = False
        repair_succeeded = False
        repair_backend: Optional[str] = None
        repair_adapter_used: bool = False
        repair_adapter_active: bool = False
        initial_violations: List[str] = []
        repair_violations: List[str] = []

        model_used_for_final_answer = False
        base_model_used = False
        fine_tuned_adapter_used = False
        adapter_mode = "none"
        prompt_tokens = 0
        completion_tokens = 0
        context_length = int(os.environ.get("TRACE_CONTEXT_LENGTH", "4096"))
        fallback_used = (selected_llm_mode == "disabled" or effective_backend == "DisabledBackend")
        failure_reason: Optional[str] = None
        explanation_text = ""
        answer_source = "deterministic_fallback"

        if not fallback_used:
            generation_attempted = True
            try:
                # Debug logging to verify Gemma prompt grounding of selected chunks
                debug_chunks = [c.get("chunk_id") or c.get("id") for c in knowledge_res.get("all_chunks", [])]
                debug_sources = [c.get("source_id") for c in knowledge_res.get("all_chunks", [])]
                logger.debug(f"[TRACE DEBUG] Grounding check: Selected KB Chunks: {debug_chunks}, Sources: {debug_sources}")

                gen_res = backend.generate(prompt_text)
                explanation_text = ResponseValidator.normalize_generated_answer(gen_res.text)
                effective_backend = gen_res.backend
                prompt_tokens = gen_res.prompt_tokens
                completion_tokens = gen_res.completion_tokens
                generation_succeeded = True
                base_model_used = True
                fine_tuned_adapter_used = gen_res.adapter_used
                adapter_mode = "lora" if getattr(backend, "adapter_loaded", False) else ("merged_before_gguf_conversion" if fine_tuned_adapter_used else "none")
            except Exception as exc:
                generation_succeeded = False
                failure_reason = f"generation_error_{type(exc).__name__}: {str(exc)}"
                if strict_lora:
                    audit_id = f"audit_{hashlib.sha256((case_id + str(time.time())).encode('utf-8')).hexdigest()[:16]}"
                    strict_provenance = {
                        "status": "error",
                        "real_classifier_checkpoint": True,
                        "real_retrieval_checkpoint": True,
                        "real_faiss_search": include_retrieval,
                        "requested_mode": selected_llm_mode,
                        "requested_backend": requested_backend,
                        "effective_backend": effective_backend,
                        "base_model_path": str(backend.model_path) if hasattr(backend, "model_path") else None,
                        "base_model_loaded": getattr(backend, "is_loaded", False),
                        "adapter_requested": True,
                        "adapter_load_attempted": True,
                        "adapter_initialization_accepted": getattr(backend, "adapter_initialization_accepted", False),
                        "adapter_generation_verified": False,
                        "adapter_loaded": False,
                        "fine_tuned_adapter_used": False,
                        "adapter_path": str(getattr(backend, "lora_path", "")) if getattr(backend, "lora_path", None) else None,
                        "adapter_mode": "none",
                        "adapter_scale": float(getattr(backend, "lora_scale", 1.0)),
                        "generation_attempted": True,
                        "generation_succeeded": False,
                        "model_used_for_final_answer": False,
                        "fallback_used": False,
                        "failure_reason": failure_reason,
                        "initialization_error": failure_reason,
                        "audit_id": audit_id,
                    }
                    raise RuntimeError(f"Strict LoRA execution failed: {failure_reason}. Provenance: {json.dumps(strict_provenance)}") from exc
                fallback_used = True

        # 10. Initial Post-Generation Validation with Citation Stripping (Section 7)
        from utils.reasoning_policy import AnswerMode, DiagnosisAnchor
        anchor = DiagnosisAnchor(
            concept=primary_label,
            family=primary_family,
            confidence=float(bridge.get("confidence", {}).get("final_fused_confidence", 0.85)),
            bridge_concept=primary_label,
            evidence_findings=bridge.get("supported_findings", []),
        )

        if generation_succeeded and explanation_text:
            post_val = ResponseValidator.validate_post_generation(
                answer=explanation_text,
                question=user_question,
                anchor=anchor,
                mode=AnswerMode.HYBRID,
                bridge_result=bridge,
                permitted_citations=knowledge_res.get("permitted_citations", []),
            )
            initial_violations = list(post_val.violations)

            # Separate citations from clinical validation: strip unsupported citations and revalidate (Section 7)
            if not post_val.valid and post_val.violations == ["unpermitted_citation_id"]:
                cleaned_text = ResponseValidator.strip_unsupported_citations(explanation_text, knowledge_res.get("permitted_citations", []))
                cleaned_val = ResponseValidator.validate_post_generation(
                    answer=cleaned_text,
                    question=user_question,
                    anchor=anchor,
                    mode=AnswerMode.HYBRID,
                    bridge_result=bridge,
                    permitted_citations=knowledge_res.get("permitted_citations", []),
                )
                if cleaned_val.valid:
                    explanation_text = cleaned_text
                    post_val = cleaned_val

            if post_val.valid:
                model_used_for_final_answer = True
                answer_source = "gemma_lora" if gen_res.adapter_used else "gemma_base"
            elif not fallback_used:
                # Constrained Repair Pass (Section 4 & 5)
                repair_attempted = True
                failure_reason = "guardrail_rejected:" + ",".join(post_val.violations)
                repair_prompt = f"""[REPAIR INSTRUCTION]
Your previous generated answer failed safety guardrail validation.
Revise the answer to correct ONLY the listed validation violations.
Preserve useful question-specific information. Do not replace the response with a generic ECG summary.
Do not change any structured model output. Do not invent patient information or citations.

<user_question>
{user_question}
</user_question>

<immutable_case_evidence>
Primary Candidate Label: {primary_label}
Primary Family: {primary_family}
Decision Status: {bridge.get("decision")}
OOD Status: {bridge.get("ood", {}).get("ood_status")}
Confidence Level: {bridge.get("confidence", {}).get("confidence_level")}
</immutable_case_evidence>

[PREVIOUS GENERATED ANSWER]
{explanation_text}

[EXACT VALIDATION VIOLATIONS TO CORRECT]
{post_val.violations}

[PERMITTED CITATIONS]
{[c.get("citation_id") for c in knowledge_res.get("permitted_citations", []) if isinstance(c, dict)]}
"""
                try:
                    repair_res = backend.generate(repair_prompt)
                    repaired_raw = ResponseValidator.normalize_generated_answer(repair_res.text)
                    repair_backend = repair_res.backend
                    repair_adapter_used = repair_res.adapter_used
                    repair_adapter_active = getattr(backend, "adapter_loaded", False) or repair_adapter_used

                    if strict_lora and not repair_adapter_active:
                        audit_id = f"audit_{hashlib.sha256((case_id + str(time.time())).encode('utf-8')).hexdigest()[:16]}"
                        strict_provenance = {
                            "status": "error",
                            "requested_mode": selected_llm_mode,
                            "requested_backend": requested_backend,
                            "effective_backend": effective_backend,
                            "repair_attempted": True,
                            "repair_adapter_active": False,
                            "failure_reason": "strict_lora_repair_missing_adapter",
                            "audit_id": audit_id,
                        }
                        raise RuntimeError(f"Strict LoRA execution failed: Repair response generated without active adapter. Provenance: {json.dumps(strict_provenance)}")

                    # Validate repair response with citation stripping
                    repair_val = ResponseValidator.validate_post_generation(
                        answer=repaired_raw,
                        question=user_question,
                        anchor=anchor,
                        mode=AnswerMode.HYBRID,
                        bridge_result=bridge,
                        permitted_citations=knowledge_res.get("permitted_citations", []),
                    )
                    if not repair_val.valid and repair_val.violations == ["unpermitted_citation_id"]:
                        cleaned_repair = ResponseValidator.strip_unsupported_citations(repaired_raw, knowledge_res.get("permitted_citations", []))
                        cleaned_repair_val = ResponseValidator.validate_post_generation(
                            answer=cleaned_repair,
                            question=user_question,
                            anchor=anchor,
                            mode=AnswerMode.HYBRID,
                            bridge_result=bridge,
                            permitted_citations=knowledge_res.get("permitted_citations", []),
                        )
                        if cleaned_repair_val.valid:
                            repaired_raw = cleaned_repair
                            repair_val = cleaned_repair_val

                    repair_violations = list(repair_val.violations)

                    if repair_val.valid:
                        explanation_text = repaired_raw
                        post_val = repair_val
                        repair_succeeded = True
                        model_used_for_final_answer = True
                        answer_source = "gemma_lora_repaired" if repair_res.adapter_used else "gemma_base_repaired"
                    else:
                        repair_succeeded = False
                        fallback_used = True
                        failure_reason = "guardrail_rejected:" + ",".join(repair_val.violations)
                except Exception as exc:
                    repair_succeeded = False
                    fallback_used = True
                    if not failure_reason:
                        failure_reason = f"repair_error_{type(exc).__name__}: {str(exc)}"

                if not post_val.valid and fallback_used and strict_lora:
                    audit_id = f"audit_{hashlib.sha256((case_id + str(time.time())).encode('utf-8')).hexdigest()[:16]}"
                    strict_provenance = {
                        "status": "error",
                        "real_classifier_checkpoint": True,
                        "real_retrieval_checkpoint": True,
                        "real_faiss_search": include_retrieval,
                        "requested_mode": selected_llm_mode,
                        "requested_backend": requested_backend,
                        "effective_backend": effective_backend,
                        "generation_attempted": True,
                        "repair_attempted": repair_attempted,
                        "failure_reason": failure_reason or f"guardrail_rejected: {post_val.violations}",
                        "audit_id": audit_id,
                    }
                    raise RuntimeError(f"Strict LoRA execution failed guardrail validation after repair attempt: {post_val.violations}. Provenance: {json.dumps(strict_provenance)}")
        else:
            fallback_used = True
            post_val = ResponseValidator.validate_post_generation(
                answer="N/A",
                question=user_question,
                anchor=anchor,
                mode=AnswerMode.HYBRID,
                bridge_result=bridge,
                permitted_citations=knowledge_res.get("permitted_citations", []),
            )

        # Execute Deterministic Operational Fallback if LLM failed or fallback active (Section 2 & 3)
        if fallback_used or not explanation_text or not post_val.valid:
            if not failure_reason:
                failure_reason = "generation_or_validation_failed" if generation_attempted else "llm_disabled"
            from prompt_builder.real_explanation_generator import RealExplanationGenerator
            _, explanation_text = RealExplanationGenerator().generate_explanation(
                context_package={
                    "concept": primary_label,
                    "primary_label": primary_label,
                    "family": primary_family,
                    "primary_family": primary_family,
                    "primary_probability": prob_resolved,
                    "decision": bridge.get("decision", "supported"),
                    "bridge": bridge,
                    "classifier": classifier,
                },
                instruction=user_question,
                intent=question_route_obj.intent,
                sub_intent=question_route_obj.sub_intent,
                failure_reason=failure_reason,
            )
            effective_backend = "deterministic_phrase_library"
            answer_source = "deterministic_fallback"

            fallback_val = ResponseValidator.validate_post_generation(
                answer=explanation_text,
                question=user_question,
                anchor=anchor,
                mode=AnswerMode.HYBRID,
                bridge_result=bridge,
                permitted_citations=knowledge_res.get("permitted_citations", []),
            )

            if not fallback_val.valid:
                explanation_text = f"The language-model answer could not be validated. The structured ECG results remain available (Leading candidate: {primary_label}). Failure reason: {failure_reason}."
                post_val = fallback_val
            else:
                post_val = fallback_val

        # Structural Upstream Immutability Assertion (Section 6)
        def assert_upstream_unchanged(before: Dict[str, Any], after: Dict[str, Any]) -> None:
            for key, val_before in before.items():
                assert key in after, f"Upstream key '{key}' missing from snapshot after execution"
                val_after = after[key]
                if isinstance(val_before, dict) and isinstance(val_after, dict):
                    assert_upstream_unchanged(val_before, val_after)
                elif isinstance(val_before, list) and isinstance(val_after, list):
                    assert len(val_before) == len(val_after), f"Upstream list length changed for key '{key}'"
                    for item_b, item_a in zip(val_before, val_after):
                        if isinstance(item_b, dict) and isinstance(item_a, dict):
                            assert_upstream_unchanged(item_b, item_a)
                        else:
                            assert item_b == item_a, f"Upstream list element changed for key '{key}'"
                else:
                    assert val_before == val_after, f"Upstream output mutated for key '{key}': before={val_before}, after={val_after}"

        assert_upstream_unchanged(
            before=upstream_snapshot,
            after={
                "classifier": classifier,
                "family_head": family_head,
                "retrieval": retrieval_data,
                "ood": bridge.get("ood"),
                "confidence": bridge.get("confidence"),
                "decision": bridge.get("decision"),
            }
        )

        cit_val = ResponseValidator.validate_citations(explanation_text, knowledge_res.get("permitted_citations", []))

        conv_state.history.append({"role": "user", "content": user_question})
        conv_state.history.append({"role": "assistant", "content": explanation_text})
        conv_state.history = conv_state.history[-10:]

        explanation_payload = {
            "text": explanation_text,
            "answer_source": answer_source,
            "intent": question_route_obj.intent,
            "sub_intent": question_route_obj.sub_intent,
            "patient_specific": question_route_obj.patient_specific,
            "question_received": True,
            "generation_attempted": generation_attempted,
            "generation_succeeded": generation_succeeded,
            "repair_attempted": repair_attempted,
            "repair_succeeded": repair_succeeded,
            "repair_backend": repair_backend,
            "repair_adapter_used": repair_adapter_used,
            "repair_adapter_active": repair_adapter_active,
            "fallback_used": fallback_used,
            "fallback_reason": failure_reason,
            "initial_guardrail_violations": initial_violations,
            "repair_guardrail_violations": repair_violations,
            "guardrail_status": "passed" if post_val.valid else "failed",
            "guardrail_violations": post_val.violations,
            "citations": cit_val.extracted_citations,
            "citation_status": "verified" if cit_val.valid else "unverified",
            "citation_reason": "No approved knowledge passages were available." if not knowledge_res.get("permitted_citations") else None,
            "backend": effective_backend,
            "llm_policy": system_prompt_metadata(),
            "analytical_tools": analytical_tool_context,
            "skills_activated": [],
            "validation": {
                "status": "PASS" if post_val.valid else "FAIL",
                "violations": post_val.violations,
            },
        }

        audit_id = f"audit_{hashlib.sha256((case_id + str(time.time())).encode('utf-8')).hexdigest()[:16]}"

        model_variant = "base"
        if fine_tuned_adapter_used:
            model_variant = "lora" if getattr(backend, "adapter_loaded", False) else ("merged_lora" if "llama" in requested_backend else "transformers_peft")

        runtime_provenance = {
            "real_classifier_checkpoint": True,
            "real_retrieval_checkpoint": True,
            "real_faiss_search": include_retrieval,
            "mock_neighbors_used": False,
            "ood_input": bridge.get("ood", {}).get("input_source"),
            "external_adapter_used": adaptation is not None,
            "requested_mode": selected_llm_mode,
            "requested_backend": requested_backend,
            "effective_backend": effective_backend,
            "base_model_path": str(backend.model_path) if hasattr(backend, "model_path") else None,
            "base_model_loaded": getattr(backend, "is_loaded", False),
            "model_path": str(backend.model_path) if hasattr(backend, "model_path") else None,
            "model_variant": model_variant,
            "model_loaded": getattr(backend, "is_loaded", False),
            "adapter_requested": getattr(backend, "lora_path", None) is not None,
            "adapter_load_attempted": getattr(backend, "lora_path", None) is not None,
            "adapter_loaded": getattr(backend, "adapter_loaded", False),
            "fine_tuned_adapter_used": getattr(backend, "adapter_loaded", False) or fine_tuned_adapter_used,
            "adapter_path": str(getattr(backend, "lora_path", "")) if getattr(backend, "lora_path", None) else None,
            "adapter_mode": adapter_mode,
            "adapter_scale": float(getattr(backend, "lora_scale", 1.0)) if getattr(backend, "adapter_loaded", False) else 0.0,
            "answer_source": answer_source,
            "llm_policy": system_prompt_metadata(),
            "analytical_tool_names": [
                item.get("tool") for item in analytical_tool_context.get("calls", [])
            ],
            "intent": question_route_obj.intent,
            "sub_intent": question_route_obj.sub_intent,
            "patient_specific": question_route_obj.patient_specific,
            "generation_attempted": generation_attempted,
            "generation_succeeded": generation_succeeded,
            "repair_attempted": repair_attempted,
            "repair_succeeded": repair_succeeded,
            "repair_backend": repair_backend,
            "repair_adapter_used": repair_adapter_used,
            "repair_adapter_active": repair_adapter_active,
            "initial_guardrail_violations": initial_violations,
            "repair_guardrail_violations": repair_violations,
            "model_used_for_final_answer": model_used_for_final_answer,
            "base_model_used": base_model_used,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "context_length": context_length,
            "fallback_used": fallback_used,
            "failure_reason": failure_reason,
            "initialization_error": failure_reason or getattr(backend, "load_error", None),
            "knowledge_available": len(knowledge_res.get("all_chunks", [])) > 0,
            "skills_activated": [],
            "citations_validated": cit_val.valid,
            "guardrails_validated": post_val.valid,
            "audit_id": audit_id,
        }

        combined_result = dict(structured_result)
        combined_result.update({
            "structured_result": structured_result,
            "explanation": explanation_payload,
            "provenance": runtime_provenance,
        })

        return make_json_safe(combined_result)

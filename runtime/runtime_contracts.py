# -*- coding: utf-8 -*-
"""
runtime_contracts.py

Strict contract boundary validation functions and typed dataclasses for pipeline stages in TRACE.
Ensures typed objects, required dictionary keys, array shapes, and fail-closed safety.
"""

from dataclasses import dataclass, field, is_dataclass, asdict
from typing import Any, Dict, List, Optional, Union, Tuple, Set
import numpy as np


def make_json_safe(value: Any) -> Any:
    """Recursively convert NumPy values, arrays, dataclasses, tuples, and sets into JSON-safe primitives."""
    if is_dataclass(value) and not isinstance(value, type):
        return make_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class PipelineContractError(TypeError):
    """Raised when a pipeline stage contract boundary validation fails."""
    def __init__(self, stage: str, expected_type: str, expected_shape_or_keys: str, actual_value: Any, source_module: str):
        msg = (
            f"\n[PIPELINE CONTRACT FAILURE]\n"
            f"  Stage:                  {stage}\n"
            f"  Source Module:          {source_module}\n"
            f"  Expected Type:          {expected_type}\n"
            f"  Expected Shape / Keys:  {expected_shape_or_keys}\n"
            f"  Actual Value / Type:    {type(actual_value).__name__} -> {repr(actual_value)[:150]}\n"
        )
        super().__init__(msg)


@dataclass
class KnowledgeQuery:
    question: str
    primary_label: str
    primary_family: str
    secondary_findings: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    rare_case_state: Optional[Dict[str, Any]] = None
    intent: str = "diagnosis"
    top_k: int = 6


@dataclass
class KnowledgeChunk:
    citation_id: str
    chunk_id: str
    title: str
    source: str
    source_type: str
    section: str
    date_or_version: str
    reference: str
    text: str
    relevance_score: float = 0.0
    supported_labels: List[str] = field(default_factory=list)
    supported_families: List[str] = field(default_factory=list)
    validation_state: str = "validated"

    def to_dict(self) -> Dict[str, Any]:
        return make_json_safe(self.__dict__)


@dataclass
class PermittedCitation:
    citation_id: str
    title: str
    source: str
    allowed_claims: List[str] = field(default_factory=list)


@dataclass
class SkillRoute:
    intent: str
    always_active_skills: List[str] = field(default_factory=list)
    intent_skills: List[str] = field(default_factory=list)
    resolved_skills: List[str] = field(default_factory=list)
    skill_paths: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    conflicts: Dict[str, List[str]] = field(default_factory=dict)
    priority_order: List[str] = field(default_factory=list)
    presentation_mode: str = "standard"


@dataclass
class ConversationState:
    case_id: str
    bridge_conclusion: Dict[str, Any] = field(default_factory=dict)
    primary_findings: List[str] = field(default_factory=list)
    secondary_findings: List[str] = field(default_factory=list)
    ood_state: Dict[str, Any] = field(default_factory=dict)
    signal_quality_limitations: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    confidence_level: str = "MODERATE"
    confidence_drivers: List[str] = field(default_factory=list)
    rare_case_state: Dict[str, Any] = field(default_factory=dict)
    supplied_knowledge_chunks: List[Dict[str, Any]] = field(default_factory=list)
    permitted_citation_ids: List[str] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    user_explanation_preference: str = "standard"

    def reset_case(self, new_case_id: str) -> None:
        """Reset patient-specific state while retaining harmless presentation preferences."""
        pref = self.user_explanation_preference
        self.case_id = new_case_id
        self.bridge_conclusion = {}
        self.primary_findings = []
        self.secondary_findings = []
        self.ood_state = {}
        self.signal_quality_limitations = []
        self.contradictions = []
        self.confidence_level = "MODERATE"
        self.confidence_drivers = []
        self.rare_case_state = {}
        self.supplied_knowledge_chunks = []
        self.permitted_citation_ids = []
        self.history = []
        self.user_explanation_preference = pref


@dataclass
class PromptPackage:
    question: str
    patient_context: Dict[str, Any] = field(default_factory=dict)
    ecg_statistics: Dict[str, Any] = field(default_factory=dict)
    temporal_summary: Dict[str, Any] = field(default_factory=dict)
    ecg_comparison: Dict[str, Any] = field(default_factory=dict)
    conversation_state: Dict[str, Any] = field(default_factory=dict)
    skill_route: Dict[str, Any] = field(default_factory=dict)
    skill_instructions: List[Dict[str, Any]] = field(default_factory=list)
    classifier: Dict[str, Any] = field(default_factory=dict)
    family_head: Dict[str, Any] = field(default_factory=dict)
    raw_retrieval: Dict[str, Any] = field(default_factory=dict)
    reranked_retrieval: Dict[str, Any] = field(default_factory=dict)
    retrieval_quality: Dict[str, Any] = field(default_factory=dict)
    signal_quality: Dict[str, Any] = field(default_factory=dict)
    ood: Dict[str, Any] = field(default_factory=dict)
    rare_case: Dict[str, Any] = field(default_factory=dict)
    contradictions: List[Any] = field(default_factory=list)
    confidence: Dict[str, Any] = field(default_factory=dict)
    bridge: Dict[str, Any] = field(default_factory=dict)
    knowledge_chunks: List[Any] = field(default_factory=list)
    permitted_citations: List[Any] = field(default_factory=list)
    ontology_context: Dict[str, Any] = field(default_factory=dict)
    condition_cards_context: Dict[str, Any] = field(default_factory=dict)



@dataclass
class LLMGenerationResult:
    text: str
    backend: str
    model_path: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_time_ms: float = 0.0
    adapter_used: bool = False
    adapter_status_reason: Optional[str] = None
    adapter_initialization_accepted: bool = False
    adapter_generation_verified: bool = False


@dataclass
class CitationValidationResult:
    valid: bool
    extracted_citations: List[str] = field(default_factory=list)
    permitted_citations: List[str] = field(default_factory=list)
    unsupported_citations: List[str] = field(default_factory=list)
    sources_section_added: bool = False
    corrective_retry_performed: bool = False


@dataclass
class PostGenerationValidation:
    valid: bool
    diagnosis_preserved: bool = True
    uncertainty_preserved: bool = True
    ood_preserved: bool = True
    confidence_language_valid: bool = True
    neighbor_attribution_valid: bool = True
    signal_limitations_valid: bool = True
    contradictions_disclosed: bool = True
    treatment_grounded: bool = True
    citations_valid: bool = True
    critical_failure: bool = False
    violations: List[str] = field(default_factory=list)


@dataclass
class DeterministicExplanation:
    text: str
    intent: str
    primary_finding: str
    confidence_level: str
    citations: List[str] = field(default_factory=list)
    phrase_sources: List[str] = field(default_factory=list)


@dataclass
class FeedbackSubmission:
    case_id: str
    model_version: str
    bridge_decision: str
    explanation_version: str
    clinician_action: str
    corrected_labels: List[str] = field(default_factory=list)
    corrected_families: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    review_status: str = "Pending"
    timestamp: str = ""
    target_component: str = "overall"
    audit_provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompleteDeploymentResponse:
    structured_result: Dict[str, Any]
    explanation: Dict[str, Any]


def validate_ecg_array(raw_ecg: Any, source_module: str = "main_inference") -> np.ndarray:
    """Validates raw input ECG signal numpy array."""
    if not isinstance(raw_ecg, np.ndarray):
        raise PipelineContractError("ECG Input", "np.ndarray", "shape (12, 1000) or (12, N)", raw_ecg, source_module)
    if raw_ecg.ndim != 2:
        raise PipelineContractError("ECG Input", "2D np.ndarray", "shape (12, N) or (N, 12)", raw_ecg, source_module)
    if raw_ecg.shape[0] != 12 and raw_ecg.shape[1] != 12:
        raise PipelineContractError("ECG Input", "12-lead signal", "lead dimension equal to 12", raw_ecg, source_module)
    return raw_ecg


def validate_canonical_result(canonical_result: Any, source_module: str = "main_inference") -> Dict[str, Any]:
    """Validates canonical result schema."""
    if not isinstance(canonical_result, dict):
        raise PipelineContractError("Canonical Result", "Dict", "canonical top-level result dict", canonical_result, source_module)

    required_keys = {"query", "preprocessing", "signal_quality", "diagnosis", "raw_retrieval", "reranked_retrieval", "bridge", "ood", "confidence", "evidence", "knowledge", "gemma", "provenance", "telemetry"}
    missing = required_keys - set(canonical_result.keys())
    if missing:
        raise PipelineContractError("Canonical Result", "Dict", f"missing required keys: {missing}", canonical_result, source_module)

    return canonical_result

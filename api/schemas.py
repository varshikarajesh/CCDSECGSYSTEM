# -*- coding: utf-8 -*-
"""
api/schemas.py

Pydantic schemas for the TRACE FastAPI runtime endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    name: str = Field(..., description="API Name")
    version: str = Field(..., description="API Version")
    status: str = Field(..., description="Overall system health status (healthy or degraded)")
    docs_url: str = Field(..., description="URL for OpenAPI interactive documentation")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall status ('healthy' or 'degraded')")
    diagnosis_model_loaded: bool = Field(..., description="Whether DiagnosisModel is initialized and cached")
    diagnosis_model_initialization_error: Optional[str] = Field(None, description="Initialization error message if DiagnosisModel failed to load")
    feedback_service_available: bool = Field(..., description="Whether FeedbackService is available")
    artifact_validation_results: Dict[str, Any] = Field(..., description="Physical verification status of runtime artifacts")
    missing_required_artifacts: Dict[str, Any] = Field(..., description="Map of missing required runtime artifacts")
    configured_device: str = Field(..., description="PyTorch inference execution device (cuda or cpu)")
    cuda_availability: bool = Field(..., description="Whether CUDA is available to PyTorch")
    pytorch_version: str = Field(..., description="Installed PyTorch framework version")
    configured_llm_mode: str = Field(..., description="Configured LLM execution mode ('disabled' or 'real')")
    configured_llm_backend: str = Field(..., description="Configured base-Gemma backend ('llama_cpp')")
    gguf_base_model_existence: bool = Field(..., description="Whether the GGUF base model file exists on disk")
    llm_model_variant: str = Field(..., description="Selected LLM variant; base for this bundle")
    adapter_supported: bool = Field(..., description="False: LoRA/PEFT adapters are retired")
    api_version: str = Field(..., description="TRACE API version string")


class ConfigResponse(BaseModel):
    supported_file_formats: List[str] = Field(..., description="Supported upload file extensions")
    upload_limit_mb: float = Field(..., description="Maximum allowed ECG upload size in MB")
    allowed_sampling_rate_range: Dict[str, int] = Field(..., description="Supported sampling rate range in Hz")
    available_llm_modes: List[str] = Field(..., description="Available LLM execution modes")
    available_llm_backends: List[str] = Field(..., description="Available LLM backends")
    configured_llm_mode: str = Field(..., description="Currently active LLM mode")
    configured_llm_backend: str = Field(..., description="Currently active LLM backend")
    llm_model_variant: str = Field(..., description="Selected LLM variant; base for this bundle")
    adapter_supported: bool = Field(..., description="False: LoRA/PEFT adapters are retired")
    allowed_dataset_names: List[str] = Field(..., description="Supported external dataset names")
    registered_skill_count: int = Field(..., description="Count of registered clinical reasoning skills")
    frontend_origins: List[str] = Field(..., description="Allowed CORS origins")


class ReviewFeedbackRequest(BaseModel):
    new_status: str = Field(..., description="Target review status ('Approved', 'Rejected', 'Needs Clarification')")
    reviewer_id: str = Field(..., description="Identifier of the reviewing clinician")
    notes: Optional[str] = Field(None, description="Optional review notes or comments")


class ChatRequest(BaseModel):
    question: str = Field(..., description="Clinician's follow-up question about the recording")
    conversation: Optional[List[Dict[str, str]]] = Field(None, description="Optional conversation history for context (deprecated, server state preferred)")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Advisory prose answer from the LLM or fallback")
    text: str = Field(..., description="Advisory prose answer text")
    intent: str = Field(..., description="Resolved question intent category")
    active_condition: Dict[str, str] = Field(..., description="Active condition label and clinical display name")
    evidence: Dict[str, Any] = Field(..., description="Clean audit trail of clinical evidence sources utilized")
    citations: List[Dict[str, Any]] = Field(..., description="Structured literature citations referenced")
    status: str = Field(..., description="Response generation status")


class WindowSelectRequest(BaseModel):
    window_indices: List[int] = Field(..., description="Clinician selected window indices to override selection")


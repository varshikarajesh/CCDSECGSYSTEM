# -*- coding: utf-8 -*-
"""
api/main.py

Production-oriented FastAPI application for the TRACE ECG Evidence & LLM Reasoning System.
Exposes full ECG inference, sample execution, system health/configuration, and clinician feedback workflows.

Deployment Note:
    Due to single-GPU/CPU memory constraints with heavy PyTorch encoders, FAISS indices, and GGUF/LoRA models,
    this application MUST be executed using a SINGLE worker process:

    python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import io
import json
import logging
import os
from pathlib import Path
import secrets
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import (
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from backend.diagnosis_model import DiagnosisModel
from backend.recording_analysis import RecordingAnalyzer
from long_recording_v2.inference import canonical_recording
from backend.feedback import feedback_service
from deployment_config import (
    DATA_DIR,
    DEVICE,
    PACKAGE_ROOT,
    SAMPLE_ECG_PATH,
    SKILL_REGISTRY_PATH,
    TRACE_GGUF_MODEL,
    TRACE_LLM_BACKEND,
    TRACE_LLM_MODE,
    missing_required_artifacts,
    validate_package_artifacts,
)
from api.schemas import (
    ConfigResponse,
    HealthResponse,
    ReviewFeedbackRequest,
    RootResponse,
    ChatRequest,
    ChatResponse,
    WindowSelectRequest,
)

# Audit-safe logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trace_api")

# Process-level cached singletons and concurrency controls
diagnosis_model: Optional[DiagnosisModel] = None
recording_analyzer: Optional[RecordingAnalyzer] = None
jetson_pipeline: Any = None
model_init_error: Optional[str] = None
model_lock = asyncio.Lock()

# Cache for recording analysis results with associated chat state
# Structure: { recording_id: {"result": result_dict, "chat_state": chat_state_dict} }
recording_analysis_cache: Dict[str, Dict[str, Any]] = {}


def sync_analysis_cache() -> None:
    """Synchronizes in-memory analysis cache with the recording store.
    Evicts any cache entry whose underlying JSON metadata file has expired or was evicted from disk.
    """
    global recording_analyzer
    if not recording_analyzer or not recording_analyzer.recording_store:
        return
    try:
        store = recording_analyzer.recording_store
        # Run cleanup on the store first
        store.cleanup()
        # Find all active recording IDs on disk
        active_ids = {path.stem for path in store.root.glob("rec_*.json")}
        # Evict expired keys from cache
        expired_keys = [k for k in recording_analysis_cache.keys() if k not in active_ids]
        for key in expired_keys:
            recording_analysis_cache.pop(key, None)
    except Exception as exc:
        logger.error("Failed to sync recording analysis cache: %s", exc)


# Configuration constants
MAX_UPLOAD_SIZE_BYTES = int(os.environ.get("TRACE_MAX_UPLOAD_SIZE_BYTES", 50 * 1024 * 1024))
ALLOWED_DATASETS = ["Chapman", "Georgia", "CPSC2018", "INCART", "PTB_Diagnostic"]

FORBIDDEN_OOD_KEYS = {
    "ood",
    "is_ood",
    "ood_score",
    "raw_distance",
    "distance_score",
    "embedding_confidence",
    "calibrated_threshold",
    "calibration_status",
    "out_of_distribution",
}


def assert_no_ood_fields(value):
    """Recursively asserts that no dictionary or list contains any OOD key at any nesting depth."""
    if isinstance(value, dict):
        for key, child in value.items():
            assert key.lower() not in FORBIDDEN_OOD_KEYS, f"Forbidden OOD key '{key}' found in payload"
            assert_no_ood_fields(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_ood_fields(child)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown resource lifecycle.
    Instantiates and caches exactly one process-wide JetsonECGPipeline instance on startup,
    which manages the unified model, retriever, and recording analyzer states.
    """
    global diagnosis_model, recording_analyzer, jetson_pipeline, model_init_error
    logger.info("Starting TRACE API server lifespan initialization...")

    try:
        from runtime.jetson_runtime import JetsonECGPipeline
        llm_mode = os.environ.get("TRACE_LLM_MODE", TRACE_LLM_MODE)
        llm_backend = os.environ.get("TRACE_LLM_BACKEND", TRACE_LLM_BACKEND)
        requested_device = os.environ.get("TRACE_DEVICE", "auto")

        # Gemma is advisory. Keep the deterministic ECG pipeline available on
        # developer/CPU machines and opt into the real model automatically only
        # where CUDA is actually usable. An explicit TRACE_LLM_MODE=disabled is
        # also respected. On Jetson, `auto` resolves to CUDA and real mode stays
        # unchanged.
        if llm_mode == "real" and requested_device == "auto":
            try:
                import torch
                cuda_available = bool(torch.cuda.is_available())
            except ImportError:
                cuda_available = False
            if not cuda_available:
                logger.warning(
                    "CUDA is unavailable; starting ECG inference with the advisory LLM disabled. "
                    "Set TRACE_LLM_MODE=real on a CUDA host to enable Gemma."
                )
                llm_mode = "disabled"
        enable_holter = os.environ.get("TRACE_ENABLE_EXPERIMENTAL_HOLTER", "0").lower() in ("1", "true", "yes")

        jetson_pipeline = JetsonECGPipeline(
            device=requested_device,
            llm_mode=llm_mode,
            llm_backend=llm_backend,
            enable_experimental_holter=enable_holter
        )
        diagnosis_model = jetson_pipeline.model
        recording_analyzer = jetson_pipeline.recording_analyzer
        model_init_error = None
        logger.info("JetsonECGPipeline successfully loaded and cached in process memory.")
    except Exception as exc:
        model_init_error = str(exc)
        diagnosis_model = None
        recording_analyzer = None
        jetson_pipeline = None
        logger.error("Failed to initialize JetsonECGPipeline during startup: %s", exc, exc_info=True)

    yield

    logger.info("Shutting down TRACE API server lifespan...")


# Define module-level FastAPI application
app = FastAPI(
    title="TRACE ECG Inference API",
    description="Production-oriented FastAPI application for Jetson ECG Evidence Inference and LLM Explanations",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
allowed_origins_env = os.environ.get("TRACE_ALLOWED_ORIGINS", "")
if allowed_origins_env.strip():
    origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:4174",
        "http://127.0.0.1:4174",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Security: Administrator API Key Verification Dependency
def verify_admin_api_key(
    x_admin_api_key: Optional[str] = Header(None, alias="X-Admin-API-Key")
) -> str:
    """
    Validates X-Admin-API-Key header against server-configured TRACE_ADMIN_API_KEY using constant-time comparison.
    If TRACE_ADMIN_API_KEY is not configured on the server, returns HTTP 503.
    If header is missing or incorrect, returns HTTP 401.
    """
    admin_key = os.environ.get("TRACE_ADMIN_API_KEY")
    if not admin_key or not admin_key.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Administrator endpoints are disabled because TRACE_ADMIN_API_KEY is not set on the server."
        )
    if not x_admin_api_key or not secrets.compare_digest(x_admin_api_key, admin_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing administrator API key."
        )
    return x_admin_api_key


# Custom Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Ensure consistent structured errors
    error_code = "HTTP_ERROR"
    if exc.status_code == 404:
        error_code = "NOT_FOUND"
    elif exc.status_code == 422:
        error_code = "UNPROCESSABLE_ENTITY"
    elif exc.status_code == 413:
        error_code = "REQUEST_ENTITY_TOO_LARGE"
    elif exc.status_code == 400:
        error_code = "BAD_REQUEST"
    elif exc.status_code == 503:
        error_code = "SERVICE_UNAVAILABLE"
        
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status": "error",
            "error_code": error_code,
            "message": exc.detail,
            "details": {}
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled server exception: %s", exc, exc_info=True)
    error_msg = str(exc)
    error_code = "INTERNAL_PROCESSING_ERROR"
    if "KeyError" in error_msg or "window_index" in error_msg:
        error_code = "INVALID_WINDOW_INDEX"
        error_msg = "A mismatch in the window structures was detected during processing."
        
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected internal server error occurred.",
            "status": "error",
            "error_code": error_code,
            "message": error_msg if error_code != "INTERNAL_PROCESSING_ERROR" else "Unable to process the requested evidence.",
            "details": {"exception_type": type(exc).__name__}
        },
    )


# ECG Signal Parser and Validator
def parse_and_validate_ecg_bytes(filename: str, contents: bytes) -> np.ndarray:
    """
    Validates uploaded file contents, loads .npy or CSV signal, enforces 2D (12, N) float32 layout.
    Rejects empty, oversized, NaN/Inf, or non-12-lead signals without modifying or fabricating data.
    """
    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded ECG file is empty."
        )

    suffix = Path(filename).suffix.lower() if filename else ".npy"
    if suffix not in (".npy", ".npz", ".csv", ".txt", ".hea", ".dat"):
        # Default fallback suffix if generic filename is used
        suffix = ".npy"

    raw: Optional[np.ndarray] = None

    try:
        if suffix == ".npy":
            raw = np.load(io.BytesIO(contents), allow_pickle=False)
        elif suffix == ".npz":
            package = np.load(io.BytesIO(contents), allow_pickle=False)
            keys = list(package.files)
            signal_key = next((key for key in ("ecg", "signal", "waveform", "data") if key in package), None)
            if signal_key is None:
                if len(keys) != 1:
                    raise ValueError(f"NPZ must contain ecg/signal/waveform/data; found {keys}")
                signal_key = keys[0]
            raw = package[signal_key]
        elif suffix in (".csv", ".txt"):
            import csv
            text = contents.decode("utf-8-sig")
            first_line = next(csv.reader(io.StringIO(text)), [])
            
            # Smart header detection: if any non-empty cell in first line is non-numeric, it is a header
            has_header = False
            for cell in first_line:
                cell_str = cell.strip()
                if cell_str:
                    try:
                        float(cell_str)
                    except ValueError:
                        has_header = True
                        break
                        
            raw = np.genfromtxt(
                io.StringIO(text), 
                delimiter="," if suffix == ".csv" else None,
                skip_header=1 if has_header else 0,
                dtype=np.float32
            )
        elif suffix in (".hea", ".dat"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            try:
                from ecg_loader import _load_wfdb
                raw, _, _ = _load_wfdb(Path(tmp_path))
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
        else:
            raise ValueError(f"Unsupported suffix {suffix}")
            
        if raw is None:
            raise ValueError("No numeric data parsed")
            
        from ecg_loader import _orient
        arr = _orient(raw)
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"TRACE execution error: {exc}"
        )

    return arr


# Root Endpoint
@app.get("/", response_model=RootResponse)
async def get_root():
    health_status = "degraded" if (diagnosis_model is None or model_init_error) else "healthy"
    return RootResponse(
        name="TRACE ECG Inference API",
        version="1.0.0",
        status=health_status,
        docs_url="/docs",
    )


# Health Endpoint
@app.get("/api/health")
async def get_health():
    artifacts = validate_package_artifacts()
    serialized_artifacts = {
        name: {"path": str(path), "present": present}
        for name, (path, present) in artifacts.items()
    }
    missing = {name: str(path) for name, path in missing_required_artifacts().items()}

    model_loaded = (diagnosis_model is not None and model_init_error is None)
    feedback_available = feedback_service is not None
    overall_status = "healthy" if (model_loaded and not missing and feedback_available) else "degraded"

    return {
        "status": overall_status,
        "diagnosis_model_loaded": model_loaded,
        "diagnosis_model_initialization_error": model_init_error,
        "feedback_service_available": feedback_available,
        "artifact_validation_results": serialized_artifacts,
        "missing_required_artifacts": missing,
        "configured_device": str(getattr(jetson_pipeline, "device", DEVICE)),
        "cuda_availability": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
        "configured_llm_mode": os.environ.get("TRACE_LLM_MODE", TRACE_LLM_MODE),
        "configured_llm_backend": os.environ.get("TRACE_LLM_BACKEND", TRACE_LLM_BACKEND),
        "gguf_base_model_existence": TRACE_GGUF_MODEL.is_file(),
        "llm_model_variant": "base",
        "adapter_supported": False,
        "api_version": "1.0.0",
    }


# Config Endpoint
@app.get("/api/config")
async def get_config():
    skill_count = 0
    if SKILL_REGISTRY_PATH.is_file():
        try:
            import yaml
            reg_data = yaml.safe_load(SKILL_REGISTRY_PATH.read_text(encoding="utf-8")) or {}
            skills = reg_data.get("skills", {})
            skill_count = len(skills)
        except Exception:
            skill_count = 0

    return {
        "supported_file_formats": [".npy", ".csv"],
        "upload_limit_mb": round(MAX_UPLOAD_SIZE_BYTES / (1024 * 1024), 2),
        "allowed_sampling_rate_range": {"min": 100, "max": 1000},
        "available_llm_modes": ["disabled", "real"],
        "available_llm_backends": ["llama_cpp"],
        "configured_llm_mode": os.environ.get("TRACE_LLM_MODE", TRACE_LLM_MODE),
        "configured_llm_backend": os.environ.get("TRACE_LLM_BACKEND", TRACE_LLM_BACKEND),
        "llm_model_variant": "base",
        "adapter_supported": False,
        "allowed_dataset_names": ALLOWED_DATASETS,
        "registered_skill_count": skill_count,
        "frontend_origins": origins,
    }


# Full ECG Inference Endpoint
@app.post("/api/inference")
async def run_inference(
    file: UploadFile = File(...),
    sampling_rate_hz: int = Form(100),
    top_k: int = Form(5),
    ecg_id: Optional[int] = Form(None),
    patient_id: Optional[float] = Form(None),
    dataset_name: Optional[str] = Form(None),
    lead_names_json: Optional[str] = Form(None),
    question: str = Form("What is the primary finding and diagnostic conclusion?"),
    conversation_id: Optional[str] = Form(None),
    include_retrieval: bool = Form(True),
    include_knowledge: bool = Form(True),
    include_explanation: bool = Form(True),
    llm_mode: Optional[str] = Form(None),
    llm_backend: Optional[str] = Form(None),
):
    if not 1 <= top_k <= 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="top_k must be between 1 and 20",
        )

    if diagnosis_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Diagnosis model is unavailable or degraded: {model_init_error or 'Initialization failed'}"
        )

    # Validate non-clinical parameters
    if dataset_name and dataset_name not in ALLOWED_DATASETS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid dataset_name '{dataset_name}'. Allowed dataset names are: {ALLOWED_DATASETS}"
        )
    if llm_mode and llm_mode not in ("disabled", "real"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="llm_mode must be 'disabled' or 'real'."
        )
    if llm_backend and llm_backend not in ("llama_cpp", "transformers_peft"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="llm_backend must be 'llama_cpp' or 'transformers_peft'."
        )

    lead_names = None
    if lead_names_json:
        try:
            lead_names = json.loads(lead_names_json)
            if not isinstance(lead_names, list) or len(lead_names) != 12:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="lead_names_json must contain a JSON array of exactly 12 lead names."
                )
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid lead_names_json string: {exc}"
            )

    # Validate file size and signal array
    contents = await file.read()
    try:
        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file size ({len(contents)} bytes) exceeds the {MAX_UPLOAD_SIZE_BYTES / (1024*1024):.0f} MB upload limit."
            )
        ecg = parse_and_validate_ecg_bytes(file.filename or "", contents)
    finally:
        await file.close()

    metadata = {
        "sampling_rate_hz": sampling_rate_hz,
        "top_k": top_k,
        "ecg_id": ecg_id,
        "patient_id": patient_id,
        "external_dataset": dataset_name,
        "lead_names": lead_names,
    }

    try:
        async with model_lock:
            result = await run_in_threadpool(
                diagnosis_model.predict,
                ecg,
                metadata=metadata,
                include_retrieval=include_retrieval,
                include_knowledge=include_knowledge,
                include_explanation=include_explanation,
                question=question,
                conversation_id=conversation_id,
                llm_mode=llm_mode,
                llm_backend=llm_backend,
                top_k=top_k,
            )
            if recording_analyzer is not None:
                canonical = canonical_recording(ecg, sampling_rate_hz, lead_names, 100)
                acquisition = recording_analyzer.recording_store.save(canonical, {
                    "recording_mode": "10s",
                    "sampling_rate_hz": 100,
                    "duration_seconds": canonical.shape[1] / 100.0,
                    "lead_order": lead_names or ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                    "preprocessing_state": "canonical_resampled_recording_before_window_normalization",
                })
                result["acquisition"] = acquisition
        return JSONResponse(content=result)
    except RuntimeError as exc:
        err_msg = str(exc)
        if "Strict LoRA" in err_msg or "guardrail validation" in err_msg or "checkpoint missing" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=err_msg
            )
        logger.error("Error during model inference execution: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution failed."
        )
    except Exception as exc:
        logger.error("Unexpected error during inference execution: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected internal server error during inference."
        )


@app.post("/api/recording-inference")
async def run_recording_inference(
    file: UploadFile = File(...),
    recording_mode: str = Form(...),
    sampling_rate_hz: int = Form(100),
    top_k: int = Form(5),
    manual_window_indices_json: str = Form("[]"),
    lead_names_json: Optional[str] = Form(None),
    question: str = Form("What is the primary finding and diagnostic conclusion?"),
    include_explanation: bool = Form(True),
):
    """Analyze a complete 10-second, 2-minute, or 5-minute recording, cache the state, and generate initial advisory explanation."""
    if recording_analyzer is None:
        raise HTTPException(status_code=503, detail=f"Recording analyzer unavailable: {model_init_error or 'not initialized'}")
    if recording_mode not in {"10s", "2min", "5min"}:
        raise HTTPException(status_code=400, detail="recording_mode must be 10s, 2min, or 5min")
    if not 1 <= top_k <= 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
    try:
        manual = json.loads(manual_window_indices_json)
        if not isinstance(manual, list) or any(not isinstance(v, int) for v in manual):
            raise ValueError
    except Exception:
        raise HTTPException(status_code=400, detail="manual_window_indices_json must be a JSON integer array")
    lead_names = None
    if lead_names_json:
        try:
            lead_names = json.loads(lead_names_json)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid lead_names_json: {exc}")
    contents = await file.read()
    try:
        if len(contents) > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded recording exceeds size limit")
        ecg = parse_and_validate_ecg_bytes(file.filename or "", contents)
    finally:
        await file.close()
    try:
        sync_analysis_cache()
        async with model_lock:
            result = await run_in_threadpool(
                recording_analyzer.analyze,
                ecg,
                sampling_rate_hz,
                recording_mode,
                manual,
                lead_names,
                top_k,
            )
            # Generate initial explanation if requested
            if include_explanation and jetson_pipeline is not None:
                explanation = await run_in_threadpool(
                    jetson_pipeline._explain_recording,
                    result,
                    question,
                )
                result["explanation"] = explanation
            else:
                result["explanation"] = {"status": "disabled", "text": "[Explanation disabled]"}
        
        recording_id = result.get("recording_id")
        if recording_id:
            recording_analysis_cache[recording_id] = {
                "result": result,
                "chat_state": getattr(jetson_pipeline, "chat_state", None)
            }
            
        return JSONResponse(content=result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Recording inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Recording inference failed")


@app.get("/api/recordings/{recording_id}/window")
async def get_recording_window(recording_id: str, start_seconds: float, end_seconds: float):
    """Return a clinician-selected interval from the bounded temporary recording cache."""
    if recording_analyzer is None:
        raise HTTPException(status_code=503, detail="Recording analyzer unavailable")
    try:
        payload = await run_in_threadpool(
            recording_analyzer.recording_store.window,
            recording_id,
            start_seconds,
            end_seconds,
        )
        return JSONResponse(content=payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/recordings/{recording_id}/chat", response_model=ChatResponse)
async def run_recording_chat(
    recording_id: str,
    body: ChatRequest,
):
    """Case-scoped chatbot endpoint reusing stored case/recording analysis and state."""
    if jetson_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Jetson ECG pipeline is not initialized or is degraded."
        )
        
    sync_analysis_cache()
    cache_entry = recording_analysis_cache.get(recording_id)
    if not cache_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording analysis state is missing or expired. Please run recording inference again."
        )
        
    result = cache_entry["result"]
    
    # Restore persistent dialogue state of this recording to JetsonECGPipeline
    jetson_pipeline.chat_state = cache_entry.get("chat_state")
    
    try:
        async with model_lock:
            chat_res = await run_in_threadpool(
                jetson_pipeline.answer_question,
                result,
                body.question,
                None, # Use the server-owned context history
            )
            
        # Save dialogue state back to cache entry
        cache_entry["chat_state"] = jetson_pipeline.chat_state
        
        # Get natural condition details
        from runtime.jetson_runtime import get_abbreviation_mapping
        ont_map = get_abbreviation_mapping()
        
        label = jetson_pipeline.chat_state.get("last_subject", "UNKNOWN")
        name = ont_map.get(label.upper(), label) if label else "Unknown"
        active_condition = {
            "label": label,
            "name": name
        }
        
        intent = jetson_pipeline.chat_state.get("last_question_type", "UNKNOWN")
        clinical_chunks = jetson_pipeline.chat_state.get("last_citations", [])
        
        chunk_ids = [chunk.get("chunk_id") or chunk.get("id") for chunk in clinical_chunks if chunk]
        
        evidence = {
            "source_type": "clinical_knowledge" if intent == "clinical_knowledge" else "model_prediction",
            "chunk_ids": [cid for cid in chunk_ids if cid]
        }
        
        citations = []
        for chunk in clinical_chunks:
            citations.append({
                "title": chunk.get("source_title") or chunk.get("title") or "Clinical Reference",
                "organization_or_authors": chunk.get("organization_or_authors") or chunk.get("author", ""),
                "year": chunk.get("date_or_version") or chunk.get("year", ""),
                "section": chunk.get("section", ""),
                "page_or_locator": chunk.get("page", ""),
                "doi": chunk.get("doi", ""),
                "url": chunk.get("url", ""),
            })
            
        return ChatResponse(
            answer=chat_res.get("text", ""),
            text=chat_res.get("text", ""),
            intent=intent,
            active_condition=active_condition,
            evidence=evidence,
            citations=citations,
            status=chat_res.get("status", "generated")
        )
    except Exception as exc:
        logger.error("Chat resolution failed for recording %s: %s", recording_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing follow-up question."
        )


@app.post("/api/recordings/{recording_id}/windows/select")
async def override_recording_windows(
    recording_id: str,
    body: WindowSelectRequest,
):
    """Update manual window selections for a stored recording without re-uploading the signal."""
    if recording_analyzer is None:
        raise HTTPException(status_code=503, detail="Recording analyzer unavailable")
        
    sync_analysis_cache()
    cache_entry = recording_analysis_cache.get(recording_id)
    if not cache_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording analysis state is missing or expired. Please run recording inference again."
        )
        
    try:
        # Load raw signal and metadata
        signal, metadata = await run_in_threadpool(
            recording_analyzer.recording_store.load,
            recording_id
        )
        mode = metadata.get("recording_mode", "2min")
        rate = metadata.get("sampling_rate_hz", 100)
        lead_names = metadata.get("lead_order")
        
        async with model_lock:
            result = await run_in_threadpool(
                recording_analyzer.analyze,
                signal,
                rate,
                mode,
                body.window_indices,
                lead_names,
                20,
                None
            )
            # Narrative generation is advisory and must not discard a valid
            # deterministic selected-window recomputation.
            if jetson_pipeline is not None:
                try:
                    explanation = await run_in_threadpool(
                        jetson_pipeline._explain_recording,
                        result,
                        "What is the primary finding and diagnostic conclusion?",
                    )
                    result["explanation"] = explanation
                except Exception as explanation_exc:
                    logger.warning("Selected-window explanation unavailable: %s", explanation_exc)
                    result["explanation"] = {"status": "unavailable", "text": "Selected windows were reanalyzed; narrative explanation is temporarily unavailable."}
            else:
                result["explanation"] = {"status": "disabled", "text": "[Explanation disabled]"}
                
        # Cache updated results
        result["recording_id"] = recording_id
        cache_entry["result"] = result
        # Keep chat state since it belongs to recording
        
        return JSONResponse(content=result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to override windows for recording %s: %s", recording_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Window override failed")


# Sample ECG Inference Endpoint
@app.post("/api/sample")
async def run_sample_inference(
    sampling_rate_hz: int = Form(100),
    ecg_id: Optional[int] = Form(None),
    patient_id: Optional[float] = Form(None),
    dataset_name: Optional[str] = Form(None),
    lead_names_json: Optional[str] = Form(None),
    question: str = Form("What is the primary finding and diagnostic conclusion?"),
    conversation_id: Optional[str] = Form(None),
    include_retrieval: bool = Form(True),
    include_knowledge: bool = Form(True),
    include_explanation: bool = Form(True),
    llm_mode: Optional[str] = Form(None),
    llm_backend: Optional[str] = Form(None),
):
    """
    Executes inference on the packaged sample ECG waveform specified by SAMPLE_ECG_PATH in deployment_config.py.
    Documented contract source: SAMPLE_ECG_PATH points to 'sample_data/deployment_test_ecg_378.npy',
    so default ecg_id falls back to 378 if omitted.
    """
    if diagnosis_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Diagnosis model is unavailable or degraded: {model_init_error or 'Initialization failed'}"
        )

    if not SAMPLE_ECG_PATH.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Packaged sample ECG file not found at {SAMPLE_ECG_PATH}"
        )

    if dataset_name and dataset_name not in ALLOWED_DATASETS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid dataset_name '{dataset_name}'. Allowed dataset names are: {ALLOWED_DATASETS}"
        )
    if llm_mode and llm_mode not in ("disabled", "real"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="llm_mode must be 'disabled' or 'real'."
        )
    if llm_backend and llm_backend not in ("llama_cpp", "transformers_peft"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="llm_backend must be 'llama_cpp' or 'transformers_peft'."
        )

    lead_names = None
    if lead_names_json:
        try:
            lead_names = json.loads(lead_names_json)
            if not isinstance(lead_names, list) or len(lead_names) != 12:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="lead_names_json must contain a JSON array of exactly 12 lead names."
                )
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid lead_names_json string: {exc}"
            )

    try:
        ecg = np.load(SAMPLE_ECG_PATH, allow_pickle=False)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load packaged sample ECG waveform: {exc}"
        )

    # Documented source: SAMPLE_ECG_PATH filename is deployment_test_ecg_378.npy
    effective_ecg_id = ecg_id if ecg_id is not None else 378

    metadata = {
        "sampling_rate_hz": sampling_rate_hz,
        "ecg_id": effective_ecg_id,
        "patient_id": patient_id,
        "external_dataset": dataset_name,
        "lead_names": lead_names,
    }

    try:
        async with model_lock:
            result = await run_in_threadpool(
                diagnosis_model.predict,
                ecg,
                metadata=metadata,
                include_retrieval=include_retrieval,
                include_knowledge=include_knowledge,
                include_explanation=include_explanation,
                question=question,
                conversation_id=conversation_id,
                llm_mode=llm_mode,
                llm_backend=llm_backend,
            )
        return JSONResponse(content=result)
    except RuntimeError as exc:
        err_msg = str(exc)
        if "Strict LoRA" in err_msg or "guardrail validation" in err_msg or "checkpoint missing" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=err_msg
            )
        logger.error("Error during sample inference execution: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sample inference execution failed."
        )
    except Exception as exc:
        logger.error("Unexpected error during sample inference execution: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected internal server error during sample inference."
        )


# Clinician Feedback Submission Endpoint (Public for Clinicians)
@app.post("/api/feedback", status_code=status.HTTP_201_CREATED)
async def submit_clinician_feedback(payload: Dict[str, Any]):
    """
    Public clinician feedback endpoint.
    Passes JSON body directly to FeedbackService for validation, PII anonymization, and repository insertion.
    Does NOT log raw payload to protect PII.
    """
    if feedback_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback service is currently unavailable."
        )

    try:
        result = await run_in_threadpool(feedback_service.submit_feedback, payload)
        if result.get("status") == "Error":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result.get("message", "Feedback validation failed.")
            )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected failure submitting clinician feedback: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record clinician feedback due to internal server error."
        )


# Safe Neighbor Waveform Lookup Endpoint
@app.get("/api/labels")
async def get_label_registry():
    """Return the validated PTB-XL display-name registry used by the UI."""
    registry_path = PACKAGE_ROOT / "knowledge" / "label_abbreviation_registry.json"
    if not registry_path.is_file():
        raise HTTPException(status_code=503, detail="Diagnostic label registry unavailable")
    try:
        return JSONResponse(content=json.loads(registry_path.read_text(encoding="utf-8")))
    except Exception as exc:
        logger.error("Failed to load label registry: %s", exc)
        raise HTTPException(status_code=500, detail="Diagnostic label registry is invalid")


@app.get("/api/retrieval/neighbors/{ecg_id}/waveform")
async def get_neighbor_waveform(ecg_id: int):
    """
    Return the exact source waveform for a retrieved PTB-XL ECG ID.
    Prefer optional metadata, then use the canonical records100 ID layout.
    """
    metadata_path = Path(os.getenv("TRACE_ECG_METADATA_CSV", str(DATA_DIR / "ecg_metadata.csv"))).expanduser().resolve()
    api_package_root = Path(__file__).resolve().parent.parent
    bundled_records_root = api_package_root / "data" / "records100"
    configured_records_root = Path(os.getenv("TRACE_PTBXL_RECORDS100_DIR", str(bundled_records_root))).expanduser().resolve()
    # The self-contained deployment dataset is authoritative. A configured
    # external root is only a fallback when that specific record is absent.
    records_root = bundled_records_root.resolve()
    rec_path: Optional[Path] = None
    # PTB-XL's own registry is the authoritative ECG-ID to waveform mapping.
    ptbxl_registry = api_package_root / "data" / "ptbxl_database.csv"
    if ptbxl_registry.is_file():
        try:
            import csv
            with ptbxl_registry.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if int(row.get("ecg_id", -1)) == int(ecg_id):
                        filename_lr = str(row.get("filename_lr") or "").strip()
                        if filename_lr:
                            rec_path = api_package_root / "data" / filename_lr
                        break
        except Exception as exc:
            logger.warning("Unable to use PTB-XL registry %s: %s", ptbxl_registry, exc)

    if metadata_path.is_file():
        try:
            import pandas as pd
            df = pd.read_csv(metadata_path)
            matching = df[df["ecg_id"] == ecg_id]
            if rec_path is None and not matching.empty:
                candidate = Path(str(matching.iloc[0]["record_path"])).expanduser()
                rec_path = candidate if candidate.is_absolute() else records_root.parent / candidate
        except Exception as exc:
            logger.warning("Ignoring unusable ECG metadata CSV %s: %s", metadata_path, exc)

    if rec_path is None:
        bucket = (int(ecg_id) // 1000) * 1000
        relative_record = Path(f"{bucket:05d}") / f"{int(ecg_id):05d}_lr"
        candidate_roots = [records_root]
        if configured_records_root not in candidate_roots:
            candidate_roots.append(configured_records_root)
        rec_path = next(
            (root / relative_record for root in candidate_roots
             if (root / relative_record).with_suffix(".hea").is_file()
             and (root / relative_record).with_suffix(".dat").is_file()),
            records_root / relative_record,
        )

    if rec_path.suffix.lower() in {".hea", ".dat"}:
        rec_path = rec_path.with_suffix("")
    header_path = rec_path.with_suffix(".hea")
    data_path = rec_path.with_suffix(".dat")
    if not header_path.is_file() or not data_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original waveform files are missing for retrieved ECG ID {ecg_id}."
        )

    try:
        import wfdb
        record = wfdb.rdrecord(str(rec_path))
        signal = np.asarray(record.p_signal, dtype=np.float32).T
        if signal.ndim != 2 or signal.shape[0] != 12:
            raise ValueError(f"expected 12 leads, received shape {signal.shape}")
        lead_name_map = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}
        lead_order = [lead_name_map.get(str(name).upper(), str(name)) for name in (record.sig_name or [])]
        sampling_rate = int(round(float(record.fs)))
    except Exception as exc:
        logger.error("Unable to read retrieved ECG %s from %s: %s", ecg_id, rec_path, exc)
        raise HTTPException(status_code=500, detail=f"Waveform is present but could not be decoded for ECG ID {ecg_id}.")

    return {
        "ecg_id": ecg_id,
        "waveform_available": True,
        "sampling_rate_hz": sampling_rate,
        "lead_order": lead_order,
        "values": signal.tolist()
    }


# Clinician Feedback Review Queue List Endpoint (Administrator Only)
@app.get("/api/feedback/reviews")
async def get_feedback_reviews(
    status_filter: Optional[str] = None,
    admin_key: str = Security(verify_admin_api_key),
):
    """
    [ADMINISTRATOR ONLY] Retrieve pending/filtered feedback records from the review queue.
    Requires X-Admin-API-Key request header.
    """
    if feedback_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback service is currently unavailable."
        )

    try:
        if hasattr(feedback_service, "get_reviews"):
            reviews = await run_in_threadpool(feedback_service.get_reviews, status_filter)
        else:
            reviews = await run_in_threadpool(feedback_service.get_pending_reviews)
        return reviews
    except Exception as exc:
        logger.error("Error retrieving review queue records: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve review queue records."
        )


# Clinician Feedback Review Transition Endpoint (Administrator Only)
@app.post("/api/feedback/reviews/{feedback_id}")
async def review_feedback_item(
    feedback_id: int,
    body: ReviewFeedbackRequest,
    admin_key: str = Security(verify_admin_api_key),
):
    """
    [ADMINISTRATOR ONLY] Transition review status for a specific clinician feedback record.
    Requires X-Admin-API-Key request header.
    """
    if feedback_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback service is currently unavailable."
        )

    try:
        res = await run_in_threadpool(
            feedback_service.review_feedback,
            feedback_id=feedback_id,
            new_status=body.new_status,
            reviewer_id=body.reviewer_id,
            notes=body.notes,
        )
        return res
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc)
        )
    except Exception as exc:
        logger.error("Error updating review status for feedback_id %d: %s", feedback_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update review status."
        )


# Clinician Feedback Analytics Endpoint (Administrator Only)
@app.get("/api/feedback/analytics")
async def get_feedback_analytics(
    admin_key: str = Security(verify_admin_api_key),
):
    """
    [ADMINISTRATOR ONLY] Generate feedback analytics summary.
    Requires X-Admin-API-Key request header.
    """
    if feedback_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback service is currently unavailable."
        )

    try:
        report = await run_in_threadpool(feedback_service.get_analytics)
        return report
    except Exception as exc:
        logger.error("Error generating feedback analytics: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate feedback analytics report."
        )


# Clinician Feedback Approved Export Endpoint (Administrator Only)
@app.post("/api/feedback/export")
async def export_approved_feedback_data(
    admin_key: str = Security(verify_admin_api_key),
):
    """
    [ADMINISTRATOR ONLY] Export approved clinician feedback records to CSV/JSON format.
    Requires X-Admin-API-Key request header.
    """
    if feedback_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback service is currently unavailable."
        )

    try:
        export_res = await run_in_threadpool(feedback_service.export_approved_data)
        return export_res
    except Exception as exc:
        logger.error("Error exporting approved feedback: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export approved feedback data."
        )

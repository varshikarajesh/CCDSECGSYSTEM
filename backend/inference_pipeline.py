"""Fail-closed real-execution ECG classifier and FAISS retrieval pipeline."""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTBXL_DIR = os.path.join(PROJECT_ROOT, "ptbxl_five_superclass")
for path in (PROJECT_ROOT, PTBXL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import Config
from labels_stage5 import load_complete_scp_mapping
from model_multihead import Stage5MultiHeadECGResNet1D
from scp_family_mapping import SCPFamilyMapper
from deployment_config import (
    CLASSIFIER_CHECKPOINT_PATH,
    ENCODER_CHECKPOINT_PATH,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    ORDERED_SCP_LABELS_PATH,
)
from preprocessing.ecg_preprocessor import preprocess_raw_ecg
from retrieval.retrieval_encoder import CheckpointRetrievalEncoder, JointV7RetrievalEncoder
from retrieval.retrieval_wrapper import RetrievalWrapper
from runtime.runtime_contracts import make_json_safe, validate_ecg_array

logger = logging.getLogger(__name__)


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
        if checkpoint and all(isinstance(key, str) for key in checkpoint):
            return checkpoint
    raise RuntimeError("Checkpoint does not contain a recognizable state dictionary")


def load_retrieval_encoder(checkpoint_path, device):
    """Load the exact registered retrieval architecture; fail closed on unknown formats."""
    checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False);state=_extract_state_dict(checkpoint)
    if isinstance(checkpoint,dict) and checkpoint.get("version")=="V7_GRADED_PROJECTION" and checkpoint.get("model_class")=="JointECGModelV3":
        encoder=JointV7RetrievalEncoder().to(device);encoder.load_packaged_state_dict(state)
    else:
        encoder=CheckpointRetrievalEncoder().to(device);encoder.load_state_dict(state,strict=True)
    return encoder.eval()


class InferencePipeline:
    def __init__(self, config_path: Optional[str] = None):
        del config_path
        self.config = Config()
        requested = os.environ.get("TRACE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("TRACE_DEVICE requests CUDA but CUDA is unavailable")
        self.device = torch.device(requested)

        _, self.scp_codes = load_complete_scp_mapping(os.path.join(self.config.data_dir, self.config.scp_statements_csv))
        if len(self.scp_codes) != 71:
            raise RuntimeError(f"Expected 71 SCP labels, found {len(self.scp_codes)}")
        ordered_manifest = json.loads(ORDERED_SCP_LABELS_PATH.read_text(encoding="utf-8"))
        ordered_labels = ordered_manifest.get("labels", ordered_manifest) if isinstance(ordered_manifest, dict) else ordered_manifest
        if list(self.scp_codes) != list(ordered_labels):
            raise RuntimeError("Runtime SCP label order does not match the validated checkpoint manifest")
        self.family_mapper = SCPFamilyMapper(self.scp_codes)

        if not CLASSIFIER_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"Classifier checkpoint missing: {CLASSIFIER_CHECKPOINT_PATH}")
        self.model = Stage5MultiHeadECGResNet1D(
            self.config,
            num_scp=len(self.scp_codes),
            num_families=len(self.family_mapper.families),
            num_rare=22,
        )
        classifier_checkpoint = self.model.load_stage5_checkpoint(str(CLASSIFIER_CHECKPOINT_PATH), self.device)
        classifier_state = _extract_state_dict(classifier_checkpoint)
        if not any(key in classifier_state for key in ("head_family.weight", "fc_family.weight")):
            raise RuntimeError("Selected classifier checkpoint does not contain a trained independent family head")
        self.model.to(self.device).eval()

        if not ENCODER_CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"Retrieval checkpoint missing: {ENCODER_CHECKPOINT_PATH}")
        encoder = load_retrieval_encoder(ENCODER_CHECKPOINT_PATH,self.device)


        # Public contract verification
        dummy = torch.zeros((1, 12, 1000), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = encoder(dummy)
        if out.shape != (1, 128) or not torch.isfinite(out).all():
            raise RuntimeError(f"Retrieval encoder produced invalid tensor shape/values: {out.shape}")

        norm = torch.linalg.vector_norm(out, dim=-1).item()
        if abs(norm - 1.0) > 1e-3:
            raise RuntimeError(f"Retrieval encoder output is not L2-normalized: norm={norm}")

        self.retrieval_encoder = encoder


        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("The real FAISS Python package is required") from exc
        if not FAISS_INDEX_PATH.is_file() or not FAISS_METADATA_PATH.is_file():
            raise FileNotFoundError("Matched FAISS index and metadata are required")
        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        metadata = json.loads(FAISS_METADATA_PATH.read_text(encoding="utf-8"))
        if faiss_index.d != 128:
            raise RuntimeError(f"FAISS dimension {faiss_index.d} does not match the 128-D retrieval encoder")
        if faiss_index.ntotal != len(metadata):
            raise RuntimeError(f"FAISS/metadata mismatch: index={faiss_index.ntotal}, metadata={len(metadata)}")
        if any(item.get("faiss_row") != idx for idx, item in enumerate(metadata)):
            raise RuntimeError("FAISS metadata rows are not sequential and aligned")
        self.retrieval_wrapper = RetrievalWrapper(self.retrieval_encoder, faiss_index, metadata)

    def run(
        self,
        ecg_signal: Any,
        metadata: Optional[Dict[str, Any]] = None,
        include_retrieval: bool = True,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        meta = dict(metadata or {})
        effective_top_k = int(meta.get("top_k", top_k))
        ecg = validate_ecg_array(np.asarray(ecg_signal), source_module="backend.inference_pipeline")
        if ecg.shape[0] != 12:
            ecg = ecg.T
        if not np.isfinite(ecg).all():
            raise ValueError("ECG contains NaN or infinite values")

        sampling_rate = int(meta.get("sampling_rate_hz", meta.get("sampling_rate", 100)))
        prep = preprocess_raw_ecg(ecg, sampling_rate=sampling_rate)
        quality = prep["quality"]
        signal = prep.get("preprocessed_signal", prep.get("signal"))
        if signal is None or tuple(signal.shape) != (12, 1000):
            raise RuntimeError(f"Preprocessing did not produce (12, 1000): {getattr(signal, 'shape', None)}")
        tensor = torch.from_numpy(np.asarray(signal, dtype=np.float32)).unsqueeze(0).to(self.device)

        self.model.eval()
        self.retrieval_encoder.eval()

        with torch.inference_mode():
            output = self.model(tensor)
        scp_probs = output["scp_predictions"].squeeze(0).cpu().numpy()
        family_probs = output["family_predictions"].squeeze(0).cpu().numpy()
        embedding = output["embeddings"].squeeze(0).cpu().numpy()
        scp = {code: float(scp_probs[idx]) for idx, code in enumerate(self.scp_codes)}
        families = {name: float(family_probs[idx]) for idx, name in enumerate(self.family_mapper.families)}

        raw_retrieval = {"raw_neighbors": [], "top_k": 0, "available": False}
        if include_retrieval:
            raw_retrieval = self.retrieval_wrapper.search(
                tensor,
                top_k=effective_top_k,
                exclude_patient_id=meta.get("patient_id"),
                exclude_ecg_id=meta.get("ecg_id"),
            )
            raw_retrieval["available"] = True

        sorted_scp = sorted(scp.items(), key=lambda item: item[1], reverse=True)
        sorted_family = sorted(families.items(), key=lambda item: item[1], reverse=True)
        return make_json_safe({
            "status": "ok",
            "classifier": {
                "probabilities": scp,
                "primary_label": sorted_scp[0][0],
                "primary_probability": sorted_scp[0][1],
                "mapped_family": self.family_mapper.get_family_name(sorted_scp[0][0]),
                "top_predictions": [{"label": label, "probability": prob} for label, prob in sorted_scp[:5]],
            },
            "family_head": {
                "probabilities": families,
                "primary_family": sorted_family[0][0],
                "primary_probability": sorted_family[0][1],
            },
            "model_embedding": embedding,
            "retrieval": raw_retrieval,
            "signal_quality": quality,
            "preprocessing": {
                "sampling_rate_hz": 100,
                "processed_shape": [12, 1000],
                "version": prep.get("preprocessing_version"),
            },
            "metadata": meta,
        })

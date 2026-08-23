# -*- coding: utf-8 -*-
"""
retrieval/retrieval_wrapper.py

Production wrapper for Retrieval Branch.
Executes query embedding computation, queries FAISS index, excludes same-patient records,
and returns RAW retrieval results preserving raw FAISS ranks and similarities.
Computes dynamic verification status against active checkpoint and index build manifest.
Does NOT consume diagnosis probabilities, reranked results, or ground truth.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import torch

from retrieval.retrieval_encoder import CheckpointRetrievalEncoder
from preprocessing.ecg_preprocessor import compute_signal_hash
from deployment_config import ENCODER_CHECKPOINT_PATH, FAISS_INDEX_PATH

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def _get_checkpoint_sha256(ckpt_path: Path) -> str:
    """Computes SHA-256 hash of checkpoint file."""
    if not ckpt_path.is_file():
        return "missing"
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class RetrievalWrapper:
    """Raw FAISS Query Retrieval Wrapper with dynamic provenance verification."""

    def __init__(
        self,
        encoder: Any,
        faiss_index: Any = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ):
        self.encoder = encoder
        self.faiss_index = faiss_index
        self.metadata_list = metadata_list or []

    @torch.no_grad()
    def search(
        self,
        retrieval_input_tensor: torch.Tensor,
        top_k: int = 5,
        exclude_patient_id: Optional[float] = None,
        exclude_ecg_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Input: (1, 12, 1000) preprocessed retrieval tensor.
        Returns raw retrieval results preserving raw FAISS ranks and similarities.
        """
        self.encoder.eval()
        device = next(self.encoder.parameters()).device
        tensor = retrieval_input_tensor.to(device)

        # Forward pass outputs 128D L2-normalized vector
        query_embedding_tensor = self.encoder(tensor)  # (1, 128)
        query_embedding_np = query_embedding_tensor.cpu().numpy()[0]  # (128,)

        query_checksum = compute_signal_hash(query_embedding_np)
        raw_neighbors: List[Dict[str, Any]] = []

        if self.faiss_index is not None and len(self.metadata_list) > 0:
            query_matrix = query_embedding_np.reshape(1, -1).astype(np.float32)
            k_search = min(top_k * 5, self.faiss_index.ntotal)
            similarities, indices = self.faiss_index.search(query_matrix, k_search)

            sims_row = similarities[0]
            idxs_row = indices[0]

            rank_counter = 1
            for sim, idx in zip(sims_row, idxs_row):
                if idx < 0 or idx >= len(self.metadata_list):
                    continue

                meta = self.metadata_list[idx]
                pid = meta.get("patient_id", None)
                eid = meta.get("ecg_id", None)

                # Patient & ECG exclusion filter
                if exclude_patient_id is not None and pid is not None and pid == exclude_patient_id:
                    continue
                if exclude_ecg_id is not None and eid is not None and eid == exclude_ecg_id:
                    continue

                # Normalization check: if similarities are bounded inner-products on L2-normalized vectors
                raw_sim = float(sim)

                raw_neighbors.append({
                    "raw_rank": rank_counter,
                    "faiss_row": int(idx),
                    "raw_similarity": round(raw_sim, 4),
                    "ecg_id": eid,
                    "patient_id": pid,
                    "scp_codes": meta.get("scp_codes", []),
                    "families": meta.get("families",meta.get("diagnostic_classes", [])),
                    "rhythm_labels": meta.get("rhythm_labels", []),
                    "morphology_labels": meta.get("morphology_labels", [])
                })

                rank_counter += 1
                if len(raw_neighbors) >= top_k:
                    break
        else:
            raise RuntimeError("Real FAISS index and aligned metadata are required; mock retrieval is disabled")

        metric_type_name = "INNER_PRODUCT"
        if self.faiss_index is not None and hasattr(self.faiss_index, "metric_type"):
            if getattr(self.faiss_index, "metric_type") == 1:
                metric_type_name = "L2"

        # Dynamic verification check against active checkpoint and index build manifest
        ckpt_sha256 = _get_checkpoint_sha256(ENCODER_CHECKPOINT_PATH)
        manifest_path = FAISS_INDEX_PATH.parent / "faiss_manifest.json"

        is_compatible = False
        ver_status = "unverified"
        reason = "FAISS index unverified against active retrieval encoder checkpoint"

        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_ckpt = manifest.get("checkpoint_sha256")
                manifest_vcount = manifest.get("faiss_config", {}).get("vector_count") or manifest.get("vector_count")
                manifest_meta_count = manifest.get("audit", {}).get("metadata_row_count") or manifest.get("metadata_row_count")

                if (
                    manifest_ckpt == ckpt_sha256 and
                    manifest_vcount == self.faiss_index.ntotal and
                    manifest_meta_count == len(self.metadata_list) and
                    self.faiss_index.ntotal == len(self.metadata_list) and
                    self.faiss_index.ntotal > 500  # Requires full dataset index (e.g. PTB-XL >500)
                ):
                    is_compatible = True
                    ver_status = "verified"
                    reason = "FAISS production index dynamically verified against the active retrieval checkpoint and aligned metadata manifest."
                else:
                    reason = (
                        f"FAISS manifest mismatch or partial dataset index (index={self.faiss_index.ntotal}, "
                        f"metadata={len(self.metadata_list)}, manifest_ckpt_match={manifest_ckpt == ckpt_sha256})"
                    )
            except Exception as exc:
                reason = f"Failed to parse FAISS manifest: {exc}"
        else:
            reason = "FAISS candidate index uses candidate embeddings; unverified training provenance against active 3-stream conv encoder"

        retrieval_provenance = {
            "encoder_index_compatible": is_compatible,
            "verification_status": ver_status,
            "checkpoint_sha256": ckpt_sha256,
            "preprocessing_version": "1.0",
            "faiss_metric": metric_type_name,
            "reason": reason
        }

        return {
            "raw_neighbors": raw_neighbors,
            "query_embedding_checksum": query_checksum,
            "retrieval_provenance": retrieval_provenance,
        }

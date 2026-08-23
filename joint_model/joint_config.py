# -*- coding: utf-8 -*-
"""
joint_model/joint_config.py

Configuration data structure for Joint V3 ECG Model Training & Evaluation.
Routes exploratory and corrected run directories and holds strict tolerances.
"""

import os
from pathlib import Path
import torch
from dataclasses import dataclass, field
from deployment_config import PACKAGE_ROOT, DATA_DIR, CLASSIFIER_CHECKPOINT_PATH, ENCODER_CHECKPOINT_PATH

FAISS_V3_DIR = PACKAGE_ROOT / "faiss" / "v3"


@dataclass
class JointConfigV3:
    """Configuration data structure for Joint V3 ECG Model Training & Evaluation."""
    
    run_type: str = "corrected"  # Default to corrected for final run
    
    # Compute Hardware Controls
    device: str = "cuda"
    require_cuda_for_training: bool = True
    mixed_precision: bool = True
    amp_dtype: str = "float16"
    pin_memory: bool = True
    non_blocking_transfers: bool = True
    num_workers: int = 4
    persistent_workers: bool = True
    
    # Model Hyperparameters
    num_scp: int = 71
    num_superclasses: int = 5
    num_clinical_families: int = 10
    dropout: float = 0.20
    joint_embedding_dim: int = 128
    retrieval_embedding_dim: int = 128
    
    # Gradient Isolation in Gate Network
    detach_class_probabilities_in_gate: bool = True
    
    # Checkpoints & Pretrained Initialization
    classifier_checkpoint_path: Path = field(default_factory=lambda: CLASSIFIER_CHECKPOINT_PATH)
    retrieval_checkpoint_path: Path = field(default_factory=lambda: ENCODER_CHECKPOINT_PATH)
    
    # Staged Training Schedule
    v3_stage: str = "V3-B"
    batch_size: int = 32
    grad_accum_steps: int = 1
    stage_a_epochs: int = 5
    stage_b_epochs: int = 10
    early_stopping_patience: int = 3
    stop_after_stage_b_epoch: int = 10
    
    # Learning Rates
    lr_fusion: float = 2e-4
    lr_classifier: float = 2e-5
    lr_retrieval: float = 2e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    
    # Loss Weights
    w_scp: float = 1.0
    w_superclass: float = 1.0
    w_clinical_family: float = 0.5
    w_supcon: float = 0.5
    w_triplet: float = 0.2
    w_consistency: float = 0.1
    temperature_supcon: float = 0.07
    triplet_margin: float = 0.30
    
    # Validation Score Metric Weights (Composite Score Selection)
    score_w_auprc: float = 0.35
    score_w_auroc: float = 0.20
    score_w_recall5: float = 0.20
    score_w_mrr: float = 0.10
    
    # Safety drop tolerances
    maximum_superclass_macro_auprc_drop: float = 0.02
    maximum_per_class_sensitivity_drop: float = 0.05
    minimum_diagnostic_recall_at_5: float = 0.70
    
    # Output Artifact Paths
    output_dir: Path = field(default_factory=lambda: PACKAGE_ROOT / "outputs" / "joint_v3")
    best_checkpoint_path: Path = field(default_factory=lambda: PACKAGE_ROOT / "outputs" / "joint_v3" / "checkpoints" / "joint_v3_best.pth")
    last_checkpoint_path: Path = field(default_factory=lambda: PACKAGE_ROOT / "outputs" / "joint_v3" / "checkpoints" / "joint_v3_last.pth")
    
    # FAISS v3 Directory Artifacts
    faiss_v3_dir: Path = field(default_factory=lambda: FAISS_V3_DIR)
    faiss_joint_index_path: Path = field(default_factory=lambda: FAISS_V3_DIR / "faiss_joint_v3.bin")
    faiss_joint_metadata_path: Path = field(default_factory=lambda: FAISS_V3_DIR / "faiss_joint_v3_metadata.json")
    faiss_joint_manifest_path: Path = field(default_factory=lambda: FAISS_V3_DIR / "faiss_joint_v3_manifest.json")
    
    # Seed
    seed: int = 42

    def __post_init__(self):
        # Rename any existing legacy outputs to avoid loading old corrupt checkpoints
        legacy_dir = PACKAGE_ROOT / "outputs" / "joint_v3_invalid_previous"
        default_output = PACKAGE_ROOT / "outputs" / "joint_v3"
        if default_output.is_dir() and not legacy_dir.is_dir():
            best_pth = default_output / "checkpoints" / "joint_v3_best.pth"
            if best_pth.is_file():
                try:
                    default_output.rename(legacy_dir)
                    print(f"Proactively renamed legacy output directory {default_output.name} to {legacy_dir.name}")
                except Exception as e:
                    print(f"Warning: Proactive legacy directory rename failed: {e}")

        # Route output directory based on run_type
        if self.run_type == "exploratory":
            self.output_dir = PACKAGE_ROOT / "outputs" / "joint_v3" / "runs" / "exploratory_stage_b_10"
            self.best_checkpoint_path = self.output_dir / "checkpoints" / "joint_v3_exploratory_best.pth"
            self.last_checkpoint_path = self.output_dir / "checkpoints" / "joint_v3_exploratory_last.pth"
        elif self.run_type == "corrected":
            self.output_dir = PACKAGE_ROOT / "outputs" / "joint_v3" / "runs" / "corrected_joint_hierarchical_resnet"
            self.best_checkpoint_path = self.output_dir / "checkpoints" / "joint_v3_best.pth"
            self.last_checkpoint_path = self.output_dir / "checkpoints" / "joint_v3_last.pth"

        # Deployment is inference-only. Never create historical training or
        # FAISS-V3 output directories while loading the V7 architecture.

    def validate_compute_device(self) -> torch.device:
        """Enforces require_cuda_for_training when configured."""
        cuda_available = torch.cuda.is_available()
        if self.require_cuda_for_training and not cuda_available:
            raise RuntimeError(
                "CRITICAL: require_cuda_for_training is set to True, but PyTorch cannot detect a CUDA GPU. "
                "Failing closed to prevent accidental silent CPU execution."
            )
        if self.device == "cuda" and cuda_available:
            return torch.device("cuda")
        return torch.device("cpu")

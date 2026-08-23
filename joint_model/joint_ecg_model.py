# -*- coding: utf-8 -*-
"""
joint_model/joint_ecg_model.py

Joint V3 ECG Classification-and-Retrieval Neural Architecture.
Features 3 Explicit Heads:
  1. 71 SCP Statement Head (scp_logits)
  2. 5 Diagnostic Superclass Primary Head (superclass_logits: NORM, MI, STTC, CD, HYP)
  3. 10 Clinical Family Auxiliary Head (clinical_family_logits)

Gated Fusion enforces gradient detachment on class probabilities (.detach()) to protect
classification head calibration from contrastive/triplet retrieval gradients.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from ptbxl_five_superclass.model_multihead import Stage5MultiHeadECGResNet1D
from ptbxl_five_superclass.config import Config as ClassifierConfig
from retrieval.retrieval_encoder import CheckpointRetrievalEncoder
from joint_model.joint_config import JointConfigV3


class JointECGModelV3(nn.Module):
    """
    Joint V3 ECG Architecture combining ResNet-1D classifier, Checkpoint-faithful Retrieval Encoder,
    and a gradient-isolated gated fusion layer projecting a 128D L2-normalized joint embedding.
    """

    def __init__(self, joint_config: JointConfigV3):
        super().__init__()
        self.config = joint_config

        # 1. Sub-encoders
        cls_cfg = ClassifierConfig()
        self.classifier_encoder = Stage5MultiHeadECGResNet1D(
            config=cls_cfg,
            num_scp=joint_config.num_scp,
            num_families=joint_config.num_clinical_families,
            num_rare=22,
            disable_dropout=False
        )

        self.retrieval_encoder = CheckpointRetrievalEncoder()

        # 2. Primary 5-Superclass Head (NORM, MI, STTC, CD, HYP)
        self.head_superclass = nn.Linear(cls_cfg.embedding_dim, joint_config.num_superclasses)

        # 3. Classifier Embedding Projection (128 -> 128)
        self.classifier_projection = nn.Sequential(
            nn.Linear(cls_cfg.embedding_dim, joint_config.joint_embedding_dim),
            nn.LayerNorm(joint_config.joint_embedding_dim),
            nn.GELU()
        )

        # 4. Trainable Gate Network
        # Input: cls_proj (128) + ret_emb (128) + superclass_probs (5) + clinical_family_probs (10) = 271
        gate_in_dim = (
            joint_config.joint_embedding_dim
            + joint_config.retrieval_embedding_dim
            + joint_config.num_superclasses
            + joint_config.num_clinical_families
        )
        self.gate_network = nn.Sequential(
            nn.Linear(gate_in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(joint_config.dropout),
            nn.Linear(128, joint_config.joint_embedding_dim)
        )

        # 5. Joint Retrieval Projection Layer
        self.joint_projection = nn.Sequential(
            nn.Linear(joint_config.joint_embedding_dim, joint_config.joint_embedding_dim),
            nn.LayerNorm(joint_config.joint_embedding_dim)
        )

    def forward(self, ecg: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        ecg: (B, 12, 1000)
        Returns structured dictionary containing all 3 head logits/probabilities, sub-embeddings, and joint_embedding.
        """
        # Forward pass through classifier
        cls_output = self.classifier_encoder(ecg)
        cls_emb = cls_output["embeddings"]                         # (B, 128)
        scp_logits = cls_output["scp_logits"]                      # (B, 71)
        clinical_family_logits = cls_output["family_logits"]        # (B, 10)
        clinical_family_probs = cls_output["family_predictions"]    # (B, 10)

        # Primary 5-superclass predictions
        features = self.classifier_encoder.head_drop(cls_emb)
        superclass_logits = self.head_superclass(features)         # (B, 5)
        superclass_probs = torch.sigmoid(superclass_logits)        # (B, 5)

        # Forward pass through retrieval encoder
        ret_emb = self.retrieval_encoder(ecg)                      # (B, 128)

        # Project classifier embedding
        cls_proj = self.classifier_projection(cls_emb)             # (B, 128)

        # Detach class probabilities inside gate to isolate classification heads from retrieval gradients
        if self.config.detach_class_probabilities_in_gate:
            s_probs_gate = superclass_probs.detach()
            cf_probs_gate = clinical_family_probs.detach()
        else:
            s_probs_gate = superclass_probs
            cf_probs_gate = clinical_family_probs

        gate_input = torch.cat([cls_proj, ret_emb, s_probs_gate, cf_probs_gate], dim=-1) # (B, 271)
        gate = torch.sigmoid(self.gate_network(gate_input))                                # (B, 128)

        fused_embedding = gate * cls_proj + (1.0 - gate) * ret_emb                         # (B, 128)
        joint_embedding = F.normalize(self.joint_projection(fused_embedding), p=2, dim=-1)   # (B, 128)

        return {
            "scp_logits": scp_logits,
            "superclass_logits": superclass_logits,
            "superclass_probabilities": superclass_probs,
            "clinical_family_logits": clinical_family_logits,
            "clinical_family_probabilities": clinical_family_probs,
            "classifier_embedding": cls_proj,
            "retrieval_embedding": ret_emb,
            "joint_embedding": joint_embedding,
            "gate": gate,
        }

    def load_pretrained_checkpoints(self, device: torch.device):
        """Loads baseline classifier and retrieval weights using explicit map_location."""
        if self.config.classifier_checkpoint_path.is_file():
            self.classifier_encoder.load_stage5_checkpoint(
                str(self.config.classifier_checkpoint_path), device=device
            )
            print(f"Loaded baseline classifier checkpoint: {self.config.classifier_checkpoint_path.name}")

        if self.config.retrieval_checkpoint_path.is_file():
            ret_ckpt = torch.load(self.config.retrieval_checkpoint_path, map_location=device, weights_only=False)
            state_dict = ret_ckpt.get("model_state_dict", ret_ckpt)
            
            # Extract and log keys
            model_state = self.retrieval_encoder.state_dict()
            missing = [k for k in model_state if k not in state_dict]
            unexpected = [k for k in state_dict if k not in model_state]
            
            print(f"Retrieval Loader - Missing keys: {missing}")
            print(f"Retrieval Loader - Unexpected keys: {unexpected}")
            
            self.retrieval_encoder.load_state_dict(state_dict, strict=False)
            print(f"Loaded baseline retrieval checkpoint: {self.config.retrieval_checkpoint_path.name}")

    def freeze_encoders(self):
        """Stage A: Freeze both pretrained encoders. Train only gate, projection, and superclass head."""
        for param in self.classifier_encoder.parameters():
            param.requires_grad = False
        for param in self.retrieval_encoder.parameters():
            param.requires_grad = False

        for param in self.head_superclass.parameters():
            param.requires_grad = True
        for param in self.classifier_projection.parameters():
            param.requires_grad = True
        for param in self.gate_network.parameters():
            param.requires_grad = True
        for param in self.joint_projection.parameters():
            param.requires_grad = True

    def unfreeze_upper_stages(self):
        """Stage B: Unfreeze upper encoder stages."""
        for param in self.classifier_encoder.stage3.parameters():
            param.requires_grad = True
        for param in self.classifier_encoder.fc1.parameters():
            param.requires_grad = True
        for param in self.classifier_encoder.head_scp.parameters():
            param.requires_grad = True
        for param in self.classifier_encoder.head_family.parameters():
            param.requires_grad = True
        for param in self.retrieval_encoder.fuse.parameters():
            param.requires_grad = True
        for param in self.retrieval_encoder.proj.parameters():
            param.requires_grad = True


def configure_training_mode(model, stage: str):
    """Configures explicit training modes for warmup stage to keep sub-encoders in eval mode."""
    model.train()

    if stage == "stage_a":
        model.classifier_encoder.eval()
        model.retrieval_encoder.eval()

        model.classifier_projection.train()
        model.head_superclass.train()
        model.gate_network.train()
        model.joint_projection.train()

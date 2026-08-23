# -*- coding: utf-8 -*-
"""
retrieval/retrieval_encoder.py

Hierarchical 1D Convolutional Retrieval Encoder for 12-lead ECG signals.
Includes Version 1 HierarchicalEncoder and ImprovedHierarchicalEncoder
with Squeeze-and-Excitation attention and learned branch fusion.
Produces 128-dimensional L2-normalized embedding vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SqueezeExcitation1D(nn.Module):
    """
    Squeeze-and-Excitation block for 1D feature maps.
    Computes temporal average pooling followed by two FC layers to rescale channels.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced_dim = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, reduced_dim, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(reduced_dim, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        w = self.avg_pool(x).squeeze(-1)           # (b, c)
        w = self.act(self.fc1(w))                  # (b, c // reduction)
        w = self.sigmoid(self.fc2(w)).unsqueeze(-1) # (b, c, 1)
        return x * w


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with Conv1D, BatchNorm, GELU, Dropout, Conv1D, BatchNorm, SE attention.
    Shortcut uses 1x1 Conv + BatchNorm when stride > 1 or in_channels != out_channels.
    Same padding is used for odd kernel sizes (kernel_size // 2).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.10,
        use_se: bool = False,
        reduction: int = 8
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if use_se:
            self.se = SqueezeExcitation1D(out_channels, reduction=reduction)
        else:
            self.se = nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.act(out + shortcut)


class BranchAttention(nn.Module):
    """
    Learned branch-attention module predicting scalar importance weights
    for Rhythm, Morphology, and Global physiological branches.
    Input: (B, 3, 128)
    Output: (B, 128) weighted sum, and (B, 3) softmax attention weights.
    """
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, stacked_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # stacked_features: (B, 3, 128)
        logits = self.attn_net(stacked_features)       # (B, 3, 1)
        weights = torch.softmax(logits, dim=1)         # (B, 3, 1) across the 3 branches
        weighted_feat = torch.sum(stacked_features * weights, dim=1)  # (B, 128)
        return weighted_feat, weights.squeeze(-1)      # (B, 128), (B, 3)


class ImprovedHierarchicalEncoder(nn.Module):
    """
    Improved Version 1 Hierarchical ECG Encoder.
    
    Physiological Branches:
    1. Rhythm Branch: Large temporal kernels (15, 11) for rhythm regularity, atrial activity, timing.
    2. Morphology Branch: Small temporal kernels (5, 3) for QRS, ST-T, waveform detail.
    3. Global Branch: Medium temporal kernels (9, 7, 5) for whole-ECG context.
    
    Fusion:
    - SqueezeExcitation1D (reduction ratio 8) per branch
    - AdaptiveAvgPool1d(1) per branch -> 128D per branch
    - Branch Attention: Linear(128, 64) -> GELU -> Linear(64, 1) -> Softmax
    - Concatenation of all 3 branches (384D)
    - Fusion Head: Linear(384 + 128, 256) -> LayerNorm(256) -> GELU -> Dropout(0.20) -> Linear(256, 128)
    - Final L2 normalization -> (B, 128)
    """
    def __init__(self, in_channels: int = 12, embed_dim: int = 128, dropout: float = 0.20):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Rhythm Branch
        self.rhythm_stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.rhythm_res1 = ResidualBlock1D(32, 64, kernel_size=15, stride=2, dropout=0.10)
        self.rhythm_res2 = ResidualBlock1D(64, 128, kernel_size=11, stride=2, dropout=0.10)
        self.rhythm_se = SqueezeExcitation1D(128, reduction=8)
        self.rhythm_pool = nn.AdaptiveAvgPool1d(1)

        # 2. Morphology Branch
        self.morph_stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU()
        )
        self.morph_res1 = ResidualBlock1D(32, 64, kernel_size=5, stride=2, dropout=0.10)
        self.morph_res2 = ResidualBlock1D(64, 128, kernel_size=3, stride=2, dropout=0.10)
        self.morph_se = SqueezeExcitation1D(128, reduction=8)
        self.morph_pool = nn.AdaptiveAvgPool1d(1)

        # 3. Global Branch
        self.global_stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.global_res1 = ResidualBlock1D(64, 128, kernel_size=7, stride=2, dropout=0.10)
        self.global_res2 = ResidualBlock1D(128, 128, kernel_size=5, stride=2, dropout=0.10)
        self.global_se = SqueezeExcitation1D(128, reduction=8)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 4. Learned Branch Fusion
        self.branch_attention = BranchAttention(128)
        self.fusion_head = nn.Sequential(
            nn.Linear(384 + 128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim)
        )

        # Optional classification head for family loss during staged training
        self.family_head: Optional[nn.Module] = None

    def create_family_head(self, num_families: int):
        self.family_head = nn.Linear(self.embed_dim, num_families)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_family_logits: bool = False
    ) -> torch.Tensor:
        """
        x: (B, 12, 1000)
        Returns L2-normalized embedding (B, 128). Optionally returns attention weights and family logits.
        """
        # Rhythm path
        r = self.rhythm_stem(x)
        r = self.rhythm_res1(r)
        r = self.rhythm_res2(r)
        r = self.rhythm_se(r)
        r_feat = self.rhythm_pool(r).squeeze(-1)   # (B, 128)

        # Morphology path
        m = self.morph_stem(x)
        m = self.morph_res1(m)
        m = self.morph_res2(m)
        m = self.morph_se(m)
        m_feat = self.morph_pool(m).squeeze(-1)   # (B, 128)

        # Global path
        g = self.global_stem(x)
        g = self.global_res1(g)
        g = self.global_res2(g)
        g = self.global_se(g)
        g_feat = self.global_pool(g).squeeze(-1)   # (B, 128)

        # Stack branch features: (B, 3, 128)
        stacked = torch.stack([r_feat, m_feat, g_feat], dim=1)
        weighted_feat, attn_weights = self.branch_attention(stacked)  # (B, 128), (B, 3)

        # Concatenate raw branch features: (B, 384)
        concat_feat = torch.cat([r_feat, m_feat, g_feat], dim=-1)

        # Combined representation: (B, 512)
        fused_input = torch.cat([concat_feat, weighted_feat], dim=-1)
        raw_embed = self.fusion_head(fused_input)  # (B, 128)

        l2_embed = F.normalize(raw_embed, p=2, dim=-1)

        outputs = [l2_embed]
        if return_attention:
            outputs.append(attn_weights)
        if return_family_logits:
            fam_logits = self.family_head(l2_embed) if self.family_head is not None else None
            outputs.append(fam_logits)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class HierarchicalEncoder(ImprovedHierarchicalEncoder):
    """Alias for Version 1 / Improved Hierarchical Encoder backwards compatibility."""
    pass


class RetrievalEncoder1D(nn.Module):
    """Legacy 1D CNN Retrieval Encoder producing 128D L2-normalized embeddings."""
    def __init__(self, in_channels: int = 12, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        pooled = self.global_pool(x).squeeze(-1)
        raw_embed = self.projection(pooled)
        return F.normalize(raw_embed, p=2, dim=-1)


class RetrievalConvBlock(nn.Module):
    """1D Conv + BatchNorm + ReLU sub-block matching checkpoint structure."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 2, padding: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CheckpointRetrievalEncoder(nn.Module):
    """
    Checkpoint-faithful 3-stream 1D Convolutional Retrieval Encoder for 12-lead ECG signals.
    Matches retrieval/best_retrieval.pth state_dict keys and exact shapes strictly:
    - rhythm.0.conv: Conv1d(12, 32, kernel_size=15, stride=1, padding=7)
    - rhythm.0.bn: BatchNorm1d(32)
    - rhythm.1.conv: Conv1d(32, 64, kernel_size=15, stride=1, padding=7)
    - rhythm.1.bn: BatchNorm1d(64)
    - morphology.0.conv: Conv1d(12, 32, kernel_size=5, stride=1, padding=2)
    - morphology.0.bn: BatchNorm1d(32)
    - morphology.1.conv: Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
    - morphology.1.bn: BatchNorm1d(64)
    - global_enc.0.conv: Conv1d(12, 64, kernel_size=9, stride=1, padding=4)
    - global_enc.0.bn: BatchNorm1d(64)
    - fuse: nn.Linear(192, 256)
    - proj: nn.Linear(256, 128)
    - Output: (B, 128) L2-normalized float32 tensor
    """
    def __init__(self):
        super().__init__()
        self.rhythm = nn.Sequential(
            RetrievalConvBlock(12, 32, kernel_size=15, stride=1, padding=7),
            RetrievalConvBlock(32, 64, kernel_size=15, stride=1, padding=7),
        )
        self.morphology = nn.Sequential(
            RetrievalConvBlock(12, 32, kernel_size=5, stride=1, padding=2),
            RetrievalConvBlock(32, 64, kernel_size=5, stride=1, padding=2),
        )
        self.global_enc = nn.Sequential(
            RetrievalConvBlock(12, 64, kernel_size=9, stride=1, padding=4),
        )
        self.fuse = nn.Linear(192, 256)
        self.proj = nn.Linear(256, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 12, 1000)
        r_out = self.rhythm(x)
        m_out = self.morphology(x)
        g_out = self.global_enc(x)

        r_feat = torch.mean(r_out, dim=-1)
        m_feat = torch.mean(m_out, dim=-1)
        g_feat = torch.mean(g_out, dim=-1)

        combined = torch.cat([r_feat, m_feat, g_feat], dim=-1)
        fused = self.fuse(combined)
        projected = self.proj(fused)
        return F.normalize(projected, p=2, dim=-1)


class JointV7RetrievalEncoder(nn.Module):
    """Deployment adapter exposing only V7's normalized 128-D joint embedding."""
    def __init__(self):
        super().__init__()
        from joint_model.joint_config import JointConfigV3
        from joint_model.joint_ecg_model import JointECGModelV3
        self.model = JointECGModelV3(JointConfigV3())

    def load_packaged_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        embedding = output["joint_embedding"]
        return F.normalize(embedding.float(), p=2, dim=-1)



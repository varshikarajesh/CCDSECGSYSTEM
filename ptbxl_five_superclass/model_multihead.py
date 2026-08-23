"""
Multi-Head ECG ResNet1D for Joint SCP, Clinical Family, and Rare Label Specialist Classification.
Architecture:
Shared 1D ResNet Encoder -> 128-D Embedding -> Parallel Heads:
├── SCP Head (71 SCPs)
├── Clinical Family Head (10 Families)
└── Rare Label Specialist Head (N_rare Labels)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional

try:
    from config import Config
except ModuleNotFoundError:
    from ptbxl_five_superclass.config import Config


class ResidualBlock1D(nn.Module):
    """1D Residual Block for Stage5MultiHeadECGResNet1D."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.10
    ):
        super().__init__()
        padding1 = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        kernel_size2 = 5
        padding2 = kernel_size2 // 2
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size2, stride=1, padding=padding2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.act(out + res)



class Stage5MultiHeadECGResNet1D(nn.Module):
    """
    Multi-Head 1D Residual CNN for PTB-XL ECG Classification.
    Shared Encoder -> 128-D Embedding -> Parallel Heads.
    """
    def __init__(
        self, 
        config: Config, 
        num_scp: int = 71, 
        num_families: int = 10, 
        num_rare: int = 15,
        disable_dropout: bool = False
    ):
        super().__init__()
        self.config = config
        self.num_scp = num_scp
        self.num_families = num_families
        self.num_rare = num_rare
        p_dropout = 0.0 if disable_dropout else config.dropout
        
        # Stem
        self.stem_conv = nn.Conv1d(
            in_channels=config.input_channels,
            out_channels=config.stem_channels,
            kernel_size=15,
            stride=2,
            padding=7,
            bias=False
        )
        self.stem_bn = nn.BatchNorm1d(config.stem_channels)
        self.stem_act = nn.GELU()
        self.stem_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual Stage 1: 64 channels
        self.stage1 = self._make_stage(
            in_channels=config.stem_channels,
            out_channels=config.stage_channels[0],
            num_blocks=config.blocks_per_stage[0],
            stride=1,
            dropout=p_dropout
        )
        
        # Residual Stage 2: 128 channels
        self.stage2 = self._make_stage(
            in_channels=config.stage_channels[0],
            out_channels=config.stage_channels[1],
            num_blocks=config.blocks_per_stage[1],
            stride=2,
            dropout=p_dropout
        )
        
        # Residual Stage 3: 256 channels
        self.stage3 = self._make_stage(
            in_channels=config.stage_channels[1],
            out_channels=config.stage_channels[2],
            num_blocks=config.blocks_per_stage[2],
            stride=2,
            dropout=p_dropout
        )
        
        # Shared Embedding Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(config.stage_channels[2], config.embedding_dim)
        self.head_act = nn.GELU()
        self.head_drop = nn.Dropout(p=p_dropout)
        
        # Parallel Prediction Heads
        # 1. Existing SCP Head (71 outputs)
        self.head_scp = nn.Linear(config.embedding_dim, num_scp)
        
        # 2. New Clinical Family Head (10 outputs)
        self.head_family = nn.Linear(config.embedding_dim, num_families)
        
        # 3. New Rare Label Specialist Head (N_rare outputs)
        self.head_rare = nn.Linear(config.embedding_dim, num_rare)
        
        self._initialize_weights()

    def _make_stage(
        self, 
        in_channels: int, 
        out_channels: int, 
        num_blocks: int, 
        stride: int, 
        dropout: float
    ) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, dropout=dropout))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem_conv(x)
        out = self.stem_bn(out)
        out = self.stem_act(out)
        out = self.stem_pool(out)
        
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        
        out = self.global_pool(out)
        out = self.flatten(out)
        embedding = self.fc1(out)
        embedding = self.head_act(embedding)
        return embedding

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        embedding = self.forward_features(x)
        features = self.head_drop(embedding)
        
        logits_scp = self.head_scp(features)
        logits_family = self.head_family(features)
        logits_rare = self.head_rare(features)
        
        probs_scp = torch.sigmoid(logits_scp)
        probs_family = torch.sigmoid(logits_family)
        probs_rare = torch.sigmoid(logits_rare)
        
        return {
            "scp_logits": logits_scp,
            "family_logits": logits_family,
            "rare_logits": logits_rare,
            "scp_predictions": probs_scp,
            "family_predictions": probs_family,
            "rare_predictions": probs_rare,
            "embeddings": embedding
        }

    def freeze_encoder_and_scp_head(self):
        """Stage 1 Freezing: Freeze Encoder & SCP Head. Train Family & Rare heads only."""
        for param in self.parameters():
            param.requires_grad = False
            
        for param in self.head_family.parameters():
            param.requires_grad = True
        for param in self.head_rare.parameters():
            param.requires_grad = True

    def unfreeze_last_encoder_stage(self):
        """Stage 2 Fine-Tuning: Unfreeze Stage 3 of Encoder and FC1 embedding layer."""
        for param in self.stage3.parameters():
            param.requires_grad = True
        for param in self.fc1.parameters():
            param.requires_grad = True

    def load_stage5_checkpoint(self, checkpoint_path: str, device: torch.device) -> dict:
        """
        Loads the validated Stage 5 weights into encoder (stem, stage1-3, fc1)
        and 71-label SCP head (head_scp). Reinitializes new family and rare heads.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        current_dict = self.state_dict()
        for k, v in state_dict.items():
            if k == 'fc2.weight':
                current_dict['head_scp.weight'] = v
            elif k == 'fc2.bias':
                current_dict['head_scp.bias'] = v
            elif k in current_dict and current_dict[k].shape == v.shape:
                current_dict[k] = v
                
        self.load_state_dict(current_dict)
        
        # Reinitialize only if heads are NOT present in the checkpoint
        if 'head_family.weight' not in state_dict and 'fc_family.weight' not in state_dict:
            nn.init.kaiming_normal_(self.head_family.weight, mode='fan_out', nonlinearity='relu')
            if self.head_family.bias is not None:
                nn.init.constant_(self.head_family.bias, 0)
            
        if 'head_rare.weight' not in state_dict and 'fc_rare.weight' not in state_dict:
            nn.init.kaiming_normal_(self.head_rare.weight, mode='fan_out', nonlinearity='relu')
            if self.head_rare.bias is not None:
                nn.init.constant_(self.head_rare.bias, 0)
            
        return checkpoint

"""
Configuration file for PTB-XL Five-Superclass 12-lead ECG Classification (Stage 1).
"""
import os
import torch
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Config:
    # Workspace & Paths
    project_root: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    database_csv: str = "ptbxl_database.csv"
    scp_statements_csv: str = "scp_statements.csv"
    records_folder: str = "records100"
    
    output_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    checkpoints_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "checkpoints"))
    tables_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "tables"))
    plots_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "plots"))
    logs_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "logs"))

    # Superclasses & Target Construction
    superclasses: List[str] = field(default_factory=lambda: ["NORM", "MI", "STTC", "CD", "HYP"])
    scp_likelihood_threshold: float = 0.0
    number_of_outputs: int = 5
    
    # ECG Signal Properties
    input_channels: int = 12
    input_length: int = 1000  # 10s @ 100 Hz
    sampling_rate: int = 100
    lead_order: List[str] = field(default_factory=lambda: [
        "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
    ])
    
    # Patient Splits (Official PTB-XL strat_fold)
    train_folds: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    val_fold: int = 9
    test_fold: int = 10
    
    # Preprocessing
    zscore_eps: float = 1e-8
    std_near_zero_thresh: float = 1e-6

    # Model Parameters
    stem_channels: int = 64
    stage_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    blocks_per_stage: List[int] = field(default_factory=lambda: [2, 2, 2])
    embedding_dim: int = 128
    dropout: float = 0.30

    # Loss & Weights
    maximum_pos_weight: float = 10.0

    # Hyperparameters & Optimization
    random_seed: int = 42
    epochs: int = 15
    minimum_epochs_before_early_stopping: int = 8
    batch_size: int = 128
    num_workers: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    betas: Tuple[float, float] = (0.9, 0.999)
    gradient_clipping: float = 1.0
    mixed_precision: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Scheduler Settings
    scheduler_mode: str = "max"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    minimum_learning_rate: float = 1e-6

    # Early Stopping Settings
    early_stopping_monitor: str = "val_macro_auprc"
    early_stopping_mode: str = "max"
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 0.001

    # Threshold Tuning
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_step: float = 0.01

    # Sanity Check Parameters
    tiny_subset_size: int = 64
    tiny_subset_epochs: int = 100
    tiny_batch_size: int = 32
    tiny_learning_rate: float = 0.001
    disable_weight_decay_sanity: bool = True
    disable_dropout_sanity: bool = True

    def create_directories(self):
        """Create all output directories if they do not exist."""
        # Deployment is inference-only; output/training directories are not
        # created as a side effect of constructing the classifier config.

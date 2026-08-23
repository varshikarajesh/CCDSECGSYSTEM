"""
Stage 5 SCP Statement Mapping, Target Construction, and Positive Weight Calculation.
Automatically discovers the complete PTB-XL SCP statement set (71 statements) from scp_statements.csv.
"""
import os
import ast
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any
try:
    from config import Config
except ModuleNotFoundError:
    from ptbxl_five_superclass.config import Config

logger = logging.getLogger(__name__)

def load_complete_scp_mapping(scp_statements_path: str) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Loads scp_statements.csv and automatically constructs the complete SCP label space
    for all SCP statements in PTB-XL.
    
    Returns:
        scp_info: Dict[scp_code, {'description': str, 'category': str, 'superclass': str}]
        scp_codes: Sorted List of all unique SCP statement codes
    """
    if not os.path.exists(scp_statements_path):
        raise FileNotFoundError(f"scp_statements.csv not found at {scp_statements_path}")
        
    df_scp = pd.read_csv(scp_statements_path, index_col=0)
    
    scp_info = {}
    for code, row in df_scp.iterrows():
        code_str = str(code).strip()
        desc = str(row.get('description', '')).strip() if pd.notna(row.get('description')) else ""
        category = str(row.get('Statement Category', '')).strip() if pd.notna(row.get('Statement Category')) else ""
        superclass = str(row.get('diagnostic_class', '')).strip() if pd.notna(row.get('diagnostic_class')) else "N/A"
        
        # Determine diagnostic category / statement type
        diag_val = row.get('diagnostic', 0.0)
        form_val = row.get('form', 0.0)
        rhythm_val = row.get('rhythm', 0.0)
        
        diag_cat = "Diagnostic" if (pd.notna(diag_val) and diag_val == 1.0) else (
            "Form" if (pd.notna(form_val) and form_val == 1.0) else (
                "Rhythm" if (pd.notna(rhythm_val) and rhythm_val == 1.0) else "Other"
            )
        )
        
        scp_info[code_str] = {
            "description": desc,
            "category": category,
            "diagnostic_category": diag_cat,
            "superclass": superclass
        }
        
    scp_codes = sorted(list(scp_info.keys()))
    logger.info(f"Automatically discovered complete SCP ontology with {len(scp_codes)} SCP statements from {scp_statements_path}.")
    return scp_info, scp_codes


def construct_scp_targets(
    df: pd.DataFrame, 
    scp_codes: List[str], 
    scp_likelihood_threshold: float = 0.0
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, int]]:
    """
    Constructs N_scp multi-hot target vectors for each ECG across all discovered SCP statements.
    Excludes ECGs without any present SCP codes.
    """
    scp_to_idx = {code: i for i, code in enumerate(scp_codes)}
    valid_indices = []
    targets_raw = []
    exclusion_counts = {"train": 0, "val": 0, "test": 0, "total": 0}
    
    for idx, row in df.iterrows():
        scp_codes_raw = row['scp_codes']
        if isinstance(scp_codes_raw, str):
            try:
                scp_dict = ast.literal_eval(scp_codes_raw)
            except Exception:
                scp_dict = {}
        elif isinstance(scp_codes_raw, dict):
            scp_dict = scp_codes_raw
        else:
            scp_dict = {}

        target_vec = np.zeros(len(scp_codes), dtype=np.float32)
        has_scp = False
        
        for code, likelihood in scp_dict.items():
            if likelihood > scp_likelihood_threshold:
                code_str = str(code).strip()
                if code_str in scp_to_idx:
                    target_vec[scp_to_idx[code_str]] = 1.0
                    has_scp = True

        fold = row['strat_fold']
        split_name = "train" if fold in range(1, 9) else ("val" if fold == 9 else "test")

        if has_scp and target_vec.sum() > 0:
            valid_indices.append(idx)
            targets_raw.append(target_vec)
        else:
            exclusion_counts[split_name] += 1
            exclusion_counts["total"] += 1

    df_filtered = df.loc[valid_indices].copy().reset_index(drop=True)
    targets = np.array(targets_raw, dtype=np.float32)
    
    logger.info(f"Complete SCP Target Construction Complete: {len(df_filtered)} included ECGs, {exclusion_counts['total']} excluded.")
    return df_filtered, targets, exclusion_counts


def compute_scp_positive_weights(
    train_targets: np.ndarray, 
    max_weight: float = 10.0
) -> np.ndarray:
    """Calculates pos_weight for BCEWithLogitsLoss on training targets."""
    n_samples = len(train_targets)
    n_positives = train_targets.sum(axis=0)
    n_negatives = n_samples - n_positives
    
    pos_weights = np.zeros(train_targets.shape[1], dtype=np.float32)
    for c in range(train_targets.shape[1]):
        if n_positives[c] > 0:
            w = n_negatives[c] / n_positives[c]
            pos_weights[c] = min(w, max_weight)
        else:
            pos_weights[c] = 1.0
            
    return pos_weights


def export_scp_statement_mapping_table(
    scp_info: Dict[str, Dict[str, Any]],
    scp_codes: List[str],
    df_clean: pd.DataFrame,
    targets_clean: np.ndarray,
    config: Config
) -> pd.DataFrame:
    """
    Saves outputs/tables/scp_statement_mapping.csv containing:
    SCP Code, Description, Diagnostic category, Superclass, Number of positives
    """
    scp_to_idx = {code: i for i, code in enumerate(scp_codes)}
    total_positives = targets_clean.sum(axis=0)

    rows = []
    for code_str in scp_codes:
        info = scp_info.get(code_str, {})
        idx = scp_to_idx[code_str]
        num_pos = int(total_positives[idx])
        
        rows.append({
            "SCP Code": code_str,
            "Description": info.get("description", ""),
            "Diagnostic category": info.get("diagnostic_category", ""),
            "Superclass": info.get("superclass", ""),
            "Number of positives": num_pos
        })

    df_mapping = pd.DataFrame(rows)
    out_path = os.path.join(config.tables_dir, "scp_statement_mapping.csv")
    df_mapping.to_csv(out_path, index=False)
    logger.info(f"Saved complete SCP statement mapping table to {out_path}")
    return df_mapping

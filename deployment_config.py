"""Package-relative, fail-closed configuration for the headless Jetson runtime."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Any

PACKAGE_ROOT = Path(__file__).resolve().parent
ACTIVE_MODELS_YAML_PATH = PACKAGE_ROOT / "models" / "active_models.yaml"

def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        # Simple fallback parser if PyYAML is unavailable
        res: Dict[str, Any] = {}
        curr_section = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue
                if ":" in line_str and not line.startswith("  "):
                    k, v = line_str.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if v:
                        res[k] = v
                    else:
                        res[k] = {}
                        curr_section = k
                elif ":" in line_str and line.startswith("  ") and curr_section:
                    k, v = line_str.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if isinstance(res[curr_section], dict):
                        res[curr_section][k] = v
        return res

ACTIVE_MODELS_CONFIG = _load_yaml_file(ACTIVE_MODELS_YAML_PATH)

DATA_DIR = PACKAGE_ROOT / "data"
FAISS_DIR = PACKAGE_ROOT / "faiss_assets"
RETRIEVAL_DIR = PACKAGE_ROOT / "retrieval"
KNOWLEDGE_DIR = PACKAGE_ROOT / "knowledge"
BRIDGE_DIR = PACKAGE_ROOT / "bridge"
EVIDENCE_DIR = KNOWLEDGE_DIR
SAMPLE_DATA_DIR = PACKAGE_ROOT / "sample_data"
LOGS_DIR = PACKAGE_ROOT / "logs"
PTBXL_DIR = PACKAGE_ROOT / "ptbxl_five_superclass"
CHECKPOINTS_DIR = PTBXL_DIR / "outputs" / "checkpoints"

# Registry-driven default paths with fallbacks
classifier_cfg = ACTIVE_MODELS_CONFIG.get("classifier", {})
retrieval_cfg = ACTIVE_MODELS_CONFIG.get("retrieval", {})
knowledge_cfg = ACTIVE_MODELS_CONFIG.get("knowledge", {})
skills_cfg = ACTIVE_MODELS_CONFIG.get("skills", {})
llm_cfg = ACTIVE_MODELS_CONFIG.get("llm", {})

ENCODER_CHECKPOINT_PATH = PACKAGE_ROOT / retrieval_cfg.get("checkpoint", "retrieval/best_retrieval.pth")
FAISS_INDEX_PATH = PACKAGE_ROOT / retrieval_cfg.get("faiss_index", "faiss_assets/faiss_index_candidate.bin")
FAISS_METADATA_PATH = PACKAGE_ROOT / retrieval_cfg.get("faiss_metadata", "faiss_assets/faiss_metadata_candidate.json")
FAISS_CANDIDATE_METADATA_PATH = FAISS_METADATA_PATH
ECG_METADATA_PATH = DATA_DIR / "ecg_metadata.csv"
SCP_STATEMENTS_PATH = DATA_DIR / "scp_statements.csv"
TEST_INDICES_PATH = DATA_DIR / "test_indices.npy"
SAMPLE_ECG_PATH = SAMPLE_DATA_DIR / "deployment_test_ecg_378.npy"

CLASSIFIER_CHECKPOINT_PATH = PACKAGE_ROOT / classifier_cfg.get("checkpoint", "ptbxl_five_superclass/assets/config4_family_rare_hardneg.pt")
ORDERED_SCP_LABELS_PATH = PACKAGE_ROOT / classifier_cfg.get("labels", "ptbxl_five_superclass/assets/ordered_scp_labels.json")
RARE_LABEL_LIST_PATH = PTBXL_DIR / "outputs" / "rare_label_list.json"
SCP_THRESHOLDS_PATH = PACKAGE_ROOT / classifier_cfg.get("thresholds", "ptbxl_five_superclass/assets/validation_scp_thresholds.json")

BRIDGE_ALIGNMENT_MODEL_PATH = BRIDGE_DIR / "best_alignment_model.pt"
BRIDGE_CONCEPT_MODEL_PATH = BRIDGE_DIR / "best_concept_model.pt"
BRIDGE_FAMILY_MODEL_PATH = BRIDGE_DIR / "best_family_model.pt"
CONFIDENCE_BREAKDOWN_PATH = EVIDENCE_DIR / "confidence_breakdown.json"
EVIDENCE_TEMPLATES_DIR = EVIDENCE_DIR / "templates"
CONCEPT_TEMPLATES_PATH = EVIDENCE_TEMPLATES_DIR / "concept_templates.json"
FAMILY_TEMPLATES_PATH = EVIDENCE_TEMPLATES_DIR / "family_templates.json"
LABEL_TO_CONCEPT_MAP_PATH = KNOWLEDGE_DIR / "label_to_concept_mapping.json"
LABEL_TO_FAMILY_MAP_PATH = EVIDENCE_TEMPLATES_DIR / "label_to_family_mapping.json"
RARE_LABEL_PROFILES_PATH = EVIDENCE_TEMPLATES_DIR / "rare_label_profiles.json"

KB_PATH = PACKAGE_ROOT / knowledge_cfg.get("database", "knowledge/kb.json")
KB_EMBEDDINGS_PATH = PACKAGE_ROOT / knowledge_cfg.get("embeddings", "knowledge/embeddings_kb.npy")
KB_ID_LIST_PATH = PACKAGE_ROOT / knowledge_cfg.get("ids", "knowledge/id_list.json")
CLINICAL_ONTOLOGY_PATH = PACKAGE_ROOT / knowledge_cfg.get("ontology", "knowledge/ontology/clinical_ontology.json")
CONDITION_CARDS_PATH = PACKAGE_ROOT / knowledge_cfg.get("condition_cards", "knowledge/ontology/condition_cards.json")
SKILL_REGISTRY_PATH = PACKAGE_ROOT / skills_cfg.get("registry", "prompt_builder/skills/registry.yaml")

LORA_DIR = PACKAGE_ROOT / llm_cfg.get("lora_path", "gemma3_ecg_lora_v2_4_refined/gemma3_ecg_lora_v2_4_refined_FIXED")
REFINED_LORA_PATH = Path(os.environ.get("TRACE_LORA_PATH", str(LORA_DIR)))
LORA_ADAPTER_CONFIG_PATH = REFINED_LORA_PATH / "adapter_config.json"
LORA_ADAPTER_WEIGHTS_PATH = REFINED_LORA_PATH / "adapter_model.safetensors"
PACKAGED_GGUF_MODEL_PATH = PACKAGE_ROOT / llm_cfg.get("gguf_model", "gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf")

BASE_MODEL_NAME = os.environ.get("TRACE_BASE_MODEL", str(PACKAGED_GGUF_MODEL_PATH))
TRACE_GGUF_MODEL = Path(os.environ.get("TRACE_GGUF_MODEL", str(PACKAGED_GGUF_MODEL_PATH)))
TRACE_LLM_MODE = os.environ.get("TRACE_LLM_MODE", llm_cfg.get("mode", "disabled")).lower()
TRACE_LLM_BACKEND = os.environ.get("TRACE_LLM_BACKEND", llm_cfg.get("backend", "llama_cpp")).lower()
TRACE_LORA_SCALE = float(os.environ.get("TRACE_LORA_SCALE", llm_cfg.get("lora_scale", 1.0)))
TRACE_LLM_MODEL_VARIANT = str(llm_cfg.get("model_variant", "base")).strip().lower()
TRACE_LORA_ENABLED_BY_DEFAULT = bool(llm_cfg.get("adapter_included", False)) and TRACE_LLM_MODEL_VARIANT == "lora"
DEVICE = os.environ.get("TRACE_DEVICE", "cuda")
BASE_MODEL_PACKAGED_LOCALLY = PACKAGED_GGUF_MODEL_PATH.is_file()
EXTERNAL_HF_CACHE_REQUIRED = False

ArtifactRecord = Tuple[Path, bool]


def validate_package_artifacts() -> Dict[str, ArtifactRecord]:
    """Return physical artifact state without copying or generating anything."""
    return {
        "package_root": (PACKAGE_ROOT, PACKAGE_ROOT.is_dir()),
        "active_models_config": (ACTIVE_MODELS_YAML_PATH, ACTIVE_MODELS_YAML_PATH.is_file()),
        "classifier_checkpoint": (CLASSIFIER_CHECKPOINT_PATH, CLASSIFIER_CHECKPOINT_PATH.is_file()),
        "ordered_scp_labels": (ORDERED_SCP_LABELS_PATH, ORDERED_SCP_LABELS_PATH.is_file()),
        "scp_thresholds": (SCP_THRESHOLDS_PATH, SCP_THRESHOLDS_PATH.is_file()),
        "ecg_metadata": (ECG_METADATA_PATH, ECG_METADATA_PATH.is_file()),
        "scp_statements": (SCP_STATEMENTS_PATH, SCP_STATEMENTS_PATH.is_file()),
        "faiss_index": (FAISS_INDEX_PATH, FAISS_INDEX_PATH.is_file()),
        "faiss_metadata": (FAISS_METADATA_PATH, FAISS_METADATA_PATH.is_file()),
        "encoder_checkpoint": (ENCODER_CHECKPOINT_PATH, ENCODER_CHECKPOINT_PATH.is_file()),
        "knowledge_base": (KB_PATH, KB_PATH.is_file()),
        "knowledge_embeddings": (KB_EMBEDDINGS_PATH, KB_EMBEDDINGS_PATH.is_file()),
        "knowledge_ids": (KB_ID_LIST_PATH, KB_ID_LIST_PATH.is_file()),
        "bridge_alignment": (BRIDGE_ALIGNMENT_MODEL_PATH, BRIDGE_ALIGNMENT_MODEL_PATH.is_file()),
        "concept_templates": (CONCEPT_TEMPLATES_PATH, CONCEPT_TEMPLATES_PATH.is_file()),
        "skill_registry": (SKILL_REGISTRY_PATH, SKILL_REGISTRY_PATH.is_file()),
        "gguf_base_model": (TRACE_GGUF_MODEL, TRACE_GGUF_MODEL.is_file()),
        "lora_adapter_config": (LORA_ADAPTER_CONFIG_PATH, LORA_ADAPTER_CONFIG_PATH.is_file()),
        "lora_adapter_weights": (LORA_ADAPTER_WEIGHTS_PATH, LORA_ADAPTER_WEIGHTS_PATH.is_file()),
        "sample_ecg": (SAMPLE_ECG_PATH, SAMPLE_ECG_PATH.is_file()),
    }


def missing_required_artifacts() -> Dict[str, Path]:
    optional = {"sample_ecg", "lora_adapter_config", "lora_adapter_weights"}
    if os.environ.get("TRACE_LLM_MODE", TRACE_LLM_MODE).lower() != "real":
        optional.add("gguf_base_model")
    return {name: path for name, (path, present) in validate_package_artifacts().items() if not present and name not in optional}

# TRACE optimized Jetson bundle contents

This directory is the complete inference deployment root. Runtime paths are
resolved relative to this directory; the parent development repository is not
required.

## Production assets

| Component | Bundled path |
|---|---|
| PTB-XL classifier | `ptbxl_five_superclass/assets/config4_family_rare_hardneg.pt` |
| Ordered 71-label manifest | `ptbxl_five_superclass/assets/ordered_scp_labels.json` |
| Classifier thresholds | `ptbxl_five_superclass/assets/validation_scp_thresholds.json` |
| V7 encoder | `faiss/v7/joint_v7_retrieval.pth` |
| V7 FAISS index | `faiss/v7/faiss_joint_v7.bin` |
| V7 FAISS metadata | `faiss/v7/faiss_joint_v7_metadata.json` |
| Validated knowledge | `knowledge/kb.json` |
| Knowledge embeddings | `knowledge/embeddings_kb.npy` |
| Knowledge IDs | `knowledge/id_list.json` |
| Ontology/cards/provenance | `knowledge/ontology/`, `knowledge/source_registry.json`, `knowledge/provenance.py` |
| Base Gemma | `gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf` |
| Feedback database | `backend/database/clinician_feedback.db` |

LoRA is not bundled and cannot be activated by the deployment backend.

## Runtime code

- `run_pipeline.py`, `runtime/jetson_runtime.py`, `ecg_loader.py`
- `backend/` including Bridge V4, statistics, recording analysis and feedback
- `preprocessing/`
- `retrieval/` and `joint_model/` inference architectures
- `long_recording_v2/` selector and optional Holter architecture
- `ptbxl_five_superclass/` classifier architecture
- Active `utils/` and `prompt_builder/` modules only
- `knowledge/label_to_concept_mapping.json` for Bridge V4 citation queries
- `api/`, `ui/streamlit_app.py`, `streamlit_components/`

The package contains 70 active Python source files. Retired Bridge V2/V3,
top-level bridge models, OOD logic, legacy chat/composer code and Python caches
are excluded. The authoritative bridge is
`backend/bridge/evidence_bridge_v4.py`.

## Mutable deployment paths

- `backend/database/clinician_feedback.db`
- `outputs/feedback/`
- `knowledge/approved_cases/`
- `datasets/future_training/`
- `logs/` and `tmp/`

Except for the database, mutable directories are created lazily when their
corresponding export or runtime action is requested. They are intentionally excluded from immutable hash enforcement.
All immutable files and assets are recorded in `bundle_manifest.json`.

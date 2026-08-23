# `final_version_optimized` deployment specification

## Purpose

Create a clean, inference-only sibling of `final_version` without changing the
verified working bundle. The optimized package must run from the project root:

```text
<PROJECT_ROOT>
```

with this entry point:

```text
python final_version_optimized\run_pipeline.py ...
```

The optimized package must remain self-contained. It must never resolve code,
models, FAISS data, the knowledge base or Gemma from `final_version` or another
development directory.

## Authoritative architecture

```text
ECG loader
  -> preprocessing and quality checks
  -> window screening / optional Holter localization
  -> PTB-XL 71-label + five-family classifier
  -> frozen V7 encoder and production FAISS retrieval
  -> ECG statistics
  -> deterministic Bridge V4
  -> validated knowledge and citations
  -> GPU-only base Gemma advisory explanation
  -> CLI / API / Streamlit interface
  -> clinician feedback storage
```

Bridge V4 remains diagnostic authority. Gemma cannot edit its decision. The
experimental Holter branch may localize windows but is not diagnostic authority.

## Required Python source

### Deployment entry points and contracts

```text
__init__.py
run_pipeline.py
runtime/jetson_runtime.py
deployment_config.py
ecg_loader.py
runtime/runtime_contracts.py
clinician_feedback_system.py
ui/streamlit_app.py
```

### API

```text
api/__init__.py
api/main.py
api/schemas.py
```

### Active backend

```text
backend/__init__.py
backend/diagnosis_model.py
backend/inference_pipeline.py
backend/recording_analysis.py
backend/recording_store.py
backend/diagnosis/__init__.py
backend/diagnosis/external_dataset_adapter.py
```

### Authoritative Bridge V4

```text
backend/bridge/__init__.py
backend/bridge/evidence_bridge.py
backend/bridge/evidence_bridge_v4.py
backend/bridge/ecg_statistics.py
backend/bridge/statistics_knowledge.py
backend/bridge/decision_reasoning.py
```

### Feedback API and database

```text
backend/database/connection.py
backend/database/models.py
backend/feedback/__init__.py
backend/feedback/analytics.py
backend/feedback/feedback_export.py
backend/feedback/feedback_repository.py
backend/feedback/feedback_service.py
backend/feedback/feedback_validation.py
backend/feedback/review_queue.py
backend/feedback/schemas.py
```

### Preprocessing

```text
preprocessing/__init__.py
preprocessing/ecg_preprocessor.py
preprocessing/signal_quality.py
preprocessing/preprocessing_config.py
preprocessing/preprocessing_manifest.py
preprocessing/validation.py
preprocessing/branch_adapters.py
```

`branch_adapters.py` is retained for checkpoint preprocessing compatibility even
though the selected V7 path currently receives the canonical tensor directly.

### Long-recording screening and optional Holter localization

```text
long_recording_v2/__init__.py
long_recording_v2/inference.py
long_recording_v2/model.py
```

### PTB-XL classifier

```text
ptbxl_five_superclass/config.py
ptbxl_five_superclass/labels_stage5.py
ptbxl_five_superclass/model_multihead.py
ptbxl_five_superclass/scp_family_mapping.py
```

`labels_stage5.py` is mandatory. `backend/inference_pipeline.py` imports it by
its short module name after adding this directory to `sys.path`; a simple package
import graph can therefore miss it.

### V7 model and retrieval

```text
joint_model/__init__.py
joint_model/joint_config.py
joint_model/joint_ecg_model.py
retrieval/retrieval_encoder.py
retrieval/retrieval_wrapper.py
```

These architecture modules must remain at stable import paths for checkpoint
compatibility.

### Knowledge provenance and Gemma prompt runtime

```text
knowledge/provenance.py
prompt_builder/gemma_prompt_builder.py
prompt_builder/system_runtime.py
utils/evidence_fields.py
utils/evidence_hierarchy.py
utils/knowledge_retriever.py
utils/llm.py
utils/question_router.py
utils/reasoning_policy.py
utils/response_validator.py
utils/runtime_lifecycle.py
```

### Streamlit interface

```text
streamlit_components/__init__.py
streamlit_components/api_client.py
streamlit_components/ecg_visualization.py
streamlit_components/neighbor_visualization.py
streamlit_components/result_panels.py
streamlit_components/session_state.py
```

## Required immutable assets

### Active registry

```text
models/active_models.yaml
```

### Classifier

```text
data/scp_statements.csv
ptbxl_five_superclass/assets/config4_family_rare_hardneg.pt
ptbxl_five_superclass/assets/ordered_scp_labels.json
ptbxl_five_superclass/assets/validation_scp_thresholds.json
```

### V7 retrieval and FAISS

```text
faiss/v7/joint_v7_retrieval.pth
faiss/v7/faiss_joint_v7.bin
faiss/v7/faiss_joint_v7_metadata.json
faiss/v7/faiss_manifest.json
```

### Validated knowledge and provenance

```text
knowledge/kb.json
knowledge/embeddings_kb.npy
knowledge/id_list.json
knowledge/source_registry.json
knowledge/validation_report.json
knowledge/ecg_statistics_rules.json
knowledge/ontology/clinical_ontology.json
knowledge/ontology/condition_cards.json
knowledge/KNOWLEDGE_GOVERNANCE.md
knowledge/label_to_concept_mapping.json
```

The label-to-concept mapping is an active Bridge V4 dependency. It is loaded
directly by `TraceableKnowledgeIndex` when constructing citation queries.

### Base Gemma

```text
gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf
```

No LoRA adapter may be included. Real LLM mode requires a CUDA-enabled
`llama-cpp-python`, all model layers offloaded (`n_gpu_layers=-1`), K/Q/V
offload and fail-closed behavior when CUDA offload is unavailable.

### Deployment diagnostics

```text
sample_data/deployment_test_ecg_378.npy
backend/bridge/BRIDGE_V4_CONTRACT.md
```

The sample ECG is retained solely for installation smoke tests.

## Required mutable paths

```text
backend/database/clinician_feedback.db
outputs/
logs/
tmp/
datasets/future_training/
knowledge/approved_cases/
```

Mutable files must not be included in immutable hash verification after initial
creation. The feedback database must contain its expected schema but no copied
patient data.

## Deployment documentation to regenerate or retain

```text
README.md
requirements/jetson-runtime.txt
requirements/jetson-system.txt
docs/BUNDLE_CONTENTS.md
bundle_manifest.json
docs/FINAL_VERSION_OPTIMIZED_SPEC.md
```

`bundle_manifest.json` and `docs/BUNDLE_CONTENTS.md` must be regenerated from the
optimized package, not copied unchanged.

## Exclude from the optimized package

### Entire legacy directories

```text
bridge/
clinical_reasoning/
api/services/
backend/retrieval/
backend/database/migrations/
```

The top-level `bridge/` directory is not Bridge V4. It contains retired wrapper
code and three retired learned bridge models. Active Bridge V4 is under
`backend/bridge/`.

### Retired Bridge and OOD source

```text
backend/bridge/confidence.py
backend/bridge/confidence_reasoning.py
backend/bridge/confidence_score.py
backend/bridge/contradiction_detector.py
backend/bridge/evidence_bridge_v2.py
backend/bridge/evidence_bridge_v3.py
backend/bridge/evidence_formatter.py
backend/bridge/evidence_models.py
backend/bridge/evidence_reranker.py
backend/bridge/evidence_rules.py
backend/bridge/family_support.py
backend/bridge/rare_case_rules.py
backend/bridge/retrieval_adapter.py
backend/diagnosis/ood_detector.py
backend/database/seed.py
backend/feedback/feedback_models.py
```

### Retired prompt and utility pipelines

```text
prompt_builder/gemma_wrapper.py
utils/bridge.py
utils/chat_formatter.py
utils/chat_session.py
utils/composer_input_factory.py
utils/composer_registry.py
utils/confidence.py
utils/confidence_threshold.py
utils/config.py
utils/context_builder.py
utils/context_lineage.py
utils/diagnosis_aggregator.py
utils/encoder.py
utils/evidence.py
utils/knowledge.py
utils/knowledge_memory.py
utils/knowledge_summarizer.py
utils/pipeline_runner.py
utils/pipeline_schema.py
utils/rare_case.py
utils/response_composer.py
utils/retrieval.py
utils/singleton_registry.py
utils/skill_loader.py
```

### Unused assets and generated files

```text
bridge/best_alignment_model.pt
bridge/best_concept_model.pt
bridge/best_family_model.pt
evidence/confidence_breakdown.json
Retired evidence templates other than the active label-to-concept mapping
ptbxl_five_superclass/outputs/rare_label_list.json
outputs/llm_language_demo.json
RETIRED_LORA_ASSETS.md
all __pycache__/ directories
all *.pyc files
```

The retired LoRA ledger remains available in the original `final_version` and
does not need to ship on the Jetson.

## Do not flatten or merge these boundaries

Do not combine active modules into one giant Python file. Stable module paths
are required by model loading and separating preprocessing, model architecture,
fusion, KB, UI and feedback makes failures isolatable. Optimization here means
removing parallel obsolete implementations, not destroying active boundaries.

## Construction procedure

1. Create `final_version_optimized` as a new sibling; do not modify or rename
   `final_version`.
2. Copy only the required files and create the required empty mutable paths.
3. Rewrite all package-root resolution against `final_version_optimized`.
4. Search the optimized source for references to `final_version`, absolute
   development paths, V2, V3, OOD and LoRA.
5. Initialize an empty feedback database using the optimized package.
6. Generate a new immutable manifest with SHA-256 for every immutable file.
7. Run all gates below from outside both deployment directories.
8. Promote only after the optimized result matches the working bundle.

## Mandatory validation gates

```text
Asset check                                      PASS
All Python files parse                           PASS
All active imports resolve inside optimized root PASS
No parent/development code resolution            PASS
10-second classifier result parity               PASS
10-second V7 retrieval result parity              PASS
2-minute recording analysis parity               PASS
5-minute recording analysis parity               PASS
Manual-window selection                          PASS
Bridge version exactly 4.x                       PASS
Bridge V4 decision parity                        PASS
Citation provenance validation                   PASS
Semantic KB retrieval                            PASS
API import and health endpoint                   PASS
Recording-inference API                          PASS
Feedback create/review/export on temporary DB    PASS
Streamlit import and API connection               PASS
Gemma disabled mode                              PASS
Gemma real mode on CUDA                          PASS on target Jetson
CPU fallback in Gemma real mode                  REJECTED
Manifest verification                            PASS
```

Numerical comparison must ignore timestamps, generated IDs, elapsed time and
temporary output paths. Classifier probabilities, selected labels, retrieval
indices/scores, selected windows, Bridge V4 findings and citations must remain
within their documented deterministic tolerances.

## Promotion rule

`final_version` remains the rollback package. `final_version_optimized` becomes
deployable only after every applicable gate passes and its own manifest is
sealed. Do not delete `final_version` during construction or validation.

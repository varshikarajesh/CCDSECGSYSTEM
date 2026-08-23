# TRACE final deployment runtime

This folder is the deployment-only entry point for the selected TRACE system:

1. ECG loading and canonical 12-lead validation.
2. Preprocessing and signal quality.
3. Coarse-to-fine window screening for 2/5-minute recordings.
4. Selected PTB-XL multi-label classifier.
5. Frozen V7 encoder and production FAISS index.
6. Optional experimental Holter localization (never diagnostic authority).
7. ECG statistics.
8. Authoritative deterministic Bridge V4.
9. Validated knowledge retrieval and traceable citations.
10. Optional base-Gemma advisory explanation.
11. Optional immutable clinician-feedback snapshot.

No training, recalibration, index building, validation, model promotion, or artifact mutation is exposed here.

## Files

- `run_pipeline.py`: the only command clinicians/deployment software need to call.
- `runtime/jetson_runtime.py`: model lifecycle and component orchestration.
- `ecg_loader.py`: safe `.npy`, `.npz`, `.csv`, `.txt`, and optional WFDB loading.
- `__init__.py`: Python package export.
- `api/`: FastAPI inference and feedback endpoints.
- `ui/streamlit_app.py` and `streamlit_components/`: clinician interface.
- `backend/database/clinician_feedback.db`: empty, versioned feedback database.
- `bundle_manifest.json`: file sizes and SHA-256 hashes.

This directory is self-contained. It includes the authoritative runtime modules,
selected checkpoints, V7 FAISS assets, validated knowledge assets, base Gemma,
preprocessing, recording selection, Bridge V4, and feedback storage. It does not
resolve code or model paths from the parent development repository.

## Authoritative component mapping

| Component | Runtime source |
|---|---|
| Preprocessing and signal quality | `preprocessing/ecg_preprocessor.py`, `preprocessing/signal_quality.py` |
| Window selection and optional Holter localization | `long_recording_v2/inference.py`, `backend/recording_analysis.py` |
| PTB-XL classifier and V7/FAISS inference | `backend/inference_pipeline.py` |
| ECG statistics | `backend/bridge/ecg_statistics.py` |
| Deterministic fusion | `backend/bridge/evidence_bridge_v4.py` |
| Validated knowledge | `utils/knowledge_retriever.py`, `knowledge/` assets |
| Base-Gemma system policy and read-only tools | `prompt_builder/system_runtime.py`, `utils/llm.py` |
| Feedback database | `clinician_feedback_system.py`, when explicitly enabled |

These remain separate internally because they are the tested model implementations. `run_pipeline.py` is the single operational entry point and prevents deployment users from calling training or promotion scripts.

## Jetson Orin Nano commands

Install a CUDA-enabled llama.cpp binding before using `--llm real`:

```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1
python3 -m pip install --no-cache-dir --force-reinstall "llama-cpp-python>=0.3.34,<0.4"
```

The GGUF is intentionally distributed separately because it exceeds GitHub's
2 GiB per-object LFS limit. Place it at
`gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf`. Without it, use the default
`--llm disabled`; the ECG pipeline, Bridge decision, retrieval and statistics
continue to run unchanged.

The runtime sets `n_gpu_layers=-1`, offloads K/Q/V operations, enables flash
attention, and targets GPU 0. It fails at model startup if the installed
binding has no CUDA offload; it never silently generates Gemma tokens on CPU.
Tokenization, sampling, file I/O, and request orchestration still use a small
amount of CPU because llama.cpp cannot move those operations to CUDA.

From the project root:

```bash
python3 run_pipeline.py --check-assets
```

Ten-second inference without Gemma:

```bash
python3 run_pipeline.py \
  --ecg sample_data/deployment_test_ecg_378.npy \
  --sampling-rate 100 \
  --mode 10s \
  --llm disabled \
  --output outputs/case_001.json
```

Five-minute inference with Gemma:

```bash
python3 run_pipeline.py \
  --ecg recordings/case_001.npy \
  --sampling-rate 100 \
  --mode 5min \
  --llm real \
  --question "What abnormalities occurred, when did they occur, and what evidence supports them?" \
  --output outputs/case_001.json
```

Keep the analyzed case and Gemma active for multiple questions:

```bash
python3 run_pipeline.py \
  --ecg recordings/case_001.npy \
  --sampling-rate 100 \
  --mode 5min \
  --llm real \
  --chat \
  --output outputs/case_001_chat.json
```

Enter questions at `Clinician>`. Use `/decision` to display the immutable Bridge
V4 result and `/exit` to end the session. ECG inference and model loading happen
once; follow-up turns reuse the same evidence and loaded Gemma instance.

Clinician-selected windows can be added without disabling automatic selection:

```bash
python3 run_pipeline.py \
  --ecg recordings/case_001.npy \
  --sampling-rate 100 \
  --mode 5min \
  --manual-windows 3 8 17 \
  --output outputs/case_001.json
```

To store an immutable response for later clinician correction:

```bash
python3 run_pipeline.py \
  --ecg recordings/case_001.npy \
  --sampling-rate 100 \
  --mode 5min \
  --save-feedback-snapshot \
  --output outputs/case_001.json
```

## API and clinician interface

Start the API from the project root. On CPU-only development machines the API
keeps ECG inference available and disables only the advisory Gemma layer. On a
CUDA host, set `TRACE_LLM_MODE=real` to enable Gemma.

```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
streamlit run ui/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

The UI talks to the local API at port 8000. Keep both services on the Jetson;
the diagnostic pipeline and knowledge base are offline.

## Input contract

- Exactly 12 ECG leads.
- Accepted orientation: `[12, samples]` or `[samples, 12]`.
- Supply the true sampling rate.
- Duration must match `10s`, `2min`, or `5min` within one second.
- Default lead order is `I II III aVR aVL aVF V1 V2 V3 V4 V5 V6`.
- For another order, supply all names using `--lead-names`.

## Safety contract

- Bridge V4 remains the machine-decision authority.
- Base Gemma is advisory and cannot modify Bridge V4 output. LoRA is retired.
- The experimental Holter model can localize windows only.
- Retrieved ECGs are examples, not proof of a diagnosis.
- The output requires clinician review and does not decide procedures such as stenting.

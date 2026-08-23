# Private repository asset policy

This repository contains the runnable ECG decision-support source, configuration,
validated knowledge base, FAISS retrieval assets, selected classifier/retriever
checkpoints, manifests, requirements and frontend dependency manifests.

## Included in Git

- Python and frontend source code
- Deployment configuration and runtime contracts
- `requirements/`
- `package.json`, `package-lock.json` and `pnpm-lock.yaml`
- Validated knowledge-base files and embeddings
- FAISS index and metadata
- Selected `.pt`/`.pth` checkpoints below the hosting limit
- Documentation and bundle manifests

## Installed separately

- `gemma-3-4b-it-GGUF/*.gguf`

The selected GGUF is larger than GitHub's 2 GiB per-object LFS limit and is not
stored in this repository. Publish it through approved private model storage, then
place it back at the manifest-defined relative path before using `--llm real`.
ECG inference remains available without this file using the default
`--llm disabled` mode.

## Intentionally excluded

- `node_modules/` and generated frontend `dist/`
- Python bytecode and test/build caches
- Temporary ECG acquisition/session caches
- Mutable clinician-feedback databases and their WAL/SHM files
- Generated outputs and logs
- Local secrets and `.env` files
- `data/records100/` PTB-XL waveform files

`data/ptbxl_database.csv` remains included for retrieval metadata. The waveform
corpus must be distributed separately only when its license and institutional data
governance permit redistribution. Without that corpus, inference and FAISS search
still run, but the interface cannot display retrieved reference waveforms.

## Teammate checkout

```bash
git clone <private-repository-url>
cd final_version_optimized
```

The clone directory name is arbitrary; if you clone directly into `CCDSECG`, use
`cd CCDSECG` instead.

Install frontend dependencies from a lockfile rather than committing
`node_modules`. On the Jetson, install the platform-appropriate CUDA/PyTorch and
llama.cpp packages described by the deployment requirements before starting the
API and frontend.

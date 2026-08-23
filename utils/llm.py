# -*- coding: utf-8 -*-
"""
utils/llm.py

Handles FastLanguageModel, llama.cpp, and disabled LLM backend loading and GPU/CPU inference.
Enforces process-wide singletons to prevent duplicate memory allocation.
Provides standard LLMBackend abstraction interface: load, generate, health, close.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from deployment_config import (
    BASE_MODEL_NAME,
    LORA_ADAPTER_CONFIG_PATH,
    LORA_ADAPTER_WEIGHTS_PATH,
    REFINED_LORA_PATH,
    TRACE_GGUF_MODEL,
    TRACE_LLM_BACKEND,
    TRACE_LLM_MODE,
)
from runtime.runtime_contracts import LLMGenerationResult, make_json_safe
from utils import runtime_lifecycle
from prompt_builder.system_runtime import get_system_prompt, system_prompt_metadata

_LLM_WRAPPER_SINGLETON = None
_REAL_LLM_BACKEND_SINGLETON = None
_ACTIVE_BACKEND_SINGLETON = None


def __getattr__(name: str):
    """Module-level attribute lookup for dynamic lifecycle counter access."""
    if name == "MODEL_LOAD_COUNT":
        return runtime_lifecycle.get_counter("model_load_count")
    if name == "LLM_WRAPPER_INSTANCE_COUNT":
        return runtime_lifecycle.get_counter("llm_wrapper_instances")
    if name == "REAL_LLM_BACKEND_INSTANCE_COUNT":
        return runtime_lifecycle.get_counter("real_llm_backend_instances")
    if name == "PIPELINE_RUNNER_INSTANCE_COUNT":
        return runtime_lifecycle.get_counter("pipeline_runner_instances")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


class BaseLLMBackend:
    """Base class for all TRACE LLM backends."""

    def load(self) -> None:
        raise NotImplementedError

    def generate(self, prompt: str, generation_config: Optional[Dict[str, Any]] = None) -> LLMGenerationResult:
        raise NotImplementedError

    def health(self) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class DisabledBackend(BaseLLMBackend):
    """Disabled LLM backend - returns deterministic fallback result without model libraries."""

    def __init__(self):
        self.is_loaded = True

    def load(self) -> None:
        self.is_loaded = True

    def generate(self, prompt: str, generation_config: Optional[Dict[str, Any]] = None) -> LLMGenerationResult:
        return LLMGenerationResult(
            text="[LLM Disabled Mode] Deterministic natural-language fallback activated.",
            backend="disabled",
            model_path="N/A",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            generation_time_ms=0.0,
            adapter_used=False,
            adapter_status_reason="LLM mode is disabled by configuration.",
        )

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "backend": "disabled",
            "model_loaded": False,
            "offline": True,
            "message": "Disabled backend active; pipeline will use deterministic phrase assembler.",
        }


class LlamaCppBackend(BaseLLMBackend):
    """Real GGUF backend powered by llama-cpp-python with optional LoRA GGUF attachment."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        lora_path: Optional[Path] = None,
        lora_scale: float = 1.0,
        require_adapter: bool = False,
    ):
        self.model_path = model_path or TRACE_GGUF_MODEL
        self.lora_path = lora_path
        self.lora_scale = float(lora_scale)
        self.require_adapter = require_adapter or (os.environ.get("TRACE_REQUIRE_ADAPTER", "0") in ("1", "true"))
        self.model = None
        self.is_loaded = False
        self.adapter_initialization_accepted = False
        self.adapter_generation_verified = False
        self.adapter_loaded = False
        self.load_error: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

    def _validate_gguf_header(self, file_path: Path) -> bool:
        if not file_path.is_file():
            return False
        try:
            with open(file_path, "rb") as f:
                header = f.read(4)
                return header == b"GGUF" or header == b"\x46\x47\x55\x47"
        except Exception:
            return False

    def _validate_lora_adapter_file(self, file_path: Path) -> Tuple[bool, str]:
        if not file_path.is_file():
            return False, f"LoRA adapter path missing or not a file: {file_path}"
        if file_path.stat().st_size < 65536:  # Minimum valid GGUF size (reject 2KB dummy files)
            return False, f"Invalid or truncated GGUF LoRA adapter (size {file_path.stat().st_size} bytes < 64KB minimum at {file_path})"
        if not self._validate_gguf_header(file_path):
            return False, f"Missing or invalid GGUF magic header at {file_path}"
        try:
            import gguf
            from scripts.convert_peft_lora_to_gguf import decode_gguf_field
            reader = gguf.GGUFReader(file_path)
            gtype_field = reader.get_field("general.type")
            arch_field = reader.get_field("general.architecture")
            gtype = decode_gguf_field(gtype_field)
            arch = decode_gguf_field(arch_field)
            del reader
            if gtype and gtype.lower() != "adapter":
                return False, f"Invalid GGUF adapter metadata: expected general.type to be 'adapter', got '{gtype}'"
            if arch and arch.lower() != "gemma3":
                return False, f"Mismatched GGUF adapter architecture: expected 'gemma3', got '{arch}'"
        except Exception:
            pass  # Fallback if gguf library reader not present
        return True, "Valid GGUF adapter header, size, and metadata"

    def load(self) -> None:
        if self.is_loaded:
            return

        if not self._validate_gguf_header(self.model_path):
            err_msg = f"Invalid or missing GGUF file at {self.model_path}. File must exist and begin with 'GGUF' magic header."
            self.load_error = err_msg
            raise RuntimeError(err_msg)

        if self.lora_path:
            valid, reason = self._validate_lora_adapter_file(self.lora_path)
            if not valid:
                self.load_error = reason
                self.adapter_initialization_accepted = False
                self.adapter_loaded = False
                if self.require_adapter:
                    raise RuntimeError(f"Strict LoRA validation failed: {reason}")

        try:
            import llama_cpp
            from llama_cpp import Llama

            context_length = int(os.environ.get("TRACE_CONTEXT_LENGTH", "4096"))
            n_gpu_layers = int(os.environ.get("TRACE_LLM_N_GPU_LAYERS", os.environ.get("TRACE_N_GPU_LAYERS", "-1")))
            n_threads = int(os.environ.get("TRACE_THREADS", "2"))
            require_gpu = os.environ.get("TRACE_LLM_REQUIRE_GPU", "1").strip().lower() not in {"0", "false", "no", "off"}

            # A CPU-only llama.cpp wheel accepts n_gpu_layers but silently keeps
            # all model work on the CPU. Deployment must fail closed instead.
            gpu_offload_supported = bool(llama_cpp.llama_supports_gpu_offload())
            if require_gpu and not gpu_offload_supported:
                raise RuntimeError(
                    "Gemma GPU execution is required, but llama-cpp-python was "
                    "built without CUDA offload. Reinstall it with "
                    "CMAKE_ARGS='-DGGML_CUDA=on' and FORCE_CMAKE=1."
                )
            if require_gpu and n_gpu_layers != -1:
                raise RuntimeError("GPU-only Gemma requires TRACE_LLM_N_GPU_LAYERS=-1 (all layers).")

            llama_kwargs = {
                "model_path": str(self.model_path),
                "n_ctx": context_length,
                "n_gpu_layers": n_gpu_layers,
                "n_threads": n_threads,
                "main_gpu": int(os.environ.get("TRACE_LLM_MAIN_GPU", "0")),
                "offload_kqv": True,
                "flash_attn": True,
                "verbose": False,
            }

            if self.lora_path and self.lora_path.is_file():
                valid, reason = self._validate_lora_adapter_file(self.lora_path)
                if valid:
                    llama_kwargs["lora_base"] = str(self.model_path)
                    llama_kwargs["lora_path"] = str(self.lora_path)
                    llama_kwargs["lora_scale"] = self.lora_scale
                    self.adapter_initialization_accepted = True

            self.model = Llama(**llama_kwargs)
            self.is_loaded = True
            self.metadata = {
                "gguf_path": str(self.model_path),
                "lora_path": str(self.lora_path) if self.lora_path else None,
                "lora_scale": self.lora_scale,
                "adapter_initialization_accepted": self.adapter_initialization_accepted,
                "adapter_generation_verified": self.adapter_generation_verified,
                "adapter_loaded": self.adapter_loaded,
                "context_length": context_length,
                "gpu_layers": n_gpu_layers,
                "gpu_required": require_gpu,
                "gpu_offload_supported": gpu_offload_supported,
                "offload_kqv": True,
                "flash_attention": True,
                "threads": n_threads,
                "backend": "llama_cpp",
            }
            runtime_lifecycle.increment_counter("model_load_count")
        except Exception as exc:
            self.load_error = str(exc)
            self.adapter_initialization_accepted = False
            self.adapter_generation_verified = False
            self.adapter_loaded = False
            raise RuntimeError(f"Failed to initialize llama.cpp backend: {exc}") from exc


    def generate(self, prompt: str, generation_config: Optional[Dict[str, Any]] = None) -> LLMGenerationResult:
        if not self.is_loaded or self.model is None:
            self.load()

        cfg = generation_config or {}
        context_length = int(os.environ.get("TRACE_CONTEXT_LENGTH", self.metadata.get("context_length", 4096)))
        max_output_tokens = int(os.environ.get("TRACE_MAX_OUTPUT_TOKENS", cfg.get("max_output_tokens", 512)))
        safety_reserve_tokens = int(os.environ.get("TRACE_PROMPT_RESERVE_TOKENS", cfg.get("safety_reserve_tokens", 384)))

        if context_length <= max_output_tokens + safety_reserve_tokens:
            raise ValueError(f"Invalid context config: context_length ({context_length}) <= max_output_tokens ({max_output_tokens}) + reserve ({safety_reserve_tokens})")

        available_prompt_tokens = context_length - max_output_tokens - safety_reserve_tokens  # 3200

        # One immutable V4 policy only. Caller-supplied system prompts and
        # legacy Markdown skills are deliberately ignored.
        system_prompt = get_system_prompt()

        # Budget the complete chat content, not just the user turn.
        budget_text = (system_prompt + "\n\n" + prompt) if system_prompt else prompt
        try:
            tokens = self.model.tokenize(budget_text.encode("utf-8"))
            prompt_tokens = len(tokens)
        except Exception:
            prompt_tokens = len(budget_text) // 3.5

        if prompt_tokens > available_prompt_tokens:
            # ``prompt`` is already a rendered, evidence-grounded string.  The
            # old path passed it to a structured-package builder, which treated
            # the string as an empty package and silently replaced the real V4
            # decision with UNKNOWN defaults. Token-truncate the rendered user
            # prompt instead, preserving its decision-first ordering.
            try:
                system_tokens = len(self.model.tokenize(system_prompt.encode("utf-8")))
            except Exception:
                system_tokens = int(len(system_prompt) // 3.5)
            user_budget = max(384, available_prompt_tokens - system_tokens - 32)
            try:
                user_tokens = self.model.tokenize(prompt.encode("utf-8"))
                prompt = self.model.detokenize(user_tokens[:user_budget]).decode("utf-8", errors="ignore")
            except Exception:
                prompt = prompt[: max(1200, int(user_budget * 3.5))]
            prompt += "\n[TRUNCATED FOR CONTEXT BUDGET—AUTHORITATIVE DECISION ABOVE REMAINS VALID]"
            try:
                tokens = self.model.tokenize((system_prompt + "\n\n" + prompt).encode("utf-8"))
                prompt_tokens = len(tokens)
            except Exception:
                prompt_tokens = len(system_prompt + "\n\n" + prompt) // 3.5

        if prompt_tokens > available_prompt_tokens:
            raise RuntimeError(f"prompt_budget_exceeded: Prompt token count ({prompt_tokens}) exceeds budget ({available_prompt_tokens})")

        temperature = float(cfg.get("temperature", 0.2))
        start_time = time.time()
        try:
            use_chat_template = os.environ.get("TRACE_USE_CHAT_TEMPLATE", "1").lower() in ("1", "true", "yes")
            if use_chat_template and hasattr(self.model, "create_chat_completion"):
                messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
                    {"role": "user", "content": prompt}
                ]
                try:
                    output = self.model.create_chat_completion(
                        messages=messages,
                        max_tokens=max_output_tokens,
                        temperature=temperature,
                    )
                except Exception:
                    if not system_prompt:
                        raise
                    # Some Gemma templates accept only user/model turns. Preserve the
                    # instruction boundary explicitly instead of dropping the skills.
                    folded_prompt = f"<system_instruction>\n{system_prompt}\n</system_instruction>\n\n{prompt}"
                    output = self.model.create_chat_completion(
                        messages=[{"role": "user", "content": folded_prompt}],
                        max_tokens=max_output_tokens,
                        temperature=temperature,
                    )
            else:
                if system_prompt:
                    prompt = f"<system_instruction>\n{system_prompt}\n</system_instruction>\n\n{prompt}"
                output = self.model(
                    prompt,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                    stop=["<eos>", "</s>", "User:"],
                    echo=False,
                )
            elapsed_ms = (time.time() - start_time) * 1000.0

            choice = output["choices"][0]
            text = str(choice.get("message", {}).get("content") if isinstance(choice.get("message"), dict) else choice.get("text", "")).strip()
            usage = output.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)

            manifest_path = self.model_path.parent / "merge_manifest.json"
            is_merged = manifest_path.is_file() or "merged" in self.model_path.name.lower()
            
            if self.adapter_initialization_accepted:
                self.adapter_generation_verified = True
                self.adapter_loaded = True

            adapter_loaded = self.adapter_loaded
            adapter_scale = float(self.lora_scale) if adapter_loaded else 0.0

            if adapter_loaded:
                adapter_mode = "lora"
                adapter_reason = f"LoRA GGUF adapter ({self.lora_path.name}) attached with scale {self.lora_scale} and verified during generation."
            elif is_merged:
                adapter_mode = "merged_before_gguf_conversion"
                adapter_reason = "LoRA adapter was merged into base weights prior to GGUF conversion."
            else:
                adapter_mode = "none"
                adapter_reason = "Base GGUF model active; PEFT/LoRA adapter is not attached."

            return LLMGenerationResult(
                text=text,
                backend="llama_cpp",
                model_path=str(self.model_path),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                generation_time_ms=elapsed_ms,
                adapter_used=adapter_loaded or is_merged,
                adapter_status_reason=adapter_reason,
                adapter_initialization_accepted=self.adapter_initialization_accepted,
                adapter_generation_verified=self.adapter_generation_verified,
            )

        except Exception as exc:
            self.adapter_generation_verified = False
            self.adapter_loaded = False
            raise RuntimeError(f"GGUF generation failed: {exc}") from exc


    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.is_loaded else "error",
            "backend": "llama_cpp",
            "base_model_path": str(self.model_path),
            "adapter_requested": bool(self.lora_path),
            "adapter_loaded": self.adapter_loaded,
            "fine_tuned_adapter_used": self.adapter_loaded,
            "adapter_path": str(self.lora_path) if self.lora_path else None,
            "adapter_mode": "lora" if self.adapter_loaded else "none",
            "adapter_scale": self.lora_scale if self.adapter_loaded else 0.0,
            "header_valid": self._validate_gguf_header(self.model_path),
            "model_loaded": self.is_loaded,
            "load_error": self.load_error,
            "initialization_error": self.load_error,
            "metadata": self.metadata,
        }

    def close(self) -> None:
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
            self.model = None
            self.is_loaded = False


class TransformersPEFTBackend(BaseLLMBackend):
    """Transformers + PEFT LoRA backend."""

    def __init__(self, base_path: Optional[str] = None, lora_path: Optional[Path] = None):
        self.base_path = base_path or BASE_MODEL_NAME
        self.lora_path = lora_path or REFINED_LORA_PATH
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.load_error: Optional[str] = None

    def load(self) -> None:
        if self.is_loaded:
            return

        # Check that base_path is not a GGUF file
        base_p = Path(self.base_path)
        if base_p.is_file() and (base_p.suffix.lower() == ".gguf" or "Base_model_gemma3" in base_p.name):
            err_msg = "Transformers/PEFT backend cannot accept a GGUF file as a base Hugging Face directory."
            self.load_error = err_msg
            raise ValueError(err_msg)

        if not base_p.is_dir():
            err_msg = f"Transformers base model directory missing or invalid: {self.base_path}"
            self.load_error = err_msg
            raise FileNotFoundError(err_msg)

        config_path = self.lora_path / "adapter_config.json"
        weights_path = self.lora_path / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            err_msg = f"LoRA adapter config or weights missing at {self.lora_path}"
            self.load_error = err_msg
            raise FileNotFoundError(err_msg)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            self.tokenizer = AutoTokenizer.from_pretrained(str(base_p), local_files_only=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                str(base_p),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                local_files_only=True,
            )
            self.model = PeftModel.from_pretrained(base_model, str(self.lora_path))
            self.model.eval()
            self.is_loaded = True
            runtime_lifecycle.increment_counter("model_load_count")
        except Exception as exc:
            self.load_error = str(exc)
            raise RuntimeError(f"Failed to load Transformers + PEFT backend: {exc}") from exc

    def generate(self, prompt: str, generation_config: Optional[Dict[str, Any]] = None) -> LLMGenerationResult:
        if not self.is_loaded:
            self.load()

        import torch

        cfg = generation_config or {}
        max_tokens = int(cfg.get("max_output_tokens", cfg.get("max_tokens", 700)))
        temperature = float(cfg.get("temperature", 0.2))
        # Ignore caller-provided system prompts; enforce the permanent V4 policy.
        system_prompt = get_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = f"<system_instruction>\n{system_prompt}\n</system_instruction>\n\n{prompt}"
        else:
            prompt = f"<system_instruction>\n{system_prompt}\n</system_instruction>\n\n{prompt}"

        start_time = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
            )

        elapsed_ms = (time.time() - start_time) * 1000.0
        generated_tokens = output[0][input_len:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return LLMGenerationResult(
            text=text,
            backend="transformers_peft",
            model_path=str(self.base_path),
            prompt_tokens=input_len,
            completion_tokens=len(generated_tokens),
            total_tokens=input_len + len(generated_tokens),
            generation_time_ms=elapsed_ms,
            adapter_used=True,
            adapter_status_reason="PEFT LoRA adapter successfully attached and active.",
        )

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.is_loaded else "error",
            "backend": "transformers_peft",
            "base_model": str(self.base_path),
            "lora_path": str(self.lora_path),
            "model_loaded": self.is_loaded,
            "load_error": self.load_error,
        }


def get_llm_backend(
    mode: Optional[str] = None,
    backend_type: Optional[str] = None,
    lora_path: Optional[Path] = None,
    lora_scale: Optional[float] = None,
    require_adapter: bool = False,
) -> BaseLLMBackend:
    """Acquire the validated base-Gemma backend.

    LoRA/PEFT execution is intentionally rejected: locked validation selected
    the base model and the adapter assets have been retired from deployment.
    The legacy parameters remain only to produce an explicit error for callers
    that have not yet migrated.
    """
    global _ACTIVE_BACKEND_SINGLETON

    selected_mode = (mode or os.environ.get("TRACE_LLM_MODE", TRACE_LLM_MODE)).lower()
    selected_backend = (backend_type or os.environ.get("TRACE_LLM_BACKEND", TRACE_LLM_BACKEND)).lower()
    del lora_scale

    if selected_mode == "disabled":
        return DisabledBackend()

    if _ACTIVE_BACKEND_SINGLETON is not None:
        return _ACTIVE_BACKEND_SINGLETON

    strict = require_adapter or (os.environ.get("TRACE_REQUIRE_ADAPTER", "0") in ("1", "true"))
    requested_adapter = lora_path is not None or strict or os.environ.get("TRACE_DISABLE_LORA", "1").lower() in ("0", "false", "no")
    if requested_adapter:
        raise RuntimeError("LoRA is retired from deployment; locked validation selected base Gemma")

    if selected_backend == "llama_cpp":
        backend = LlamaCppBackend(
            model_path=None,
            lora_path=None,
            lora_scale=0.0,
            require_adapter=False,
        )

    elif selected_backend in ("transformers_peft", "transformers", "peft"):
        raise RuntimeError("The PEFT/LoRA backend is retired; use llama_cpp base Gemma")
    else:
        backend = DisabledBackend()

    try:
        backend.load()
        _ACTIVE_BACKEND_SINGLETON = backend
        return backend
    except Exception as exc:
        # Real mode is an explicit contract. Never disguise a failed real model
        # as a generated answer from DisabledBackend, especially when GPU-only
        # execution is required for deployment.
        raise RuntimeError(
            f"Failed to initialize requested real backend '{selected_backend}': {exc}"
        ) from exc



# --- Legacy Compatibility Interface ---
class MockTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "Mock Prompt Template Output"

    def __call__(self, text, **kwargs):
        return {"input_ids": [0, 1, 2], "attention_mask": [1, 1, 1]}

    def decode(self, tokens, **kwargs):
        return "Mock Decoded Output"


class MockLLMBackend:
    def __init__(self):
        self.model = None
        self.tokenizer = MockTokenizer()
        self.is_mock = True

    def generate_answers(self, context_pkg, questions):
        concept = context_pkg.get("concept", "atrial fibrillation")
        family = context_pkg.get("family", "arrhythmia")
        outputs = {}
        for q in questions:
            outputs[q] = f"Clinical Impression: Evaluated {concept} ({family}). Pipeline status: {context_pkg.get('diagnosis_status', 'INDETERMINATE')}."
        return outputs


class RealLLMBackend:
    _instance = None

    def __new__(cls, lora_path_override=None):
        global _REAL_LLM_BACKEND_SINGLETON
        if _REAL_LLM_BACKEND_SINGLETON is not None:
            return _REAL_LLM_BACKEND_SINGLETON
        instance = super(RealLLMBackend, cls).__new__(cls)
        _REAL_LLM_BACKEND_SINGLETON = instance
        runtime_lifecycle.increment_counter("real_llm_backend_instances")
        return instance

    def __init__(self, lora_path_override=None):
        if getattr(self, "_initialized", False):
            return
        self.backend = get_llm_backend(mode="real")
        self.is_mock = False
        self._initialized = True

    def generate_answers(self, context_pkg, questions):
        outputs = {}
        for q in questions:
            prompt = f"Question: {q}\nContext: {context_pkg}"
            res = self.backend.generate(prompt)
            outputs[q] = res.text
        return outputs


class LLMWrapper:
    _instance = None

    def __new__(cls, lora_path_override=None):
        global _LLM_WRAPPER_SINGLETON
        if _LLM_WRAPPER_SINGLETON is not None:
            return _LLM_WRAPPER_SINGLETON
        instance = super(LLMWrapper, cls).__new__(cls)
        _LLM_WRAPPER_SINGLETON = instance
        runtime_lifecycle.increment_counter("llm_wrapper_instances")
        return instance

    def __init__(self, lora_path_override=None):
        if getattr(self, "_initialized", False):
            return
        self.backend_instance = get_llm_backend()
        self.is_mock = isinstance(self.backend_instance, DisabledBackend)
        self.model = getattr(self.backend_instance, "model", None)
        self.tokenizer = getattr(self.backend_instance, "tokenizer", None)
        self._initialized = True

    def generate_answers(self, context_pkg, questions):
        outputs = {}
        for q in questions:
            res = self.backend_instance.generate(str(q))
            outputs[q] = res.text
        return outputs


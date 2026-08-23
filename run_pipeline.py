"""One-command deployment entry point for TRACE on NVIDIA Jetson Orin Nano."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE
for path in (str(ROOT),):
    if path not in sys.path:
        sys.path.insert(0, path)

from ecg_loader import CANONICAL_LEADS, load_ecg
from runtime.jetson_runtime import JetsonECGPipeline, asset_status


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Run the deployment-only TRACE V4/V7 ECG pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    command.add_argument("--ecg", type=Path, help=".npy/.npz/.csv/.txt/.hea/.dat 12-lead ECG")
    command.add_argument("--sampling-rate", type=int, help="Input sampling rate in Hz")
    command.add_argument("--mode", choices=("10s", "2min", "5min"), help="Infer from duration when omitted")
    command.add_argument("--lead-names", nargs=12, metavar="LEAD", help="Input channel order")
    command.add_argument("--manual-windows", nargs="*", type=int, default=[], help="Window indices selected by clinician")
    command.add_argument("--top-k", type=int, default=5, help="Retrieved ECG examples")
    command.add_argument("--question", default="What is the diagnostic conclusion and the evidence supporting it?")
    command.add_argument("--chat", action="store_true", help="Keep the analyzed ECG and Gemma active for repeated questions")
    command.add_argument("--patient-id", help="Optional identifier; feedback storage hashes it")
    command.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    command.add_argument(
        "--llm",
        choices=("real", "disabled"),
        default="disabled",
        help="Gemma is opt-in so ECG inference still runs when CUDA/llama.cpp is unavailable",
    )
    command.add_argument("--llm-backend", choices=("llama_cpp",), default="llama_cpp")
    command.add_argument("--enable-experimental-holter", action="store_true", help="Localization only; never diagnostic authority")
    command.add_argument("--save-feedback-snapshot", action="store_true", help="Store immutable response for later clinician feedback")
    command.add_argument("--output", type=Path, help="Output JSON; stdout when omitted")
    command.add_argument("--check-assets", action="store_true", help="Check active assets without loading models")
    return command


def _write(payload: object, output: Path | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, output)
        print(f"Saved: {output.resolve()}")
    else:
        print(text)


def main() -> int:
    args = parser().parse_args()
    if args.check_assets:
        state = asset_status(include_llm=args.llm == "real")
        _write(state, args.output)
        return 0 if state["ready"] else 2
    if args.ecg is None:
        print("Input error: --ecg is required unless --check-assets is used", file=sys.stderr)
        return 2
    try:
        loaded = load_ecg(
            args.ecg,
            sampling_rate_hz=args.sampling_rate,
            mode=args.mode,
            lead_names=args.lead_names or CANONICAL_LEADS,
        )
        runtime = JetsonECGPipeline(
            device=args.device,
            llm_mode=args.llm,
            llm_backend=args.llm_backend,
            enable_experimental_holter=args.enable_experimental_holter,
        )
        result = runtime.run(
            loaded.signal,
            sampling_rate_hz=loaded.sampling_rate_hz,
            mode=loaded.mode,
            lead_names=loaded.lead_names,
            manual_window_indices=args.manual_windows,
            top_k=args.top_k,
            question=args.question,
            include_llm=args.llm == "real" and not args.chat,
            patient_id=args.patient_id,
            save_feedback_snapshot=args.save_feedback_snapshot,
        )
        result.setdefault("input", {})
        result["input"].update({
            "source_path": loaded.source_path,
            "duration_seconds": loaded.duration_seconds,
            "sampling_rate_hz": loaded.sampling_rate_hz,
            "mode": loaded.mode,
            "lead_names": list(loaded.lead_names),
        })
        if args.chat:
            if args.llm != "real":
                raise ValueError("--chat requires --llm real")
            history = []
            print("\nECG analysis complete. Gemma is active. Ask follow-up questions.")
            print("Commands: /decision, /exit\n")
            while True:
                try:
                    question = input("Clinician> ").strip()
                except EOFError:
                    break
                if not question:
                    continue
                if question.lower() in {"/exit", "exit", "quit", "/quit"}:
                    break
                if question.lower() == "/decision":
                    decision = result.get("final_diagnostic_decision", result.get("recording_bridge", result.get("bridge", {})))
                    print("\nGemma> " + json.dumps(decision, ensure_ascii=False, indent=2, default=str) + "\n")
                    continue
                explanation = runtime.answer_question(result, question, conversation=history)
                if explanation.get("status") == "cleared":
                    history.clear()
                answer = str(explanation.get("text", ""))
                print("\nGemma> " + answer + "\n")
                if explanation.get("status") not in ("cleared", "context", "decision"):
                    history.append({"question": question, "answer": answer})
            result["interactive_chat"] = {
                "turns": history,
                "bridge_decision_modified": False,
                "clinical_evidence_source": "immutable_bridge_v4_case_evidence",
            }
        _write(result, args.output)
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"Input/asset error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(json.dumps({"status": "error", "type": type(exc).__name__, "message": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

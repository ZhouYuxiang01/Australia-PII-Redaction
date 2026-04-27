#!/usr/bin/env python3
"""Evaluate a JSON dataset with batched live model inference.

This keeps the deployed redaction contract: same prompt, same LoRA adapter,
same qwen_redact parsing/repair/policy path. It only removes HTTP and
single-request generation overhead.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from eval_deployed_api_dataset import load_dataset, load_supported_labels, read_completed, write_summary
from qwen_redact import (
    apply_policy,
    build_response,
    load_json,
    normalize_text,
    parse_annotated_output,
    repair_offsets_to_input,
    safe_postprocess_spans,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = PROJECT_ROOT / "configs" / "policies" / "qwen-safe-default-v1.json"
DEFAULT_BASE_MODEL = Path("/home/admin/model/Qwen3.5-9B-Base")
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "outputs" / "qwen3_5_9b_base_lora_tagged_28_fastretry"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions-out", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--supported-labels", type=Path)
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH, type=Path)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, type=Path)
    parser.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def chunk_rows(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def pending_rows(rows: list[dict[str, Any]], completed: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["id"]) not in completed]


def system_prompt(target_labels: list[str]) -> str:
    return (
        "You are an Australian PII redaction system. Return the input text with each supported PII span wrapped as "
        "<pii type=\"TYPE\">exact text</pii>. Preserve every character. Do not explain.\n"
        "Supported types:\n- " + "\n- ".join(target_labels)
    )


class BatchedLiveModelBackend:
    def __init__(self, *, base_model: Path, adapter_dir: Path, max_new_tokens: int) -> None:
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.max_new_tokens = max_new_tokens
        self._loaded = False
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir, trust_remote_code=True, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(
            str(self.base_model),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base, self.adapter_dir)
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._loaded = True

    def generate_batch(self, texts: list[str], target_labels: list[str]) -> list[str]:
        self.load()
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        prompt = system_prompt(target_labels)
        prompt_texts = [
            self._tokenizer.apply_chat_template(
                [{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for text in texts
        ]
        encoded = self._tokenizer(prompt_texts, padding=True, return_tensors="pt")
        target_device = getattr(self._model, "device", None) or next(self._model.parameters()).device
        encoded = {key: value.to(target_device) for key, value in encoded.items()}
        prompt_width = encoded["input_ids"].shape[-1]
        eos_token_id = [self._tokenizer.eos_token_id, self._tokenizer.convert_tokens_to_ids("<|im_end|>")]
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=eos_token_id,
            )
        return [
            self._tokenizer.decode(output_ids[index][prompt_width:], skip_special_tokens=True).strip()
            for index in range(len(texts))
        ]

    def clear_cuda_cache(self) -> None:
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def build_redaction_payload(
    *,
    text: str,
    annotated_output: str,
    policy: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    text = normalize_text(text)
    parsed_text, spans = parse_annotated_output(annotated_output)
    parsed_text = normalize_text(parsed_text)
    spans, repair_warnings, repaired = repair_offsets_to_input(text, parsed_text, spans)
    spans, post_warnings = safe_postprocess_spans(text, spans, policy)
    spans = apply_policy(spans, policy)
    payload = build_response(
        text=text,
        spans=spans,
        policy=policy,
        raw_offset_mapping_applied=repaired,
        warnings=[*repair_warnings, *post_warnings],
    )
    payload["input_text"] = text
    payload["model_output"] = annotated_output
    payload["demo"] = {"backend": "batched_live_model", "latency_ms": round(latency_ms, 1)}
    return payload


def is_cuda_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "cuda" in message and ("out of memory" in message or "cublas" in message)


def generate_with_oom_split(
    backend: BatchedLiveModelBackend,
    batch: list[dict[str, Any]],
    target_labels: list[str],
) -> list[str]:
    try:
        return backend.generate_batch([row["text"] for row in batch], target_labels)
    except RuntimeError as exc:
        if len(batch) == 1 or not is_cuda_oom(exc):
            raise
        backend.clear_cuda_cache()
        midpoint = len(batch) // 2
        print(f"CUDA OOM at batch={len(batch)}; retrying as {midpoint}+{len(batch) - midpoint}", flush=True)
        return [
            *generate_with_oom_split(backend, batch[:midpoint], target_labels),
            *generate_with_oom_split(backend, batch[midpoint:], target_labels),
        ]


def append_predictions(
    handle: Any,
    *,
    batch: list[dict[str, Any]],
    outputs: list[str],
    policy: dict[str, Any],
    latency_ms: float,
) -> list[dict[str, Any]]:
    per_row_latency = latency_ms / len(batch) if batch else 0.0
    items: list[dict[str, Any]] = []
    for row, annotated_output in zip(batch, outputs, strict=True):
        payload = build_redaction_payload(
            text=row["text"],
            annotated_output=annotated_output,
            policy=policy,
            latency_ms=per_row_latency,
        )
        item = {
            "id": row["id"],
            "text": row["text"],
            "gold_labels": row["gold_labels"],
            "test_metadata": row["test_metadata"],
            "api_response": payload,
            "error": None,
            "latency_ms": round(per_row_latency, 1),
        }
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        items.append(item)
    handle.flush()
    return items


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rows = load_dataset(args.dataset, args.limit)
    supported_labels = load_supported_labels(args.supported_labels)
    policy = load_json(args.policy_path)
    target_labels = list(policy.get("type_actions", {}).keys()) or supported_labels
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(args.predictions_out) if args.resume else {}
    pending = pending_rows(rows, completed)
    backend = BatchedLiveModelBackend(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"rows: {len(rows)}", flush=True)
    print(f"already done: {len(completed)}", flush=True)
    print(f"pending: {len(pending)}", flush=True)
    print(f"batch_size: {args.batch_size}", flush=True)
    print(f"predictions_out: {args.predictions_out}", flush=True)
    print(f"summary_out: {args.summary_out}", flush=True)

    mode = "a" if args.resume else "w"
    with args.predictions_out.open(mode, encoding="utf-8") as handle:
        for batch in chunk_rows(pending, args.batch_size):
            started = time.perf_counter()
            outputs = generate_with_oom_split(backend, batch, target_labels)
            latency_ms = (time.perf_counter() - started) * 1000.0
            items = append_predictions(handle, batch=batch, outputs=outputs, policy=policy, latency_ms=latency_ms)
            for item in items:
                completed[str(item["id"])] = item
            print(
                f"[{len(completed)}/{len(rows)}] batch={len(batch)} ok {latency_ms / 1000.0:.1f}s "
                f"({latency_ms / max(len(batch), 1) / 1000.0:.2f}s/row)",
                flush=True,
            )
            write_summary(args.summary_out, rows, list(completed.values()), supported_labels)

    write_summary(args.summary_out, rows, list(completed.values()), supported_labels)
    print("done", flush=True)


if __name__ == "__main__":
    main()

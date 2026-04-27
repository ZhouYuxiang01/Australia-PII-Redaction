#!/usr/bin/env python3
"""Run LoRA inference on processed held-out test and score raw/postprocessed outputs."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from postprocess_qwen_cleaned_predictions import normalize_cleaned_pair, postprocess_row, prf
from redaction_utils import parse_annotated


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def span_key(span: dict[str, Any]) -> tuple[str, int, int, str]:
    return (span["type"], int(span["start"]), int(span["end"]), span["value"])


def value_pairs(spans: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (span["type"], normalize_cleaned_pair(span["type"], span["value"]))
        for span in spans
        if span.get("value") is not None
    }


def add_counts(counter: Counter[str], gold: set[Any], pred: set[Any], prefix: str) -> None:
    counter[f"{prefix}_tp"] += len(gold & pred)
    counter[f"{prefix}_fp"] += len(pred - gold)
    counter[f"{prefix}_fn"] += len(gold - pred)
    counter[f"{prefix}_sample_exact"] += int(gold == pred)


def metrics(counter: Counter[str], prefix: str, rows: int) -> dict[str, float | int]:
    payload = prf(counter[f"{prefix}_tp"], counter[f"{prefix}_fp"], counter[f"{prefix}_fn"])
    return {
        "rows": rows,
        "sample_exact_acc": counter[f"{prefix}_sample_exact"] / rows if rows else 0.0,
        **payload,
    }


def make_system_prompt(target_labels: list[str]) -> str:
    return (
        "You are a PII annotator for Australian context.\n"
        "Return the SAME text with supported PII wrapped as <pii type=\"TYPE\">VALUE</pii>.\n"
        "Preserve every character exactly. Do not paraphrase, summarize, or explain.\n"
        "Wrap every occurrence of supported PII. Do not deduplicate.\n"
        "If no supported PII is present, return the input unchanged.\n"
        "Supported types:\n- " + "\n- ".join(target_labels)
    )


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") is not None:
                done.add(str(row["id"]))
    return done


def build_prompt(tokenizer: Any, system_prompt: str, text: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def infer_batch(
    model: Any,
    tokenizer: Any,
    system_prompt: str,
    texts: list[str],
    *,
    max_input_len: int,
    max_new_tokens: int,
) -> list[str]:
    prompts = [build_prompt(tokenizer, system_prompt, text) for text in texts]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len,
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    decoded = []
    for row in outputs:
        generated_ids = row[prompt_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        decoded.append(strip_think_blocks(text))
    return decoded


def score_existing_rows(path: Path) -> tuple[Counter[str], Counter[str], int, int, int]:
    counter: Counter[str] = Counter()
    per_type: Counter[str] = Counter()
    rows = 0
    parse_fail = 0
    roundtrip_fail = 0
    if not path.exists():
        return counter, per_type, rows, parse_fail, roundtrip_fail

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            parse_fail += int(row.get("parse_error") is not None)
            roundtrip_fail += int(not row.get("roundtrip_ok"))
            gold_spans = row.get("gold_spans", [])
            pred_spans = row.get("pred_spans", [])
            gold_strict = {span_key(span) for span in gold_spans}
            pred_strict = {span_key(span) for span in pred_spans}
            gold_value = value_pairs(gold_spans)
            pred_value = value_pairs(pred_spans)
            post_value = postprocess_row(
                {"text": row["text"], "pred_spans": pred_spans},
                add_date_variants=True,
                collapse_work_contact=True,
                add_encoded_emails=True,
            )
            add_counts(counter, gold_strict, pred_strict, "strict")
            add_counts(counter, gold_value, pred_value, "value")
            add_counts(counter, gold_value, post_value, "post")
            for label_type, _ in gold_value & post_value:
                per_type[f"{label_type}::tp"] += 1
            for label_type, _ in post_value - gold_value:
                per_type[f"{label_type}::fp"] += 1
            for label_type, _ in gold_value - post_value:
                per_type[f"{label_type}::fn"] += 1
    return counter, per_type, rows, parse_fail, roundtrip_fail


def per_type_payload(counter: Counter[str], target_labels: list[str]) -> dict[str, dict[str, float | int]]:
    return {
        label_type: prf(counter[f"{label_type}::tp"], counter[f"{label_type}::fp"], counter[f"{label_type}::fn"])
        for label_type in target_labels
    }


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    target_labels: list[str],
    counts: Counter[str],
    per_type: Counter[str],
    rows: int,
    total_rows: int,
    parse_fail: int,
    roundtrip_fail: int,
    start_time: float,
) -> None:
    payload = {
        "rows_completed": rows,
        "rows_total": total_rows,
        "base_model": str(args.base_model),
        "adapter_dir": str(args.adapter_dir),
        "test_path": str(args.test_path),
        "predictions_out": str(args.predictions_out),
        "batch_size": args.batch_size,
        "max_input_len": args.max_input_len,
        "max_new_tokens": args.max_new_tokens,
        "parse_fail": parse_fail,
        "roundtrip_fail": roundtrip_fail,
        "strict_span": metrics(counts, "strict", rows),
        "raw_value_level": metrics(counts, "value", rows),
        "postprocessed_value_level": metrics(counts, "post", rows),
        "postprocessed_per_type": per_type_payload(per_type, target_labels),
        "elapsed_seconds": time.time() - start_time,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=Path("../../model/Qwen3.5-9B-Base"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("../outputs/qwen3_5_9b_base_lora_tagged_28_fastretry"))
    parser.add_argument("--test-path", type=Path, default=Path("../data/processed/test_ground_truth.jsonl"))
    parser.add_argument("--predictions-out", type=Path, default=Path("../outputs/qwen3_5_9b_base_lora_tagged_28_fastretry/processed_test_predictions_optimized.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("../outputs/qwen3_5_9b_base_lora_tagged_28_fastretry/processed_test_summary_optimized.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-len", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.base_model = args.base_model.expanduser().resolve()
    args.adapter_dir = args.adapter_dir.expanduser().resolve()
    args.test_path = args.test_path.expanduser().resolve()
    args.predictions_out = args.predictions_out.expanduser().resolve()
    args.summary_out = args.summary_out.expanduser().resolve()

    target_labels = json.loads((args.adapter_dir / "target_labels.json").read_text(encoding="utf-8"))
    system_prompt = make_system_prompt(target_labels)
    test_items = load_jsonl(args.test_path)
    if args.max_examples is not None:
        test_items = test_items[: args.max_examples]

    done = completed_ids(args.predictions_out) if args.resume else set()
    pending = [item for item in test_items if str(item.get("id")) not in done]
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)

    counts, per_type, scored_rows, parse_fail, roundtrip_fail = score_existing_rows(args.predictions_out) if args.resume else (Counter(), Counter(), 0, 0, 0)
    start_time = time.time()

    print(f"target labels: {len(target_labels)}", flush=True)
    print(f"test rows: {len(test_items)} | already done: {len(done)} | pending: {len(pending)}", flush=True)
    print(f"predictions_out: {args.predictions_out}", flush=True)
    print(f"summary_out: {args.summary_out}", flush=True)

    if not pending:
        write_summary(
            args.summary_out,
            args=args,
            target_labels=target_labels,
            counts=counts,
            per_type=per_type,
            rows=scored_rows,
            total_rows=len(test_items),
            parse_fail=parse_fail,
            roundtrip_fail=roundtrip_fail,
            start_time=start_time,
        )
        print("No pending rows; summary refreshed from existing predictions.", flush=True)
        return 0

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading base model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("Loading LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    if torch.cuda.is_available():
        print(f"GPU allocated after load: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB", flush=True)

    mode = "a" if args.resume else "w"
    with args.predictions_out.open(mode, encoding="utf-8", newline="\n") as out_f:
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            generated_batch = infer_batch(
                model,
                tokenizer,
                system_prompt,
                [item["text"] for item in batch],
                max_input_len=args.max_input_len,
                max_new_tokens=args.max_new_tokens,
            )

            for item, generated in zip(batch, generated_batch):
                text = item["text"]
                gold_spans = item.get("labels", [])
                pred_plain = None
                pred_spans: list[dict[str, Any]] = []
                err = None
                try:
                    pred_plain, pred_spans = parse_annotated(generated, strict=False)
                    if pred_plain != text:
                        roundtrip_fail += 1
                except Exception as exc:  # noqa: BLE001 - evaluation should keep going.
                    parse_fail += 1
                    err = repr(exc)

                pred_spans = [span for span in pred_spans if span.get("type") in target_labels]
                gold_strict = {span_key(span) for span in gold_spans}
                pred_strict = {span_key(span) for span in pred_spans}
                gold_value = value_pairs(gold_spans)
                pred_value = value_pairs(pred_spans)
                row_for_post = {"text": text, "pred_spans": pred_spans}
                post_value = postprocess_row(
                    row_for_post,
                    add_date_variants=True,
                    collapse_work_contact=True,
                    add_encoded_emails=True,
                )

                add_counts(counts, gold_strict, pred_strict, "strict")
                add_counts(counts, gold_value, pred_value, "value")
                add_counts(counts, gold_value, post_value, "post")
                for label_type, _ in gold_value & post_value:
                    per_type[f"{label_type}::tp"] += 1
                for label_type, _ in post_value - gold_value:
                    per_type[f"{label_type}::fp"] += 1
                for label_type, _ in gold_value - post_value:
                    per_type[f"{label_type}::fn"] += 1

                scored_rows += 1
                out_f.write(
                    json.dumps(
                        {
                            "id": item.get("id"),
                            "text": text,
                            "gold_spans": gold_spans,
                            "generated": generated,
                            "pred_plain": pred_plain,
                            "pred_spans": pred_spans,
                            "parse_error": err,
                            "roundtrip_ok": pred_plain == text,
                            "gt_pairs": sorted([{"type": t, "value": v} for t, v in gold_value], key=lambda x: (x["type"], x["value"])),
                            "pred_pairs": sorted([{"type": t, "value": v} for t, v in pred_value], key=lambda x: (x["type"], x["value"])),
                            "postprocessed_pairs": sorted([{"type": t, "value": v} for t, v in post_value], key=lambda x: (x["type"], x["value"])),
                            "strict_tp": sorted(list(gold_strict & pred_strict)),
                            "strict_fp": sorted(list(pred_strict - gold_strict)),
                            "strict_fn": sorted(list(gold_strict - pred_strict)),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            out_f.flush()
            done_now = scored_rows
            if done_now % args.print_every == 0 or done_now == len(test_items):
                write_summary(
                    args.summary_out,
                    args=args,
                    target_labels=target_labels,
                    counts=counts,
                    per_type=per_type,
                    rows=done_now,
                    total_rows=len(test_items),
                    parse_fail=parse_fail,
                    roundtrip_fail=roundtrip_fail,
                    start_time=start_time,
                )
                strict = metrics(counts, "strict", done_now)
                raw_value = metrics(counts, "value", done_now)
                post = metrics(counts, "post", done_now)
                elapsed = time.time() - start_time
                rate = (done_now - len(done)) / max(elapsed, 0.01)
                eta = (len(test_items) - done_now) / max(rate, 0.01)
                print(
                    f"{done_now}/{len(test_items)} | "
                    f"strict F1={strict['f1']:.4f} | "
                    f"raw value F1={raw_value['f1']:.4f} | "
                    f"post value F1={post['f1']:.4f} | "
                    f"parse_fail={parse_fail} roundtrip_fail={roundtrip_fail} | "
                    f"{rate:.2f} samples/s | ETA={eta / 60:.1f} min",
                    flush=True,
                )

    write_summary(
        args.summary_out,
        args=args,
        target_labels=target_labels,
        counts=counts,
        per_type=per_type,
        rows=scored_rows,
        total_rows=len(test_items),
        parse_fail=parse_fail,
        roundtrip_fail=roundtrip_fail,
        start_time=start_time,
    )
    print("done", flush=True)
    print(f"predictions saved: {args.predictions_out}", flush=True)
    print(f"summary saved: {args.summary_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

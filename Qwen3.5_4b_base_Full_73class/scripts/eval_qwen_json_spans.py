#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a Qwen structured-generation SFT model for 73-class AU PII JSON spans.

Supports:
- Qwen SFT JSONL rows from build_sft_dataset_73.py
- OPF-style JSONL rows: {"example_id":..., "text":..., "spans": {"LABEL: value": [[start,end]]}}

Outputs:
- predictions.jsonl
- metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PREFIX = """You are a PII span extraction model for Australian context.
Return only compact JSON with this exact top-level shape: {"spans":[...]}.
Each span object must contain start, end, type, and value.
start and end are Python-style character offsets into the input text; end is exclusive.
Use only the supported types listed below. Preserve overlapping spans when present.
If no supported PII is present, return {"spans":[]}.
Supported types:"""


def load_taxonomy_labels(path: Path) -> list[str]:
    try:
        import yaml
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        return [str(x["code"]) for x in doc["classes"]]
    except Exception:
        labels = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith("- code:"):
                labels.append(s.split(":", 1)[1].strip())
        if not labels:
            raise
        return labels


def build_system_prompt(labels: list[str]) -> str:
    return DEFAULT_SYSTEM_PREFIX + "\n- " + "\n- ".join(labels)


def load_jsonl(path: Path, max_examples: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


def row_id(row: dict[str, Any], idx: int) -> str:
    return str(row.get("id") or row.get("example_id") or f"row-{idx}")


def normalize_gold_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(row.get("text", ""))
    raw = row.get("spans") or []
    out = []

    if isinstance(raw, list):
        for s in raw:
            if not isinstance(s, dict):
                continue
            label = s.get("type") or s.get("label")
            if label is None:
                continue
            try:
                start = int(s["start"])
                end = int(s["end"])
                out.append({
                    "start": start,
                    "end": end,
                    "type": str(label),
                    "value": str(s.get("value", text[start:end])),
                })
            except Exception:
                continue

    elif isinstance(raw, dict):
        for k, offsets in raw.items():
            label = str(k).split(":", 1)[0]
            for off in offsets:
                try:
                    if isinstance(off, dict):
                        start, end = int(off["start"]), int(off["end"])
                    else:
                        start, end = int(off[0]), int(off[1])
                    out.append({"start": start, "end": end, "type": label, "value": text[start:end]})
                except Exception:
                    continue

    out.sort(key=lambda x: (x["start"], x["end"], x["type"]))
    return out


def make_messages(row: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    if isinstance(row.get("messages"), list):
        msgs = []
        for m in row["messages"]:
            if m.get("role") == "assistant":
                break
            if m.get("role") in {"system", "user"}:
                msgs.append({"role": m["role"], "content": str(m.get("content", ""))})
        if msgs:
            return msgs
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(row.get("text", ""))},
    ]


def extract_json_object(s: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = s.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    candidates = [raw]
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(raw[first:last + 1])

    last_err = None
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj, None
        except Exception as e:
            last_err = str(e)
    return None, last_err or "no_json_object_found"


def normalize_pred_obj(obj: dict[str, Any], text: str, allowed_labels: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spans_raw = obj.get("spans")
    if not isinstance(spans_raw, list):
        return [], [{"reason": "missing_or_nonlist_spans", "object": obj}]

    valid = []
    invalid = []
    seen = set()

    for s in spans_raw:
        if not isinstance(s, dict):
            invalid.append({"reason": "span_not_object", "span": s})
            continue
        try:
            start = int(s["start"])
            end = int(s["end"])
            label = str(s["type"])
            value = str(s.get("value", text[start:end] if 0 <= start <= end <= len(text) else ""))
        except Exception as e:
            invalid.append({"reason": f"bad_span_fields:{e}", "span": s})
            continue

        if label not in allowed_labels:
            invalid.append({"reason": "unknown_type", "span": s})
            continue
        if not (0 <= start < end <= len(text)):
            invalid.append({"reason": "offset_out_of_range", "span": s, "text_length": len(text)})
            continue

        actual = text[start:end]
        if actual != value:
            invalid.append({"reason": "offset_value_mismatch", "span": s, "actual": actual})
            value = actual

        key = (start, end, label)
        if key in seen:
            continue
        seen.add(key)
        valid.append({"start": start, "end": end, "type": label, "value": value})

    valid.sort(key=lambda x: (x["start"], x["end"], x["type"]))
    return valid, invalid


def span_sets(spans: list[dict[str, Any]]):
    typed = {(s["start"], s["end"], s["type"]) for s in spans}
    untyped = {(s["start"], s["end"]) for s in spans}
    return typed, untyped


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def encode_chat(tokenizer, messages, device):
    """
    Compatible with transformers versions where apply_chat_template returns:
    - Tensor
    - dict / BatchEncoding with input_ids
    - BatchEncoding-like object with .data
    """
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # BatchEncoding path
    if hasattr(encoded, "data") and isinstance(encoded.data, dict) and "input_ids" in encoded.data:
        input_ids = encoded.data["input_ids"]
        attention_mask = encoded.data.get("attention_mask")

    # dict path
    elif isinstance(encoded, dict) and "input_ids" in encoded:
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")

    # plain tensor path
    else:
        input_ids = encoded
        attention_mask = None

    # Some tokenizer versions can return Python lists.
    import torch
    if not hasattr(input_ids, "dim"):
        input_ids = torch.tensor(input_ids, dtype=torch.long)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)

    model_inputs = {"input_ids": input_ids}

    if attention_mask is not None:
        if not hasattr(attention_mask, "dim"):
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        model_inputs["attention_mask"] = attention_mask.to(device)

    return model_inputs, int(input_ids.shape[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Fine-tuned Qwen model directory")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--taxonomy", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    labels = load_taxonomy_labels(args.taxonomy)
    allowed = set(labels)
    system_prompt = build_system_prompt(labels)
    rows = load_jsonl(args.input, args.max_examples)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / "predictions.jsonl"
    metrics_path = args.out_dir / "metrics.json"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if args.device == "cuda" else None,
    )
    model.eval()

    counters = Counter()
    invalid_reasons = Counter()
    per_label_tp = Counter()
    per_label_fp = Counter()
    per_label_fn = Counter()

    typed_tp = typed_fp = typed_fn = 0
    untyped_tp = untyped_fp = untyped_fn = 0
    total_gen_tokens = 0
    start_time = time.time()

    with pred_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows):
            text = str(row.get("text", ""))
            gold = normalize_gold_spans(row)
            messages = make_messages(row, system_prompt)

            model_inputs, input_len = encode_chat(tokenizer, messages, model.device)

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "temperature": args.temperature if args.temperature > 0 else None,
                "top_p": args.top_p if args.temperature > 0 else None,
                "pad_token_id": tokenizer.eos_token_id,
            }
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            with torch.inference_mode():
                output_ids = model.generate(**model_inputs, **gen_kwargs)

            new_ids = output_ids[0, input_len:]
            total_gen_tokens += int(new_ids.numel())
            generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

            obj, parse_error = extract_json_object(generated)
            if obj is None:
                pred = []
                invalid = [{"reason": f"json_parse_error:{parse_error}"}]
                counters["json_invalid_rows"] += 1
            else:
                counters["json_valid_rows"] += 1
                pred, invalid = normalize_pred_obj(obj, text, allowed)

            if invalid:
                counters["rows_with_invalid_spans"] += 1
                for item in invalid:
                    invalid_reasons[item.get("reason", "unknown")] += 1

            gold_t, gold_u = span_sets(gold)
            pred_t, pred_u = span_sets(pred)

            typed_tp += len(gold_t & pred_t)
            typed_fp += len(pred_t - gold_t)
            typed_fn += len(gold_t - pred_t)

            untyped_tp += len(gold_u & pred_u)
            untyped_fp += len(pred_u - gold_u)
            untyped_fn += len(gold_u - pred_u)

            for s in gold:
                k = (s["start"], s["end"], s["type"])
                if k in pred_t:
                    per_label_tp[s["type"]] += 1
                else:
                    per_label_fn[s["type"]] += 1
            for s in pred:
                k = (s["start"], s["end"], s["type"])
                if k not in gold_t:
                    per_label_fp[s["type"]] += 1

            if not gold and pred:
                counters["false_positive_empty_gold_rows"] += 1
            if gold and not pred:
                counters["missed_positive_rows"] += 1

            out.write(json.dumps({
                "id": row_id(row, idx),
                "text": text,
                "gold_spans": gold,
                "predicted_spans": pred,
                "invalid_items": invalid,
                "generated": generated,
            }, ensure_ascii=False) + "\n")

            if (idx + 1) % 50 == 0:
                print(f"[progress] {idx+1}/{len(rows)}")

    elapsed = time.time() - start_time
    per_label = {}
    for lab in sorted(set(per_label_tp) | set(per_label_fp) | set(per_label_fn)):
        per_label[lab] = prf(per_label_tp[lab], per_label_fp[lab], per_label_fn[lab])

    metrics = {
        "input": str(args.input),
        "model": args.model,
        "n_rows": len(rows),
        "elapsed_s": elapsed,
        "generated_tokens": total_gen_tokens,
        "generated_tokens_per_s": total_gen_tokens / elapsed if elapsed else None,
        "json_valid_rate": counters["json_valid_rows"] / len(rows) if rows else 0,
        "json_invalid_rows": counters["json_invalid_rows"],
        "rows_with_invalid_spans": counters["rows_with_invalid_spans"],
        "invalid_reasons": dict(invalid_reasons),
        "false_positive_empty_gold_rows": counters["false_positive_empty_gold_rows"],
        "false_positive_empty_gold_row_rate": counters["false_positive_empty_gold_rows"] / len(rows) if rows else 0,
        "missed_positive_rows": counters["missed_positive_rows"],
        "typed_exact": prf(typed_tp, typed_fp, typed_fn),
        "untyped_exact": prf(untyped_tp, untyped_fp, untyped_fn),
        "per_label_typed_exact": per_label,
    }

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

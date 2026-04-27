#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repair and re-evaluate Qwen JSON-span predictions by resolving offsets from generated values.

Input:
  predictions.jsonl produced by eval_qwen_json_spans.py

Repair policy:
  1. Parse the raw generated JSON.
  2. For each span, trust the generated `value` and `type` first.
  3. If `value` appears once in the input text, rewrite start/end to that occurrence.
  4. If `value` appears multiple times, choose the occurrence nearest to the model's original start.
  5. If `value` cannot be found, keep it invalid and exclude it from repaired spans.

This tests whether Qwen learned "which values to extract" but failed exact offsets.

Example:
  python scripts/repair_qwen_predictions.py \
    --pred ./outputs/qwen3_5_4b_base_full_73class/eval_smoke_20/predictions.jsonl \
    --out-dir ./outputs/qwen3_5_4b_base_full_73class/eval_smoke_20_repaired
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def extract_json_object(s: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = (s or "").strip()
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


def find_all(text: str, value: str) -> list[int]:
    if not value:
        return []
    starts = []
    pos = 0
    while True:
        i = text.find(value, pos)
        if i == -1:
            break
        starts.append(i)
        pos = i + 1
    return starts


def repair_generated_spans(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    text = row.get("text", "")
    generated = row.get("generated", "")
    obj, err = extract_json_object(generated)

    stats = Counter()
    repaired = []
    invalid = []
    seen = set()

    if obj is None:
        return [], [{"reason": f"json_parse_error:{err}"}], Counter({"json_parse_error": 1})

    spans = obj.get("spans")
    if not isinstance(spans, list):
        return [], [{"reason": "missing_or_nonlist_spans"}], Counter({"missing_or_nonlist_spans": 1})

    for s in spans:
        if not isinstance(s, dict):
            invalid.append({"reason": "span_not_object", "span": s})
            stats["span_not_object"] += 1
            continue

        label = s.get("type")
        value = s.get("value")
        if label is None or value is None:
            invalid.append({"reason": "missing_type_or_value", "span": s})
            stats["missing_type_or_value"] += 1
            continue

        label = str(label)
        value = str(value)
        matches = find_all(text, value)

        if len(matches) == 1:
            start = matches[0]
            end = start + len(value)
            stats["unique_value_repaired"] += 1
        elif len(matches) > 1:
            try:
                orig_start = int(s.get("start", matches[0]))
            except Exception:
                orig_start = matches[0]
            start = min(matches, key=lambda x: abs(x - orig_start))
            end = start + len(value)
            stats["multi_value_repaired"] += 1
        else:
            invalid.append({"reason": "value_not_found", "span": s})
            stats["value_not_found"] += 1
            continue

        key = (start, end, label)
        if key in seen:
            stats["deduped"] += 1
            continue
        seen.add(key)
        repaired.append({"start": start, "end": end, "type": label, "value": value})

    repaired.sort(key=lambda x: (x["start"], x["end"], x["type"]))
    return repaired, invalid, stats


def span_sets(spans: list[dict[str, Any]]):
    typed = {(s["start"], s["end"], s["type"]) for s in spans}
    untyped = {(s["start"], s["end"]) for s in spans}
    return typed, untyped


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    rows = load_jsonl(args.pred)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_path = args.out_dir / "predictions_repaired.jsonl"
    metrics_path = args.out_dir / "metrics_repaired.json"

    typed_tp = typed_fp = typed_fn = 0
    untyped_tp = untyped_fp = untyped_fn = 0
    repair_stats = Counter()
    invalid_reasons = Counter()
    per_label_tp = Counter()
    per_label_fp = Counter()
    per_label_fn = Counter()
    false_positive_empty_gold_rows = 0
    missed_positive_rows = 0

    with out_path.open("w", encoding="utf-8") as out:
        for row in rows:
            gold = row.get("gold_spans") or []
            repaired, invalid, stats = repair_generated_spans(row)
            repair_stats.update(stats)
            for item in invalid:
                invalid_reasons[item.get("reason", "unknown")] += 1

            gold_t, gold_u = span_sets(gold)
            pred_t, pred_u = span_sets(repaired)

            typed_tp += len(gold_t & pred_t)
            typed_fp += len(pred_t - gold_t)
            typed_fn += len(gold_t - pred_t)

            untyped_tp += len(gold_u & pred_u)
            untyped_fp += len(pred_u - gold_u)
            untyped_fn += len(gold_u - pred_u)

            for s in gold:
                key = (s["start"], s["end"], s["type"])
                if key in pred_t:
                    per_label_tp[s["type"]] += 1
                else:
                    per_label_fn[s["type"]] += 1

            for s in repaired:
                key = (s["start"], s["end"], s["type"])
                if key not in gold_t:
                    per_label_fp[s["type"]] += 1

            if not gold and repaired:
                false_positive_empty_gold_rows += 1
            if gold and not repaired:
                missed_positive_rows += 1

            out.write(json.dumps({
                "id": row.get("id"),
                "text": row.get("text", ""),
                "gold_spans": gold,
                "predicted_spans_repaired": repaired,
                "invalid_after_repair": invalid,
                "generated": row.get("generated", ""),
            }, ensure_ascii=False) + "\n")

    per_label = {}
    for lab in sorted(set(per_label_tp) | set(per_label_fp) | set(per_label_fn)):
        per_label[lab] = prf(per_label_tp[lab], per_label_fp[lab], per_label_fn[lab])

    metrics = {
        "input": str(args.pred),
        "n_rows": len(rows),
        "repair_stats": dict(repair_stats),
        "invalid_after_repair_count": sum(invalid_reasons.values()),
        "invalid_after_repair_reasons": dict(invalid_reasons),
        "false_positive_empty_gold_rows": false_positive_empty_gold_rows,
        "false_positive_empty_gold_row_rate": false_positive_empty_gold_rows / len(rows) if rows else 0,
        "missed_positive_rows": missed_positive_rows,
        "typed_exact_repaired": prf(typed_tp, typed_fp, typed_fn),
        "untyped_exact_repaired": prf(untyped_tp, untyped_fp, untyped_fn),
        "per_label_typed_exact_repaired": per_label,
    }

    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

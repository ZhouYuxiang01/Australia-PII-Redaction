#!/usr/bin/env python
"""Analyze value-level errors from the Qwen cleaned-200 evaluator."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def as_pairs(items: list[list[str]]) -> set[tuple[str, str]]:
    return {(item[0], item[1]) for item in items}


def related(a: str, b: str) -> bool:
    return bool(a and b and (a in b or b in a))


def classify_fp(fp_pair: tuple[str, str], row: dict[str, Any]) -> str:
    fp_type, fp_value = fp_pair
    gt = as_pairs(row["gt_pairs"])
    fn = as_pairs(row["fn_pairs"])

    for gt_type, gt_value in fn:
        if gt_value == fp_value and gt_type != fp_type:
            return "wrong_type_same_value"

    for gt_type, gt_value in fn:
        if gt_type == fp_type and related(fp_value, gt_value):
            if fp_value.startswith(("id: ", "dob: ", "tfn: ", "crn ")) or ": " in fp_value:
                return "included_field_label_or_prefix"
            return "same_type_partial_or_joined_value"

    for gt_type, gt_value in gt:
        if gt_type != fp_type and related(fp_value, gt_value):
            return "wrong_type_related_value"

    return "extra_prediction"


def classify_fn(fn_pair: tuple[str, str], row: dict[str, Any]) -> str:
    fn_type, fn_value = fn_pair
    pred = as_pairs(row["pred_pairs"])
    fp = as_pairs(row["fp_pairs"])

    for pred_type, pred_value in fp:
        if pred_value == fn_value and pred_type != fn_type:
            return "wrong_type_same_value"

    for pred_type, pred_value in fp:
        if pred_type == fn_type and related(pred_value, fn_value):
            if pred_value.startswith(("id: ", "dob: ", "tfn: ", "crn ")) or ": " in pred_value:
                return "field_label_prefix_mismatch"
            return "same_type_partial_or_joined_value"

    for pred_type, pred_value in pred:
        if pred_type != fn_type and related(pred_value, fn_value):
            return "wrong_type_related_value"

    return "missed_no_related_prediction"


def add_example(bucket: dict[str, list[dict[str, Any]]], key: str, row: dict[str, Any], pair: tuple[str, str], related_pairs: list[list[str]], limit: int) -> None:
    if len(bucket[key]) >= limit:
        return
    bucket[key].append(
        {
            "id": row["id"],
            "difficulty": row["difficulty"],
            "pair": {"type": pair[0], "value": pair[1]},
            "related_pairs": [{"type": item[0], "value": item[1]} for item in related_pairs],
            "text_preview": row["text"][:260].replace("\n", " | "),
        }
    )


def analyze(predictions_path: Path, summary_path: Path | None, example_limit: int) -> dict[str, Any]:
    rows = load_jsonl(predictions_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path else None

    fp_by_type = Counter()
    fn_by_type = Counter()
    fp_categories = Counter()
    fn_categories = Counter()
    fp_type_categories = Counter()
    fn_type_categories = Counter()
    roundtrip_failures: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if not row["round_trip_ok"]:
            roundtrip_failures.append(
                {
                    "id": row["id"],
                    "difficulty": row["difficulty"],
                    "sample_exact_match": row["sample_exact_match"],
                    "fp_count": len(row["fp_pairs"]),
                    "fn_count": len(row["fn_pairs"]),
                    "text_preview": row["text"][:220].replace("\n", " | "),
                }
            )

        for item in row["fp_pairs"]:
            pair = (item[0], item[1])
            category = classify_fp(pair, row)
            fp_by_type[pair[0]] += 1
            fp_categories[category] += 1
            fp_type_categories[f"{pair[0]}::{category}"] += 1
            add_example(examples, f"fp::{pair[0]}::{category}", row, pair, row["fn_pairs"], example_limit)

        for item in row["fn_pairs"]:
            pair = (item[0], item[1])
            category = classify_fn(pair, row)
            fn_by_type[pair[0]] += 1
            fn_categories[category] += 1
            fn_type_categories[f"{pair[0]}::{category}"] += 1
            add_example(examples, f"fn::{pair[0]}::{category}", row, pair, row["fp_pairs"], example_limit)

    overall = summary["overall"] if summary else {
        "tp": sum(len(row["tp_pairs"]) for row in rows),
        "fp": sum(len(row["fp_pairs"]) for row in rows),
        "fn": sum(len(row["fn_pairs"]) for row in rows),
    }

    return {
        "predictions_path": str(predictions_path),
        "summary_path": str(summary_path) if summary_path else None,
        "rows": len(rows),
        "overall": overall,
        "parse_fail_rows": sum(not row["parse_ok"] for row in rows),
        "roundtrip_fail_rows": len(roundtrip_failures),
        "sample_error_rows": sum(not row["sample_exact_match"] for row in rows),
        "rows_with_fp": sum(bool(row["fp_pairs"]) for row in rows),
        "rows_with_fn": sum(bool(row["fn_pairs"]) for row in rows),
        "rows_with_both_fp_fn": sum(bool(row["fp_pairs"]) and bool(row["fn_pairs"]) for row in rows),
        "fp_by_type": fp_by_type.most_common(),
        "fn_by_type": fn_by_type.most_common(),
        "fp_categories": fp_categories.most_common(),
        "fn_categories": fn_categories.most_common(),
        "fp_type_categories": fp_type_categories.most_common(),
        "fn_type_categories": fn_type_categories.most_common(),
        "roundtrip_failures": roundtrip_failures,
        "examples": dict(examples),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    lines = ["# Qwen 9B Adapter Cleaned-200 Error Analysis", ""]
    lines.append(f"Predictions: `{payload['predictions_path']}`")
    if payload["summary_path"]:
        lines.append(f"Summary: `{payload['summary_path']}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Rows: {payload['rows']}")
    lines.append(f"- Parse failures: {payload['parse_fail_rows']}")
    lines.append(f"- Round-trip failures: {payload['roundtrip_fail_rows']}")
    lines.append(f"- Sample-level error rows: {payload['sample_error_rows']}")
    lines.append(f"- Rows with FP / FN / both: {payload['rows_with_fp']} / {payload['rows_with_fn']} / {payload['rows_with_both_fp_fn']}")
    lines.append(f"- Value-level P/R/F1: {overall['precision']:.4f} / {overall['recall']:.4f} / {overall['f1']:.4f}")
    lines.append(f"- TP / FP / FN: {overall['tp']} / {overall['fp']} / {overall['fn']}")
    lines.append("")
    lines.append("## Error Categories")
    lines.append("")
    lines.append("### False Positives")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("| --- | ---: |")
    for category, count in payload["fp_categories"]:
        lines.append(f"| `{category}` | {count} |")
    lines.append("")
    lines.append("### False Negatives")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("| --- | ---: |")
    for category, count in payload["fn_categories"]:
        lines.append(f"| `{category}` | {count} |")
    lines.append("")
    lines.append("## FP By Type")
    lines.append("")
    lines.append("| Type | Count |")
    lines.append("| --- | ---: |")
    for label_type, count in payload["fp_by_type"]:
        lines.append(f"| `{label_type}` | {count} |")
    lines.append("")
    lines.append("## FN By Type")
    lines.append("")
    lines.append("| Type | Count |")
    lines.append("| --- | ---: |")
    for label_type, count in payload["fn_by_type"]:
        lines.append(f"| `{label_type}` | {count} |")
    lines.append("")
    lines.append("## Top Type-Category Pairs")
    lines.append("")
    lines.append("| Side | Type | Category | Count |")
    lines.append("| --- | --- | --- | ---: |")
    for key, count in payload["fp_type_categories"][:12]:
        label_type, category = key.split("::", 1)
        lines.append(f"| FP | `{label_type}` | `{category}` | {count} |")
    for key, count in payload["fn_type_categories"][:12]:
        label_type, category = key.split("::", 1)
        lines.append(f"| FN | `{label_type}` | `{category}` | {count} |")
    lines.append("")
    lines.append("## Round-Trip Failures")
    lines.append("")
    lines.append("| ID | Difficulty | Sample Exact | FP | FN |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for item in payload["roundtrip_failures"]:
        lines.append(f"| `{item['id']}` | `{item['difficulty']}` | {str(item['sample_exact_match']).lower()} | {item['fp_count']} | {item['fn_count']} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The model has no parse failures and no TRAP false positives in the saved summary. Most remaining errors are value-boundary or label-prefix issues, not fundamental extraction failures.")
    lines.append("")
    lines.append("The largest mechanically recoverable group is field-label inclusion: values such as `DOB: 25/06/1969`, `TFN: 832 109 111`, `ID: 405997905`, and `CRN 585 024 614V` should be normalized to the identifier value before scoring or final output.")
    lines.append("")
    lines.append("The second major group is date value shape. The model often outputs a complete date such as `September 13, 1966` while the value-level ground truth stores `September 13` and `1966` separately. For redaction, the complete-date span is arguably preferable; for this benchmark, a value normalizer or controlled split would improve the score.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Qwen cleaned-200 value-level prediction errors.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--example-limit", type=int, default=5)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    payload = analyze(args.predictions, args.summary, args.example_limit)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    write_markdown(args.md_out, payload)
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

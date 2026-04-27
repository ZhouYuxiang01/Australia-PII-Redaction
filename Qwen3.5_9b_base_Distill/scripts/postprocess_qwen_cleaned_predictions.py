#!/usr/bin/env python
"""Conservative value-level scorer for Qwen cleaned predictions."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015]")
MONTH_DAY_RE = re.compile(
    r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+0?(\d{1,2})$",
    re.IGNORECASE,
)

PREFIX_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "DATE_OF_BIRTH": [
        re.compile(r"^(?:d\.?\s*o\.?\s*b\.?|dob|date\s+of\s+birth|born)\s*[:=\-]?\s+", re.IGNORECASE),
    ],
    "AU_TFN": [
        re.compile(r"^(?:t\.?\s*f\.?\s*n\.?|tfn|tax\s+file\s+number)\s*[:=\-]?\s+", re.IGNORECASE),
    ],
    "STUDENT_ID": [
        re.compile(r"^student\s+id\s*[:=\-]?\s+", re.IGNORECASE),
        re.compile(r"^sid\s*[:=\-]?\s+", re.IGNORECASE),
        re.compile(r"^id\s*[:=\-]?\s+", re.IGNORECASE),
    ],
    "CENTRELINK_REFERENCE_NUMBER": [
        re.compile(r"^crn\s+(?=crn[:\s])", re.IGNORECASE),
        re.compile(r"^crn\s+(?!:)", re.IGNORECASE),
        re.compile(r"^centrelink(?:\s+reference(?:\s+number)?)?\s*[:=\-]?\s+", re.IGNORECASE),
    ],
    "AU_DRIVERS_LICENCE": [
        re.compile(r"^\(?(?:nsw|vic|qld|wa|sa|tas|act|nt)\)?\s*[:=\-]\s+", re.IGNORECASE),
        re.compile(r"^(?:licen[cs]e|driver'?s?\s+licen[cs]e|dl)\s*[:=\-]?\s+", re.IGNORECASE),
    ],
    "EMPLOYEE_NUMBER": [
        re.compile(r"^(?:emp(?:loyee)?\s*(?:id|number|no\.?)|emp\s*id)\s*[:=\-]?\s+", re.IGNORECASE),
    ],
    "PERSONNEL_NUMBER": [
        re.compile(r"^(?:personnel\s*(?:id|number|no\.?))\s*[:=\-]?\s+", re.IGNORECASE),
    ],
}


def normalize_value(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = DASH_RE.sub("-", text)
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


def strip_known_prefix(label_type: str, value: str) -> str:
    cleaned = value.strip()
    patterns = PREFIX_PATTERNS.get(label_type, [])
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            new_value = pattern.sub("", cleaned).strip()
            if new_value != cleaned:
                cleaned = new_value
                changed = True
    return cleaned


def normalize_cleaned_pair(label_type: str, value: Any) -> str:
    cleaned = strip_known_prefix(label_type, normalize_value(value))
    if label_type == "ADDRESS":
        cleaned = re.sub(r"\s*,\s*", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    elif label_type in {"AU_PHONE", "WORK_PHONE"}:
        compact = re.sub(r"(?!^\+)[^\d]", "", cleaned)
        if compact:
            cleaned = compact
    elif label_type in {"AU_TFN", "BSB", "AU_BANK_ACCOUNT", "PAYMENT_CARD_NUMBER"}:
        compact = re.sub(r"[\s\-]", "", cleaned)
        if compact.isdigit():
            cleaned = compact
    elif label_type == "STUDENT_ID":
        compact = re.sub(r"[\s\-]", "", cleaned)
        if compact.isdigit():
            cleaned = compact
    elif label_type == "DATE_OF_BIRTH":
        match = MONTH_DAY_RE.match(cleaned)
        if match:
            month, day = match.groups()
            cleaned = f"{month.lower()} {int(day)}"
    return cleaned


def pair_dict(label_type: str, value: str) -> dict[str, str]:
    return {"type": label_type, "value": value}


def postprocess_row(row: dict[str, Any], *, add_date_variants: bool, collapse_work_contact: bool, add_encoded_emails: bool) -> set[tuple[str, str]]:
    """Return normalized predicted pairs without adding, deleting, or relabeling spans.

    The option flags are accepted for CLI compatibility with older reports, but are
    intentionally no-ops. The cleaned-200-specific rules they once controlled did
    not transfer to the processed-test annotation policy.
    """
    pairs: set[tuple[str, str]] = set()
    text = row["text"]

    for span in row.get("pred_spans", []):
        label_type = span["type"]
        start = int(span.get("start", -1))
        end = int(span.get("end", -1))
        raw_value = span.get("value")
        if raw_value is None and start >= 0 and end <= len(text) and start < end:
            raw_value = text[start:end]
        if raw_value is None:
            continue

        value = normalize_cleaned_pair(label_type, raw_value)
        if not value:
            continue

        pairs.add((label_type, value))

    return pairs


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def add_type_counts(counter: Counter[str], pairs: set[tuple[str, str]], suffix: str) -> None:
    for label_type, _ in pairs:
        counter[f"{label_type}::{suffix}"] += 1


def metric_payload(counter: Counter[str], label_types: list[str]) -> dict[str, dict[str, float | int]]:
    return {
        label_type: prf(counter[f"{label_type}::tp"], counter[f"{label_type}::fp"], counter[f"{label_type}::fn"])
        for label_type in label_types
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(args.summary.read_text(encoding="utf-8")) if args.summary else None
    label_types = summary["trained_types"] if summary else sorted({item[0] for row in rows for item in row["gt_pairs"] + row["pred_pairs"]})

    total_counts = Counter()
    type_counts = Counter()
    difficulty_counts: dict[str, Counter[str]] = defaultdict(Counter)
    out_rows: list[dict[str, Any]] = []

    for row in rows:
        gold = {(item[0], normalize_cleaned_pair(item[0], item[1])) for item in row["gt_pairs"]}
        original_pred = {(item[0], normalize_cleaned_pair(item[0], item[1])) for item in row["pred_pairs"]}
        pred = postprocess_row(
            row,
            add_date_variants=args.add_date_variants,
            collapse_work_contact=args.collapse_work_contact,
            add_encoded_emails=args.add_encoded_emails,
        )
        tp = gold & pred
        fp = pred - gold
        fn = gold - pred
        original_fp = original_pred - gold
        original_fn = gold - original_pred

        total_counts["rows"] += 1
        total_counts["sample_exact"] += int(gold == pred)
        total_counts["tp"] += len(tp)
        total_counts["fp"] += len(fp)
        total_counts["fn"] += len(fn)

        difficulty = row.get("difficulty", "UNKNOWN")
        difficulty_counts[difficulty]["rows"] += 1
        difficulty_counts[difficulty]["sample_exact"] += int(gold == pred)
        difficulty_counts[difficulty]["tp"] += len(tp)
        difficulty_counts[difficulty]["fp"] += len(fp)
        difficulty_counts[difficulty]["fn"] += len(fn)

        add_type_counts(type_counts, tp, "tp")
        add_type_counts(type_counts, fp, "fp")
        add_type_counts(type_counts, fn, "fn")

        out = dict(row)
        out["postprocessed_pairs"] = sorted([pair_dict(t, v) for t, v in pred], key=lambda x: (x["type"], x["value"]))
        out["postprocessed_tp_pairs"] = sorted([pair_dict(t, v) for t, v in tp], key=lambda x: (x["type"], x["value"]))
        out["postprocessed_fp_pairs"] = sorted([pair_dict(t, v) for t, v in fp], key=lambda x: (x["type"], x["value"]))
        out["postprocessed_fn_pairs"] = sorted([pair_dict(t, v) for t, v in fn], key=lambda x: (x["type"], x["value"]))
        out["postprocessed_sample_exact_match"] = gold == pred
        out["fixed_fp_pairs"] = sorted([pair_dict(t, v) for t, v in original_fp - fp], key=lambda x: (x["type"], x["value"]))
        out["fixed_fn_pairs"] = sorted([pair_dict(t, v) for t, v in original_fn - fn], key=lambda x: (x["type"], x["value"]))
        out_rows.append(out)

    difficulty_payload = []
    for difficulty, counts in sorted(difficulty_counts.items()):
        metrics = prf(counts["tp"], counts["fp"], counts["fn"])
        difficulty_payload.append(
            {
                "difficulty": difficulty,
                "rows": counts["rows"],
                "sample_exact_acc": counts["sample_exact"] / counts["rows"] if counts["rows"] else 0.0,
                **metrics,
            }
        )

    per_type = metric_payload(type_counts, label_types)
    output_prediction_path = args.predictions_out
    output_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with output_prediction_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "source_predictions": str(args.predictions),
        "source_summary": str(args.summary) if args.summary else None,
        "predictions_out": str(output_prediction_path),
        "rules": {
            "strip_known_prefixes": True,
            "add_date_variants": args.add_date_variants,
            "collapse_work_contact": args.collapse_work_contact,
            "add_encoded_emails": args.add_encoded_emails,
        },
        "overall": {
            "rows": total_counts["rows"],
            "sample_exact_acc": total_counts["sample_exact"] / total_counts["rows"] if total_counts["rows"] else 0.0,
            **prf(total_counts["tp"], total_counts["fp"], total_counts["fn"]),
        },
        "difficulty_breakdown": difficulty_payload,
        "per_type_breakdown": per_type,
        "fixed_fp_count": sum(len(row["fixed_fp_pairs"]) for row in out_rows),
        "fixed_fn_count": sum(len(row["fixed_fn_pairs"]) for row in out_rows),
        "rows_improved": sum(
            len(row["postprocessed_fp_pairs"]) + len(row["postprocessed_fn_pairs"]) < len(row["fp_pairs"]) + len(row["fn_pairs"])
            for row in out_rows
        ),
        "rows_worsened": sum(
            len(row["postprocessed_fp_pairs"]) + len(row["postprocessed_fn_pairs"]) > len(row["fp_pairs"]) + len(row["fn_pairs"])
            for row in out_rows
        ),
    }


def write_markdown(path: Path, payload: dict[str, Any], original_summary: dict[str, Any] | None) -> None:
    lines = ["# Qwen Cleaned-200 Postprocessed Evaluation", ""]
    lines.append(f"Source predictions: `{payload['source_predictions']}`")
    lines.append(f"Postprocessed predictions: `{payload['predictions_out']}`")
    lines.append("")
    if original_summary:
        original = original_summary["overall"]
        current = payload["overall"]
        lines.append("## Before vs After")
        lines.append("")
        lines.append("| Metric | Before | After | Delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        for key in ["precision", "recall", "f1", "sample_exact_acc"]:
            lines.append(f"| `{key}` | {original[key]:.4f} | {current[key]:.4f} | {current[key] - original[key]:+.4f} |")
        for key in ["tp", "fp", "fn"]:
            lines.append(f"| `{key}` | {original[key]} | {current[key]} | {current[key] - original[key]:+d} |")
    else:
        current = payload["overall"]
        lines.append("## Overall")
        lines.append("")
        lines.append(f"- Value-level P/R/F1: {current['precision']:.4f} / {current['recall']:.4f} / {current['f1']:.4f}")
        lines.append(f"- TP / FP / FN: {current['tp']} / {current['fp']} / {current['fn']}")
    lines.append("")
    lines.append("## Rule Impact")
    lines.append("")
    lines.append(f"- Fixed FP pairs: {payload['fixed_fp_count']}")
    lines.append(f"- Fixed FN pairs: {payload['fixed_fn_count']}")
    lines.append(f"- Rows improved: {payload['rows_improved']}")
    lines.append(f"- Rows worsened: {payload['rows_worsened']}")
    lines.append("")
    lines.append("## Difficulty")
    lines.append("")
    lines.append("| Difficulty | Sample Exact | Precision | Recall | F1 | TP | FP | FN |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in payload["difficulty_breakdown"]:
        lines.append(
            f"| `{item['difficulty']}` | {item['sample_exact_acc']:.4f} | {item['precision']:.4f} | "
            f"{item['recall']:.4f} | {item['f1']:.4f} | {item['tp']} | {item['fp']} | {item['fn']} |"
        )
    lines.append("")
    lines.append("## Weakest Remaining Types")
    lines.append("")
    lines.append("| Type | Precision | Recall | F1 | TP | FP | FN |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    active = [
        (metrics["f1"], label_type, metrics)
        for label_type, metrics in payload["per_type_breakdown"].items()
        if metrics["tp"] + metrics["fp"] + metrics["fn"] > 0
    ]
    for _, label_type, metrics in sorted(active)[:12]:
        lines.append(
            f"| `{label_type}` | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process Qwen cleaned-200 predictions and rescore.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--add-date-variants", action="store_true")
    parser.add_argument("--collapse-work-contact", action="store_true")
    parser.add_argument("--add-encoded-emails", action="store_true")
    args = parser.parse_args()

    original_summary = json.loads(args.summary.read_text(encoding="utf-8")) if args.summary else None
    payload = evaluate(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    write_markdown(args.md_out, payload, original_summary)
    print(f"Wrote {args.predictions_out}")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

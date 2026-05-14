from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ZERO_EXAMPLE_LABELS = [
    "BANK_ACCOUNT_INFORMATION",
    "FIRST_NAME",
    "HOME_PHONE",
    "LAST_NAME",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def merge_augmented_dataset(
    stage1_records: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    training_labels: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage2_records = [stage2_row_to_record(row) for row in stage2_rows]
    merged = stage1_records + stage2_records
    validation = validate_merged_records(merged, training_labels)
    stats = build_statistics(merged, training_labels)
    distribution = {
        "per_label_count": stats["per_label_count"],
        "label_distribution": stats["label_distribution"],
        "non_pii_distribution": stats["non_pii_distribution"],
        "all_80_classes_represented": stats["all_80_classes_represented"],
        "missing_classes": stats["missing_classes"],
    }
    warnings = build_warning_examples(merged)
    audit = {
        **validation,
        **stats,
        "stage1_record_count": len(stage1_records),
        "stage2_record_count": len(stage2_records),
        "merged_into_model_training": False,
        "student_training_started": False,
    }
    return merged, audit, distribution, warnings


def stage2_row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("context", ""))
    value = str(row.get("span_value", ""))
    start = text.find(value)
    if start < 0:
        start = 0
        text = value if not text else text
    end = start + len(value)
    context_type = str(row.get("context_type", "unknown"))
    subtype = "candidate_level_negative" if is_candidate_level_negative_context(context_type) else "qwen_5way_ranking"
    weight = row.get("training_weight", default_stage2_weight(row))
    span = {
        "start": start,
        "end": end,
        "value": value,
        "type_distribution": row.get("type_distribution", {}),
        "top_type": row.get("top_type"),
        "source": "qwen_5way_ranking",
        "training_weight": weight,
        "format_candidates": row.get("candidate_labels", []),
        "rule_verified": False,
        "teacher_verdicts": row.get("verdicts", {}),
        "source_prompt_id": row.get("source_prompt_id", row.get("id")),
        "context_type": context_type,
    }
    return {
        "id": f"STAGE2-{row.get('id')}",
        "text": text,
        "metadata": {
            "source_type": "qwen_5way_ranking",
            "data_category": "stage2_teacher",
            "subtype": subtype,
            "ambiguity_group": row.get("ambiguity_group"),
            "context_type": context_type,
            "language": "en-AU",
        },
        "spans": [span],
    }


def default_stage2_weight(row: dict[str, Any]) -> float:
    context_type = row.get("context_type")
    if context_type == "hard_negative_context":
        return 0.8
    if context_type == "reverse_negative_context":
        return 0.5
    if str(row.get("ambiguity_group", "")).startswith("zero_example_"):
        return 0.6
    if context_type == "strong_positive_context":
        return 0.7
    return 0.5


def is_candidate_level_negative_context(context_type: str) -> bool:
    return context_type in {"reverse_negative_context", "reverse_negative", "hard_negative_context", "hard_negative"}


def validate_merged_records(records: list[dict[str, Any]], training_labels: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    labels_outside: Counter[str] = Counter()
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    offset_mismatch_count = 0
    for record in records:
        record_id = str(record.get("id"))
        if record_id in seen_ids:
            duplicate_ids.append(record_id)
            errors.append(f"{record_id}: duplicated record id")
        seen_ids.add(record_id)
        text = str(record.get("text", ""))
        for span in record.get("spans", []):
            start = span.get("start")
            end = span.get("end")
            value = str(span.get("value", ""))
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start or end > len(text):
                offset_mismatch_count += 1
                errors.append(f"{record_id}: invalid offsets for {value!r}")
            elif text[start:end] != value:
                offset_mismatch_count += 1
                errors.append(f"{record_id}: offset mismatch for {value!r}")
            dist = span.get("type_distribution", {})
            if not isinstance(dist, dict) or not dist:
                errors.append(f"{record_id}: missing type_distribution")
                continue
            total = sum(float(v) for v in dist.values())
            if abs(total - 1.0) > 0.02:
                errors.append(f"{record_id}: distribution sum {total}")
            top_type = span.get("top_type")
            if top_type != max(dist, key=dist.get):
                errors.append(f"{record_id}: top_type is not argmax")
            labels = set(dist) | {top_type} | set(span.get("format_candidates", []))
            for label in labels:
                if label not in training_labels:
                    labels_outside[str(label)] += 1
                    errors.append(f"{record_id}: label outside training space: {label}")
            weight = span.get("training_weight")
            if not isinstance(weight, (int, float)) or not 0 <= float(weight) <= 1:
                errors.append(f"{record_id}: invalid training_weight {weight}")
    return {
        "validation_error_count": len(errors),
        "validation_errors": errors[:100],
        "labels_outside_training_space": dict(sorted(labels_outside.items())),
        "offset_mismatch_count": offset_mismatch_count,
        "duplicate_record_ids": duplicate_ids[:100],
        "duplicate_record_id_count": len(duplicate_ids),
    }


def build_statistics(records: list[dict[str, Any]], training_labels: set[str]) -> dict[str, Any]:
    source_distribution: Counter[str] = Counter()
    label_distribution: Counter[str] = Counter()
    per_label_count: Counter[str] = Counter()
    non_pii_distribution = {"top_type_count": 0, "candidate_count": 0, "probability_sum": 0.0}
    high_entropy_sample_count = 0
    total_spans = 0
    for record in records:
        spans = record.get("spans", [])
        if not spans:
            source_distribution["document_level_negative"] += 1
            continue
        for span in spans:
            total_spans += 1
            source_distribution[source_key(record, span)] += 1
            top_type = str(span.get("top_type"))
            label_distribution[top_type] += 1
            per_label_count[top_type] += 1
            dist = span.get("type_distribution", {})
            if "NON_PII" in dist:
                non_pii_distribution["candidate_count"] += 1
                non_pii_distribution["probability_sum"] += float(dist["NON_PII"])
            if top_type == "NON_PII":
                non_pii_distribution["top_type_count"] += 1
            if entropy(dist) >= 1.5:
                high_entropy_sample_count += 1
    represented = {label for label, count in per_label_count.items() if count > 0}
    missing = sorted(training_labels - represented)
    zero_missing = [label for label in ZERO_EXAMPLE_LABELS if per_label_count.get(label, 0) == 0]
    non_pii_distribution["probability_mean"] = round(
        non_pii_distribution["probability_sum"] / non_pii_distribution["candidate_count"],
        6,
    ) if non_pii_distribution["candidate_count"] else 0.0
    non_pii_distribution["probability_sum"] = round(non_pii_distribution["probability_sum"], 6)
    return {
        "total_records": len(records),
        "total_spans": total_spans,
        "source_distribution": dict(sorted(source_distribution.items())),
        "label_distribution": dict(sorted(label_distribution.items())),
        "non_pii_distribution": non_pii_distribution,
        "per_label_count": {label: per_label_count.get(label, 0) for label in sorted(training_labels)},
        "zero_example_labels_after_merge": zero_missing,
        "high_entropy_sample_count": high_entropy_sample_count,
        "all_80_classes_represented": len(missing) == 0,
        "missing_classes": missing,
    }


def source_key(record: dict[str, Any], span: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    if metadata.get("subtype") == "candidate_level_negative":
        return "candidate_level_negative"
    return str(span.get("source") or metadata.get("source_type") or "unknown")


def entropy(distribution: dict[str, Any]) -> float:
    value = 0.0
    for probability in distribution.values():
        p = float(probability)
        if p > 0:
            value -= p * math.log2(p)
    return value


def build_warning_examples(records: list[dict[str, Any]], limit: int = 50) -> dict[str, Any]:
    strong_positive_not_confident = []
    for record in records:
        for span in record.get("spans", []):
            context_type = span.get("context_type") or record.get("metadata", {}).get("context_type")
            dist = span.get("type_distribution", {})
            top_type = span.get("top_type")
            top_prob = float(dist.get(top_type, 0.0)) if top_type in dist else 0.0
            if context_type == "strong_positive_context" and top_prob < 0.50:
                strong_positive_not_confident.append(
                    {
                        "id": record.get("id"),
                        "value": span.get("value"),
                        "top_type": top_type,
                        "top_probability": round(top_prob, 6),
                        "context": record.get("text"),
                        "type_distribution": dist,
                    }
                )
    return {
        "strong_positive_not_confident_count": len(strong_positive_not_confident),
        "strong_positive_not_confident_examples": strong_positive_not_confident[:limit],
        "merged_into_model_training": False,
        "student_training_started": False,
    }


def merge_project(root: Path | str = ".") -> dict[str, Any]:
    root = Path(root)
    stage1_path = root / "data" / "processed" / "stage1_v3_2_canonical.jsonl"
    stage2_path = root / "data" / "generated" / "stage2_full_teacher_converted.jsonl"
    hard_negative_stage2_path = root / "data" / "generated" / "stage2_hard_negative_teacher_converted.jsonl"
    training_path = root / "pii_schema" / "training_label_space_80.json"
    quality_path = root / "reports" / "stage2_full_teacher_quality_report.json"
    warning_path = root / "reports" / "stage2_full_teacher_warning_examples.json"
    output_path = root / "data" / "processed" / "stage2_v3_2_augmented.jsonl"
    audit_path = root / "reports" / "stage2_augmented_audit.json"
    distribution_path = root / "reports" / "stage2_augmented_label_distribution.json"
    warnings_path = root / "reports" / "stage2_augmented_warning_examples.json"

    stage1_records = load_jsonl(stage1_path)
    stage2_rows_by_input = {str(stage2_path): load_jsonl(stage2_path)}
    if hard_negative_stage2_path.exists():
        stage2_rows_by_input[str(hard_negative_stage2_path)] = load_jsonl(hard_negative_stage2_path)
    stage2_rows = [row for rows in stage2_rows_by_input.values() for row in rows]
    training_labels = set(json.loads(training_path.read_text(encoding="utf-8")))
    merged, audit, distribution, warnings = merge_augmented_dataset(stage1_records, stage2_rows, training_labels)
    audit["inputs"] = {
        "stage1_path": str(stage1_path),
        "stage2_path": str(stage2_path),
        "stage2_paths": list(stage2_rows_by_input),
        "stage2_row_count_by_input": {path: len(rows) for path, rows in stage2_rows_by_input.items()},
        "training_label_space": str(training_path),
        "stage2_quality_report": str(quality_path),
        "stage2_warning_examples": str(warning_path),
    }
    if quality_path.exists():
        audit["stage2_quality_warning_counts"] = json.loads(quality_path.read_text(encoding="utf-8")).get("warning_counts", {})
    write_jsonl(output_path, merged)
    write_json(audit_path, audit)
    write_json(distribution_path, distribution)
    write_json(warnings_path, warnings)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    audit = merge_project(args.root)
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["validation_error_count"] != 0:
        raise SystemExit("augmented dataset validation failed")
    if audit["labels_outside_training_space"]:
        raise SystemExit("labels outside training space found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

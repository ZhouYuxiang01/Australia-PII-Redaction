from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ZERO_EXAMPLE_LABELS = [
    "BANK_ACCOUNT_INFORMATION",
    "FIRST_NAME",
    "HOME_PHONE",
    "LAST_NAME",
]

SPLITS = ("train", "dev", "test")


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


def source_record_key(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    if metadata.get("subtype") == "doc_level":
        return "document_level_negative"
    if metadata.get("subtype") == "candidate_level_negative":
        return "candidate_level_negative"
    if metadata.get("source_type") == "qwen_5way_ranking":
        return "qwen_5way_ranking"
    return str(metadata.get("source_type", "unknown"))


def source_span_key(record: dict[str, Any], span: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    if metadata.get("subtype") == "candidate_level_negative":
        return "candidate_level_negative"
    return str(span.get("source") or metadata.get("source_type") or "unknown")


def build_audit_v2(records: list[dict[str, Any]], base_audit: dict[str, Any]) -> dict[str, Any]:
    record_distribution: Counter[str] = Counter()
    span_distribution: Counter[str] = Counter()
    for record in records:
        record_distribution[source_record_key(record)] += 1
        for span in record.get("spans", []):
            span_distribution[source_span_key(record, span)] += 1
    audit = dict(base_audit)
    audit.pop("source_distribution", None)
    audit["source_record_distribution"] = dict(sorted(record_distribution.items()))
    audit["source_span_distribution"] = dict(sorted(span_distribution.items()))
    audit["student_training_started"] = False
    audit["merged_into_model_training"] = False
    return audit


def group_key(record: dict[str, Any]) -> str:
    record_id = str(record.get("id", ""))
    metadata = record.get("metadata", {})
    if metadata.get("subtype") == "candidate_level_negative" and record.get("spans"):
        span = record.get("spans", [{}])[0]
        return "|".join(
            [
                "stage2-near-duplicate",
                str(metadata.get("ambiguity_group")),
                str(metadata.get("context_type")),
                str(span.get("value")),
                str(record.get("text")),
                ",".join(span.get("format_candidates", [])),
            ]
        )
    if record_id.startswith("STAGE2-STAGE2-FULL-BASE-"):
        base = record_id.rsplit("-SC", 1)[0]
        return base
    if record_id.startswith("AU-PII-"):
        parts = record_id.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:3])
    return record_id


def split_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[group_key(record)].append(record)
    split_groups: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for key in sorted(groups):
        split = split_for_key(key)
        split_groups[split].extend(groups[key])
    return split_groups


def split_for_key(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def validate_records(records: list[dict[str, Any]], training_labels: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    labels_outside: Counter[str] = Counter()
    offset_mismatch_count = 0
    for record in records:
        text = str(record.get("text", ""))
        for span in record.get("spans", []):
            start = span.get("start")
            end = span.get("end")
            value = str(span.get("value", ""))
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start or end > len(text) or text[start:end] != value:
                offset_mismatch_count += 1
                errors.append(f"{record.get('id')}: invalid offset for {value!r}")
            dist = span.get("type_distribution", {})
            if not isinstance(dist, dict) or not dist:
                errors.append(f"{record.get('id')}: missing distribution")
                continue
            if abs(sum(float(v) for v in dist.values()) - 1.0) > 0.02:
                errors.append(f"{record.get('id')}: distribution does not sum to 1")
            if span.get("top_type") != max(dist, key=dist.get):
                errors.append(f"{record.get('id')}: top_type is not argmax")
            labels = set(dist) | {span.get("top_type")} | set(span.get("format_candidates", []))
            for label in labels:
                if label not in training_labels:
                    labels_outside[str(label)] += 1
    return {
        "validation_error_count": len(errors),
        "validation_errors": errors[:100],
        "labels_outside_training_space": dict(sorted(labels_outside.items())),
        "offset_mismatch_count": offset_mismatch_count,
    }


def build_split_report(splits: dict[str, list[dict[str, Any]]], training_labels: set[str]) -> dict[str, Any]:
    report = {"splits": {}, "student_training_started": False}
    for split, records in splits.items():
        report["splits"][split] = split_statistics(records, training_labels)
    return report


def split_statistics(records: list[dict[str, Any]], training_labels: set[str]) -> dict[str, Any]:
    span_count = 0
    label_distribution: Counter[str] = Counter()
    source_record_distribution: Counter[str] = Counter()
    source_span_distribution: Counter[str] = Counter()
    non_pii_count = 0
    high_entropy_count = 0
    for record in records:
        source_record_distribution[source_record_key(record)] += 1
        for span in record.get("spans", []):
            span_count += 1
            top_type = str(span.get("top_type"))
            label_distribution[top_type] += 1
            if top_type == "NON_PII":
                non_pii_count += 1
            source_span_distribution[source_span_key(record, span)] += 1
            if entropy(span.get("type_distribution", {})) >= 1.5:
                high_entropy_count += 1
    missing_labels = sorted(label for label in training_labels if label_distribution.get(label, 0) == 0)
    return {
        "record_count": len(records),
        "span_count": span_count,
        "label_distribution": dict(sorted(label_distribution.items())),
        "source_record_distribution": dict(sorted(source_record_distribution.items())),
        "source_span_distribution": dict(sorted(source_span_distribution.items())),
        "non_pii_count": non_pii_count,
        "high_entropy_sample_count": high_entropy_count,
        "zero_example_labels_missing": [label for label in ZERO_EXAMPLE_LABELS if label_distribution.get(label, 0) == 0],
        "missing_label_count": len(missing_labels),
        "missing_labels": missing_labels,
    }


def entropy(distribution: dict[str, Any]) -> float:
    value = 0.0
    for probability in distribution.values():
        p = float(probability)
        if p > 0:
            value -= p * math.log2(p)
    return value


def full_distribution(span: dict[str, Any], training_labels: list[str]) -> dict[str, float]:
    dist = span.get("type_distribution", {})
    return {label: round(float(dist.get(label, 0.0)), 6) for label in training_labels}


def build_qwen_spancls_examples(records: list[dict[str, Any]], split: str, training_labels: list[str]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        text = str(record.get("text", ""))
        for span_index, span in enumerate(record.get("spans", [])):
            examples.append(
                {
                    "id": f"{record.get('id')}::span-{span_index}",
                    "record_id": record.get("id"),
                    "text": text,
                    "start": span["start"],
                    "end": span["end"],
                    "value": span["value"],
                    "target_distribution": full_distribution(span, training_labels),
                    "top_type": span["top_type"],
                    "training_weight": span.get("training_weight", 1.0),
                    "source": source_span_key(record, span),
                    "split": split,
                }
            )
    return examples


def build_opf_hard_records(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        spans = []
        for span in record.get("spans", []):
            if span.get("top_type") == "NON_PII":
                continue
            spans.append(
                {
                    "start": span["start"],
                    "end": span["end"],
                    "value": span["value"],
                    "label": span["top_type"],
                    "source": source_span_key(record, span),
                }
            )
        output.append(
            {
                "id": record.get("id"),
                "text": record.get("text", ""),
                "metadata": {**record.get("metadata", {}), "split": split},
                "spans": spans,
            }
        )
    return output


def validate_qwen_examples(examples: list[dict[str, Any]], training_labels: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    labels_outside: Counter[str] = Counter()
    for example in examples:
        text = str(example["text"])
        if text[example["start"] : example["end"]] != example["value"]:
            errors.append(f"{example['id']}: offset mismatch")
        if not example["target_distribution"]:
            errors.append(f"{example['id']}: empty target distribution")
        for label in set(example["target_distribution"]) | {example["top_type"]}:
            if label not in training_labels:
                labels_outside[str(label)] += 1
        if abs(sum(float(v) for v in example["target_distribution"].values()) - 1.0) > 0.02:
            errors.append(f"{example['id']}: distribution does not sum to 1")
    return {
        "example_count": len(examples),
        "empty_span_record_count": 0,
        "validation_error_count": len(errors),
        "validation_errors": errors[:100],
        "labels_outside_training_space": dict(sorted(labels_outside.items())),
    }


def validate_opf_records(records: list[dict[str, Any]], training_labels: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    labels_outside: Counter[str] = Counter()
    offset_mismatch_count = 0
    for record in records:
        text = str(record.get("text", ""))
        for span in record.get("spans", []):
            if text[span["start"] : span["end"]] != span["value"]:
                offset_mismatch_count += 1
                errors.append(f"{record.get('id')}: offset mismatch")
            if span["label"] not in training_labels or span["label"] == "NON_PII":
                labels_outside[str(span["label"])] += 1
    return {
        "record_count": len(records),
        "validation_error_count": len(errors),
        "validation_errors": errors[:100],
        "offset_mismatch_count": offset_mismatch_count,
        "labels_outside_training_space": dict(sorted(labels_outside.items())),
    }


def build_stage3_project(root: Path | str = ".") -> dict[str, Any]:
    root = Path(root)
    records = load_jsonl(root / "data" / "processed" / "stage2_v3_2_augmented.jsonl")
    training_labels = json.loads((root / "pii_schema" / "training_label_space_80.json").read_text(encoding="utf-8"))
    training_label_set = set(training_labels)
    base_audit = json.loads((root / "reports" / "stage2_augmented_audit.json").read_text(encoding="utf-8"))
    audit_v2 = build_audit_v2(records, base_audit)
    write_json(root / "reports" / "stage2_augmented_audit_v2.json", audit_v2)

    splits = split_records(records)
    split_dir = root / "data" / "splits"
    train_dir = root / "data" / "train"
    split_report = build_split_report(splits, training_label_set)
    qwen_report = {"splits": {}, "student_training_started": False}
    opf_report = {"splits": {}, "student_training_started": False}
    for split in SPLITS:
        write_jsonl(split_dir / f"{split}.jsonl", splits[split])
        qwen_examples = build_qwen_spancls_examples(splits[split], split, training_labels)
        opf_records = build_opf_hard_records(splits[split], split)
        write_jsonl(train_dir / f"qwen_spancls_{split}.jsonl", qwen_examples)
        write_jsonl(train_dir / f"opf_hard_{split}.jsonl", opf_records)
        qwen_report["splits"][split] = {
            **validate_qwen_examples(qwen_examples, training_label_set),
            "label_distribution": split_report["splits"][split]["label_distribution"],
            "source_distribution": split_report["splits"][split]["source_span_distribution"],
        }
        opf_report["splits"][split] = {
            **validate_opf_records(opf_records, training_label_set),
            "source_record_distribution": split_report["splits"][split]["source_record_distribution"],
        }
    split_validation = validate_records(records, training_label_set)
    split_report["validation"] = split_validation
    split_report["split_ratios"] = {split: round(len(splits[split]) / len(records), 6) for split in SPLITS}
    write_json(root / "reports" / "stage3_split_report.json", split_report)
    write_json(root / "reports" / "stage3_qwen_spancls_dataset_report.json", qwen_report)
    write_json(root / "reports" / "stage3_opf_hard_dataset_report.json", opf_report)
    return split_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    report = build_stage3_project(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    validation = report["validation"]
    if validation["validation_error_count"] != 0:
        raise SystemExit("split source validation failed")
    if validation["labels_outside_training_space"]:
        raise SystemExit("labels outside training space found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

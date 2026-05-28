from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .taxonomy import RAW_TO_TARGET_OVERRIDES, Taxonomy, canonical_code, load_taxonomy


def load_csv_labels(csv_path: Path) -> tuple[int, list[str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        name = row.get("Name", "").strip()
        if not name:
            continue
        label = canonical_code(name)
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return len(rows), labels


def collect_raw_label_counts(raw_path: Path) -> tuple[list[str], Counter[str]]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_types = [str(label) for label in raw.get("pii_types", [])]
    counts: Counter[str] = Counter()
    for record in raw.get("records", []):
        for label in record.get("positive_sample", {}).get("labels", []):
            if "type" in label:
                counts[str(label["type"])] += 1
    return raw_types, counts


def build_alias_map(raw_labels: list[str], taxonomy: Taxonomy) -> tuple[dict[str, str], dict[str, Any]]:
    csv_labels = set(taxonomy.labels)
    alias_map: dict[str, str] = {}
    exact_matches: list[str] = []
    normalized_matches: list[dict[str, str]] = []
    proposed_alias_mappings: dict[str, str] = {}
    raw_only_unmapped: list[str] = []

    for raw_label in sorted(set(raw_labels)):
        if raw_label in csv_labels:
            alias_map[raw_label] = raw_label
            exact_matches.append(raw_label)
            continue

        normalized = canonical_code(raw_label)
        if normalized in csv_labels:
            alias_map[raw_label] = normalized
            normalized_matches.append({"raw": raw_label, "canonical": normalized})
            continue

        proposed = RAW_TO_TARGET_OVERRIDES.get(raw_label)
        if proposed and proposed in csv_labels:
            alias_map[raw_label] = proposed
            proposed_alias_mappings[raw_label] = proposed
            continue

        raw_only_unmapped.append(raw_label)

    return alias_map, {
        "exact_matches": exact_matches,
        "normalized_matches": normalized_matches,
        "proposed_alias_mappings": proposed_alias_mappings,
        "raw_only_unmapped": raw_only_unmapped,
    }


def reconcile_project(root: Path | str = ".", enforce_expected_counts: bool = True) -> dict[str, Any]:
    root = Path(root)
    docs_csv = root / "docs" / "Data Sensitivity.csv"
    raw_path = root / "data" / "raw" / "au_pii_19000_final.json"
    stage1_path = root / "data" / "processed" / "stage1_v3_2.jsonl"

    csv_row_count, canonical_labels = load_csv_labels(docs_csv)
    taxonomy = load_taxonomy(docs_csv)
    raw_labels, raw_label_counts = collect_raw_label_counts(raw_path)
    alias_map, match_report = build_alias_map(raw_labels, taxonomy)

    mapped_counts: Counter[str] = Counter()
    for raw_label, count in raw_label_counts.items():
        if raw_label in alias_map:
            mapped_counts[alias_map[raw_label]] += count

    csv_only_zero_example = [label for label in canonical_labels if mapped_counts[label] == 0]
    training_labels = canonical_labels + ["NON_PII"]
    report = {
        "csv_row_count": csv_row_count,
        "csv_effective_label_count": len(canonical_labels),
        "raw_label_count": len(set(raw_labels)),
        "exact_matches": match_report["exact_matches"],
        "normalized_matches": match_report["normalized_matches"],
        "raw_only_unmapped": match_report["raw_only_unmapped"],
        "csv_only_zero_example": csv_only_zero_example,
        "proposed_alias_mappings": match_report["proposed_alias_mappings"],
        "final_training_class_count": len(training_labels),
    }

    schema_dir = root / "pii_schema"
    reports_dir = root / "reports"
    schema_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    _write_json(schema_dir / "canonical_labels_79.json", canonical_labels)
    _write_json(schema_dir / "training_label_space_80.json", training_labels)
    _write_json(schema_dir / "label_aliases_v3_2.json", dict(sorted(alias_map.items())))
    _write_json(reports_dir / "label_reconciliation.json", report)

    if report["raw_only_unmapped"]:
        raise ValueError(f"Unmapped raw labels: {report['raw_only_unmapped']}")
    if enforce_expected_counts and len(canonical_labels) != 79:
        raise ValueError(f"CSV effective label count must be 79, got {len(canonical_labels)}")
    if enforce_expected_counts and (len(training_labels) != 80 or "NON_PII" not in training_labels):
        raise ValueError("training label space must contain 79 PII labels plus NON_PII")

    canonical_path = root / "data" / "processed" / "stage1_v3_2_canonical.jsonl"
    audit_path = root / "reports" / "stage1_canonical_audit.json"
    audit = remap_stage1_jsonl(stage1_path, canonical_path, alias_map, set(training_labels))
    _write_json(audit_path, audit)
    if audit["validation_error_count"]:
        raise ValueError(f"canonical stage1 validation failed with {audit['validation_error_count']} errors")
    return report


def remap_stage1_jsonl(
    source_path: Path,
    output_path: Path,
    alias_map: dict[str, str],
    training_labels: set[str],
) -> dict[str, Any]:
    audit = {
        "input_records": 0,
        "output_records": 0,
        "span_count": 0,
        "offset_mismatch_count": 0,
        "labels_outside_training_space": Counter(),
        "validation_errors": [],
    }
    output_rows: list[str] = []

    with source_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            audit["input_records"] += 1
            record = json.loads(line)
            remapped = remap_record(record, alias_map)
            validate_record(remapped, training_labels, audit, line_number)
            output_rows.append(json.dumps(remapped, ensure_ascii=False, separators=(",", ":")))
            audit["output_records"] += 1

    audit["labels_outside_training_space"] = dict(audit["labels_outside_training_space"])
    audit["validation_error_count"] = len(audit["validation_errors"])
    if audit["validation_error_count"] == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(output_rows) + ("\n" if output_rows else ""), encoding="utf-8")
    return audit


def remap_record(record: dict[str, Any], alias_map: dict[str, str]) -> dict[str, Any]:
    record = dict(record)
    spans = []
    for span in record.get("spans", []):
        new_span = dict(span)
        new_span["top_type"] = map_label(str(new_span["top_type"]), alias_map)
        new_span["type_distribution"] = remap_distribution(new_span.get("type_distribution", {}), alias_map)
        new_span["format_candidates"] = sorted(
            {
                map_label(str(candidate), alias_map)
                for candidate in new_span.get("format_candidates", [])
            }
        )
        spans.append(new_span)
    record["spans"] = spans
    return record


def map_label(label: str, alias_map: dict[str, str]) -> str:
    if label == "NON_PII":
        return label
    return alias_map.get(label, label)


def remap_distribution(distribution: dict[str, float], alias_map: dict[str, str]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for label, probability in distribution.items():
        canonical = map_label(str(label), alias_map)
        merged[canonical] = merged.get(canonical, 0.0) + float(probability)
    return {label: round(probability, 6) for label, probability in merged.items()}


def validate_record(record: dict[str, Any], training_labels: set[str], audit: dict[str, Any], line_number: int) -> None:
    text = str(record.get("text", ""))
    for span in record.get("spans", []):
        audit["span_count"] += 1
        if text[span["start"] : span["end"]] != span.get("value"):
            audit["offset_mismatch_count"] += 1
            audit["validation_errors"].append(f"line {line_number}: offset mismatch for {span.get('value')!r}")
        field_labels = [("top_type", span.get("top_type"))]
        field_labels.extend(("type_distribution", label) for label in span.get("type_distribution", {}))
        field_labels.extend(("format_candidates", label) for label in span.get("format_candidates", []))
        for field, label in field_labels:
            if label not in training_labels:
                audit["labels_outside_training_space"][label] += 1
                audit["validation_errors"].append(f"line {line_number}: {field} label outside training space: {label}")
        dist = span.get("type_distribution", {})
        if abs(sum(float(v) for v in dist.values()) - 1.0) > 0.02:
            audit["validation_errors"].append(f"line {line_number}: distribution does not sum to 1")
        if dist and span.get("top_type") != max(dist, key=dist.get):
            audit["validation_errors"].append(f"line {line_number}: top_type is not argmax")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    reconcile_project(Path(args.root))
    print("taxonomy reconciliation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .format_filter import FormatFilter
from .taxonomy import Taxonomy, load_taxonomy


LABEL_SMOOTHING = 0.05
HIGH_CONFIDENCE_THRESHOLD = 0.85


def build_records(
    raw_path: Path | str,
    taxonomy_csv_path: Path | str,
    include_hard_negatives: bool = True,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    records = raw.get("records", [])
    if not isinstance(records, list):
        raise ValueError("raw dataset must contain a records list")
    if limit is not None:
        records = records[:limit]

    taxonomy = load_taxonomy(taxonomy_csv_path, raw_types=raw.get("pii_types", []))
    output: list[dict[str, Any]] = []
    audit = {
        "input_records": len(records),
        "positive_records": 0,
        "hard_negative_records": 0,
        "span_count": 0,
        "label_counts": Counter(),
        "format_mismatch_count": 0,
        "offset_mismatch_count": 0,
    }

    for record in records:
        positive = _convert_positive_record(record, taxonomy, audit)
        output.append(positive)
        audit["positive_records"] += 1
        if include_hard_negatives:
            for index, negative in enumerate(record.get("hard_negatives", [])):
                output.append(_convert_doc_level_negative(record, negative, index))
                audit["hard_negative_records"] += 1

    audit["label_counts"] = dict(audit["label_counts"])
    return output, audit


def _convert_positive_record(record: dict[str, Any], taxonomy: Taxonomy, audit: dict[str, Any]) -> dict[str, Any]:
    positive = record.get("positive_sample", {})
    text = str(positive.get("text", ""))
    metadata = record.get("input", {}).get("metadata", {})
    spans = []

    for label in positive.get("labels", []):
        span = _convert_label(label, text, taxonomy, audit)
        if span is not None:
            spans.append(span)

    return {
        "id": str(record.get("id", "unknown")),
        "text": text,
        "metadata": {
            "source_type": metadata.get("source_type", "unknown"),
            "data_category": "A+B",
            "language": metadata.get("language", "en-AU"),
        },
        "spans": spans,
    }


def _convert_label(
    label: dict[str, Any],
    text: str,
    taxonomy: Taxonomy,
    audit: dict[str, Any],
) -> dict[str, Any] | None:
    start = int(label["start"])
    end = int(label["end"])
    value = str(label.get("value", text[start:end]))
    if text[start:end] != value:
        audit["offset_mismatch_count"] += 1
        return None

    target_type = taxonomy.map_raw_type(str(label["type"]))
    confidence = float(label.get("confidence", 0.0))
    rule_verified = FormatFilter.rule_verified(target_type, value)
    candidates = FormatFilter.get_candidates(value)
    format_mismatch = target_type not in candidates
    if format_mismatch:
        audit["format_mismatch_count"] += 1
        candidates.add(target_type)

    weight = _training_weight(confidence, rule_verified)
    source = "sonnet_high_conf" if confidence >= HIGH_CONFIDENCE_THRESHOLD else "sonnet_low_conf"
    distribution = {target_type: round(1.0 - LABEL_SMOOTHING, 6), "NON_PII": LABEL_SMOOTHING}

    audit["span_count"] += 1
    audit["label_counts"][target_type] += 1
    return {
        "start": start,
        "end": end,
        "value": value,
        "type_distribution": distribution,
        "top_type": target_type,
        "source": source,
        "training_weight": weight,
        "format_candidates": sorted(candidates),
        "rule_verified": rule_verified,
        "format_mismatch": format_mismatch,
        "teacher_confidence": confidence,
        "raw_type": str(label["type"]),
    }


def _training_weight(confidence: float, rule_verified: bool) -> float:
    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        weight = 0.8
    else:
        weight = 0.4 + 0.4 * max(0.0, min(confidence, HIGH_CONFIDENCE_THRESHOLD)) / HIGH_CONFIDENCE_THRESHOLD
    if rule_verified:
        weight = min(1.0, weight + 0.2)
    return round(weight, 6)


def _convert_doc_level_negative(record: dict[str, Any], negative: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"{record.get('id', 'unknown')}-HN-{index + 1:02d}",
        "text": str(negative.get("text", "")),
        "metadata": {
            "source_type": "hard_negative",
            "data_category": "D",
            "subtype": "doc_level",
            "language": "en-AU",
            "note": negative.get("note", ""),
        },
        "spans": [],
    }


def validate_records(records: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for record in records:
        text = record.get("text", "")
        for span in record.get("spans", []):
            if text[span["start"] : span["end"]] != span["value"]:
                errors.append(f"{record.get('id')}: offset mismatch for {span.get('value')!r}")
            dist = span.get("type_distribution", {})
            if abs(sum(dist.values()) - 1.0) > 0.02:
                errors.append(f"{record.get('id')}: distribution sum {sum(dist.values())}")
            if span.get("top_type") != max(dist, key=dist.get):
                errors.append(f"{record.get('id')}: top_type is not argmax")
            if not set(k for k, v in dist.items() if v > 1e-4).issubset(set(span.get("format_candidates", []))):
                errors.append(f"{record.get('id')}: distribution outside format candidates")
            weight = span.get("training_weight", 0)
            if not 0 <= weight <= 1:
                errors.append(f"{record.get('id')}: invalid training_weight {weight}")
    return errors


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/au_pii_19000_final.json")
    parser.add_argument("--taxonomy", default="docs/Data Sensitivity.csv")
    parser.add_argument("--output", default="data/processed/stage1_v3_2.jsonl")
    parser.add_argument("--audit", default="reports/stage1_audit.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-hard-negatives", action="store_true")
    args = parser.parse_args(argv)

    records, audit = build_records(
        args.raw,
        args.taxonomy,
        include_hard_negatives=not args.no_hard_negatives,
        limit=args.limit,
    )
    errors = validate_records(records)
    audit["validation_errors"] = errors[:100]
    audit["validation_error_count"] = len(errors)
    if errors:
        raise SystemExit(f"validation failed with {len(errors)} errors; see {args.audit}")

    write_jsonl(Path(args.output), records)
    Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} records to {args.output}")
    print(f"wrote audit to {args.audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ["train", "dev", "test"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_jsonl(path: Path):
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            yield line_no, json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8")


def validate_and_convert_split(path: Path, output_path: Path, canonical_labels: list[str]) -> dict[str, Any]:
    label_set = set(canonical_labels)
    converted_rows = []
    errors = []
    label_counts: Counter[str] = Counter()
    empty_span_records = 0
    for line_no, record in iter_jsonl(path):
        text = str(record.get("text", ""))
        spans = record.get("spans", [])
        if not isinstance(spans, list):
            errors.append({"path": str(path), "line": line_no, "error": "spans_not_list"})
            continue
        span_map: dict[str, list[list[int]]] = defaultdict(list)
        if not spans:
            empty_span_records += 1
        for span_idx, span in enumerate(spans):
            if not isinstance(span, dict):
                errors.append({"path": str(path), "line": line_no, "span_index": span_idx, "error": "span_not_object"})
                continue
            label = str(span.get("label", ""))
            if label == "NON_PII":
                errors.append({"path": str(path), "line": line_no, "span_index": span_idx, "label": label, "error": "non_pii_span_not_allowed"})
                continue
            if label not in label_set:
                errors.append({"path": str(path), "line": line_no, "span_index": span_idx, "label": label, "error": "label_outside_canonical_79"})
                continue
            start = span.get("start")
            end = span.get("end")
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
                errors.append({"path": str(path), "line": line_no, "span_index": span_idx, "label": label, "error": "offset_not_integer"})
                continue
            if not (0 <= start < end <= len(text)):
                errors.append({"path": str(path), "line": line_no, "span_index": span_idx, "label": label, "start": start, "end": end, "text_length": len(text), "error": "offset_out_of_bounds"})
                continue
            value = text[start:end]
            if span.get("value") is not None and str(span.get("value")) != value:
                errors.append({"path": str(path), "line": line_no, "span_index": span_idx, "label": label, "expected": span.get("value"), "actual": value, "error": "value_offset_mismatch"})
                continue
            span_map[label].append([start, end])
            label_counts[label] += 1
        converted_rows.append(
            {
                "id": record.get("id"),
                "text": text,
                "spans": dict(sorted(span_map.items())),
                "metadata": record.get("metadata", {}),
            }
        )
    if errors:
        return {"ok": False, "input_path": str(path), "output_path": str(output_path), "errors": errors[:500], "error_count": len(errors)}
    write_jsonl(output_path, converted_rows)
    return {
        "ok": True,
        "input_path": str(path),
        "output_path": str(output_path),
        "record_count": len(converted_rows),
        "empty_span_records": empty_span_records,
        "span_count": sum(label_counts.values()),
        "label_counts": dict(sorted(label_counts.items())),
        "errors": [],
        "error_count": 0,
    }


def prepare_opf(root: Path | str = ".") -> dict[str, Any]:
    root = Path(root)
    canonical_labels = read_json(root / "pii_schema" / "canonical_labels_79.json")
    training_labels = read_json(root / "pii_schema" / "training_label_space_80.json")
    if len(canonical_labels) != 79:
        raise ValueError(f"canonical_labels_79.json must contain 79 labels, got {len(canonical_labels)}")
    if "NON_PII" not in training_labels:
        raise ValueError("training_label_space_80.json must include NON_PII")
    if "NON_PII" in canonical_labels:
        raise ValueError("canonical_labels_79.json must not include NON_PII")
    missing = sorted(set(canonical_labels) - set(training_labels))
    if missing:
        raise ValueError(f"canonical labels missing from training label space: {missing[:20]}")
    label_space = {
        "category_version": "au_pii_79_v1",
        "span_class_names": ["O", *canonical_labels],
    }
    write_json(root / "pii_schema" / "opf_label_space_79.json", label_space)
    split_reports = {}
    all_errors = []
    for split in SPLITS:
        report = validate_and_convert_split(
            root / "data" / "train" / f"opf_hard_{split}.jsonl",
            root / "data" / "train" / f"opf_{split}_opf_format.jsonl",
            canonical_labels,
        )
        split_reports[split] = {key: value for key, value in report.items() if key != "errors"}
        all_errors.extend({"split": split, **error} for error in report.get("errors", []))
    status = {
        "opf_label_space_path": "pii_schema/opf_label_space_79.json",
        "span_class_count_including_o": len(label_space["span_class_names"]),
        "pii_span_class_count": len(canonical_labels),
        "non_pii_in_span_classes": "NON_PII" in label_space["span_class_names"],
        "split_reports": split_reports,
        "validation_error_count": len(all_errors),
        "converted_paths": {
            split: f"data/train/opf_{split}_opf_format.jsonl" for split in SPLITS
        },
    }
    write_json(root / "reports" / "stage3b_opf_hard_prepare_report.json", status)
    write_json(root / "reports" / "stage3b_opf_hard_errors.json", {"error_count": len(all_errors), "errors": all_errors})
    if all_errors:
        raise SystemExit("OPF dataset validation failed; see reports/stage3b_opf_hard_errors.json")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    print(json.dumps(prepare_opf(args.root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

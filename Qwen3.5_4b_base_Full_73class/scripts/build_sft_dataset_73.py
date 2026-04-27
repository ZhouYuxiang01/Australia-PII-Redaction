#!/usr/bin/env python3
"""Build Qwen SFT data for the 73-class canonical AU PII taxonomy.

The output uses JSON span lists instead of inline XML-like tags because the
full taxonomy contains intentional overlapping spans such as
GEOLOCATION_INFORMATION containing LATITUDE and LONGITUDE.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in lean stdlib environments
    yaml = None


DEFAULT_SYSTEM_PREFIX = """You are a PII span extraction model for Australian context.
Return only compact JSON with this exact top-level shape: {"spans":[...]}.
Each span object must contain start, end, type, and value.
start and end are Python-style character offsets into the input text; end is exclusive.
Use only the supported types listed below. Preserve overlapping spans when present.
If no supported PII is present, return {"spans":[]}.
Supported types:"""


@dataclass(frozen=True)
class TaxonomyMapping:
    class_labels: list[str]
    source_to_class: dict[str, str]


def load_taxonomy_mapping(path: Path) -> TaxonomyMapping:
    doc = _load_taxonomy_doc(path)
    classes = doc.get("classes", [])
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"taxonomy has no classes: {path}")

    class_labels: list[str] = []
    source_to_class: dict[str, str] = {}
    for entry in classes:
        code = str(entry["code"])
        if code in class_labels:
            raise ValueError(f"duplicate class code in taxonomy: {code}")
        class_labels.append(code)
        for source_type in entry.get("source_types", []):
            source_type = str(source_type)
            if source_type in source_to_class:
                raise ValueError(f"duplicate source type in taxonomy: {source_type}")
            source_to_class[source_type] = code

    return TaxonomyMapping(class_labels=class_labels, source_to_class=source_to_class)


def _load_taxonomy_doc(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return _parse_minimal_taxonomy_yaml(text)


def _parse_minimal_taxonomy_yaml(text: str) -> dict[str, Any]:
    classes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_source_types = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "classes:":
            continue
        if stripped.startswith("- code:"):
            current = {"code": stripped.split(":", 1)[1].strip(), "source_types": []}
            classes.append(current)
            in_source_types = False
            continue
        if current is None:
            continue
        if stripped == "source_types:":
            in_source_types = True
            continue
        if in_source_types and stripped.startswith("- "):
            current["source_types"].append(stripped[2:].strip())
            continue
        in_source_types = False
    return {"classes": classes}


def build_system_prompt(class_labels: list[str]) -> str:
    return DEFAULT_SYSTEM_PREFIX + "\n- " + "\n- ".join(class_labels)


def _normalise_label(label: dict[str, Any], text: str, mapping: TaxonomyMapping) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source_type = str(label.get("type", ""))
    if source_type not in mapping.source_to_class:
        return None, {"reason": f"unknown_source_type:{source_type}", "label": label}

    try:
        start = int(label["start"])
        end = int(label["end"])
    except (KeyError, TypeError, ValueError) as exc:
        return None, {"reason": f"invalid_offsets:{exc}", "label": label}

    if not (0 <= start < end <= len(text)):
        return None, {"reason": "offset_out_of_range", "label": label, "text_length": len(text)}

    value = str(label.get("value", text[start:end]))
    actual = text[start:end]
    if actual != value:
        return None, {"reason": "offset_value_mismatch", "label": label, "actual": actual}

    return {
        "start": start,
        "end": end,
        "type": mapping.source_to_class[source_type],
        "value": value,
    }, None


def map_source_labels(text: str, labels: list[dict[str, Any]], mapping: TaxonomyMapping) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    deduped_after_mapping = 0

    for label in labels:
        span, problem = _normalise_label(label, text, mapping)
        if problem is not None:
            dropped.append(problem)
            continue
        assert span is not None
        key = (span["start"], span["end"], span["type"])
        if key in seen:
            deduped_after_mapping += 1
            continue
        seen.add(key)
        spans.append(span)

    spans.sort(key=lambda item: (item["start"], -(item["end"] - item["start"]), item["type"]))
    overlap_count = 0
    for idx, left in enumerate(spans):
        for right in spans[idx + 1 :]:
            if right["start"] >= left["end"]:
                break
            overlap_count += 1

    return spans, {
        "dropped": dropped,
        "deduped_after_mapping": deduped_after_mapping,
        "overlap_count": overlap_count,
    }


def build_assistant_json(spans: list[dict[str, Any]]) -> str:
    return json.dumps({"spans": spans}, ensure_ascii=False, separators=(",", ":"))


def build_row(example_id: str, text: str, spans: list[dict[str, Any]], system_prompt: str) -> dict[str, Any]:
    used_labels = sorted({span["type"] for span in spans})
    return {
        "id": example_id,
        "text": text,
        "spans": spans,
        "used_labels": used_labels,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
            {"role": "assistant", "content": build_assistant_json(spans)},
        ],
    }


def _split_records(records: list[dict[str, Any]], seed: int, train_ratio: float, dev_ratio: float, test_ratio: float) -> dict[str, list[dict[str, Any]]]:
    total = train_ratio + dev_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)
    return {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:dev_end],
        "test": shuffled[dev_end:],
    }


def _choose_train_negatives(record: dict[str, Any], rng: random.Random, train_negatives_per_record: int) -> list[dict[str, Any]]:
    negatives = list(record.get("hard_negatives", []))
    if train_negatives_per_record < 0:
        return negatives
    if len(negatives) <= train_negatives_per_record:
        return negatives
    return rng.sample(negatives, train_negatives_per_record)


def build_dataset_rows(
    raw_path: Path,
    taxonomy_path: Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    train_negatives_per_record: int = 1,
    keep_all_eval_negatives: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    records = raw.get("records", [])
    if not isinstance(records, list):
        raise ValueError("raw input must contain a records list")

    mapping = load_taxonomy_mapping(taxonomy_path)
    system_prompt = build_system_prompt(mapping.class_labels)
    splits = _split_records(records, seed, train_ratio, dev_ratio, test_ratio)
    rng = random.Random(seed)

    output: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    audit_rows: list[dict[str, Any]] = []
    total_dropped = 0
    total_deduped = 0
    total_overlaps = 0

    for split_name, split_records in splits.items():
        for record in split_records:
            rec_id = str(record.get("id", "unknown"))
            positive = record.get("positive_sample", {})
            text = str(positive.get("text", ""))
            labels = list(positive.get("labels", []))
            spans, audit = map_source_labels(text, labels, mapping)
            total_dropped += len(audit["dropped"])
            total_deduped += int(audit["deduped_after_mapping"])
            total_overlaps += int(audit["overlap_count"])
            audit_rows.append({"id": f"{rec_id}::pos", **audit})
            output[split_name].append(build_row(f"{rec_id}::pos", text, spans, system_prompt))
            for label in labels:
                source_counts[split_name][str(label.get("type"))] += 1
            for span in spans:
                class_counts[split_name][span["type"]] += 1

            if split_name == "train" or keep_all_eval_negatives:
                negatives = (
                    _choose_train_negatives(record, rng, train_negatives_per_record)
                    if split_name == "train"
                    else list(record.get("hard_negatives", []))
                )
                for idx, negative in enumerate(negatives):
                    negative_text = str(negative.get("text", ""))
                    output[split_name].append(build_row(f"{rec_id}::neg::{idx}", negative_text, [], system_prompt))

    meta = {
        "schema": "json-spans",
        "span_format": {"spans": [{"start": "int", "end": "int", "type": "str", "value": "str"}]},
        "source_version": raw.get("version"),
        "raw_pii_type_count": len(raw.get("pii_types", [])),
        "class_count": len(mapping.class_labels),
        "target_labels": mapping.class_labels,
        "source_to_class": mapping.source_to_class,
        "seed": seed,
        "train_ratio": train_ratio,
        "dev_ratio": dev_ratio,
        "test_ratio": test_ratio,
        "train_negatives_per_record": train_negatives_per_record,
        "keep_all_eval_negatives": keep_all_eval_negatives,
        "record_count": len(records),
        "split_sizes": {name: len(rows) for name, rows in output.items()},
        "positive_record_counts": {name: len(splits[name]) for name in output},
        "class_counts": {name: dict(counter) for name, counter in class_counts.items()},
        "source_type_counts": {name: dict(counter) for name, counter in source_counts.items()},
        "deduped_after_mapping_count": total_deduped,
        "overlap_count": total_overlaps,
        "dropped_span_count": total_dropped,
    }
    audit = {
        "dropped_span_count": total_dropped,
        "deduped_after_mapping_count": total_deduped,
        "overlap_count": total_overlaps,
        "rows": audit_rows,
    }
    return output, meta, audit


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_outputs(rows: dict[str, list[dict[str, Any]]], meta: dict[str, Any], audit: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "qwen_sft_train.jsonl", rows["train"])
    write_jsonl(out_dir / "qwen_sft_dev.jsonl", rows["dev"])
    write_jsonl(out_dir / "qwen_sft_test.jsonl", rows["test"])
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(out_dir / "audit.jsonl", audit["rows"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("../data/raw/au_pii_19000_final.json"))
    parser.add_argument("--taxonomy", type=Path, default=Path("../configs/taxonomy_v1.1.1.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("../data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--train-negatives-per-record", type=int, default=1)
    parser.add_argument("--eval-negatives", choices=["all", "none"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, meta, audit = build_dataset_rows(
        raw_path=args.raw,
        taxonomy_path=args.taxonomy,
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        train_negatives_per_record=args.train_negatives_per_record,
        keep_all_eval_negatives=args.eval_negatives == "all",
    )
    write_outputs(rows, meta, audit, args.out_dir)
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "split_sizes": meta["split_sizes"],
        "class_count": meta["class_count"],
        "raw_pii_type_count": meta["raw_pii_type_count"],
        "deduped_after_mapping_count": meta["deduped_after_mapping_count"],
        "overlap_count": meta["overlap_count"],
        "dropped_span_count": meta["dropped_span_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

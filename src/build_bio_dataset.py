import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def load_jsonl(path: str):
    records = []
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return records
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    if isinstance(text, dict):
        return text
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            value = json.loads(snippet)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    return {}


def flatten_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten_values(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(flatten_values(item))
        return out
    text = str(value).strip()
    return [text] if text else []


def all_matches(text: str, value: str):
    matches = []
    start = 0
    while True:
        idx = text.find(value, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(value)))
        start = idx + 1
    if matches:
        return matches

    lowered_text = text.lower()
    lowered_value = value.lower()
    start = 0
    while True:
        idx = lowered_text.find(lowered_value, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(value)))
        start = idx + 1
    return matches


def build_spans(text: str, entities: dict[str, Any]):
    candidates = []
    for entity_type, raw_value in entities.items():
        for value in flatten_values(raw_value):
            for start, end in all_matches(text, value):
                candidates.append((start, end, entity_type, value))

    candidates.sort(key=lambda item: (item[1] - item[0]), reverse=True)
    occupied = [None] * len(text)
    spans = []

    for start, end, entity_type, value in candidates:
        if start < 0 or end > len(text) or start >= end:
            continue
        if any(occupied[idx] is not None for idx in range(start, end)):
            continue
        for idx in range(start, end):
            occupied[idx] = entity_type
        spans.append({"start": start, "end": end, "label": entity_type, "text": value})

    spans.sort(key=lambda item: item["start"])
    return spans, occupied


def build_label_space(records: list[dict[str, Any]]):
    entity_types = set()
    for record in records:
        entities = parse_json_object(record.get("target_text", ""))
        entity_types.update(str(key) for key in entities.keys())
    ordered = sorted(entity_types)
    label_names = ["O"]
    for entity in ordered:
        label_names.append(f"B-{entity}")
        label_names.append(f"I-{entity}")
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    return label_names, label_to_id


def encode_record(record, tokenizer, label_to_id, max_length: int):
    text = record.get("input_text", "")
    entities = parse_json_object(record.get("target_text", ""))
    spans, char_labels = build_spans(text, entities)

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
    )

    labels = []
    previous_entity = None
    for (start, end), attention in zip(encoded["offset_mapping"], encoded["attention_mask"]):
        if attention == 0 or end <= start:
            labels.append(-100)
            previous_entity = None
            continue

        token_entities = [char_labels[idx] for idx in range(start, min(end, len(char_labels))) if char_labels[idx] is not None]
        if not token_entities:
            labels.append(label_to_id["O"])
            previous_entity = None
            continue

        entity = token_entities[0]
        is_begin = start == 0 or char_labels[start - 1] != entity or previous_entity != entity
        label_name = f"B-{entity}" if is_begin else f"I-{entity}"
        labels.append(label_to_id.get(label_name, label_to_id["O"]))
        previous_entity = entity

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="把 teacher span 标签转换成 BIO token 数据")
    parser.add_argument("--train_input", required=True, help="train jsonl 路径")
    parser.add_argument("--validation_input", default=None, help="validation jsonl 路径")
    parser.add_argument("--output_dir", default="data/processed_bio", help="输出目录")
    parser.add_argument("--model_name", default="gpt2", help="student tokenizer/model 名称")
    parser.add_argument("--max_length", type=int, default=512, help="最大长度")
    args = parser.parse_args()

    train_records = load_jsonl(args.train_input)
    val_records = load_jsonl(args.validation_input) if args.validation_input else []
    all_records = train_records + val_records

    label_names, label_to_id = build_label_space(all_records)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = [encode_record(row, tokenizer, label_to_id, args.max_length) for row in train_records]
    val_rows = [encode_record(row, tokenizer, label_to_id, args.max_length) for row in val_records]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "val.jsonl", val_rows)

    with (output_dir / "label_map.json").open("w", encoding="utf-8") as f:
        json.dump({"labels": label_names, "label_to_id": label_to_id}, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(train_rows)} BIO train rows to {output_dir / 'train.jsonl'}")
    print(f"Saved {len(val_rows)} BIO validation rows to {output_dir / 'val.jsonl'}")
    print(f"Saved label map with {len(label_names)} labels to {output_dir / 'label_map.json'}")


if __name__ == "__main__":
    main()

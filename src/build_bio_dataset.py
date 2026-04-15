import argparse
import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ENTITY_KEY_ALIASES = {
    "person": "PERSON",
    "name": "PERSON",
    "full_name": "PERSON",
    "date_of_birth": "DATE_OF_BIRTH",
    "dob": "DATE_OF_BIRTH",
    "address": "ADDRESS",
    "email": "EMAIL_ADDRESS",
    "email_address": "EMAIL_ADDRESS",
    "phone": "AU_PHONE",
    "phone_number": "AU_PHONE",
    "tfn": "AU_TFN",
    "tax_file_number": "AU_TFN",
    "passport_number": "AU_PASSPORT",
    "driver_license": "AU_DRIVERS_LICENCE",
    "driver_licence": "AU_DRIVERS_LICENCE",
    "medicare_number": "MEDICARE_NUMBER",
    "medicare_expiry": "MEDICARE_EXPIRY",
    "ihi": "IHI",
    "ihl": "IHI",
    "bank_name": "AU_BANK_NAME",
    "bank": "AU_BANK_NAME",
    "bsb": "BSB",
    "account_number": "AU_BANK_ACCOUNT",
    "card_number": "PAYMENT_CARD_NUMBER",
    "credit_card_number": "PAYMENT_CARD_NUMBER",
    "card_expiry": "CREDIT_CARD_EXPIRY",
    "expiration_date": "CREDIT_CARD_EXPIRY",
    "expiry_date": "CREDIT_CARD_EXPIRY",
    "credit_card_expiry": "CREDIT_CARD_EXPIRY",
    "cvv": "CREDIT_CARD_CVV",
    "credit_card_cvv": "CREDIT_CARD_CVV",
    "gross_salary": "SALARY",
    "current_salary": "SALARY",
    "salary": "SALARY",
    "salary_expectation": "SALARY_WAGE_EXPECTATION",
    "position": "EMPLOYMENT_INFORMATION",
    "role": "EMPLOYMENT_INFORMATION",
    "organization": "EMPLOYMENT_INFORMATION",
    "contract_type": "CONTRACT_TYPE",
    "employee_id": "EMPLOYEE_NUMBER",
    "emp_id": "EMPLOYEE_NUMBER",
    "personnel_id": "PERSONNEL_NUMBER",
    "id": "STUDENT_ID",
    "id_number": "STUDENT_ID",
    "username": "USERNAME",
    "ip_address": "IP_ADDRESS",
    "imei": "DEVICE_ID",
    "device_id": "DEVICE_ID",
    "mac_address": "DEVICE_ID",
    "auth_token": "COOKIE_INFORMATION",
    "session_id": "COOKIE_INFORMATION",
    "location": "GEOLOCATION_INFORMATION",
    "geolocation": "GEOLOCATION_INFORMATION",
    "latitude": "LATITUDE",
    "longitude": "LONGITUDE",
    "social_handle": "SOCIAL_MEDIA_ACCOUNT",
    "social_media_account": "SOCIAL_MEDIA_ACCOUNT",
    "social_id": "SOCIAL_MEDIA_ID",
    "url": "WEBSITE_HISTORY",
    "website": "WEBSITE_HISTORY",
    "gender": "GENDER",
    "pronouns": "PRONOUN",
    "sexual_orientation": "SEXUAL_ORIENTATION",
    "religion": "RELIGION_BELIEF",
    "ethnicity": "RACIAL_ETHNIC_ORIGIN",
    "indigenous_status": "ABORIGINALITY",
    "ses_background": "SOCIO_ECONOMIC_STATUS",
    "marital_status": "MARITAL_STATUS",
    "military_status": "MILITARY_VETERAN_STATUS",
    "caring_responsibilities": "CARING_RESPONSIBILITIES",
    "debt": "PERSONAL_DEBT",
    "registration": "VEHICLE_REGO",
}


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


def normalize_entity_type(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return ENTITY_KEY_ALIASES.get(cleaned, cleaned.upper())


def load_reference_confidences(path: str | None):
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        return {}

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
        data = data["records"]
    if not isinstance(data, list):
        return {}

    lookup = {}
    for record in data:
        text = record.get("input", {}).get("text") or record.get("input_text", "")
        labels = record.get("positive_sample", {}).get("labels", [])
        normalized = []
        for item in labels:
            if not isinstance(item, dict):
                continue
            normalized.append({
                "start": int(item.get("start", -1)),
                "end": int(item.get("end", -1)),
                "label": normalize_entity_type(item.get("type", "")),
                "text": str(item.get("value", "")).strip(),
                "confidence": float(item.get("confidence", 0.9)),
            })
        if text:
            lookup[text] = normalized
    return lookup


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


def match_confidence(span: dict[str, Any], reference_spans: list[dict[str, Any]], default_confidence: float):
    best_conf = default_confidence
    best_score = 0.0
    for ref in reference_spans:
        if ref.get("label") != span.get("label"):
            continue
        if (ref.get("start"), ref.get("end")) == (span.get("start"), span.get("end")):
            return float(ref.get("confidence", default_confidence))
        if ref.get("text") and span.get("text") and ref["text"].lower() == span["text"].lower():
            return float(ref.get("confidence", default_confidence))
        overlap = min(span["end"], ref["end"]) - max(span["start"], ref["start"])
        if overlap <= 0:
            continue
        denom = max(span["end"] - span["start"], ref["end"] - ref["start"], 1)
        score = overlap / denom
        if score > best_score:
            best_score = score
            best_conf = float(ref.get("confidence", default_confidence))
    return best_conf


def encode_record(record, tokenizer, label_to_id, max_length: int, reference_lookup=None, default_confidence: float = 0.9):
    text = record.get("input_text", "")
    entities = parse_json_object(record.get("target_text", ""))
    spans, char_labels = build_spans(text, entities)
    reference_spans = reference_lookup.get(text, []) if reference_lookup else []

    char_confidences = [default_confidence] * len(text)
    for span in spans:
        confidence = match_confidence(span, reference_spans, default_confidence)
        for idx in range(span["start"], span["end"]):
            if 0 <= idx < len(char_confidences):
                char_confidences[idx] = confidence

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
    )

    labels = []
    label_confidences = []
    previous_entity = None
    for (start, end), attention in zip(encoded["offset_mapping"], encoded["attention_mask"]):
        if attention == 0 or end <= start:
            labels.append(-100)
            label_confidences.append(0.0)
            previous_entity = None
            continue

        token_entities = [char_labels[idx] for idx in range(start, min(end, len(char_labels))) if char_labels[idx] is not None]
        token_conf_values = [char_confidences[idx] for idx in range(start, min(end, len(char_confidences)))]
        token_confidence = max(token_conf_values) if token_conf_values else default_confidence

        if not token_entities:
            labels.append(label_to_id["O"])
            label_confidences.append(float(token_confidence))
            previous_entity = None
            continue

        entity = token_entities[0]
        is_begin = start == 0 or char_labels[start - 1] != entity or previous_entity != entity
        label_name = f"B-{entity}" if is_begin else f"I-{entity}"
        labels.append(label_to_id.get(label_name, label_to_id["O"]))
        label_confidences.append(float(token_confidence))
        previous_entity = entity

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "label_confidences": label_confidences,
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
    parser.add_argument("--raw_reference_input", default=None, help="可选原始数据文件，用于读取 span-level confidence")
    parser.add_argument("--default_confidence", type=float, default=0.9, help="未命中原始 confidence 时的默认值")
    args = parser.parse_args()

    train_records = load_jsonl(args.train_input)
    val_records = load_jsonl(args.validation_input) if args.validation_input else []
    all_records = train_records + val_records

    label_names, label_to_id = build_label_space(all_records)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    reference_lookup = load_reference_confidences(args.raw_reference_input)

    train_rows = [encode_record(row, tokenizer, label_to_id, args.max_length, reference_lookup, args.default_confidence) for row in train_records]
    val_rows = [encode_record(row, tokenizer, label_to_id, args.max_length, reference_lookup, args.default_confidence) for row in val_records]

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

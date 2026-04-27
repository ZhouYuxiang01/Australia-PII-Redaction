import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


CANONICAL_ENTITY_TYPES = {
    "ABORIGINALITY",
    "ADDRESS",
    "AUDIO_INFORMATION",
    "AU_BANK_ACCOUNT",
    "AU_BANK_NAME",
    "AU_DRIVERS_LICENCE",
    "AU_PASSPORT",
    "AU_PHONE",
    "AU_TFN",
    "BSB",
    "CAMERA_FOOTAGE_AUDIO",
    "CARING_RESPONSIBILITIES",
    "CENTRELINK_REFERENCE_NUMBER",
    "CITIZENSHIP_STATUS",
    "CONTRACT_TYPE",
    "COOKIE_INFORMATION",
    "COUNSELLING_RECORDS",
    "CREDIT_CARD_CVV",
    "CREDIT_CARD_EXPIRY",
    "CRIMINAL_RECORDS",
    "DATE_OF_BIRTH",
    "DEVICE_ID",
    "DISABILITY_OR_SPECIFIC_CONDITION",
    "EMAIL_ADDRESS",
    "EMPLOYEE_NUMBER",
    "EMPLOYMENT_INFORMATION",
    "FACIAL_RECOGNITION",
    "FINGERPRINT",
    "GENDER",
    "GEOLOCATION_INFORMATION",
    "HASHED_PAYMENT_CARD_NUMBER",
    "IHI",
    "IP_ADDRESS",
    "LATITUDE",
    "LONGITUDE",
    "MARITAL_STATUS",
    "MEDICAL_CERTIFICATE",
    "MEDICAL_INFORMATION",
    "MEDICARE_EXPIRY",
    "MEDICARE_NUMBER",
    "MILITARY_VETERAN_STATUS",
    "NATIONALITY",
    "NATIONAL_IDENTITY_CARD",
    "NEXT_OF_KIN",
    "NUMBER_PLATE",
    "PASSPORT_EXPIRY",
    "PASSPORT_START_DATE",
    "PAYMENT_CARD_NUMBER",
    "PENSION_CARD_NUMBER",
    "PERSON",
    "PERSONAL_DEBT",
    "PERSONNEL_NUMBER",
    "PRONOUN",
    "RACIAL_ETHNIC_ORIGIN",
    "RELIGION_BELIEF",
    "SALARY",
    "SALARY_WAGE_EXPECTATION",
    "SANCTIONS",
    "SCHOLARSHIP",
    "SEXUAL_ORIENTATION",
    "SIGNATURE",
    "SOCIAL_MEDIA_ACCOUNT",
    "SOCIAL_MEDIA_HISTORY",
    "SOCIAL_MEDIA_ID",
    "SOCIO_ECONOMIC_STATUS",
    "SPECIAL_CONSIDERATION",
    "STUDENT_ID",
    "SUBJECT_RESULTS",
    "UAC_ID",
    "USERNAME",
    "USI",
    "VEHICLE_REGO",
    "VOICE_RECOGNITION",
    "WAM_SCORE",
    "WEBSITE_HISTORY",
    "WORKERS_COMPENSATION_CLAIM",
    "WORK_EMAIL",
    "WORK_PHONE",
}


ENTITY_KEY_ALIASES = {
    "person": "PERSON",
    "name": "PERSON",
    "full_name": "PERSON",
    "employee": "PERSON",
    "applicant": "PERSON",
    "date_of_birth": "DATE_OF_BIRTH",
    "dob": "DATE_OF_BIRTH",
    "address": "ADDRESS",
    "email": "EMAIL_ADDRESS",
    "email_address": "EMAIL_ADDRESS",
    "phone": "AU_PHONE",
    "phone_number": "AU_PHONE",
    "mobile": "AU_PHONE",
    "tfn": "AU_TFN",
    "tax_file_number": "AU_TFN",
    "passport_number": "AU_PASSPORT",
    "driver_license": "AU_DRIVERS_LICENCE",
    "driver_licence": "AU_DRIVERS_LICENCE",
    "drivers_licence": "AU_DRIVERS_LICENCE",
    "medicare_number": "MEDICARE_NUMBER",
    "medicare_expiry": "MEDICARE_EXPIRY",
    "individual_healthcare_identifier": "IHI",
    "ihi": "IHI",
    "ihl": "IHI",
    "bank_name": "AU_BANK_NAME",
    "bank": "AU_BANK_NAME",
    "bsb": "BSB",
    "account_number": "AU_BANK_ACCOUNT",
    "account": "AU_BANK_ACCOUNT",
    "card_number": "PAYMENT_CARD_NUMBER",
    "credit_card_number": "PAYMENT_CARD_NUMBER",
    "card_expiry": "CREDIT_CARD_EXPIRY",
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
    "auth_token": "COOKIE_INFORMATION",
    "location": "GEOLOCATION_INFORMATION",
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


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_last_json_object(text: str) -> str:
    if not text:
        return ""

    if "```json" in text:
        parts = text.split("```json")
        last_block = parts[-1]
        if "```" in last_block:
            last_block = last_block.split("```", 1)[0]
        return last_block.strip()

    last_candidate = ""
    start_idx = None
    stack = 0
    escaped = False
    in_string = False

    for idx, ch in enumerate(text):
        if ch == "\\" and not escaped:
            escaped = True
            continue
        if ch == '"' and not escaped:
            in_string = not in_string
        if in_string:
            escaped = False
            continue

        if ch == "{" and not in_string:
            if stack == 0:
                start_idx = idx
            stack += 1
        elif ch == "}" and not in_string and stack > 0:
            stack -= 1
            if stack == 0 and start_idx is not None:
                last_candidate = text[start_idx : idx + 1].strip()
                start_idx = None
        escaped = False

    return last_candidate or text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def normalize_key(key: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
    return cleaned


def looks_masked(value: str) -> bool:
    value = str(value)
    return "*" in value or "x" in value.lower()


def add_entity(entities: dict[str, Any], key: str | None, value: Any):
    if not key or value is None:
        return
    if isinstance(value, str):
        value = value.strip()
    if value in ("", [], {}):
        return

    if key not in entities:
        entities[key] = value
        return

    existing = entities[key]
    if existing == value:
        return
    if not isinstance(existing, list):
        existing = [existing]
    if value not in existing:
        existing.append(value)
    entities[key] = existing


def resolve_canonical_key(raw_key: str, value: Any, current_entities: dict[str, Any]) -> str | None:
    normalized = normalize_key(raw_key)

    if normalized in ENTITY_KEY_ALIASES:
        key = ENTITY_KEY_ALIASES[normalized]
        if key == "PAYMENT_CARD_NUMBER" and looks_masked(value):
            return "HASHED_PAYMENT_CARD_NUMBER"
        return key

    upper = normalized.upper()
    if upper in CANONICAL_ENTITY_TYPES:
        return upper

    if normalized in {"date", "expiration_date", "expiry_date"}:
        if "MEDICARE_NUMBER" in current_entities:
            return "MEDICARE_EXPIRY"
        if "PAYMENT_CARD_NUMBER" in current_entities or "HASHED_PAYMENT_CARD_NUMBER" in current_entities:
            return "CREDIT_CARD_EXPIRY"
        if "AU_PASSPORT" in current_entities:
            if "PASSPORT_START_DATE" not in current_entities:
                return "PASSPORT_START_DATE"
            return "PASSPORT_EXPIRY"
        if "DATE_OF_BIRTH" not in current_entities:
            return "DATE_OF_BIRTH"

    if normalized in {"credit_card_number", "cardnumber"}:
        return "HASHED_PAYMENT_CARD_NUMBER" if looks_masked(value) else "PAYMENT_CARD_NUMBER"

    if normalized in {"work_email", "office_email"}:
        return "WORK_EMAIL"
    if normalized in {"work_phone", "office_phone"}:
        return "WORK_PHONE"

    return None


def normalize_entities(payload: dict[str, Any]) -> dict[str, Any]:
    normalized_entities: dict[str, Any] = {}

    def handle_item(raw_key: str, raw_value: Any):
        if normalize_key(raw_key) == "entities":
            if isinstance(raw_value, list):
                for item in raw_value:
                    if not isinstance(item, dict):
                        continue
                    item_key = item.get("type") or item.get("label") or item.get("entity") or item.get("name")
                    item_value = item.get("span") or item.get("text") or item.get("value")
                    canonical = resolve_canonical_key(item_key or "", item_value, normalized_entities)
                    add_entity(normalized_entities, canonical, item_value)
            elif isinstance(raw_value, dict):
                for nested_key, nested_value in raw_value.items():
                    canonical = resolve_canonical_key(nested_key, nested_value, normalized_entities)
                    add_entity(normalized_entities, canonical, nested_value)
            return

        canonical = resolve_canonical_key(raw_key, raw_value, normalized_entities)
        add_entity(normalized_entities, canonical, raw_value)

    for key, value in payload.items():
        handle_item(key, value)

    return normalized_entities


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_teacher_student_dataset(input_path, output_dir, val_ratio, seed):
    raw_records = load_jsonl(input_path)
    random.seed(seed)
    random.shuffle(raw_records)

    cleaned_records = []
    for record in raw_records:
        input_text = record.get("input_text", "")
        teacher_output = record.get("teacher_output", "")
        target_text = extract_last_json_object(teacher_output)
        target_payload = parse_json_object(target_text)
        normalized_payload = normalize_entities(target_payload)
        if not input_text or not normalized_payload:
            continue
        cleaned_records.append({
            "input_text": input_text,
            "target_text": json.dumps(normalized_payload, ensure_ascii=False),
        })

    if len(cleaned_records) <= 1:
        train_records = cleaned_records
        val_records = []
    elif val_ratio <= 0:
        train_records = cleaned_records
        val_records = []
    else:
        split = int(len(cleaned_records) * (1 - val_ratio))
        split = max(1, min(len(cleaned_records) - 1, split))
        train_records = cleaned_records[:split]
        val_records = cleaned_records[split:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    print(f"Saved {len(train_records)} train records to {train_path}")
    print(f"Saved {len(val_records)} validation records to {val_path}")
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="从 teacher labels 生成 student 训练数据")
    parser.add_argument("--input", required=True, help="teacher labels jsonl 文件路径")
    parser.add_argument("--output_dir", default="data/processed_teacher", help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    prepare_teacher_student_dataset(
        args.input,
        args.output_dir,
        args.val_ratio,
        args.seed,
    )


if __name__ == "__main__":
    main()

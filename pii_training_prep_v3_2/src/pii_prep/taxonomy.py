from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CSV_NAME_OVERRIDES = {
    "Aboriginality": "ABORIGINALITY",
    "Audio Information": "AUDIO_INFORMATION",
    "Bank Account Information": "BANK_ACCOUNT_INFORMATION",
    "Bank Account Number": "BANK_ACCOUNT_NUMBER",
    "Camera footage Audio": "CAMERA_FOOTAGE_AUDIO",
    "Caring responsibilities": "CARING_RESPONSIBILITIES",
    "Centrelink Reference Number": "CENTRELINK_REFERENCE_NUMBER",
    "Citizenship Status": "CITIZENSHIP_STATUS",
    "Contract Type - Fixed Term / Temporary / Permanent etc.": "CONTRACT_TYPE",
    "Cookie Information": "COOKIE_INFORMATION",
    "Credit Card Expiry Details": "CREDIT_CARD_EXPIRY",
    "Criminal Records": "CRIMINAL_RECORDS",
    "Date of Birth": "DATE_OF_BIRTH",
    "Device ID": "DEVICE_ID",
    "Disability or Specific Condition": "DISABILITY_OR_SPECIFIC_CONDITION",
    "Driver's Licence Number": "DRIVERS_LICENCE",
    "Email address": "EMAIL_ADDRESS",
    "Employee Number": "EMPLOYEE_NUMBER",
    "Employment Information": "EMPLOYMENT_INFORMATION",
    "Facial Recognition": "FACIAL_RECOGNITION",
    "First Name": "FIRST_NAME",
    "Full Address": "ADDRESS",
    "Full Name": "PERSON",
    "Hashed Payment Card Number": "HASHED_PAYMENT_CARD_NUMBER",
    "Home phone": "HOME_PHONE",
    "IHI - unique health identifier": "IHI",
    "IHI - unique health identifier".replace("-", "\u2013"): "IHI",
    "IP Address": "IP_ADDRESS",
    "Last Name": "LAST_NAME",
    "Latitude of mailing/residential address": "LATITUDE",
    "Longitude of mailing/residential address": "LONGITUDE",
    "Medicare Expiry": "MEDICARE_EXPIRY",
    "Medicare Number": "MEDICARE_NUMBER",
    "Military or Veteran Status": "MILITARY_VETERAN_STATUS",
    "Mobile phone": "MOBILE",
    "National Identity Card Details": "NATIONAL_IDENTITY_CARD",
    "Next of Kin": "NEXT_OF_KIN",
    "Number Plate": "NUMBER_PLATE",
    "Passport Expiry date": "PASSPORT_EXPIRY",
    "Passport Number": "PASSPORT_NUMBER",
    "Passport Start date": "PASSPORT_START_DATE",
    "Payment Card Number": "PAYMENT_CARD_NUMBER",
    "Pension Card / Senior Card Numbers": "PENSION_CARD_NUMBER",
    "Personnel Number / Staff Number": "PERSONNEL_NUMBER",
    "Racial or Ethnic Origin": "RACIAL_ETHNIC_ORIGIN",
    "Religion / Religious Beliefs": "RELIGION_BELIEF",
    "Salary / Wage": "SALARY",
    "Salary / Wage Expectation": "SALARY_WAGE_EXPECTATION",
    "Sexual Orientation": "SEXUAL_ORIENTATION",
    "Social Media Account": "SOCIAL_MEDIA_ACCOUNT",
    "Social Media History": "SOCIAL_MEDIA_HISTORY",
    "Social Media ID": "SOCIAL_MEDIA_ID",
    "Socio Economic Status": "SOCIO_ECONOMIC_STATUS",
    "Student ID": "STUDENT_ID",
    "TFN": "AU_TFN",
    "UAC ID": "UAC_ID",
    "User Name": "USERNAME",
    "USI - unique student identifier": "USI",
    "Vehicle REGO": "VEHICLE_REGO",
    "Voice Recognition": "VOICE_RECOGNITION",
    "Website History": "WEBSITE_HISTORY",
    "Workers Compensation Claims": "WORKERS_COMPENSATION_CLAIM",
    "Work email": "WORK_EMAIL",
    "Work phone": "WORK_PHONE",
    "WAM score": "WAM_SCORE",
}

RAW_TO_TARGET_OVERRIDES = {
    "AU_BANK_ACCOUNT": "BANK_ACCOUNT_NUMBER",
    "AU_DRIVERS_LICENCE": "DRIVERS_LICENCE",
    "AU_PASSPORT": "PASSPORT_NUMBER",
    "AU_PHONE": "MOBILE",
    "BSB": "BANK_ACCOUNT_NUMBER",
    "CREDIT_CARD_CVV": "PAYMENT_CARD_NUMBER",
}


@dataclass(frozen=True)
class TaxonomyEntry:
    code: str
    name: str
    note: str
    data_classification: str
    category_type: str


@dataclass(frozen=True)
class Taxonomy:
    labels: list[str]
    entries: dict[str, TaxonomyEntry]
    raw_to_target: dict[str, str]

    def map_raw_type(self, raw_type: str) -> str:
        return self.raw_to_target.get(raw_type, raw_type)

    def classification_for(self, label: str) -> str:
        entry = self.entries.get(label)
        return entry.data_classification if entry else "Unclassified"


def canonical_code(name: str) -> str:
    cleaned = " ".join(str(name).strip().split())
    if cleaned in CSV_NAME_OVERRIDES:
        return CSV_NAME_OVERRIDES[cleaned]
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", cleaned)
    return re.sub(r"_+", "_", cleaned).strip("_").upper()


def load_taxonomy(csv_path: Path | str, raw_types: Iterable[str] | None = None, expand_from_raw: bool = False) -> Taxonomy:
    path = Path(csv_path)
    entries: dict[str, TaxonomyEntry] = {}
    labels: list[str] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Name", "").strip()
            if not name:
                continue
            code = canonical_code(name)
            if code not in entries:
                labels.append(code)
            entries[code] = TaxonomyEntry(
                code=code,
                name=name,
                note=row.get("Note", "").strip(),
                data_classification=row.get("Data Classification", "").strip(),
                category_type=row.get("Category Type", "").strip(),
            )

    raw_to_target: dict[str, str] = dict(RAW_TO_TARGET_OVERRIDES)
    for raw_type in raw_types or []:
        target = raw_to_target.get(raw_type, raw_type)
        if target not in entries:
            if not expand_from_raw:
                raw_to_target[raw_type] = target
                continue
            entries[target] = TaxonomyEntry(
                code=target,
                name=target,
                note="Added from raw dataset labels; absent from taxonomy CSV.",
                data_classification="Unclassified",
                category_type="Raw dataset compatibility",
            )
            labels.append(target)
        raw_to_target[raw_type] = target

    if "NON_PII" not in entries:
        labels.append("NON_PII")
        entries["NON_PII"] = TaxonomyEntry(
            code="NON_PII",
            name="NON_PII",
            note="Distribution class for non-sensitive spans.",
            data_classification="Public",
            category_type="Training control",
        )

    return Taxonomy(labels=labels, entries=entries, raw_to_target=raw_to_target)

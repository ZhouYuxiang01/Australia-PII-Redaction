from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DATA_CLASSIFICATION_WEIGHTS = {
    "Highly Protected": 1.0,
    "Protected": 0.5,
    "Public": 0.0,
}


def _load_classification_weights(csv_path: str | Path) -> dict[str, str]:
    import csv
    mapping = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Name", "").strip()
            classification = row.get("Data Classification", "").strip()
            if not name or not classification:
                continue
            import re
            code = re.sub(r"[^0-9A-Za-z]+", "_", str(name).split(" - ")[0].split(" / ")[0].strip())
            code = re.sub(r"_+", "_", code).strip("_").upper()
            mapping[code] = classification
    return mapping


def _canonical_label(label: str) -> str:
    overrides = {
        "Aboriginality": "ABORIGINALITY",
        "Audio Information": "AUDIO_INFORMATION",
        "Bank Account Information": "BANK_ACCOUNT_INFORMATION",
        "Bank Account Number": "BANK_ACCOUNT_NUMBER",
        "Camera footage Audio": "CAMERA_FOOTAGE_AUDIO",
        "Caring responsibilities": "CARING_RESPONSIBILITIES",
        "Centrelink Reference Number": "CENTRELINK_REFERENCE_NUMBER",
        "Citizenship Status": "CITIZENSHIP_STATUS",
        "Contract Type": "CONTRACT_TYPE",
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
        "Fingerprint": "FINGERPRINT",
        "First Name": "FIRST_NAME",
        "Full Address": "ADDRESS",
        "Full Name": "PERSON",
        "Gender": "GENDER",
        "Geolocation Information": "GEOLOCATION_INFORMATION",
        "Hashed Payment Card Number": "HASHED_PAYMENT_CARD_NUMBER",
        "Home phone": "HOME_PHONE",
        "IHI": "IHI",
        "IP Address": "IP_ADDRESS",
        "Last Name": "LAST_NAME",
        "Latitude": "LATITUDE",
        "Longitude": "LONGITUDE",
        "Marital Status": "MARITAL_STATUS",
        "Medicare Expiry": "MEDICARE_EXPIRY",
        "Medicare Number": "MEDICARE_NUMBER",
        "Military or Veteran Status": "MILITARY_VETERAN_STATUS",
        "Mobile phone": "MOBILE",
        "National Identity Card Details": "NATIONAL_IDENTITY_CARD",
        "Nationality": "NATIONALITY",
        "Next of Kin": "NEXT_OF_KIN",
        "Number Plate": "NUMBER_PLATE",
        "Passport Expiry date": "PASSPORT_EXPIRY",
        "Passport Number": "PASSPORT_NUMBER",
        "Passport Start date": "PASSPORT_START_DATE",
        "Payment Card Number": "PAYMENT_CARD_NUMBER",
        "Pension Card": "PENSION_CARD_NUMBER",
        "Personnel Number": "PERSONNEL_NUMBER",
        "Pronoun": "PRONOUN",
        "Racial or Ethnic Origin": "RACIAL_ETHNIC_ORIGIN",
        "Religion": "RELIGION_BELIEF",
        "Salary": "SALARY",
        "Salary Expectation": "SALARY_WAGE_EXPECTATION",
        "Sexual Orientation": "SEXUAL_ORIENTATION",
        "Signature": "SIGNATURE",
        "Social Media Account": "SOCIAL_MEDIA_ACCOUNT",
        "Social Media History": "SOCIAL_MEDIA_HISTORY",
        "Social Media ID": "SOCIAL_MEDIA_ID",
        "Socio Economic Status": "SOCIO_ECONOMIC_STATUS",
        "Student ID": "STUDENT_ID",
        "TFN": "AU_TFN",
        "UAC ID": "UAC_ID",
        "User Name": "USERNAME",
        "USI": "USI",
        "Vehicle REGO": "VEHICLE_REGO",
        "Voice Recognition": "VOICE_RECOGNITION",
        "Website History": "WEBSITE_HISTORY",
        "Work email": "WORK_EMAIL",
        "Workers Compensation Claims": "WORKERS_COMPENSATION_CLAIM",
        "Work phone": "WORK_PHONE",
        "Medical Information": "MEDICAL_INFORMATION",
        "Counselling Records": "COUNSELLING_RECORDS",
        "Medical Certificate": "MEDICAL_CERTIFICATE",
        "Special Consideration": "SPECIAL_CONSIDERATION",
        "Scholarship": "SCHOLARSHIP",
        "WAM score": "WAM_SCORE",
        "Subject Results": "SUBJECT_RESULTS",
        "Sanctions": "SANCTIONS",
        "Personal Debt": "PERSONAL_DEBT",
    }
    return overrides.get(label, label.upper().replace(" ", "_"))


def _build_risk_weight_map(csv_path: str | Path) -> dict[str, float]:
    raw = _load_classification_weights(csv_path)
    weight_map: dict[str, float] = {}
    for name, classification in raw.items():
        code = _canonical_label(name)
        weight_map[code] = DATA_CLASSIFICATION_WEIGHTS.get(classification, 0.5)
    for code in ["NON_PII", "VEHICLE_REGO"]:
        if code not in weight_map:
            weight_map[code] = 0.0
    return weight_map


@dataclass
class PolicyDecision:
    label: str
    risk_score: float
    decision: str
    threshold_used: dict[str, float] = field(default_factory=lambda: {"redact": 0.60, "review": 0.25})
    label_details: dict[str, Any] = field(default_factory=dict)


class PolicyLayer:
    def __init__(
        self,
        risk_weight_map: dict[str, float] | None = None,
        csv_path: str | Path | None = None,
        redact_threshold: float = 0.60,
        review_threshold: float = 0.25,
    ):
        if risk_weight_map is not None:
            self.risk_weights = dict(risk_weight_map)
        elif csv_path is not None:
            self.risk_weights = _build_risk_weight_map(csv_path)
        else:
            self.risk_weights = {}
        self.redact_threshold = redact_threshold
        self.review_threshold = review_threshold

    def decide(
        self,
        type_distribution: dict[str, float],
    ) -> PolicyDecision:
        risk_score = 0.0
        label_contributions = {}
        for label, prob in type_distribution.items():
            if label == "NON_PII":
                continue
            weight = self.risk_weights.get(label, 0.5)
            contribution = prob * weight
            risk_score += contribution
            label_contributions[label] = {
                "probability": prob,
                "risk_weight": weight,
                "contribution": contribution,
            }

        if risk_score >= self.redact_threshold:
            decision = "redact"
        elif risk_score >= self.review_threshold:
            decision = "review"
        else:
            decision = "ignore"

        return PolicyDecision(
            label="composite",
            risk_score=round(risk_score, 6),
            decision=decision,
            threshold_used={
                "redact": self.redact_threshold,
                "review": self.review_threshold,
            },
            label_details=label_contributions,
        )


def run_integrated_pipeline(
    text: str,
    qwen_classifier: Any,
    opf_detector: Any,
    policy: PolicyLayer | None = None,
    *,
    csv_taxonomy_path: str | None = None,
) -> dict[str, Any]:
    if policy is None and csv_taxonomy_path is not None:
        policy = PolicyLayer(csv_path=csv_taxonomy_path)
    elif policy is None:
        policy = PolicyLayer()

    opf_result = opf_detector.detect_spans(text)

    if opf_result.get("error"):
        return {
            "text": text,
            "stage": "opf_error",
            "error": opf_result["error"],
            "spans": [],
            "summary": {"span_count": 0, "redact_count": 0, "review_count": 0, "ignore_count": 0},
        }

    opf_spans = opf_result["candidate_spans"]
    if not opf_spans:
        return {
            "text": text,
            "stage": "no_spans_detected",
            "spans": [],
            "summary": {"span_count": 0, "redact_count": 0, "review_count": 0, "ignore_count": 0},
            "opf_summary": opf_result.get("summary", {}),
        }

    candidate_spans = []
    for os in opf_spans:
        candidate_spans.append({
            "start": os["start"],
            "end": os["end"],
            "value": os["value"],
        })

    if qwen_classifier is not None:
        qwen_result = qwen_classifier.classify_spans(
            text, candidate_spans, output_full_distribution=False, top_k=5, include_non_pii=False
        )
    else:
        qwen_result = {"spans": []}
        for os in opf_spans:
            opt = os.get("opf_top_type", "NON_PII")
            dist = {opt: 1.0}
            qwen_result["spans"].append({
                "start": os["start"],
                "end": os["end"],
                "value": os["value"],
                "type_distribution": dist,
                "top_type": opt,
                "top_probability": 1.0,
                "top_pii_type": opt,
                "top_pii_probability": 1.0,
            })

    results = []
    redact_count = 0
    review_count = 0
    ignore_count = 0

    for i, span_result in enumerate(qwen_result.get("spans", [])):
        if i < len(opf_spans):
            span_result["opf_top_type"] = opf_spans[i].get("opf_top_type", "")
        decision = policy.decide(span_result.get("type_distribution", {}))
        span_result["risk_score"] = decision.risk_score
        span_result["decision"] = decision.decision
        span_result["policy_details"] = decision.label_details

        if decision.decision == "redact":
            redact_count += 1
        elif decision.decision == "review":
            review_count += 1
        else:
            ignore_count += 1

        results.append(span_result)

    return {
        "text": text,
        "stage": "complete",
        "spans": results,
        "summary": {
            "span_count": len(results),
            "redact_count": redact_count,
            "review_count": review_count,
            "ignore_count": ignore_count,
        },
        "opf_summary": opf_result.get("summary", {}),
    }


def redact_text(text: str, spans: list[dict[str, Any]]) -> str:
    if not spans:
        return text

    to_redact = []
    for s in spans:
        if s.get("decision") == "redact":
            to_redact.append((int(s["start"]), int(s["end"])))

    to_redact.sort(key=lambda x: (x[0], -x[1]))

    merged = []
    for start, end in to_redact:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    merged.sort(key=lambda x: x[0])

    parts = []
    pos = 0
    for start, end in merged:
        if start > pos:
            parts.append(text[pos:start])
        parts.append(f"<REDACTED>")
        pos = end
    if pos < len(text):
        parts.append(text[pos:])

    return "".join(parts)


def build_redaction_output(
    text: str,
    pipeline_result: dict[str, Any],
    merged_spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    spans = pipeline_result.get("spans", [])

    redacted_text = redact_text(text, spans)

    review_spans = [
        {
            "start": s["start"],
            "end": s["end"],
            "value": s["value"],
            "top_type": s.get("top_type"),
            "risk_score": s.get("risk_score"),
        }
        for s in spans
        if s.get("decision") == "review"
    ]

    ignore_spans = [
        {"start": s["start"], "end": s["end"], "value": s["value"]}
        for s in spans
        if s.get("decision") == "ignore"
    ]

    return {
        "original_text": text,
        "redacted_text": redacted_text,
        "span_count": len(spans),
        "redact_count": sum(1 for s in spans if s.get("decision") == "redact"),
        "review_count": len(review_spans),
        "ignore_count": len(ignore_spans),
        "review_spans": review_spans,
        "ignore_spans": ignore_spans,
    }

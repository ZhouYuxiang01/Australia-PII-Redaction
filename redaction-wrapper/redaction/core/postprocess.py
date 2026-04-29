"""Generic span post-processing: prefix stripping, work-contact collapsing,
URL-encoded email rescue, overlap resolution.

These rules are model-agnostic — they apply to any backend's spans.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .span import Span


PREFIX_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "DATE_OF_BIRTH": [
        re.compile(r"^(?:d\.?\s*o\.?\s*b\.?|dob|date\s+of\s+birth|born)\s*[:=\-]?\s+", re.IGNORECASE),
    ],
    "AU_TFN": [
        re.compile(r"^(?:t\.?\s*f\.?\s*n\.?|tfn|tax\s+file\s+number)\s*[:=\-]?\s+", re.IGNORECASE),
    ],
    "STUDENT_ID": [
        re.compile(r"^id\s*[:=\-]?\s+", re.IGNORECASE),
    ],
    "CENTRELINK_REFERENCE_NUMBER": [
        re.compile(r"^crn\s+(?=crn[:\s])", re.IGNORECASE),
        re.compile(r"^crn\s+(?!:)", re.IGNORECASE),
        re.compile(r"^centrelink(?:\s+reference(?:\s+number)?)?\s*[:=\-]?\s+", re.IGNORECASE),
    ],
}

ENCODED_EMAIL_RE = re.compile(
    r"([A-Za-z0-9._%+\-]+)%40([A-Za-z0-9.\-]+\.[A-Za-z]{2,})", re.IGNORECASE
)

SAFE_PATTERN_OVERRIDES: dict[str, str] = {
    "BANK_ACCOUNT_GROUPED_6_10": r"(?<!\d)\d(?:[ -]?\d){5,9}(?!\d)",
    "NINE_DIGIT_GROUPED": r"(?<!\d)\d(?:[ -]?\d){8}(?!\d)",
    "PAYMENT_CARD_13_19_GROUPED": r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,7}(?!\d)",
    "STUDENT_ID_8_10_GROUPED": r"(?<!\d)\d(?:[ -]?\d){7,9}(?!\d)",
    "VEHICLE_PLATE_AU": (
        r"\b(?:(?:NSW|VIC|QLD|SA|WA|TAS|ACT|NT)[ -])?"
        r"(?:[A-Z]{3}\d{2}[A-Z]|[A-Z]{1,3}-\d{1,4}[A-Z]?|[A-Z]{1,3}-\d{1,4}(?:-[A-Z]{2,4})?)\b"
    ),
    "VEHICLE_PLATE_CONTEXTUAL_TOKEN": (
        r"\b(?:(?:NSW|VIC|QLD|SA|WA|TAS|ACT|NT)[ -]?)?"
        r"(?=[A-Z0-9]{5,9}\b)(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{5,9}\b"
    ),
}

COMPETING_CONTEXT_TRIGGERS: dict[str, tuple[str, ...]] = {
    "AU_BANK_ACCOUNT": ("acct", "account", "bank account"),
    "AU_DRIVERS_LICENCE": ("driver licence", "drivers licence", "driver license", "licence no", "license number"),
    "AU_TFN": ("tfn", "tax file number"),
    "CREDIT_CARD_EXPIRY": ("card expiry", "card exp", "cc exp", "exp"),
    "MEDICARE_EXPIRY": ("medicare expiry", "medicare card expiry", "card expiry"),
    "PAYMENT_CARD_NUMBER": ("card", "card number", "payment card", "card used"),
    "PHONE": ("mobile", "phone", "tel"),
    "STUDENT_ID": ("sid", "student id", "student number"),
    "UAC_ID": ("uac", "uac id", "uac no", "uac number"),
    "VEHICLE_ID": ("rego", "vehicle registration", "number plate", "license plate", "licence plate"),
}

VEHICLE_CONTEXT_TRIGGERS = ("rego", "vehicle registration", "number plate", "license plate", "licence plate", "car rego")


@dataclass(frozen=True)
class RegistryRule:
    label: str
    patterns: tuple[re.Pattern[str], ...]
    positive_triggers: tuple[str, ...]
    negative_triggers: tuple[str, ...]
    requires_positive_trigger: bool
    left_window: int
    right_window: int
    priority: int


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _find_pii_type_enum(node: Any) -> set[str]:
    if isinstance(node, dict):
        if node.get("title") == "pii_type" and isinstance(node.get("enum"), list):
            return set(node["enum"])
        if isinstance(node.get("pii_type"), dict) and isinstance(node["pii_type"].get("enum"), list):
            return set(node["pii_type"]["enum"])
        out: set[str] = set()
        for value in node.values():
            out.update(_find_pii_type_enum(value))
        return out
    if isinstance(node, list):
        out: set[str] = set()
        for value in node:
            out.update(_find_pii_type_enum(value))
        return out
    return set()


@lru_cache(maxsize=4)
def _schema_labels(repo_root: str) -> frozenset[str]:
    schema_path = Path(repo_root) / "schemas" / "redaction-output-v1.schema.json"
    if not schema_path.exists():
        return frozenset()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return frozenset(_find_pii_type_enum(schema))


@lru_cache(maxsize=4)
def load_postprocess_rule_registry(repo_root: str | Path | None = None) -> tuple[RegistryRule, ...]:
    """Load context-anchored fallback rules from configs/postprocess.

    The registry can mention surface forms that are not model taxonomy labels
    (for example synthetic-data-only aliases). Those are skipped unless they
    normalize to an existing schema pii_type.
    """
    root = Path(repo_root) if repo_root is not None else _repo_root_from_here()
    config_dir = root / "configs" / "postprocess"
    registry_path = config_dir / "postprocess_rule_registry.json"
    surface_forms_path = config_dir / "taxonomy_surface_forms.csv"
    if not registry_path.exists() or not surface_forms_path.exists():
        return ()

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    alias_map = registry.get("label_alias_normalization", {})
    allowed_labels = set(_schema_labels(str(root)))
    if not allowed_labels:
        return ()

    safe_labels: set[str] = set()
    with surface_forms_path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            label = alias_map.get(row.get("canonical_label", ""), row.get("canonical_label", ""))
            rule_safety = (row.get("rule_safe") or "").strip().lower()
            if rule_safety in {"yes", "partial"} and label in allowed_labels:
                safe_labels.add(label)

    pattern_defs = registry.get("pattern_definitions", {})
    rules: list[RegistryRule] = []
    for raw_label, rule_cfg in registry.get("rules", {}).items():
        label = alias_map.get(raw_label, raw_label)
        if label not in safe_labels or not rule_cfg.get("enabled", False):
            continue
        patterns: list[re.Pattern[str]] = []
        for pattern_name in rule_cfg.get("patterns", []):
            pattern_src = SAFE_PATTERN_OVERRIDES.get(pattern_name, pattern_defs.get(pattern_name))
            if not pattern_src:
                continue
            try:
                patterns.append(re.compile(pattern_src, re.IGNORECASE))
            except re.error:
                continue
        if not patterns:
            continue
        rules.append(RegistryRule(
            label=label,
            patterns=tuple(patterns),
            positive_triggers=tuple(t.lower() for t in rule_cfg.get("positive_triggers", [])),
            negative_triggers=tuple(t.lower() for t in rule_cfg.get("negative_triggers", [])),
            requires_positive_trigger=bool(rule_cfg.get("require_positive_trigger", True)),
            left_window=int(rule_cfg.get("window_left_chars", 50)),
            right_window=int(rule_cfg.get("window_right_chars", 30)),
            priority=int(rule_cfg.get("priority", 50)),
        ))
    return tuple(sorted(rules, key=lambda r: r.priority, reverse=True))

CONTEXTUAL_RESCUE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "STUDENT_ID",
        re.compile(
            r"\b(?:student\s+(?:id|number)|SID)\s*[:#=\-]?\s*"
            r"(?P<value>\d{8,10}|\d{4}\s\d{5})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "UAC_ID",
        re.compile(r"\bUAC(?:\s+ID)?(?:\s*[:#=]\s*|\s+)(?P<value>\d{8,10})\b", re.IGNORECASE),
    ),
    (
        "UAC_ID",
        re.compile(
            r"\bUAC(?:\s+(?:ID|no\.?|number))?(?:\s*[:#=]\s*|\s+)"
            r"(?P<value>\d{3}\s\d{3}\s\d{3})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "DATE_OF_BIRTH",
        re.compile(
            r"\b(?:bday|birthday|d\.?\s*o\.?\s*b\.?|dob(?:\s+on\s+file)?|date\s+of\s+birth|born)"
            r"\s*[:#=\-]?\s*(?:is\s+)?(?P<value>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "AU_TFN",
        re.compile(
            r"\b(?:T\.?\s*F\.?\s*N\.?|TFN|tax\s+file\s+number)\s*[:#=\-]?\s*"
            r"(?P<value>\d{3}\s?\d{3}\s?\d{3})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PASSPORT_EXPIRY",
        re.compile(
            r"\bpassport\b[^.\n]{0,80}?\b(?:expires?|expiry(?:\s+date)?)\s*[:#=\-]?\s*"
            r"(?P<value>\d{1,2}(?:[/-]\d{1,2})?[/-]\d{2,4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PASSPORT_EXPIRY",
        re.compile(
            r"\bpassport\s+expiry\s+date\s*[:#=\-]?\s*"
            r"(?P<value>\d{1,2}(?:[/-]\d{1,2})?[/-]\d{2,4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PASSPORT_START_DATE",
        re.compile(
            r"\bpassport\s+start\s+date\s*[:#=\-]?\s*"
            r"(?P<value>\d{1,2}(?:[/-]\d{1,2})?[/-]\d{2,4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MEDICARE_NUMBER",
        re.compile(r"\bmedicare\s*[:#=\-]?\s*(?P<value>\d{4}\s\d{5}\s\d)\b", re.IGNORECASE),
    ),
    (
        "MEDICARE_EXPIRY",
        re.compile(
            r"\bmedicare\s+expiry\s*[:#=\-]?\s*"
            r"(?P<value>\d{1,2}[/-]\d{2,4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MEDICARE_EXPIRY",
        re.compile(
            r"\bmedicare\b[^\n]{0,120}?\bcard\s+expiry\s*[:#=\-]?\s*"
            r"(?P<value>\d{1,2}[/-]\d{2,4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "AU_PASSPORT",
        re.compile(r"\bpassport\s+number\s*[:#=\-]?\s*(?P<value>[A-Z]\d{7})\b", re.IGNORECASE),
    ),
    (
        "NATIONAL_IDENTITY_CARD",
        re.compile(r"\bnational\s+id\s*[:#=\-]?\s*(?P<value>NID-[A-Z]{2}-\d{6}-[A-Z])\b", re.IGNORECASE),
    ),
    (
        "CENTRELINK_REFERENCE_NUMBER",
        re.compile(
            r"\bcentrelink\s+reference\s+number\s*[:#=\-]?\s*"
            r"(?P<value>\d{3}\s\d{3}\s\d{3}[A-Z])\b",
            re.IGNORECASE,
        ),
    ),
    (
        "SCHOLARSHIP",
        re.compile(r"\bscholarship\s+ref\s*[:#=\-]?\s*(?P<value>SCH-\d{4}-\d{4}[A-Z])\b", re.IGNORECASE),
    ),
    (
        "AU_BANK_ACCOUNT",
        re.compile(r"\bBSB\s*[:#=\-]?\s*(?P<value>\d{3}-\d{3})\b", re.IGNORECASE),
    ),
    (
        "AU_BANK_ACCOUNT",
        re.compile(r"\bAccount\s+Number\s*[:#=\-]?\s*(?P<value>\d{6,10})\b", re.IGNORECASE),
    ),
    (
        "AU_BANK_ACCOUNT",
        re.compile(
            r"\b(?:acct|account(?:\s+(?:number|no\.?))?)\s*[:#=\-]?\s*"
            r"(?P<value>\d{2,4}(?:\s\d{2,4}){1,3})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "EMPLOYEE_NUMBER",
        re.compile(r"\b(?:Staff|Employee)\s+ID\s*[:#=\-]?\s*(?P<value>E\d{6})\b", re.IGNORECASE),
    ),
    (
        "PERSONNEL_NUMBER",
        re.compile(r"\bPersonnel(?:\s+(?:Number|no))?\s*[:#=\-]?\s*(?P<value>P\d{8})\b", re.IGNORECASE),
    ),
    (
        "VEHICLE_ID",
        re.compile(r"\b(?:Vehicle\s+REGO|Number\s+Plate)\s*[:#=\-]?\s*(?P<value>[A-Z]{3}\d{2}[A-Z])\b", re.IGNORECASE),
    ),
    (
        "VEHICLE_ID",
        re.compile(
            r"\b(?:Vehicle\s+Registration|Registration\s+Plate|Licen[cs]e\s+Plate|Number\s+Plate|Vehicle\s+REGO)"
            r"(?:\s*\([^)]*\))?\s*[:#=\-]?\s*(?P<value>[A-Z]{2,3}-\d{2,4}-[A-Z]{2,3}|[A-Z]{1,3}-\d{1,4}|[A-Z]{3}\d{2}[A-Z])\b",
            re.IGNORECASE,
        ),
    ),
    (
        "IHI",
        re.compile(r"\bIHI\s*[:#=\-]?\s*(?P<value>\d{4}\s\d{4}\s\d{4}\s\d{4})\b", re.IGNORECASE),
    ),
    (
        "DEVICE_ID",
        re.compile(
            r"\bDevice\s+ID\s*[:#=\-]?\s*(?P<value>[A-Z]{3}-[A-F0-9]{4}-[A-F0-9]{4}-[A-F0-9]{4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "WEBSITE_HISTORY",
        re.compile(r"\bWebsite\s+history\s+flag\s*[:#=\-]?\s*(?P<value>[A-Za-z0-9.-]+/[^\s]+)", re.IGNORECASE),
    ),
    (
        "EMAIL",
        re.compile(
            r"\b(?!(?:Placeholder)\s+email\b)(?:Personal\s+email|Work\s+email|Email(?:\s+in\s+profile)?|email\s+for\s+receipt)"
            r"\s*[:#=\-]?\s*(?P<value>[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PHONE",
        re.compile(
            r"\b(?:mobile|phone|home\s+phone|work\s+phone)\s*[:#=\-]?\s*"
            r"(?P<value>(?:\+61\s?)?\d{3,4}\s\d{3}\s\d{3}|04\s\d{2}\s\d{3}\s\d{3}|\(\d{2}\)\s\d{4}\s\d{4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PAYMENT_CARD_NUMBER",
        re.compile(
            r"\b(?:card\s+used(?:\s+last\s+time)?|card\s+number|payment\s+card)\s*[:#=\-]?\s*"
            r"(?P<value>\d{4}\s\d{4}\s\d{4}\s\d{4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "CREDIT_CARD_EXPIRY",
        re.compile(
            r"\b(?:card\s+used(?:\s+last\s+time)?|card\s+number|payment\s+card)\b[^.\n]{0,80}?\b"
            r"(?:exp|expiry|expires?)\s*[:#=\-]?\s*(?P<value>\d{1,2}/\d{2,4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "PERSON",
        re.compile(
            r"(?m)^\s*\d+\.\s+(?P<value>[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3})\s*"
            r"(?=\n\s*(?:SID|Student\s+(?:ID|number)|Email|Mobile|Phone)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "PERSON",
        re.compile(
            r"\b(?:Full\s+name|Name\s+on\s+passport|Student)\s*[:#=]\s*"
            r"(?P<value>[A-Z][A-Za-z'\-]+(?:[ \t]+[A-Z][A-Za-z'\-]+){1,3})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MEDICAL_CERTIFICATE",
        re.compile(
            r"\bMedical\s+certificate\b[^.\n]{0,80}?\bfrom\s+(?P<value>\d{1,2}/\d{1,2}/\d{4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MEDICAL_CERTIFICATE",
        re.compile(
            r"\bMedical\s+certificate\b[^.\n]{0,80}?\bto\s+(?P<value>\d{1,2}/\d{1,2}/\d{4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MEDICAL_INFORMATION",
        re.compile(r"\b(?:Medical detail|Medical reason)\s*[:#=\-]?\s*(?P<value>[^\n\r]+)", re.IGNORECASE),
    ),
    (
        "MEDICAL_INFORMATION",
        re.compile(
            r"\bReason\s*[:#=\-]?\s*(?P<value>[^\n\r]*"
            r"(?:migraine|anxiety|depression|injury|illness|condition|flare-up|symptoms|disease|syndrome|ailment)"
            r"[^\n\r]*)",
            re.IGNORECASE,
        ),
    ),
    (
        "MEDICAL_CERTIFICATE",
        re.compile(r"\bmedical certificate\.(?P<value>medical certificate)\b", re.IGNORECASE),
    ),
    (
        "SALARY",
        re.compile(r"\b(?:Salary|Current wage)\s*[:#=\-]?\s*(?P<value>\$\d{1,3}(?:,\d{3})*)\b", re.IGNORECASE),
    ),
    (
        "WORKERS_COMPENSATION_CLAIM",
        re.compile(r"\bClaim\s+number\s*[:#=\-]?\s*(?P<value>WC-\d{4}-\d{5})\b", re.IGNORECASE),
    ),
    (
        "SANCTIONS",
        re.compile(r"\bAcademic\s+sanction\s*[:#=\-]?\s*(?P<value>SAN-\d{4}-\d{3})\b", re.IGNORECASE),
    ),
    (
        "NEXT_OF_KIN",
        re.compile(r"\bnext\s+of\s+kin\s+is\s+(?P<value>[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3})\b", re.IGNORECASE),
    ),
    (
        "SOCIAL_MEDIA_ID",
        re.compile(r"\b(?:insta|instagram|social)\s*[:#=\-]?\s*(?P<value>@[A-Za-z0-9_]{3,40})\b", re.IGNORECASE),
    ),
    (
        "PRONOUN",
        re.compile(r"\bpronouns?\s*[:#=\-]?\s*(?P<value>[a-z]+/[a-z]+)\b", re.IGNORECASE),
    ),
    (
        "EMPLOYMENT_INFORMATION",
        re.compile(
            r"\bRole\s*[:#=\-]?\s*(?P<value>(?:lab demonstrator|"
            r"[A-Za-z][A-Za-z /-]*(?:manager|officer|analyst|engineer|developer|lecturer|tutor|coordinator|assistant))"
            r"[^\n\r]*)",
            re.IGNORECASE,
        ),
    ),
    (
        "CONTRACT_TYPE",
        re.compile(r"\bContract\s+type\s*[:#=\-]?\s*(?P<value>[^\n\r]+)", re.IGNORECASE),
    ),
    (
        "RELIGION_BELIEF",
        re.compile(r"\bReligion\s*/\s*Religious\s+Beliefs\s*[:#=\-]?\s*(?P<value>[^\n\r]+)", re.IGNORECASE),
    ),
    (
        "SOCIO_ECONOMIC_STATUS",
        re.compile(r"\bSocio\s+Economic\s+Status\s*[:#=\-]?\s*(?P<value>[^\n\r]+)", re.IGNORECASE),
    ),
]

def _context_before(text: str, start: int, chars: int = 48) -> str:
    return text[max(0, start - chars) : start].lower()


def _context_window(text: str, start: int, end: int, chars: int = 72) -> str:
    return text[max(0, start - chars) : min(len(text), end + chars)].lower()


def _with_type(span: Span, new_type: str, note: str) -> Span:
    if span.type == new_type:
        return span
    out = Span(**{**span.__dict__})
    out.type = new_type
    out.postprocess = [*out.postprocess, note]
    return out


def normalize_contextual_type(span: Span, text: str) -> Span:
    """Normalize labels that are equivalent under the project evaluation taxonomy."""
    before = _context_before(text, span.start, chars=80)
    line_ctx = _line_context(text, span.start, span.end)
    value = span.value
    if span.type == "BSB":
        return _with_type(span, "AU_BANK_ACCOUNT", "taxonomy_alias")
    if span.type == "SALARY_WAGE_EXPECTATION" and any(k in before for k in ["salary", "current wage", "wage:"]):
        return _with_type(span, "SALARY", "taxonomy_alias")
    if span.type == "SOCIAL_MEDIA_ACCOUNT" and value.startswith("@"):
        return _with_type(span, "SOCIAL_MEDIA_ID", "taxonomy_alias")
    if span.type == "PERSON" and "next of kin" in before:
        return _with_type(span, "NEXT_OF_KIN", "taxonomy_alias")
    if span.type == "PASSPORT_START_DATE" and any(k in before for k in ["passport expiry", "expires"]):
        return _with_type(span, "PASSPORT_EXPIRY", "taxonomy_alias")
    if span.type == "STUDENT_ID" and re.fullmatch(r"E\d{6}", value or "") and "staff id" in before:
        return _with_type(span, "EMPLOYEE_NUMBER", "taxonomy_alias")
    if span.type == "DISABILITY_OR_SPECIFIC_CONDITION" and any(k in before for k in ["medical detail", "reason:"]):
        return _with_type(span, "MEDICAL_INFORMATION", "taxonomy_alias")
    if span.type == "AU_DRIVERS_LICENCE" and any(trigger in line_ctx for trigger in VEHICLE_CONTEXT_TRIGGERS):
        out = Span(**{**span.__dict__})
        out.type = "VEHICLE_ID"
        out.confidence = min(out.confidence if out.confidence is not None else 0.8, 0.8)
        out.postprocess = [*out.postprocess, "vehicle_context_label_conflict"]
        return out
    return span


def should_drop_false_positive(span: Span, text: str) -> bool:
    ctx = _context_window(text, span.start, span.end)
    before = _context_before(text, span.start, chars=40)
    if span.type == "AU_DRIVERS_LICENCE":
        return any(marker in ctx for marker in [
            "invoice", "receipt", "claim number", "ticket", "reference", "job ref",
            "system-generated", "system generated", "not staff id", "permit ref",
        ])
    if span.type in {"UAC_ID", "AU_BANK_ACCOUNT", "PENSION_CARD_NUMBER"}:
        return any(marker in ctx for marker in [
            "invoice", "receipt", "ticket", "reference", "test token", "placeholder",
            "system-generated", "system generated", "permit ref", "not staff id",
        ])
    if span.type == "EMAIL":
        return bool(re.search(r"placeholder\s+email\s*[:#=\-]?\s*$", before, re.IGNORECASE))
    if span.type == "PAYMENT_CARD_NUMBER":
        return bool(re.search(r"(?:test\s+token|token)\s*[:#=\-]?\s*(?:tok[_-]?)?$", before, re.IGNORECASE))
    return False


def collapse_work_contact_type(span: Span, text: str) -> Span:
    """If a generic 'email' / 'phone' precedes the value, collapse WORK_* -> generic."""
    ctx = _context_before(text, span.start)
    new_type = span.type
    if span.type == "WORK_EMAIL":
        explicit = any(m in ctx for m in ["work email", "office email", "staff email", "business email"])
        if not explicit and "email" in ctx:
            new_type = "EMAIL"
    elif span.type == "WORK_PHONE":
        explicit = any(m in ctx for m in ["work phone", "office phone", "business phone", "staff phone", "work:"])
        if not explicit and any(m in ctx for m in ["phone", "ph:", "tel"]):
            new_type = "PHONE"
    if new_type == span.type:
        return span
    out = Span(**{**span.__dict__})
    out.type = new_type
    out.postprocess = [*out.postprocess, "collapse_generic_work_contact"]
    return out


def strip_prefix_from_span(span: Span) -> Span | None:
    """Strip type-specific prefixes from a span value (e.g. 'DOB: ' from a date)."""
    patterns = PREFIX_PATTERNS.get(span.type, [])
    cur = span.value
    total_shift = 0
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            m = pattern.match(cur)
            if not m:
                continue
            cur = cur[m.end():].strip()
            total_shift += m.end()
            changed = True
            break
    if not cur:
        return None
    if total_shift == 0:
        return span
    out = Span(**{**span.__dict__})
    out.start += total_shift
    out.value = out.value[total_shift:].strip()
    out.end = out.start + len(out.value)
    out.postprocess = [*out.postprocess, "strip_known_prefix"]
    return out


def add_url_encoded_email_spans(text: str, existing: list[Span]) -> list[Span]:
    """Detect URL-encoded emails (foo%40example.com) that the model commonly misses."""
    spans = list(existing)
    occupied = {(span.start, span.end, span.type) for span in spans}
    for m in ENCODED_EMAIL_RE.finditer(text):
        key = (m.start(), m.end(), "EMAIL")
        if key in occupied:
            continue
        spans.append(Span(
            start=m.start(), end=m.end(), type="EMAIL",
            value=text[m.start():m.end()], confidence=None,
            source="rule", postprocess=["url_encoded_email"],
        ))
    return spans


def add_contextual_rescue_spans(text: str, existing: list[Span]) -> list[Span]:
    """Add high-precision spans for labelled Australian identifiers the model may miss."""
    spans = list(existing)
    occupied = {(span.start, span.end, span.type): idx for idx, span in enumerate(spans)}
    for span_type, pattern in CONTEXTUAL_RESCUE_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.start("value"), match.end("value")
            value = text[start:end]
            if span_type == "EMAIL" and re.search(
                r"placeholder\s+email\s*[:#=\-]?\s*$",
                _context_before(text, start, chars=40),
                re.IGNORECASE,
            ):
                continue
            if span_type == "MEDICAL_INFORMATION" and "illness in group" in value.lower():
                continue
            key = (start, end, span_type)
            if key in occupied:
                idx = occupied[key]
                current = Span(**{**spans[idx].__dict__})
                current.confidence = 1.0
                current.source = "rule"
                current.postprocess = [*current.postprocess, "contextual_identifier_rescue"]
                spans[idx] = current
                continue
            spans.append(Span(
                start=start,
                end=end,
                type=span_type,
                value=value,
                confidence=1.0,
                source="rule",
                postprocess=["contextual_identifier_rescue"],
            ))
            occupied[key] = len(spans) - 1
    return spans


def _has_trigger(ctx: str, triggers: tuple[str, ...]) -> bool:
    return _latest_trigger_position(ctx, triggers) >= 0


def _latest_trigger_position(ctx: str, triggers: tuple[str, ...]) -> int:
    positions: list[int] = []
    for trigger in triggers:
        if not trigger:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(trigger).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        matches = list(re.finditer(pattern, ctx, re.IGNORECASE))
        if matches:
            positions.append(matches[-1].start())
    return max(positions) if positions else -1


def _has_negative_trigger(ctx: str, triggers: tuple[str, ...]) -> bool:
    for trigger in triggers:
        if not trigger:
            continue
        if "*" in trigger:
            pattern = re.escape(trigger).replace(r"\*", r".{0,32}")
            if re.search(pattern, ctx):
                return True
        elif trigger in ctx:
            return True
    return False


def _rule_context(text: str, start: int, end: int, rule: RegistryRule) -> str:
    return text[max(0, start - rule.left_window): min(len(text), end + rule.right_window)].lower()


def _line_context(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].lower()


def _trigger_context_before(text: str, start: int, rule: RegistryRule) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    return text[max(line_start, start - rule.left_window):start].lower()


def _line_context_before(text: str, start: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    return text[line_start:start].lower()


def _trim_match(text: str, start: int, end: int) -> tuple[int, int, str]:
    value = text[start:end]
    left_trim = len(value) - len(value.lstrip())
    right_trim = len(value.rstrip())
    start += left_trim
    end = start + right_trim - left_trim
    return start, end, text[start:end]


def _registry_match_value(text: str, match: re.Match[str]) -> tuple[int, int, str]:
    if "value" in match.groupdict() and match.group("value") is not None:
        return _trim_match(text, match.start("value"), match.end("value"))
    return _trim_match(text, match.start(), match.end())


def _registry_has_positive_trigger(label: str, ctx_before: str, line_before: str,
                                   triggers: tuple[str, ...]) -> bool:
    if _has_trigger(ctx_before, triggers):
        return True
    if label == "CREDIT_CARD_EXPIRY":
        return "card" in line_before and any(marker in line_before for marker in ["exp", "expiry", "expires"])
    if label == "MEDICARE_EXPIRY":
        return "medicare" in line_before and "card expiry" in line_before
    return False


def _registry_positive_trigger_position(label: str, ctx_before: str, line_before: str,
                                        triggers: tuple[str, ...]) -> int:
    if _latest_trigger_position(ctx_before, triggers) >= 0:
        return _latest_trigger_position(line_before, triggers)
    if label == "CREDIT_CARD_EXPIRY" and "card" in line_before:
        return max(line_before.rfind("exp"), line_before.rfind("expiry"), line_before.rfind("expires"))
    if label == "MEDICARE_EXPIRY" and "medicare" in line_before and "card expiry" in line_before:
        return line_before.rfind("card expiry")
    if label == "PASSPORT_EXPIRY" and "passport" in line_before:
        return max(line_before.rfind("expiry"), line_before.rfind("expires"), line_before.rfind("expiration"))
    return -1


def _has_later_competing_trigger(label: str, line_before: str, current_pos: int) -> bool:
    for other_label, triggers in COMPETING_CONTEXT_TRIGGERS.items():
        if other_label == label:
            continue
        if other_label == "PAYMENT_CARD_NUMBER" and label == "PENSION_CARD_NUMBER":
            continue
        if _latest_trigger_position(line_before, triggers) > current_pos:
            return True
    return False


def _registry_rule_context_allowed(label: str, line_before: str, line_ctx: str) -> bool:
    if label == "CREDIT_CARD_EXPIRY":
        return "card" in line_before and not any(marker in line_ctx for marker in ["medicare", "passport"])
    if label == "MEDICARE_EXPIRY":
        return "medicare" in line_ctx
    if label == "PASSPORT_EXPIRY":
        return "passport" in line_ctx
    if label == "MEDICAL_INFORMATION":
        return "special consideration" not in line_ctx
    return True


def _vehicle_registry_confidence(value: str) -> float:
    compact = value.replace(" ", "-").upper()
    if re.fullmatch(r"(?:[A-Z]{3}\d{2}[A-Z]|[A-Z]{1,3}-\d{1,4}(?:-[A-Z]{2,4})?)", compact):
        return 1.0
    return 0.8


def add_registry_contextual_spans(text: str, existing: list[Span],
                                  repo_root: str | Path | None = None) -> list[Span]:
    """Apply config-driven fallback rules from taxonomy_surface_forms + registry."""
    spans = list(existing)
    occupied = {(span.start, span.end, span.type): idx for idx, span in enumerate(spans)}
    for rule in load_postprocess_rule_registry(repo_root):
        for pattern in rule.patterns:
            for match in pattern.finditer(text):
                start, end, value = _registry_match_value(text, match)
                ctx_before = _trigger_context_before(text, start, rule)
                line_before = _line_context_before(text, start)
                line_ctx = _line_context(text, start, end)
                ctx = _rule_context(text, start, end, rule)
                trigger_pos = _registry_positive_trigger_position(
                    rule.label, ctx_before, line_before, rule.positive_triggers,
                )
                if rule.requires_positive_trigger and trigger_pos < 0:
                    continue
                if trigger_pos >= 0 and _has_later_competing_trigger(rule.label, line_before, trigger_pos):
                    continue
                if _has_negative_trigger(line_ctx, rule.negative_triggers):
                    continue
                if not _registry_rule_context_allowed(rule.label, line_before, line_ctx):
                    continue
                confidence = _vehicle_registry_confidence(value) if rule.label == "VEHICLE_ID" else 1.0
                postprocess_note = (
                    "registry_contextual_review_candidate"
                    if rule.label == "VEHICLE_ID" and confidence < 1.0
                    else "registry_contextual_rescue"
                )
                key = (start, end, rule.label)
                if key in occupied:
                    idx = occupied[key]
                    current = Span(**{**spans[idx].__dict__})
                    current.confidence = confidence
                    current.source = "rule"
                    current.postprocess = [*current.postprocess, postprocess_note]
                    spans[idx] = current
                    continue
                spans.append(Span(
                    start=start,
                    end=end,
                    type=rule.label,
                    value=value,
                    confidence=confidence,
                    source="rule",
                    postprocess=[postprocess_note],
                ))
                occupied[key] = len(spans) - 1
    return spans


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Keep longer spans first; on ties, keep earlier."""
    ordered = sorted(
        spans,
        key=lambda s: (0 if s.source == "rule" else 1, s.start, -(s.end - s.start), s.type),
    )
    kept: list[Span] = []
    for span in ordered:
        if any(span.start < old.end and old.start < span.end for old in kept):
            continue
        kept.append(span)
    return sorted(kept, key=lambda s: (s.start, s.end, s.type))


def safe_postprocess_spans(text: str, spans: list[Span], policy: dict[str, Any]) -> tuple[list[Span], list[str]]:
    """Apply the policy-driven postprocess pipeline. Returns (cleaned_spans, warnings)."""
    config = policy.get("postprocess", {})
    warnings: list[str] = []
    processed: list[Span] = []
    for span in spans:
        if not (0 <= span.start < span.end <= len(text)):
            warnings.append(f"span_dropped_invalid_offsets:{span.type}:{span.value[:40]}")
            continue
        if text[span.start:span.end] != span.value:
            span = Span(**{**span.__dict__})
            span.value = text[span.start:span.end]
            span.postprocess = [*span.postprocess, "value_reset_from_offsets"]
        if config.get("drop_common_hard_negatives", True) and should_drop_false_positive(span, text):
            continue
        if config.get("normalize_contextual_labels", True):
            span = normalize_contextual_type(span, text)
        if config.get("collapse_generic_work_contacts", True):
            span = collapse_work_contact_type(span, text)
        if config.get("strip_known_prefixes", True):
            stripped = strip_prefix_from_span(span)
            if stripped is None:
                continue
            span = stripped
        processed.append(span)
    if config.get("add_url_encoded_emails", True):
        processed = add_url_encoded_email_spans(text, processed)
    if config.get("add_builtin_contextual_identifier_spans", config.get("add_contextual_identifier_spans", False)):
        processed = add_contextual_rescue_spans(text, processed)
    if config.get("add_registry_contextual_spans", True):
        processed = add_registry_contextual_spans(text, processed)
    processed = resolve_overlaps(processed)
    return processed, warnings

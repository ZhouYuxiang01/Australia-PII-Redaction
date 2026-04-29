"""Generic span post-processing: prefix stripping, work-contact collapsing,
URL-encoded email rescue, overlap resolution.

These rules are model-agnostic — they apply to any backend's spans.
"""
from __future__ import annotations

import re
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
        "AU_BANK_ACCOUNT",
        re.compile(r"\bAccount\s+name\s*[:#=\-]?\s*(?P<value>[^\n\r]+)", re.IGNORECASE),
    ),
    (
        "AU_BANK_ACCOUNT",
        re.compile(r"\bBank\s*[:#=\-]?\s*(?P<value>[A-Z][A-Za-z &'\-]+(?:Bank|Credit Union))\b", re.IGNORECASE),
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
            r"(?m)^\s*\d+\.\s+(?P<value>[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,3})\s*$",
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
        re.compile(r"\b(?:Reason|Medical detail)\s*[:#=\-]?\s*(?P<value>[^\n\r]+)", re.IGNORECASE),
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
        re.compile(r"\bRole\s*[:#=\-]?\s*(?P<value>[^\n\r]+)", re.IGNORECASE),
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
            new_type = "EMAIL_ADDRESS"
    elif span.type == "WORK_PHONE":
        explicit = any(m in ctx for m in ["work phone", "office phone", "business phone", "staff phone", "work:"])
        if not explicit and any(m in ctx for m in ["phone", "ph:", "tel"]):
            new_type = "AU_PHONE"
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
        key = (m.start(), m.end(), "EMAIL_ADDRESS")
        if key in occupied:
            continue
        spans.append(Span(
            start=m.start(), end=m.end(), type="EMAIL_ADDRESS",
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
    if config.get("add_contextual_identifier_spans", True):
        processed = add_contextual_rescue_spans(text, processed)
    processed = resolve_overlaps(processed)
    return processed, warnings

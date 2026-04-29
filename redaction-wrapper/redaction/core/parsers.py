"""Tagged-output parsing helpers used by Qwen-style generative backends.

Generative backends produce tagged text like:
    Hello <pii type="PERSON">Alice</pii>, your TFN is <pii type="AU_TFN">123 456 789</pii>.

This module:
  - parses tagged output into (plain_text, list[Span]) with offsets in the
    parsed plain text (not the original input);
  - repairs offsets back to the original input by exact-value lookup when the
    generated plain text drifts from the input.
"""
from __future__ import annotations

import html
import re

from .span import Span


THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
PII_TAG_RE = re.compile(
    r"<pii\s+type=(?:\"(?P<type_dq>[^\"]+)\"|'(?P<type_sq>[^']+)'|(?P<type_raw>[A-Za-z0-9_:-]+))\s*>(?P<value>.*?)</pii>",
    re.DOTALL | re.IGNORECASE,
)

TYPE_ALIASES = {
    "NAME": "PERSON",
    "FULL_NAME": "PERSON",
    "PERSON_NAME": "PERSON",
    "DOB": "DATE_OF_BIRTH",
    "BIRTH_DATE": "DATE_OF_BIRTH",
    "DATEOFBIRTH": "DATE_OF_BIRTH",
    "EMAIL": "EMAIL_ADDRESS",
    "EMAILADDRESS": "EMAIL_ADDRESS",
    "E_MAIL": "EMAIL_ADDRESS",
    "PHONE": "AU_PHONE",
    "PHONE_NUMBER": "AU_PHONE",
    "MOBILE": "AU_PHONE",
    "MOBILE_NUMBER": "AU_PHONE",
    "TELEPHONE": "AU_PHONE",
    "TFN": "AU_TFN",
    "TAX_FILE_NUMBER": "AU_TFN",
    "PASSPORT": "AU_PASSPORT",
    "PASSPORT_NUMBER": "AU_PASSPORT",
    "DRIVER_LICENSE": "AU_DRIVERS_LICENCE",
    "DRIVERS_LICENSE": "AU_DRIVERS_LICENCE",
    "DRIVER_LICENCE": "AU_DRIVERS_LICENCE",
    "DRIVERS_LICENCE": "AU_DRIVERS_LICENCE",
    "LICENCE_NUMBER": "AU_DRIVERS_LICENCE",
    "MEDICARE": "MEDICARE_NUMBER",
    "MEDICARE_CARD": "MEDICARE_NUMBER",
    "CARD_NUMBER": "PAYMENT_CARD_NUMBER",
    "CREDIT_CARD": "PAYMENT_CARD_NUMBER",
    "BANK_ACCOUNT": "AU_BANK_ACCOUNT",
    "ACCOUNT_NUMBER": "AU_BANK_ACCOUNT",
    "IP": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "DEVICE": "DEVICE_ID",
    "REGISTRATION": "VEHICLE_REGO",
    "REGO": "VEHICLE_REGO",
    "CRN": "CENTRELINK_REFERENCE_NUMBER",
}


def clean_model_output(output: str) -> str:
    return THINK_RE.sub("", output or "").strip()


def canonicalize_type(label: str) -> tuple[str, str | None]:
    key = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_").upper()
    canonical = TYPE_ALIASES.get(key, key)
    if canonical == label:
        return canonical, None
    return canonical, f"type_alias:{label}->{canonical}"


def parse_annotated_output(output: str) -> tuple[str, list[Span]]:
    """Parse '<pii type=X>val</pii>' tags out of model output.

    Returns:
        plain_text: model output with all tags stripped
        spans: Span objects whose offsets index into plain_text
    """
    cleaned = clean_model_output(output)
    parts: list[str] = []
    spans: list[Span] = []
    cursor = 0
    plain_offset = 0

    for m in PII_TAG_RE.finditer(cleaned):
        before = cleaned[cursor:m.start()]
        parts.append(before)
        plain_offset += len(before)

        value = html.unescape(m.group("value"))
        raw_type = m.group("type_dq") or m.group("type_sq") or m.group("type_raw")
        span_type, type_note = canonicalize_type(raw_type)
        span_start = plain_offset
        parts.append(value)
        plain_offset += len(value)
        span_end = plain_offset
        spans.append(Span(
            start=span_start, end=span_end, type=span_type, value=value,
            confidence=None, source="model",
            postprocess=[type_note] if type_note else [],
        ))
        cursor = m.end()

    parts.append(cleaned[cursor:])
    return "".join(parts), spans


def repair_offsets_to_input(input_text: str, parsed_text: str,
                            spans: list[Span]) -> tuple[list[Span], list[str], bool]:
    """Repair offsets when the generated plain text drifts from the input.

    Strategy: for each span, look for its value verbatim in input_text. If it
    occurs exactly once, snap to that occurrence. Otherwise drop it.
    """
    warnings: list[str] = []
    if parsed_text == input_text:
        return spans, warnings, False
    repaired: list[Span] = []
    warnings.append("round_trip_mismatch_offsets_repaired_when_unique")
    for span in spans:
        matches = list(re.finditer(re.escape(span.value), input_text))
        if len(matches) == 1:
            m = matches[0]
            out = Span(**{**span.__dict__})
            out.start, out.end = m.start(), m.end()
            out.postprocess = [*out.postprocess, "offset_repaired_by_exact_value"]
            repaired.append(out)
        elif (0 <= span.start < span.end <= len(input_text)
              and input_text[span.start:span.end] == span.value):
            repaired.append(span)
        else:
            warnings.append(f"span_dropped_unrepaired:{span.type}:{span.value[:40]}")
    return repaired, warnings, True

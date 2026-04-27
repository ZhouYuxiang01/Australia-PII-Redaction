#!/usr/bin/env python
"""Qwen tagged-output parser and deterministic redaction wrapper."""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
PII_TAG_RE = re.compile(
    r"<pii\s+type=(?:\"(?P<type_dq>[^\"]+)\"|'(?P<type_sq>[^']+)'|(?P<type_raw>[A-Za-z0-9_:-]+))\s*>(?P<value>.*?)</pii>",
    re.DOTALL | re.IGNORECASE,
)
ENCODED_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+\-]+)%40([A-Za-z0-9.\-]+\.[A-Za-z]{2,})", re.IGNORECASE)

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


@dataclass
class Span:
    start: int
    end: int
    type: str
    value: str
    confidence: float | None = None
    decision: str = "AUTO_REDACT"
    replacement: str | None = None
    source: str = "model"
    postprocess: list[str] = field(default_factory=list)

    def to_schema(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "type": self.type,
            "confidence": self.confidence,
            "decision": self.decision,
            "replacement": self.replacement or f"[{self.type}]",
            "value": self.value,
            "source": self.source,
            "postprocess": self.postprocess,
        }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean_model_output(output: str) -> str:
    return THINK_RE.sub("", output or "").strip()


def canonicalize_type(label: str) -> tuple[str, str | None]:
    key = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_").upper()
    canonical = TYPE_ALIASES.get(key, key)
    if canonical == label:
        return canonical, None
    return canonical, f"type_alias:{label}->{canonical}"


def parse_annotated_output(output: str) -> tuple[str, list[Span]]:
    """Parse tagged Qwen output into plain text and spans in reconstructed text."""
    cleaned = clean_model_output(output)
    parts: list[str] = []
    spans: list[Span] = []
    cursor = 0
    plain_offset = 0

    for match in PII_TAG_RE.finditer(cleaned):
        before = cleaned[cursor : match.start()]
        parts.append(before)
        plain_offset += len(before)

        value = html.unescape(match.group("value"))
        raw_type = match.group("type_dq") or match.group("type_sq") or match.group("type_raw")
        span_type, type_note = canonicalize_type(raw_type)
        span_start = plain_offset
        parts.append(value)
        plain_offset += len(value)
        span_end = plain_offset

        spans.append(
            Span(
                start=span_start,
                end=span_end,
                type=span_type,
                value=value,
                confidence=None,
                source="model",
                postprocess=[type_note] if type_note else [],
            )
        )
        cursor = match.end()

    tail = cleaned[cursor:]
    parts.append(tail)
    plain_text = "".join(parts)
    return plain_text, spans


def repair_offsets_to_input(input_text: str, parsed_text: str, spans: list[Span]) -> tuple[list[Span], list[str], bool]:
    """Repair offsets when the generated plain text differs from the original input."""
    warnings: list[str] = []
    if parsed_text == input_text:
        return spans, warnings, False

    repaired: list[Span] = []
    warnings.append("round_trip_mismatch_offsets_repaired_when_unique")
    for span in spans:
        value = span.value
        matches = [m for m in re.finditer(re.escape(value), input_text)]
        if len(matches) == 1:
            match = matches[0]
            out = Span(**{**span.__dict__})
            out.start = match.start()
            out.end = match.end()
            out.postprocess = [*out.postprocess, "offset_repaired_by_exact_value"]
            repaired.append(out)
        elif 0 <= span.start < span.end <= len(input_text) and input_text[span.start : span.end] == value:
            repaired.append(span)
        else:
            warnings.append(f"span_dropped_unrepaired:{span.type}:{value[:40]}")
    return repaired, warnings, True


def strip_prefix_from_span(span: Span) -> Span | None:
    patterns = PREFIX_PATTERNS.get(span.type, [])
    current_value = span.value
    total_shift = 0
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            match = pattern.match(current_value)
            if not match:
                continue
            shift = match.end()
            current_value = current_value[shift:].strip()
            total_shift += shift
            changed = True
            break
    if not current_value:
        return None
    if total_shift == 0:
        return span
    out = Span(**{**span.__dict__})
    out.start += total_shift
    out.value = out.value[total_shift:].strip()
    out.end = out.start + len(out.value)
    out.postprocess = [*out.postprocess, "strip_known_prefix"]
    return out


def context_before(text: str, start: int, chars: int = 48) -> str:
    return text[max(0, start - chars) : start].lower()


def collapse_work_contact_type(span: Span, text: str) -> Span:
    ctx = context_before(text, span.start)
    new_type = span.type
    if span.type == "WORK_EMAIL":
        explicit_work = any(marker in ctx for marker in ["work email", "office email", "staff email", "business email"])
        if not explicit_work and "email" in ctx:
            new_type = "EMAIL_ADDRESS"
    elif span.type == "WORK_PHONE":
        explicit_work = any(marker in ctx for marker in ["work phone", "office phone", "business phone", "staff phone", "work:"])
        if not explicit_work and any(marker in ctx for marker in ["phone", "ph:", "tel"]):
            new_type = "AU_PHONE"
    if new_type == span.type:
        return span
    out = Span(**{**span.__dict__})
    out.type = new_type
    out.postprocess = [*out.postprocess, "collapse_generic_work_contact"]
    return out


def add_url_encoded_email_spans(text: str, existing: list[Span]) -> list[Span]:
    spans = list(existing)
    occupied = {(span.start, span.end, span.type) for span in spans}
    for match in ENCODED_EMAIL_RE.finditer(text):
        key = (match.start(), match.end(), "EMAIL_ADDRESS")
        if key in occupied:
            continue
        spans.append(
            Span(
                start=match.start(),
                end=match.end(),
                type="EMAIL_ADDRESS",
                value=text[match.start() : match.end()],
                confidence=None,
                source="rule",
                postprocess=["url_encoded_email"],
            )
        )
    return spans


def safe_postprocess_spans(text: str, spans: list[Span], policy: dict[str, Any]) -> tuple[list[Span], list[str]]:
    config = policy.get("postprocess", {})
    warnings: list[str] = []
    processed: list[Span] = []
    for span in spans:
        if not (0 <= span.start < span.end <= len(text)):
            warnings.append(f"span_dropped_invalid_offsets:{span.type}:{span.value[:40]}")
            continue
        if text[span.start : span.end] != span.value:
            # Keep value consistent with redaction offsets after any repair.
            span = Span(**{**span.__dict__})
            span.value = text[span.start : span.end]
            span.postprocess = [*span.postprocess, "value_reset_from_offsets"]
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

    processed = resolve_overlaps(processed)
    return processed, warnings


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start), s.type))
    kept: list[Span] = []
    for span in ordered:
        if any(span.start < old.end and old.start < span.end for old in kept):
            continue
        kept.append(span)
    return sorted(kept, key=lambda s: (s.start, s.end, s.type))


def apply_policy(spans: list[Span], policy: dict[str, Any]) -> list[Span]:
    default_action = policy.get("default_action", "AUTO_REDACT")
    type_actions = policy.get("type_actions", {})
    default_confidence = policy.get("confidence", {}).get("default_value")
    out: list[Span] = []
    for span in spans:
        item = Span(**{**span.__dict__})
        item.decision = type_actions.get(item.type, default_action)
        item.confidence = default_confidence
        item.replacement = f"[{item.type}]"
        out.append(item)
    return out


def redact_text(text: str, spans: list[Span], mode: str = "replace_with_tag", mask_char: str = "*") -> str:
    pieces: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        if span.decision == "PASS":
            continue
        pieces.append(text[cursor : span.start])
        if mode == "remove":
            replacement = ""
        elif mode == "mask":
            replacement = mask_char * (span.end - span.start)
        else:
            replacement = span.replacement or f"[{span.type}]"
        pieces.append(replacement)
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def build_response(
    *,
    text: str,
    spans: list[Span],
    policy: dict[str, Any],
    raw_offset_mapping_applied: bool,
    warnings: list[str],
) -> dict[str, Any]:
    mode = policy.get("redaction_mode", "replace_with_tag")
    redacted_text = redact_text(text, spans, mode=mode)
    if policy.get("confidence", {}).get("calibrated") is False:
        warnings = [*warnings, "confidence_uncalibrated_null"]
    return {
        "redacted_text": redacted_text,
        "spans": [span.to_schema() for span in spans],
        "metadata": {
            "model_version": policy.get("model_version", "unknown"),
            "taxonomy_version": policy.get("taxonomy_version", "unknown"),
            "schema_version": policy.get("schema_version", "redaction-output-v1"),
            "policy_id": policy.get("policy_id", "unknown"),
            "normalization": "NFC",
            "raw_offset_mapping_applied": raw_offset_mapping_applied,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "warnings": warnings,
    }


def run_model_inference(args: argparse.Namespace, text: str) -> str:
    """Optional remote inference path. Not exercised by local tests."""
    if not args.base_model or not args.adapter_dir:
        raise ValueError("Model inference requires --base-model and --adapter-dir, or provide --annotated-output.")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip() if args.system_prompt_file else (
        "You are an Australian PII redaction system. Return the input text with each supported PII span wrapped as "
        "<pii type=\"TYPE\">exact text</pii>. Do not explain."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if isinstance(encoded, torch.Tensor):
        encoded = {"input_ids": encoded}
    target_device = getattr(model, "device", None) or next(model.parameters()).device
    encoded = {key: value.to(target_device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[-1]
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def read_text_arg(value: str | None, path: Path | None) -> str:
    if path:
        return path.read_text(encoding="utf-8")
    if value is not None:
        return value
    raise ValueError("Expected direct text or file path")


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse Qwen PII tagged output and produce redacted JSON.")
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--annotated-output", default=None)
    parser.add_argument("--annotated-output-file", type=Path, default=None)
    parser.add_argument("--policy", type=Path, default=Path("configs/policies/qwen-safe-default-v1.json"))
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    args = parser.parse_args()

    text = normalize_text(read_text_arg(args.text, args.text_file))
    policy = load_json(args.policy)
    if args.annotated_output is not None or args.annotated_output_file is not None:
        annotated_output = read_text_arg(args.annotated_output, args.annotated_output_file)
    else:
        annotated_output = run_model_inference(args, text)

    parsed_text, spans = parse_annotated_output(annotated_output)
    parsed_text = normalize_text(parsed_text)
    spans, repair_warnings, repaired = repair_offsets_to_input(text, parsed_text, spans)
    spans, post_warnings = safe_postprocess_spans(text, spans, policy)
    spans = apply_policy(spans, policy)
    payload = build_response(
        text=text,
        spans=spans,
        policy=policy,
        raw_offset_mapping_applied=repaired,
        warnings=[*repair_warnings, *post_warnings],
    )

    output = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n", encoding="utf-8", newline="\n")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

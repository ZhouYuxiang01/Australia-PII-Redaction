"""Policy application + redaction output assembly.

A policy is a JSON file that defines:
- default_action / type_actions: AUTO_REDACT | REVIEW | PASS
- per-type confidence thresholds (block_threshold, review_threshold) — optional
- redaction_mode: replace_with_tag | mask | remove
- model_version, taxonomy_version, schema_version, policy_id
- postprocess flags

If a span has a confidence score AND the policy declares thresholds, the
confidence-based decision OVERRIDES the static type_action. The higher of the
two thresholds is treated as the AUTO_REDACT cutoff and the lower as the REVIEW
cutoff, because calibration can derive the named operating points independently:
  conf >= max(block_threshold, review_threshold) -> AUTO_REDACT
  conf >= min(block_threshold, review_threshold) -> REVIEW
  otherwise                                     -> PASS
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .span import Span


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _decision_from_confidence(span_type: str, conf: float | None,
                              type_thresholds: dict[str, dict[str, float]],
                              global_block: float | None,
                              global_review: float | None,
                              fallback: str) -> str | None:
    """Compute decision from confidence + thresholds. Return None if no thresholds available."""
    if conf is None:
        return None
    t = type_thresholds.get(span_type, {})
    block_thr = t.get("block_threshold", global_block)
    review_thr = t.get("review_threshold", global_review)
    if block_thr is None and review_thr is None:
        return None
    if block_thr is not None and review_thr is not None and block_thr < review_thr:
        block_thr, review_thr = review_thr, block_thr
    if block_thr is not None and conf >= block_thr:
        return "AUTO_REDACT"
    if review_thr is not None and conf >= review_thr:
        return "REVIEW"
    return "PASS"


def apply_policy(spans: list[Span], policy: dict[str, Any]) -> list[Span]:
    """Decorate spans with decision + replacement based on policy.

    Order of precedence for `decision`:
      1. confidence + thresholds (if both available)
      2. policy.type_actions[type]
      3. policy.default_action
    """
    default_action = policy.get("default_action", "AUTO_REDACT")
    type_actions = policy.get("type_actions", {})
    type_thresholds = policy.get("type_thresholds", {})
    global_block = policy.get("global_block_threshold")
    global_review = policy.get("global_review_threshold")
    default_confidence = policy.get("confidence", {}).get("default_value")
    out: list[Span] = []
    for span in spans:
        item = Span(**{**span.__dict__})
        # Use confidence-based decision if possible, else type_actions, else default.
        conf_decision = _decision_from_confidence(
            item.type, item.confidence, type_thresholds, global_block, global_review,
            default_action,
        )
        if conf_decision is not None:
            item.decision = conf_decision
        else:
            item.decision = type_actions.get(item.type, default_action)
        if item.confidence is None and default_confidence is not None:
            item.confidence = default_confidence
        item.replacement = f"[{item.type}]"
        out.append(item)
    return out


def redact_text(
    text: str,
    spans: list[Span],
    mode: str = "replace_with_tag",
    mask_char: str = "*",
    redact_review_types: set[str] | None = None,
) -> str:
    """Apply deterministic redaction for spans approved for automatic replacement."""
    redact_review_types = redact_review_types or set()
    pieces: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        should_redact = span.decision == "AUTO_REDACT" or (
            span.decision == "REVIEW"
            and (span.type in redact_review_types or "*" in redact_review_types)
        )
        if not should_redact:
            continue
        pieces.append(text[cursor:span.start])
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


def build_response(*, text: str, spans: list[Span], policy: dict[str, Any],
                   raw_offset_mapping_applied: bool, warnings: list[str],
                   extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    mode = policy.get("redaction_mode", "replace_with_tag")
    redact_review_types = set(policy.get("redact_review_types", []))
    redacted = redact_text(text, spans, mode=mode, redact_review_types=redact_review_types)
    visible_spans = [span for span in spans if span.decision in {"AUTO_REDACT", "REVIEW"}]
    if policy.get("confidence", {}).get("calibrated") is False:
        warnings = [*warnings, "confidence_uncalibrated_null"]
    metadata = {
        "model_version": policy.get("model_version", "unknown"),
        "taxonomy_version": policy.get("taxonomy_version", "unknown"),
        "schema_version": policy.get("schema_version", "redaction-output-v1"),
        "policy_id": policy.get("policy_id", "unknown"),
        "normalization": "NFC",
        "raw_offset_mapping_applied": raw_offset_mapping_applied,
        "redaction_mode": mode,
        "redact_review_types": sorted(redact_review_types),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "redacted_text": redacted,
        "spans": [span.to_schema() for span in visible_spans],
        "metadata": metadata,
        "warnings": warnings,
    }

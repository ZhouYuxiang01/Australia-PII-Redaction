"""Model-agnostic core: Span, normalization, postprocess, policy, redact."""
from .normalize import normalize_text
from .span import Span
from .policy import apply_policy, build_response, load_json, redact_text
from .postprocess import safe_postprocess_spans, resolve_overlaps
from .parsers import parse_annotated_output, repair_offsets_to_input

__all__ = [
    "Span",
    "normalize_text",
    "apply_policy",
    "build_response",
    "load_json",
    "redact_text",
    "safe_postprocess_spans",
    "resolve_overlaps",
    "parse_annotated_output",
    "repair_offsets_to_input",
]

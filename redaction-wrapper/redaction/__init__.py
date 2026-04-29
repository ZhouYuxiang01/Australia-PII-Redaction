"""Model-agnostic PII redaction wrapper.

Public surface:
  - core: Span, apply_policy, redact_text, build_response, parse_annotated_output, ...
  - backends: RedactionBackend ABC + QwenLoraBackend, OpfBackend, registry
  - ocr: extract_upload_text, ...
  - api: create_app(), get_app()  (FastAPI factory)

A typical embedding (no FastAPI) looks like:

    from redaction.backends import build_backend_from_path
    from redaction.core import (apply_policy, build_response, load_json,
                                normalize_text, safe_postprocess_spans)
    backend = build_backend_from_path("configs/backends/opf-v3.json")
    policy  = load_json("configs/policies/opf-v3-default-v1.json")
    text = normalize_text("Please contact Alice at alice@example.com")
    spans, diag = backend.detect_spans(text)
    spans, _ = safe_postprocess_spans(text, spans, policy)
    spans = apply_policy(spans, policy)
    payload = build_response(text=text, spans=spans, policy=policy,
                             raw_offset_mapping_applied=diag["raw_offset_mapping_applied"],
                             warnings=diag["warnings"])
"""
__version__ = "1.0.0"

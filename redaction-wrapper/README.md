# Redaction Wrapper

Model-agnostic PII redaction service. Plug in any backend that satisfies the
`RedactionBackend` interface; everything else (OCR, post-processing, policy,
schema, deterministic redaction, FastAPI, web UI) is shared.

Currently bundled backends:

| Backend | Use case | Status |
|---|---|---|
| `opf` | OPF v3 token-tagger (lightweight, fast, deployable to 24GB) | ✅ |
| `qwen_lora` | Qwen 9B LoRA tagged-output | ✅ |
| `qwen_lora` (full mode) | Qwen 4B Full SFT | ✅ |

## Quick start

```bash
# 1. Pick a backend + policy
export WRAPPER_BACKEND_CONFIG=$PWD/configs/backends/opf-v3.json
export WRAPPER_POLICY_CONFIG=$PWD/configs/policies/opf-v3-default-v1.json

# 2. Launch
./scripts/run_server.sh

# 3. Open the demo
open http://127.0.0.1:8090/         # browser UI
open http://127.0.0.1:8090/docs     # interactive FastAPI docs
```

## Layout

```
redaction-wrapper/
├── redaction/                       importable Python package
│   ├── core/                        Span, normalize, postprocess, policy, redact, parsers
│   ├── backends/                    base.py + opf.py + qwen_lora.py + registry.py
│   ├── ocr/                         image / PDF text extraction (Tesseract + pdftotext)
│   └── api/                         FastAPI server.py
├── configs/
│   ├── backends/                    one JSON per backend instance
│   └── policies/                    one JSON per policy (block/review per type)
├── schemas/
│   └── redaction-output-v1.schema.json
├── static/
│   └── redaction_demo.html          single-page demo UI
├── docs/                            api.md, backends.md
├── scripts/                         run_server.sh, cli.py
├── examples/                        demo input examples (optional override)
└── tests/
```

## Embedding without FastAPI

```python
from redaction.backends import build_backend_from_path
from redaction.core import (apply_policy, build_response, load_json,
                            normalize_text, safe_postprocess_spans)

backend = build_backend_from_path("configs/backends/opf-v3.json")
policy  = load_json("configs/policies/opf-v3-default-v1.json")

text = normalize_text("Please contact Alice Wong at alice@example.edu.au.")
spans, diag = backend.detect_spans(text)
spans, post_warnings = safe_postprocess_spans(text, spans, policy)
spans = apply_policy(spans, policy)
payload = build_response(
    text=text, spans=spans, policy=policy,
    raw_offset_mapping_applied=diag["raw_offset_mapping_applied"],
    warnings=[*diag["warnings"], *post_warnings],
)
```

## Adding a new backend

See [`docs/backends.md`](docs/backends.md). Implement the
`RedactionBackend` ABC (`redaction/backends/base.py`), add a builder in
`redaction/backends/registry.py`, drop a config JSON in `configs/backends/`,
and you're done — the API, OCR, policy and UI all flow through unchanged.

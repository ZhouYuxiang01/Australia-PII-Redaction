# Redaction Wrapper API

Model-agnostic REST surface. The same endpoints serve any registered backend
selected via `WRAPPER_BACKEND_CONFIG`.

## Base URL

```
http://127.0.0.1:8090
```

## Selecting backend / policy

Set environment variables before launching:

```bash
export WRAPPER_BACKEND_CONFIG=$PWD/configs/backends/opf-v3.json
export WRAPPER_POLICY_CONFIG=$PWD/configs/policies/opf-v3-default-v1.json
./scripts/run_server.sh
```

Switch model by pointing to a different config — for example
`configs/backends/qwen-9b-lora.json` paired with
`configs/policies/qwen-9b-lora-default-v1.json`.

Interactive FastAPI docs: `/docs` and `/redoc`.

## GET /

Returns the browser demo page (`static/redaction_demo.html`).

## GET /api/health

```json
{
  "status": "ok",
  "backend": {
    "name": "opf-v3",
    "model_version": "opf-73class-v3-full",
    "loaded": false,
    "supported_types": ["PERSON", "ADDRESS", "..."]
  },
  "policy_id": "opf-v3-default-v1",
  "model_version": "opf-73class-v3-full",
  "schema_version": "redaction-output-v1"
}
```

`loaded` becomes `true` after the first inference loads the model.

## GET /api/examples

Returns the demo example list. Override with `WRAPPER_EXAMPLES=path/to/examples.json`
(format: `[{"id": "...", "title": "...", "text": "..."}]`).

## POST /api/redact

Run inference and return redaction-ready JSON.

### Request

```json
{
  "text": "Please contact Alice Nguyen at alice.nguyen@example.edu.au."
}
```

Or by example id:

```json
{ "example_id": "student-support" }
```

If both are present, `text` wins.

### Response (matches `schemas/redaction-output-v1.schema.json`)

```json
{
  "redacted_text": "Please contact [PERSON] at [EMAIL].",
  "spans": [
    {
      "start": 15,
      "end": 27,
      "type": "PERSON",
      "confidence": null,
      "decision": "AUTO_REDACT",
      "replacement": "[PERSON]",
      "source": "model",
      "postprocess": []
    }
  ],
  "metadata": {
    "model_version": "opf-73class-v3-full",
    "taxonomy_version": "taxonomy_v1.1.1-draft-73class",
    "schema_version": "redaction-output-v1",
    "policy_id": "opf-v3-default-v1",
    "normalization": "NFC",
    "raw_offset_mapping_applied": false,
    "redaction_mode": "replace_with_tag",
    "backend_name": "opf-v3",
    "backend_route": "text_input",
    "input_length": 58,
    "latency_ms": 42.7,
    "created_at": "2026-04-28T01:00:00+00:00"
  },
  "warnings": []
}
```

Field notes:
- `start` / `end` are character offsets into the submitted NFC-normalized text (end exclusive).
- Span values and raw model output are not returned in public API responses.
- `confidence` is null when the backend doesn't expose calibrated scores. With
  a calibration-aware backend (per `Span.confidence` in [0,1]), the
  `decision` field is derived from `policy.type_thresholds`.
- Public API responses expose only actionable decisions: `AUTO_REDACT` and
  `REVIEW`. Lower-confidence candidates are filtered out before serialization.
- `redacted_text` is built deterministically from `AUTO_REDACT` spans and the
  policy's `redaction_mode` (`replace_with_tag` | `mask` | `remove`).
  `REVIEW` spans remain visible for human review.
- For tagged-generation backends, the wrapper repairs offsets by exact-value
  lookup when the model output drifts from the input. See `warnings`.

## POST /api/redact-file

Upload a file. The wrapper extracts text (PDF text layer, local RapidOCR,
PaddleOCR, or Tesseract OCR for images / scanned PDFs, or plain text), runs the
chosen backend, and
returns the same redaction payload plus OCR metadata.

```bash
curl -X POST http://127.0.0.1:8090/api/redact-file \
  -F "file=@/path/to/document.pdf"
```

Extra metadata in the response:

```json
{
  "redacted_text": "...",
  "metadata": {
    "file": {
      "name": "document.pdf",
      "content_type": "application/pdf",
      "kind": "pdf",
      "size_bytes": 12345
    },
    "ocr": {
      "text_source": "pdf_text_layer",
      "pages": 0,
      "warnings": []
    }
  }
}
```

The extracted raw text is also returned as `ocr_text` for the demo UI so users
can inspect OCR quality before trusting the redaction output.

Local RapidOCR can be enabled before launch:

```bash
export WRAPPER_OCR_PROVIDER=rapidocr
export RAPIDOCR_MIN_CONFIDENCE=0.30
./scripts/run_server.sh
```

`WRAPPER_OCR_PROVIDER=auto` uses local RapidOCR when it is installed and falls
back to local Tesseract when RapidOCR is unavailable.

Local PaddleOCR can also be selected explicitly:

```bash
export WRAPPER_OCR_PROVIDER=paddle
export PADDLEOCR_LANG=en
export PADDLEOCR_USE_GPU=false
export PADDLEOCR_MIN_CONFIDENCE=0.30
./scripts/run_server.sh
```

Set `WRAPPER_OCR_PROVIDER=paddle` or `WRAPPER_OCR_PROVIDER=rapidocr` to fail
fast if that provider is not installed.

Supported inputs:

- PDF: text layer first, OCR fallback per page.
- Images: png, jpg, jpeg, tif, tiff, bmp, webp.
- Plain text: txt, md, csv, tsv, json, log.

Limits via env vars:

| Env | Default |
|---|---|
| `WRAPPER_MAX_UPLOAD_BYTES` | 26214400 (25 MB) |
| `WRAPPER_MAX_PDF_OCR_PAGES` | 8 |
| `WRAPPER_MAX_FILE_TEXT_CHARS` | 12000 |
| `WRAPPER_MAX_TEXT_CHARS` | 3000 |

## curl examples

```bash
curl http://127.0.0.1:8090/api/health

curl -X POST http://127.0.0.1:8090/api/redact \
  -H "Content-Type: application/json" \
  -d '{"text":"Please contact Alice Nguyen at alice.nguyen@example.edu.au."}'

curl -X POST http://127.0.0.1:8090/api/redact \
  -H "Content-Type: application/json" \
  -d '{"example_id":"student-support"}'
```

## Runtime

```bash
cd /home/admin/ZYX/redaction-wrapper
./scripts/run_server.sh
```

Logs are written to `scripts/logs/redaction_wrapper_${BACKEND_NAME}_${PORT}.log`.

# Redaction Demo API

This is the live-only demo API for the Qwen3.5 9B LoRA PII redaction route.

## Base URL

```text
http://100.91.98.45:8090
```

If direct browser access is not available, use an SSH tunnel:

```bash
ssh -i C:\Users\zyx62\Desktop\5703test\.ssh\modernbert_distill_ed25519 -L 8090:127.0.0.1:8090 admin@100.91.98.45
```

Then use:

```text
http://127.0.0.1:8090
```

Interactive FastAPI docs are also available at:

```text
/docs
/redoc
```

## GET /

Returns the browser demo page.

## GET /api/health

Returns service status and model/policy metadata.

Example response:

```json
{
  "status": "ok",
  "mode": "live",
  "model_loaded": false,
  "max_concurrent_generate": 4,
  "policy_id": "qwen-safe-default-v1",
  "model_version": "qwen3.5-9b-base-lora-tagged-28-fastretry"
}
```

`model_loaded` becomes `true` after the first live inference request loads the model.

## GET /api/examples

Returns demo input examples. These are not cached model outputs; selecting one only sends its text through the live model.

Example response:

```json
{
  "examples": [
    {
      "id": "student-support",
      "title": "Student Support",
      "text": "Student record: Olivia Okonkwo, DOB November 04, 1999, email olivia.okonkwo@example.edu.au, TFN 832 109 111."
    }
  ]
}
```

## POST /api/redact

Runs live model inference and returns redaction-ready JSON.

### Request Body

Use direct text:

```json
{
  "text": "Please contact Alice Nguyen at alice.nguyen@example.edu.au."
}
```

Or use a demo example id:

```json
{
  "example_id": "student-support"
}
```

If both `text` and `example_id` are provided, `text` takes precedence.

### Response Body

```json
{
  "input_text": "Please contact Alice Nguyen at alice.nguyen@example.edu.au.",
  "redacted_text": "Please contact [PERSON] at [EMAIL_ADDRESS].",
  "spans": [
    {
      "start": 15,
      "end": 27,
      "type": "PERSON",
      "confidence": null,
      "decision": "AUTO_REDACT",
      "replacement": "[PERSON]",
      "value": "Alice Nguyen",
      "source": "model",
      "postprocess": []
    }
  ],
  "metadata": {
    "model_version": "qwen3.5-9b-base-lora-tagged-28-fastretry",
    "taxonomy_version": "qwen-pii-27-safe-v1",
    "schema_version": "redaction-output-v1",
    "policy_id": "qwen-safe-default-v1",
    "normalization": "NFC",
    "raw_offset_mapping_applied": false,
    "created_at": "2026-04-22T10:00:00+00:00"
  },
  "warnings": [
    "confidence_uncalibrated_null"
  ],
  "model_output": "Please contact <pii type=\"PERSON\">Alice Nguyen</pii> at <pii type=\"EMAIL_ADDRESS\">alice.nguyen@example.edu.au</pii>.",
  "demo": {
    "backend": "live_model",
    "latency_ms": 1234.5
  }
}
```

### Notes

- `start` and `end` are Python-style character offsets into `input_text`, where `end` is exclusive.
- `confidence` is currently `null` because this tagged-generation route has not been calibrated yet.
- `decision` comes from `configs/policies/qwen-safe-default-v1.json`.
- `redacted_text` is produced deterministically from spans and policy.
- If model output does not round-trip exactly to the input text, the wrapper tries unique-value offset repair and reports warnings.

## GET /api/metrics-summary

Returns available benchmark summaries for demo display.

Example response:

```json
{
  "cleaned_200_optimized": {
    "available": true,
    "overall": {
      "precision": 0.9919447640966629,
      "recall": 0.96962879640045,
      "f1": 0.9806598407281002
    }
  },
  "random1000_seed42": {
    "available": true,
    "overall": {
      "precision": 0.971,
      "recall": 0.987792472024415,
      "f1": 0.9793242561775088
    },
    "rows_completed": 1000,
    "rows_total": 1000
  }
}
```

## Curl Examples

Health:

```bash
curl http://127.0.0.1:8090/api/health
```

Redact text:

```bash
curl -X POST http://127.0.0.1:8090/api/redact \
  -H "Content-Type: application/json" \
  -d '{"text":"Please contact Alice Nguyen at alice.nguyen@example.edu.au."}'
```

Use an example:

```bash
curl -X POST http://127.0.0.1:8090/api/redact \
  -H "Content-Type: application/json" \
  -d '{"example_id":"student-support"}'
```

## POST /api/redact-file

Uploads a file, extracts readable text, runs the same live redaction model, and returns the extracted text plus spans.

Supported demo inputs:

- PDF: uses the embedded text layer first; if no readable text is found, OCR is attempted on rendered pages.
- Images: `png`, `jpg`, `jpeg`, `tif`, `tiff`, `bmp`, `webp` through Tesseract OCR.
- Plain text files: `txt`, `md`, `csv`, `tsv`, `json`, `log`.

Example:

```bash
curl -X POST http://127.0.0.1:8090/api/redact-file \
  -F "file=@/path/to/document.pdf"
```

Important response fields:

```json
{
  "ocr_text": "Text extracted from the uploaded file...",
  "input_text": "Text extracted from the uploaded file...",
  "redacted_text": "Model policy redaction text",
  "spans": [
    {
      "start": 15,
      "end": 27,
      "type": "PERSON",
      "value": "Alice Nguyen"
    }
  ],
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
```

Demo limits are controlled by environment variables:

- `DEMO_MAX_UPLOAD_BYTES`, default `26214400`.
- `DEMO_MAX_PDF_OCR_PAGES`, default `8`.
- `DEMO_MAX_FILE_TEXT_CHARS`, default `12000`.
- `DEMO_MAX_TEXT_CHARS`, default `3000`.
- `DEMO_MAX_CONCURRENT_GENERATE`, default `4`.

## Runtime

Start or restart the live demo:

```bash
cd /home/admin/ZYX/Qwen3.5_9b_base_Distill/scripts
tmux kill-session -t redaction_demo_api 2>/dev/null || true
tmux new-session -d -s redaction_demo_api './run_redaction_demo_api.sh'
```

Logs:

```text
/home/admin/ZYX/Qwen3.5_9b_base_Distill/scripts/logs/redaction_demo_api_live_8090.log
```

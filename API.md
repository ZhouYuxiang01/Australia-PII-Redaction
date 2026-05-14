# Australia PII Redaction API Document

Version: 1.0  
Last updated: 2026-05-14

This document describes the customer-facing HTTP API for the Australia PII
Redaction service. The API detects Australian personally identifiable
information (PII), returns structured span metadata, and can optionally produce
a redacted version of the input.

## 1. Base URL

Production:

```text
https://demo.piics45one.com
```

Local development:

```text
http://127.0.0.1:8090
```

All endpoint paths in this document are relative to the selected base URL. For
example, the production health-check endpoint is
`https://demo.piics45one.com/api/health`.

## 2. Authentication

The current wrapper does not enforce authentication in code. If the service is
deployed outside a trusted network, authentication should be provided by the
hosting layer, such as an API gateway, reverse proxy, VPN, or identity-aware
proxy.

## 3. Content Types

| Endpoint | Request Content-Type | Response Content-Type |
|---|---|---|
| `GET /api/health` | none | `application/json` |
| `GET /api/examples` | none | `application/json` |
| `POST /api/redact` | `application/json` | `application/json` |
| `POST /api/redact-file` | `multipart/form-data` | `application/json` |

## 4. Endpoints

### 4.1 Health Check

```http
GET /api/health
```

Returns service status, backend information, active policy id, model version,
and schema version.

Example response:

```json
{
  "status": "ok",
  "backend": {
    "name": "hybrid-opf-qwen",
    "model_version": "opf-qwen-hybrid",
    "loaded": true,
    "supported_types": ["PERSON", "EMAIL_ADDRESS", "DATE_OF_BIRTH"]
  },
  "policy_id": "hybrid-80class-v2-4b",
  "model_version": "opf-qwen-hybrid",
  "schema_version": "redaction-output-v1"
}
```

### 4.2 Demo Examples

```http
GET /api/examples
```

Returns sample examples configured for the demo UI.

Example response:

```json
{
  "examples": [
    {
      "id": "student-support",
      "title": "Student Support",
      "text": "Student record: Olivia Okonkwo, DOB November 04, 1999..."
    }
  ]
}
```

### 4.3 Redact Text

```http
POST /api/redact
```

Runs PII detection and redaction on a text string.

Request body:

```json
{
  "text": "Please contact Alice Nguyen at alice.nguyen@example.edu.au."
}
```

You may also request one of the configured demo examples:

```json
{
  "example_id": "student-support"
}
```

If both `text` and `example_id` are supplied, `text` is used.

Example curl:

```bash
curl -X POST http://127.0.0.1:8090/api/redact \
  -H "Content-Type: application/json" \
  -d '{"text":"Please contact Alice Nguyen at alice.nguyen@example.edu.au."}'
```

Example response:

```json
{
  "redacted_text": "Please contact [PERSON] at [EMAIL_ADDRESS].",
  "spans": [
    {
      "start": 15,
      "end": 27,
      "type": "PERSON",
      "confidence": 0.98,
      "decision": "redact",
      "replacement": "[PERSON]",
      "source": "model",
      "postprocess": [],
      "risk_score": 0.91,
      "decision_reason": "confidence_based"
    },
    {
      "start": 31,
      "end": 59,
      "type": "EMAIL_ADDRESS",
      "confidence": 0.99,
      "decision": "redact",
      "replacement": "[EMAIL_ADDRESS]",
      "source": "model",
      "postprocess": []
    }
  ],
  "metadata": {
    "model_version": "opf-qwen-hybrid",
    "taxonomy_version": "au_pii_80class",
    "schema_version": "redaction-output-v1",
    "policy_id": "hybrid-80class-v2-4b",
    "normalization": "NFC",
    "raw_offset_mapping_applied": false,
    "redaction_mode": "replace_with_tag",
    "redact_review_types": [],
    "created_at": "2026-05-14T05:00:00+00:00",
    "backend_name": "hybrid-opf-qwen",
    "backend_route": "text_input",
    "latency_ms": 42.7,
    "input_length": 60
  },
  "warnings": [],
  "input_text": "Please contact Alice Nguyen at alice.nguyen@example.edu.au."
}
```

### 4.4 Redact File

```http
POST /api/redact-file
```

Uploads a file, extracts text, then runs the same PII detection and redaction
pipeline used by `/api/redact`.

Supported inputs:

| File type | Supported extensions / content |
|---|---|
| Plain text | `.txt`, `.text`, `.md`, `.csv`, `.tsv`, `.json`, `.log` |
| Images | `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp` |
| PDF | PDF text layer first; scanned pages fall back to visual text transcription |

Example curl:

```bash
curl -X POST http://127.0.0.1:8090/api/redact-file \
  -F "file=@/path/to/document.pdf"
```

File response fields are the same as `/api/redact`, with extra file and text
extraction metadata:

```json
{
  "redacted_text": "...",
  "spans": [],
  "metadata": {
    "model_version": "opf-qwen-hybrid",
    "taxonomy_version": "au_pii_80class",
    "schema_version": "redaction-output-v1",
    "policy_id": "hybrid-80class-v2-4b",
    "normalization": "NFC",
    "raw_offset_mapping_applied": false,
    "redaction_mode": "replace_with_tag",
    "backend_name": "hybrid-opf-qwen",
    "backend_route": "file_upload",
    "latency_ms": 812.3,
    "input_length": 1450,
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
  },
  "warnings": [],
  "input_text": "...",
  "ocr_text": "..."
}
```

`ocr_text` contains the extracted text used for PII detection. The name is kept
for UI compatibility; for text-layer PDFs and plain text uploads it means
"extracted input text", not necessarily OCR output.

## 5. Response Fields

### 5.1 Top-Level Response

| Field | Type | Description |
|---|---|---|
| `redacted_text` | string | Input text after applying automatic redactions. Review-only spans may remain visible unless the policy redacts review types. |
| `spans` | array | Detected PII spans and policy decisions. |
| `metadata` | object | Model, policy, backend, timing, file, and schema metadata. |
| `warnings` | array of strings | Non-fatal issues, such as text truncation or OCR cleanup. |
| `input_text` | string | NFC-normalized text used for detection. Present in API responses for UI highlighting. |
| `ocr_text` | string | File uploads only. Extracted file text used for detection. |

### 5.2 Span Object

| Field | Type | Description |
|---|---|---|
| `start` | integer | Inclusive character offset in `input_text`. |
| `end` | integer | Exclusive character offset in `input_text`. |
| `type` | string | PII category, such as `PERSON`, `EMAIL_ADDRESS`, `DATE_OF_BIRTH`, `AU_TFN`, `MEDICARE_NUMBER`, or `STUDENT_ID`. |
| `confidence` | number or null | Calibrated confidence when available. |
| `decision` | string | Policy decision. Common values are `redact`, `review`, `AUTO_REDACT`, `REVIEW`, `ignore`, or `analysis`, depending on backend/policy. |
| `replacement` | string | Replacement token used in redaction, for example `[PERSON]`. |
| `source` | string | Source of the span, such as `model`, `rule`, or `postprocess`. |
| `postprocess` | array | Post-processing rules applied to this span. |
| `risk_score` | number | Optional policy risk score. |
| `decision_reason` | string | Optional reason for the decision. |
| `type_distribution_topk` | array | Optional top-k type distribution from the classifier. |

Additional diagnostic fields may be included for hybrid backends, such as
`top1_prob`, `top3_sum`, `non_pii_prob`, `uncertainty`, `detector_source`,
`deterministic_evidence`, and `qwen_top_type`.

### 5.3 Metadata Object

| Field | Type | Description |
|---|---|---|
| `model_version` | string | Active model or backend model version. |
| `taxonomy_version` | string | PII taxonomy version. |
| `schema_version` | string | Response schema version. Current value: `redaction-output-v1`. |
| `policy_id` | string | Active redaction policy id. |
| `normalization` | string | Text normalization applied before inference. Current value: `NFC`. |
| `raw_offset_mapping_applied` | boolean | Whether offsets were mapped back to raw input after normalization. |
| `redaction_mode` | string | `replace_with_tag`, `mask`, or `remove`. |
| `redact_review_types` | array | Review types that are also redacted by policy. |
| `backend_name` | string | Active backend name. |
| `backend_route` | string | `text_input` or `file_upload`. |
| `latency_ms` | number | Server-side processing latency in milliseconds. |
| `input_length` | integer | Length of normalized input text. |
| `created_at` | string | ISO 8601 response timestamp. |
| `file` | object | File metadata for `/api/redact-file` responses. |
| `ocr` | object | Text extraction metadata for `/api/redact-file` responses. |

## 6. Error Responses

Errors use FastAPI's standard JSON shape:

```json
{
  "detail": "Error message"
}
```

Common status codes:

| Status | Cause |
|---:|---|
| `400` | Missing `text` / `example_id`, empty upload, or unsupported file type. |
| `413` | Text or file exceeds configured size limits. |
| `422` | File could not be read or no readable text was found. |
| `500` | Server configuration, dependency, model, or extraction failure. |
| `504` | External text extraction command timed out. |

## 7. Limits

Default server-side limits:

| Environment variable | Default | Description |
|---|---:|---|
| `WRAPPER_MAX_TEXT_CHARS` | `3000` | Maximum text length for `/api/redact`. |
| `WRAPPER_MAX_UPLOAD_BYTES` | `26214400` | Maximum upload size, 25 MB. |
| `WRAPPER_MAX_FILE_TEXT_CHARS` | `12000` | Maximum extracted text length from uploaded files. Longer text is truncated with a warning. |
| `WRAPPER_MAX_PDF_OCR_PAGES` | `8` | Maximum scanned PDF pages processed by visual transcription. |

## 8. Supported PII Types

The active backend returns its supported PII types in:

```http
GET /api/health
```

The current hybrid OPF + Qwen backend supports the following PII categories:

| PII Type |
|---|
| `ABORIGINALITY` |
| `ADDRESS` |
| `AUDIO_INFORMATION` |
| `AU_BANK_ACCOUNT` |
| `AU_DRIVERS_LICENCE` |
| `AU_PASSPORT` |
| `AU_TFN` |
| `CAMERA_FOOTAGE_AUDIO` |
| `CARING_RESPONSIBILITIES` |
| `CENTRELINK_REFERENCE_NUMBER` |
| `CITIZENSHIP_STATUS` |
| `CONTRACT_TYPE` |
| `COOKIE_INFORMATION` |
| `COUNSELLING_RECORDS` |
| `CREDIT_CARD_EXPIRY` |
| `CRIMINAL_RECORDS` |
| `DATE_OF_BIRTH` |
| `DEVICE_ID` |
| `DISABILITY_OR_SPECIFIC_CONDITION` |
| `EMAIL` |
| `EMPLOYEE_NUMBER` |
| `EMPLOYMENT_INFORMATION` |
| `FACIAL_RECOGNITION` |
| `FINGERPRINT` |
| `GENDER` |
| `GEOLOCATION_INFORMATION` |
| `IHI` |
| `IP_ADDRESS` |
| `LATITUDE` |
| `LONGITUDE` |
| `MARITAL_STATUS` |
| `MEDICAL_CERTIFICATE` |
| `MEDICAL_INFORMATION` |
| `MEDICARE_EXPIRY` |
| `MEDICARE_NUMBER` |
| `MILITARY_VETERAN_STATUS` |
| `NATIONALITY` |
| `NATIONAL_IDENTITY_CARD` |
| `NEXT_OF_KIN` |
| `PASSPORT_EXPIRY` |
| `PASSPORT_START_DATE` |
| `PAYMENT_CARD_NUMBER` |
| `PENSION_CARD_NUMBER` |
| `PERSON` |
| `PERSONAL_DEBT` |
| `PERSONNEL_NUMBER` |
| `PHONE` |
| `PRONOUN` |
| `RACIAL_ETHNIC_ORIGIN` |
| `RELIGION_BELIEF` |
| `SALARY` |
| `SALARY_WAGE_EXPECTATION` |
| `SANCTIONS` |
| `SCHOLARSHIP` |
| `SEXUAL_ORIENTATION` |
| `SIGNATURE` |
| `SOCIAL_MEDIA_ACCOUNT` |
| `SOCIAL_MEDIA_HISTORY` |
| `SOCIAL_MEDIA_ID` |
| `SOCIO_ECONOMIC_STATUS` |
| `SPECIAL_CONSIDERATION` |
| `STUDENT_ID` |
| `SUBJECT_RESULTS` |
| `UAC_ID` |
| `USERNAME` |
| `USI` |
| `VEHICLE_ID` |
| `VOICE_RECOGNITION` |
| `WAM_SCORE` |
| `WEBSITE_HISTORY` |
| `WORKERS_COMPENSATION_CLAIM` |
| `WORK_EMAIL` |
| `WORK_PHONE` |

For image and scanned-PDF uploads, the VLM is used to transcribe visible text
first. PII detection then runs on the transcribed text and supports the same
PII categories listed above.

## 9. Deployment Configuration

The service selects backend and policy files through environment variables:

```bash
export WRAPPER_BACKEND_CONFIG=/path/to/configs/backends/hybrid-opf-qwen9b-hn.json
export WRAPPER_POLICY_CONFIG=/path/to/configs/policies/hybrid-80class-v2-4b.json
./scripts/run_server.sh
```

Optional Qwen-VL text extraction settings:

| Environment variable | Description |
|---|---|
| `WRAPPER_QWEN_VL_MODEL` | Local model path used for image/scanned PDF transcription. |
| `WRAPPER_QWEN_VL_DEVICE` | Inference device, for example `cuda` or `cpu`. |
| `WRAPPER_QWEN_VL_DTYPE` | Model dtype, for example `bfloat16` or `float16`. |
| `WRAPPER_QWEN_VL_MAX_NEW_TOKENS` | Maximum generated transcription tokens. |

## 10. Notes for Integration

- Character offsets refer to the normalized `input_text` returned by the API.
- Clients should not reconstruct redaction by using byte offsets; use character
  offsets or the provided `redacted_text`.
- `REVIEW` / `review` spans should be shown to a human reviewer instead of
  being silently ignored.
- The API does not return raw hidden states or model prompts.
- Input text may contain sensitive information. Use TLS and avoid logging raw
  request bodies in production.

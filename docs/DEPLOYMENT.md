# Deployment and Operations

## Runtime Overview

The deployable service is a FastAPI application in `pii-redaction-service/`. It
uses the hybrid backend selected by default:

```text
pii-redaction-service/configs/backends/hybrid-opf-qwen9b-hn.json
pii-redaction-service/configs/policies/hybrid-80class-v2-4b.json
```

Primary endpoints:

```text
GET  /api/health
GET  /api/examples
POST /api/redact
POST /api/redact-file
GET  /docs
GET  /redoc
```

## Local Launch

```bash
cd pii-redaction-service
./scripts/run_server.sh
```

Open:

```text
http://127.0.0.1:8090/
http://127.0.0.1:8090/docs
```

## Docker Launch

From the repository root:

```bash
docker compose up --build
```

If the Qwen backbone is not at the default path:

```bash
QWEN_MODEL_PATH=/path/to/Qwen3.5-9B-Base docker compose up --build
```

The Docker setup expects NVIDIA GPU support on the host.

## Required Artifacts

```text
hybrid-pii-model-runtime/runs/opf_hard_79/
hybrid-pii-model-runtime/runs/qwen9b_hn_spancls_heads/last_linear/head.pt
/home/admin/model/Qwen3.5-9B-Base
```

The first two are project artifacts. The Qwen backbone is an external local
dependency and is not included in the repository.

## Environment Variables

| Variable | Default / example | Purpose |
|---|---|---|
| `REDACTION_PII_PROJECT_ROOT` | `/home/admin/ZYX/hybrid-pii-model-runtime` | Root for runtime artifacts and label schemas. |
| `REDACTION_QWEN_BACKBONE` | `/home/admin/model/Qwen3.5-9B-Base` | Local Qwen backbone path. |
| `WRAPPER_SERVICE_ROOT` | `/home/admin/ZYX/pii-redaction-service` | Service root for configs and post-processing files. |
| `WRAPPER_BACKEND_CONFIG` | `configs/backends/hybrid-opf-qwen9b-hn.json` | Backend config override. |
| `WRAPPER_POLICY_CONFIG` | `configs/policies/hybrid-80class-v2-4b.json` | Policy config override. |
| `WRAPPER_MAX_TEXT_CHARS` | `3000` | Maximum `/api/redact` text length. |
| `WRAPPER_MAX_UPLOAD_BYTES` | `26214400` | Maximum upload size, 25 MB. |
| `WRAPPER_MAX_FILE_TEXT_CHARS` | `12000` | Maximum extracted text length from uploaded files. |
| `WRAPPER_MAX_PDF_OCR_PAGES` | `8` | Maximum scanned-PDF pages for visual transcription. |
| `WRAPPER_QWEN_VL_MODEL` | `/home/admin/model/Qwen3.5-9B-Base` locally, `/models/Qwen3.5-9B-Base` in Docker | Image/scanned-PDF text transcription model. The delivered/default route uses Qwen 3.5 9B Base. |
| `WRAPPER_QWEN_VL_DEVICE` | `cuda` or `cpu` | Visual transcription device. |
| `WRAPPER_QWEN_VL_DTYPE` | `bfloat16` or `float16` | Visual transcription dtype. |

## Multimodal Input Path

Text inputs and text-layer PDFs are read directly. Images and scanned-PDF pages
are transcribed first using the configured Qwen image-text-to-text model. In the
delivered/default deployment, this is the same local Qwen 3.5 9B Base artifact:

```text
/home/admin/model/Qwen3.5-9B-Base
```

Docker maps that model to:

```text
/models/Qwen3.5-9B-Base
```

The extracted or transcribed text then enters the same OPF + Qwen span-head +
policy pipeline as normal text input.

## Health Check

```bash
curl http://127.0.0.1:8090/api/health
```

Expected response includes:

- `status: "ok"`;
- backend name and model version;
- active `policy_id`;
- `schema_version`;
- supported PII types.

## Smoke Test

```bash
curl -X POST http://127.0.0.1:8090/api/redact \
  -H "Content-Type: application/json" \
  -d '{"text":"Please contact Alice Nguyen at alice.nguyen@example.edu.au."}'
```

Expected behavior: the response includes structured spans and a `redacted_text`
value with replacement tags such as `[PERSON]` and `[EMAIL_ADDRESS]`.

## File Upload Test

```bash
curl -X POST http://127.0.0.1:8090/api/redact-file \
  -F "file=@/path/to/document.pdf"
```

Supported inputs include plain text, markdown, CSV/TSV/JSON/log files, images,
and PDFs. Text-layer PDFs are read directly; scanned pages and images use the
configured Qwen 3.5 9B image-text-to-text transcription path by default.

## Logs and Diagnostics

Service logs are written under:

```text
pii-redaction-service/scripts/logs/
```

Evaluation and comparison logs are under:

```text
pii-redaction-service/reports/
```

Training reports are under:

```text
pii_training_prep_v3_2/reports/
```

## Common Failure Modes

| Symptom | Likely cause | Action |
|---|---|---|
| Backend fails during startup | Qwen backbone path missing or inaccessible | Set `REDACTION_QWEN_BACKBONE` or `QWEN_MODEL_PATH`. |
| OPF loading fails | Runtime artifact directory missing | Verify `hybrid-pii-model-runtime/runs/opf_hard_79/`. |
| Qwen head loading fails | `head.pt` missing or wrong path | Verify `hybrid-pii-model-runtime/runs/qwen9b_hn_spancls_heads/last_linear/head.pt`. |
| CUDA or dtype error | GPU/runtime mismatch | Check NVIDIA driver, container runtime, PyTorch CUDA build, and dtype. |
| `400` response | Empty text, unsupported file type, or invalid request | Compare request with `API.md`. |
| `413` response | Text or upload too large | Adjust wrapper max-size environment variables. |
| `422` response | File could not be read or no text was extracted | Check file type, PDF text layer, or VLM availability. |
| Slow first request | Model loading and warm-up | Use `/api/health` after service startup and keep the process warm. |

## Production Notes

- Put the service behind TLS.
- Add authentication at the API gateway, reverse proxy, VPN, or identity-aware
  proxy layer.
- Avoid logging raw request bodies because inputs may contain sensitive
  information.
- Keep Qwen and OPF artifacts on secured local storage.
- Monitor latency, GPU memory, request size, and file extraction failures.
- Treat `review` decisions as work items for a human reviewer, not as ignored
  spans.

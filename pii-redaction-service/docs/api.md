# PII Redaction Service API

The service exposes one deployable hybrid backend selected by default:

```text
configs/backends/hybrid-opf-qwen9b-hn.json
configs/policies/hybrid-80class-v2-4b.json
```

## Base URL

```text
http://127.0.0.1:8090
```

Interactive docs are available at `/docs` and `/redoc`.

## Endpoints

- `GET /`: browser demo page.
- `GET /api/health`: service status, backend metadata, policy id, and schema version.
- `GET /api/examples`: demo example list.
- `POST /api/redact`: run PII detection and redaction on text.
- `POST /api/redact-file`: upload text, PDF, or image input, extract text, then run the same redaction pipeline.

## Launch

```bash
cd pii-redaction-service
./scripts/run_server.sh
```

Logs are written under `scripts/logs/`.

# PII Redaction Service

Deployable FastAPI service and web demo for Australian PII detection and redaction.

The bundled backend is the latest hybrid model route:

- OPF high-recall candidate span detection.
- Qwen 3.5 9B hard-negative span-head type scoring.
- Risk policy, deterministic post-processing, and redaction output.
- Text, PDF, and image input through the same API/demo surface.

## Quick Start

```bash
cd pii-redaction-service
./scripts/run_server.sh
```

Then open:

```text
http://127.0.0.1:8090/
http://127.0.0.1:8090/docs
```

The default config is:

```text
configs/backends/hybrid-opf-qwen9b-hn.json
configs/policies/hybrid-80class-v2-4b.json
```

## Docker Deployment

From the repository root:

```bash
docker compose up --build
```

Then open:

```text
http://127.0.0.1:8090/
http://127.0.0.1:8090/docs
```

The compose file expects the Qwen backbone at `/home/admin/model/Qwen3.5-9B-Base`.
Override it when needed:

```bash
QWEN_MODEL_PATH=/path/to/Qwen3.5-9B-Base docker compose up --build
```

The service uses NVIDIA GPUs by default. Make sure the host has the NVIDIA
container runtime installed and that the local runtime artifacts are present:

```text
hybrid-pii-model-runtime/runs/opf_hard_79/
hybrid-pii-model-runtime/runs/qwen9b_hn_spancls_heads/last_linear/head.pt
```

If the current user cannot access Docker directly, use `sudo docker compose ...`
or re-login after adding the user to the `docker` group.

## Public Domain

The API documentation uses this public demo base URL:

```text
https://demo.piics45one.com
```

The backend itself listens on port `8090`. A reverse proxy should route the
public domain to `http://127.0.0.1:8090` or to the Docker-published `8090`
port, with TLS terminated at the proxy.

Local base URL:

```text
http://127.0.0.1:8090
```

## Testing

Lightweight tests do not load the large models:

```bash
PYTHONPATH=$PWD/pii-redaction-service:$PWD/opf-runtime pytest -q pii-redaction-service/tests
```

## Runtime Layout

```text
pii-redaction-service/
├── redaction/                 FastAPI app, schema builders, policy, OCR/file input
├── configs/                   latest hybrid backend and policy configs
├── schemas/                   API response schema
├── static/                    browser demo
├── scripts/run_server.sh      backend launcher
└── tests/                     lightweight service tests
```

The service expects companion runtime files in `../hybrid-pii-model-runtime`.
Large local artifacts such as OPF checkpoints and Qwen head weights remain
ignored by Git; the config records their expected paths.

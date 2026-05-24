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

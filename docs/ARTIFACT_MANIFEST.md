# Artifact Manifest

This manifest lists the model and configuration artifacts required by the final
hybrid backend.

## Runtime Model Artifacts

| Artifact | Path | Size | SHA-256 | Required |
|---|---|---:|---|---|
| OPF checkpoint directory | `hybrid-pii-model-runtime/runs/opf_hard_79/` | 2.7 GB | Directory; see files below | Yes |
| OPF model weights | `hybrid-pii-model-runtime/runs/opf_hard_79/model.safetensors` | 2.7 GB | `2f0ce5797197bbd5cd8f1f09d37d21167d76e583bac421e1f86cd98274e5bfd4` | Yes |
| OPF config | `hybrid-pii-model-runtime/runs/opf_hard_79/config.json` | 11 KB | `aa1e0523e3b59a9f30113e3c3a9eedd89eaae4b9fbbdbe9f7c0ac354aa28d027` | Yes |
| OPF finetune summary | `hybrid-pii-model-runtime/runs/opf_hard_79/finetune_summary.json` | 4.0 KB | `306cd97efb07e29e960e835bac1872ee8c3658d7c813adfa1eb917062260e1bf` | Recommended |
| Qwen 9B span-head | `hybrid-pii-model-runtime/runs/qwen9b_hn_spancls_heads/last_linear/head.pt` | 1.3 MB | `91c459b8992ed2eef055920ab6f79fe61e409907b49e4a556838457a383edf03` | Yes |

## External Model Dependency

| Dependency | Default path | Required | Notes |
|---|---|---|---|
| Qwen 3.5 9B Base backbone and default image-text-to-text model | `/home/admin/model/Qwen3.5-9B-Base` | Yes | Not included in this repository. Used by both `REDACTION_QWEN_BACKBONE` and `WRAPPER_QWEN_VL_MODEL` in the delivered/default route. |
| Optional alternate vision-language model | Set by `WRAPPER_QWEN_VL_MODEL` | No | Override only if deployment intentionally uses a different image/scanned-PDF transcription model. |

## Schema and Label Artifacts

| Artifact | Path | Purpose |
|---|---|---|
| OPF label space | `hybrid-pii-model-runtime/pii_schema/opf_label_space_79.json` | Token/span labels for OPF detector. |
| Qwen training label space | `hybrid-pii-model-runtime/pii_schema/training_label_space_80.json` | Span-head labels including `NON_PII`. |
| API response schema | `pii-redaction-service/schemas/redaction-output-v1.schema.json` | Customer-facing JSON response contract. |

## Runtime Configuration

| Config | Path | Purpose |
|---|---|---|
| Backend config | `pii-redaction-service/configs/backends/hybrid-opf-qwen9b-hn.json` | Selects OPF checkpoint, Qwen backbone, Qwen head, label spaces, thresholds, dtype, and device. |
| Policy config | `pii-redaction-service/configs/policies/hybrid-80class-v2-4b.json` | Defines redact/review/pass actions, confidence thresholds, post-processing, and overlap handling. |
| Per-label thresholds | `pii-redaction-service/configs/postprocess/per_label_thresholds.json` | Optional per-label post-processing thresholds. |

## Loading Path

The default service reads artifacts using:

```text
REDACTION_PII_PROJECT_ROOT=/home/admin/ZYX/hybrid-pii-model-runtime
REDACTION_QWEN_BACKBONE=/home/admin/model/Qwen3.5-9B-Base
WRAPPER_QWEN_VL_MODEL=/home/admin/model/Qwen3.5-9B-Base
WRAPPER_SERVICE_ROOT=/home/admin/ZYX/pii-redaction-service
```

To launch locally:

```bash
cd pii-redaction-service
./scripts/run_server.sh
```

To launch with Docker from the repository root:

```bash
QWEN_MODEL_PATH=/path/to/Qwen3.5-9B-Base docker compose up --build
```

## Verification Commands

```bash
sha256sum \
  hybrid-pii-model-runtime/runs/opf_hard_79/model.safetensors \
  hybrid-pii-model-runtime/runs/opf_hard_79/config.json \
  hybrid-pii-model-runtime/runs/opf_hard_79/finetune_summary.json \
  hybrid-pii-model-runtime/runs/qwen9b_hn_spancls_heads/last_linear/head.pt
```

```bash
curl http://127.0.0.1:8090/api/health
```

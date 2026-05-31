# Solution Deliverables

This handover index maps the customer-requested deliverables to the repository
files that provide implementation, operation, and evaluation evidence.

## D1 - Trained Model

| Requirement | Evidence |
|---|---|
| Model artefacts and how to load them | [ARTIFACT_MANIFEST.md](ARTIFACT_MANIFEST.md), [MODEL_CARD.md](MODEL_CARD.md), [../hybrid-pii-model-runtime/runs/README.md](../hybrid-pii-model-runtime/runs/README.md) |
| Architecture and configuration | [MODEL_CARD.md](MODEL_CARD.md), [../README.md](../README.md), [../pii-redaction-service/configs/backends/hybrid-opf-qwen9b-hn.json](../pii-redaction-service/configs/backends/hybrid-opf-qwen9b-hn.json) |
| Size and resource footprint | [ARTIFACT_MANIFEST.md](ARTIFACT_MANIFEST.md), [MODEL_CARD.md](MODEL_CARD.md) |
| Training data summary | [MODEL_CARD.md](MODEL_CARD.md), [../pii_training_prep_v3_2/README.md](../pii_training_prep_v3_2/README.md), [../pii_training_prep_v3_2/reports/stage3_split_report.json](../pii_training_prep_v3_2/reports/stage3_split_report.json) |
| Dataset preparation steps | [../pii_training_prep_v3_2/README.md](../pii_training_prep_v3_2/README.md) |
| Finetuning steps | [../pii_training_prep_v3_2/README.md](../pii_training_prep_v3_2/README.md), [../hybrid-pii-model-runtime/runs/opf_hard_79/finetune_summary.json](../hybrid-pii-model-runtime/runs/opf_hard_79/finetune_summary.json) |
| Intended use and limitations | [MODEL_CARD.md](MODEL_CARD.md), [EVALUATION_REPORT.md](EVALUATION_REPORT.md) |
| Licence | [MODEL_CARD.md](MODEL_CARD.md) |

## D2 - Inference Wrapper

| Requirement | Evidence |
|---|---|
| API surface and versioning | [../API.md](../API.md), [../pii-redaction-service/docs/api.md](../pii-redaction-service/docs/api.md) |
| Input and output schema | [../API.md](../API.md), [../pii-redaction-service/schemas/redaction-output-v1.schema.json](../pii-redaction-service/schemas/redaction-output-v1.schema.json) |
| Industry-standard API documentation | FastAPI OpenAPI at `/docs`; static summary in [../API.md](../API.md) |
| Example usage | [../API.md](../API.md), [../pii-redaction-service/README.md](../pii-redaction-service/README.md) |
| Error handling and failure modes | [../API.md](../API.md), [DEPLOYMENT.md](DEPLOYMENT.md), [EVALUATION_REPORT.md](EVALUATION_REPORT.md) |

## D3 - Reproducible Pipeline

| Requirement | Evidence |
|---|---|
| Data generation and labelling | [../pii_training_prep_v3_2/README.md](../pii_training_prep_v3_2/README.md), [../pii_training_prep_v3_2/data/README.md](../pii_training_prep_v3_2/data/README.md) |
| Training procedure | [../pii_training_prep_v3_2/README.md](../pii_training_prep_v3_2/README.md) |
| Evaluation procedure | [EVALUATION_REPORT.md](EVALUATION_REPORT.md), [../pii-redaction-service/reports/wrapper_hybrid_full_eval.log](../pii-redaction-service/reports/wrapper_hybrid_full_eval.log) |
| Environment specification | [DEPLOYMENT.md](DEPLOYMENT.md), [../pii-redaction-service/README.md](../pii-redaction-service/README.md), [../requirements-docker.txt](../requirements-docker.txt), [../Dockerfile](../Dockerfile), [../docker-compose.yml](../docker-compose.yml) |
| End-to-end reproduction instructions | [../pii_training_prep_v3_2/README.md](../pii_training_prep_v3_2/README.md), [DEPLOYMENT.md](DEPLOYMENT.md) |

## D4 - Evaluation Report

| Requirement | Evidence |
|---|---|
| Methodology and dataset description | [EVALUATION_REPORT.md](EVALUATION_REPORT.md) |
| Quantitative results against teacher/baseline | [EVALUATION_REPORT.md](EVALUATION_REPORT.md), [../README.md](../README.md) |
| Robustness and edge-case analysis | [EVALUATION_REPORT.md](EVALUATION_REPORT.md) |
| Operational characteristics | [EVALUATION_REPORT.md](EVALUATION_REPORT.md), [DEPLOYMENT.md](DEPLOYMENT.md) |
| Confirmation against success criteria | [EVALUATION_REPORT.md](EVALUATION_REPORT.md) |

## Customer Review Order

1. Start with [../README.md](../README.md) for the system overview.
2. Open [README.md](README.md) in this directory for the full handover package index, including API documentation, final report, and presentation deck.
3. Use this file to check every requested deliverable.
4. Read [MODEL_CARD.md](MODEL_CARD.md) and [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for model and metric evidence.
5. Use [DEPLOYMENT.md](DEPLOYMENT.md) and [../API.md](../API.md) for integration and operations.

# Model Card

## Model Summary

The delivered model is a hybrid Australian PII redaction backend:

```text
OPF candidate detector + Qwen 3.5 9B hard-negative span-head + risk/redaction policy
```

It is deployed through the FastAPI wrapper in `pii-redaction-service/`. The model
does not use Qwen as a free-form generator for final redaction output. OPF first
locates candidate spans, Qwen hidden states are used by a lightweight span
classifier to score candidate PII types, and a deterministic policy layer makes
the final `redact`, `review`, or `ignore` decision.

## Intended Use

The model is intended for Australian PII detection and redaction in:

- plain text;
- text files and structured text files;
- PDFs with an embedded text layer;
- images and scanned PDFs after visual text transcription.

The system is designed for privacy review workflows where uncertain spans should
be escalated to human review instead of being silently discarded.

## Out-of-Scope Use

The model should not be treated as:

- a legal compliance guarantee;
- a replacement for final human review in high-risk workflows;
- a universal PII detector for jurisdictions or taxonomies not covered by the
  Australian label schema;
- a cryptographic anonymisation system;
- a model that understands non-textual document semantics beyond transcribed
  visible text.

## Architecture

| Component | Role | Main artifact/config |
|---|---|---|
| OPF / Privacy Filter | High-recall span candidate detection over full text | `hybrid-pii-model-runtime/runs/opf_hard_79/` |
| Qwen 3.5 9B Base | Frozen backbone for span embeddings and type calibration | `REDACTION_QWEN_BACKBONE`, default `/home/admin/model/Qwen3.5-9B-Base` |
| Qwen span-head | Lightweight classifier over OPF/fallback candidate spans | `hybrid-pii-model-runtime/runs/qwen9b_hn_spancls_heads/last_linear/head.pt` |
| Policy layer | Thresholding, risk, deterministic rescue rules, overlap handling, final redaction | `pii-redaction-service/configs/policies/hybrid-80class-v2-4b.json` |
| API wrapper | FastAPI service, file input, response schema, web demo | `pii-redaction-service/` |

Runtime backend config:

```text
pii-redaction-service/configs/backends/hybrid-opf-qwen9b-hn.json
```

Policy config:

```text
pii-redaction-service/configs/policies/hybrid-80class-v2-4b.json
```

## Label Space

The OPF detector uses a 79-class Australian PII span label space plus `O` for
non-entity tokens:

```text
hybrid-pii-model-runtime/pii_schema/opf_label_space_79.json
```

The Qwen span-head uses an 80-class training label space that includes `NON_PII`
for candidate-level hard negatives:

```text
hybrid-pii-model-runtime/pii_schema/training_label_space_80.json
```

The API reports the active supported categories from:

```http
GET /api/health
```

## Training Data Summary

The final reproducible training route is documented in
`pii_training_prep_v3_2/README.md`. It includes source data, converted teacher
outputs, hard negatives, dataset construction scripts, OPF-format data, and Qwen
span-classification data.

Stage-3 OPF dataset:

| Split | Records | Validation errors | Offset mismatches |
|---|---:|---:|---:|
| Train | 81,334 | 0 | 0 |
| Dev | 10,046 | 0 | 0 |
| Test | 9,671 | 0 | 0 |

Stage-3 Qwen span-head dataset:

| Split | Span examples | Validation errors |
|---|---:|---:|
| Train | 113,354 | 0 |
| Dev | 13,850 | 0 |
| Test | 13,588 | 0 |

The span-head training data includes direct PII spans, teacher-derived soft
targets, candidate-level hard negatives, document-level negatives, and ranking
examples. `NON_PII` appears only in the Qwen span-head training space, not in the
OPF span label space.

## Training Procedure

OPF:

- base checkpoint: `/home/admin/.opf/privacy_filter`;
- label space: `au_pii_79_v1`;
- epochs: 1;
- learning rate: `1e-5`;
- serialized parameter dtype: `bfloat16`;
- best validation loss: `0.03343182026375741`;
- validation token accuracy: `0.9897951856971046`.

Qwen span-head:

- backbone: `/home/admin/model/Qwen3.5-9B-Base`;
- hidden size: 4096;
- maximum input length during caching: 1536;
- backbone parameters frozen;
- selected head: `last_linear`;
- temperature: `1.004767`;
- selected checkpoint: `hybrid-pii-model-runtime/runs/qwen9b_hn_spancls_heads/last_linear/head.pt`.

Reproduction commands are in `pii_training_prep_v3_2/README.md`.

## Evaluation Summary

Full details are in [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

Qwen 9B hard-negative span-head:

| Split | Span examples | Micro F1 | Macro F1 | Weighted F1 | Top-1 acc. | Top-3 acc. | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 113,354 | 0.9864 | 0.9744 | 0.9860 | 0.9864 | 0.9995 | 0.0733 |
| Dev | 13,850 | 0.9863 | 0.9742 | 0.9860 | 0.9863 | 0.9999 | 0.0853 |
| Test | 13,588 | 0.9875 | 0.9755 | 0.9872 | 0.9875 | 0.9998 | 0.0771 |

End-to-end wrapper evaluation on 9,659 test records:

| Metric | Value |
|---|---:|
| Exact precision / recall / F1 | 0.7684 / 0.9056 / 0.8314 |
| Overlap precision / recall / F1 | 0.8312 / 0.9741 / 0.8970 |
| Type accuracy on overlap | 0.9863 |
| Latency p50 / p95 | 0.153s / 0.3084s |

## Resource Footprint

| Artifact | Size |
|---|---:|
| OPF checkpoint directory | 2.7 GB |
| Qwen span-head checkpoint | 1.3 MB |
| Qwen 3.5 9B Base backbone | External local dependency, not included in this repo |

Runtime defaults use CUDA and `bf16` for the hybrid backend. Large models are not
loaded by the lightweight unit tests.

## Limitations

- Exact-boundary F1 is lower than overlap F1 because boundary alignment remains
  difficult for addresses, names, document fragments, and noisy OCR-like text.
- Ambiguous numeric strings may require context or deterministic rules to avoid
  false positives.
- Low-confidence or conflicting spans are intentionally sent to review, which can
  increase human-review volume.
- Image and scanned-PDF performance depends on the quality of visual text
  transcription before PII detection.
- The model was designed around the included Australian PII taxonomy and should
  be re-evaluated before being used for materially different jurisdictions or
  document classes.

## Licence and Third-Party Models

Project-created code, configuration, and lightweight classifier artifacts should
be distributed according to the project or client agreement. The delivered system
also depends on third-party model assets whose licences must be reviewed before
redistribution:

- OPF / Privacy Filter: `openai/privacy-filter`;
- Qwen 3.5 9B Base: `Qwen/Qwen3.5-9B-Base`.

The Qwen backbone is not included in this repository and must be supplied in the
deployment environment.

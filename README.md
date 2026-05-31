# Australia PII Redaction

This repository contains the deployable mainline for automatic Australian PII identification and redaction. The current `main` branch is intentionally trimmed to the latest hybrid route:

```text
OPF candidate detector + Qwen 3.5 9B hard-negative span-head + risk/redaction policy
```

Older training routes, ablations, historical reports, and previous model experiments are preserved in the `legacy-models-archive` branch.

## Repository Structure

- `pii-redaction-service/`: deployable FastAPI service, web demo, post-processing policy, backend config, API schema, file input, and tests.
- `hybrid-pii-model-runtime/`: minimal runtime support for the latest hybrid backend, including label spaces, risk metadata, Qwen span-head inference code, and expected local artifact paths.
- `opf-runtime/`: OPF runtime package used by the hybrid service when `opf` is not already installed in the selected Python environment.
- `pii_training_prep_v3_2/`: reproducibility-focused training pipeline for the final hybrid route, including stage-3 dataset construction, Qwen span embedding cache, Qwen span-head training, model selection, and lightweight tests.
- `docs/`: customer handover documents mapping the requested deliverables to model, API, deployment, artifact, and evaluation evidence.
- `API.md`: customer-facing API description.

## Customer Handover Documents

For customer review and deliverable sign-off, start with:

- `docs/README.md`: handover package index, including links to API documentation, final report, and presentation deck.
- `docs/DELIVERABLES.md`: checklist mapping D1-D4 requirements to repository evidence.
- `docs/MODEL_CARD.md`: trained model architecture, intended use, training data, metrics, limitations, and licence notes.
- `docs/EVALUATION_REPORT.md`: evaluation methodology, datasets, component metrics, end-to-end wrapper results, robustness, and success criteria.
- `docs/ARTIFACT_MANIFEST.md`: required runtime artifacts, sizes, checksums, paths, and loading instructions.
- `docs/DEPLOYMENT.md`: local/Docker launch, environment variables, health checks, smoke tests, and operations notes.

## Open-Source Model References

The current mainline builds on:

- OPF / Privacy Filter: [openai/privacy-filter](https://huggingface.co/openai/privacy-filter), used as the first-stage token-classification / sequence-labeling model adapted to the AU PII label space.
- Qwen 3.5 9B Base: [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base), used as the frozen backbone for the Qwen span-head that estimates type probabilities for candidate spans. Qwen multimodal capability is also used for image/scanned-PDF text transcription.

## Current Mainline Approach

The deployment route is in `pii-redaction-service/` and uses an OPF + Qwen 9B hybrid backend. It is not a single end-to-end generative model that directly writes redacted output. Instead, it is a layered PII detection system:

1. OPF recalls possible PII-like spans.
2. Qwen span-head estimates candidate type probabilities and top-k distributions.
3. Policy/rules decide whether each span should be automatically redacted, reviewed, or ignored.
4. The service returns structured JSON plus deterministic redacted text.

The system is designed around a privacy-safe three-way decision:

- `REDACT`: high confidence; redact automatically.
- `REVIEW`: likely PII, but evidence is not strong enough; send to human review.
- `IGNORE`: ignore only when the system is confident the span is not PII.

The core principle is to prefer additional human review over incorrectly ignoring true PII.

## Stage 1: OPF Candidate Detection

OPF is the first-layer token-classification / sequence-labeling model. Its job is high-recall candidate detection rather than final decision-making.

At implementation level, OPF tokenizes input text, outputs a label distribution for each token, and uses sequence decoding to merge consecutive tokens into entity spans. The runtime label space is:

```text
hybrid-pii-model-runtime/pii_schema/opf_label_space_79.json
```

It contains:

- `O`: non-entity token.
- 79 AU PII span classes, including `PERSON`, `DATE_OF_BIRTH`, `EMAIL_ADDRESS`, `MOBILE`, `ADDRESS`, `AU_TFN`, `MEDICARE_NUMBER`, `STUDENT_ID`, `UAC_ID`, and `USI`.

OPF's span label space does not include `NON_PII`. `NON_PII` is used only by the downstream Qwen span-head candidate classifier. For OPF, ordinary text and hard negatives are learned through the `O` label, meaning "do not produce a span".

In the hybrid pipeline, OPF is responsible for:

- Locating possible PII spans in full text.
- Providing initial type and character offsets.
- Supplying candidate boundaries for the Qwen span-head and policy layer.

Historical OPF-format data scale from the archived reports:

- train: 81,298 records, 109,877 PII spans, 64,387 empty-span hard negatives.
- dev: 10,034 records, 13,412 PII spans, 7,977 empty-span hard negatives.
- test: 9,659 records, 13,513 PII spans, 7,557 empty-span hard negatives.
- Data validation: 0 offset mismatches and 0 labels outside the label space.

The full OPF training/data-prep code is archived in `legacy-models-archive` under `pii_training_prep_v3_2/`.

## Stage 2: Qwen Span-Head Type Calibration

The Qwen component uses `Qwen/Qwen3.5-9B-Base` as a frozen backbone plus a lightweight span-level classification head. The service does not ask Qwen to freely generate final redaction JSON; it uses Qwen hidden states to classify candidate spans produced by OPF and fallback regex/context rules.

Each candidate contains:

- original text
- span `start/end`
- span value
- candidate source
- type distribution / top-k probabilities

The archived training route cached span-related hidden states from Qwen:

- backbone: `/home/admin/model/Qwen3.5-9B-Base`
- hidden size: 4096
- max length: 1536
- backbone parameters frozen
- train/dev/test cached span examples: 113,354 / 13,850 / 13,588

The selected head is the `last_linear` classifier, trained on soft target distributions rather than only one-hot labels. This lets teacher data and rule-derived data express primary and secondary type probabilities.

Current 9B hard-negative span-head result from archived reports:

- best epoch: 11
- best dev NLL: 0.084690
- temperature: 1.004767
- test top-1 accuracy after temperature: 0.987489
- test top-3 accuracy after temperature: 0.999779
- test NLL after temperature: 0.077113

The lightweight head artifact path is:

```text
hybrid-pii-model-runtime/runs/qwen9b_hn_spancls_heads/last_linear/head.pt
```

This file contains only the span classification head trained in this project. It does not include the original Qwen 9B backbone.

## Metrics

The following metrics come from historical training/evaluation reports preserved in `legacy-models-archive`.

### OPF Span Detection

| Split | Records | PII spans | Detection precision | Detection recall | Detection F1 | Span precision | Span recall | Span F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dev | 10,034 | 13,412 | 0.9631 | 0.9817 | 0.9724 | 0.9437 | 0.9782 | 0.9606 |
| Test | 9,659 | 13,513 | 0.9745 | 0.9842 | 0.9793 | 0.9675 | 0.9775 | 0.9725 |

OPF is effective as a first-stage high-recall detector, but per-label weaknesses still exist, especially for ambiguous numeric/contextual fields. The downstream Qwen head and policy layer are therefore still required.

### Qwen 9B HN Span-Head Classification

| Split | Span examples | Micro F1 | Macro F1 | Weighted F1 | Top-1 acc. | Top-3 acc. | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 113,354 | 0.9864 | 0.9744 | 0.9860 | 0.9864 | 0.9995 | 0.0733 |
| Dev | 13,850 | 0.9863 | 0.9742 | 0.9860 | 0.9863 | 0.9999 | 0.0853 |
| Test | 13,588 | 0.9875 | 0.9755 | 0.9872 | 0.9875 | 0.9998 | 0.0771 |

For this single-label multi-class task, micro precision/recall/F1 is equivalent to top-1 accuracy. Macro F1 better reflects smaller-class performance.

### Historical End-to-End Wrapper Comparisons

These are historical comparisons from earlier wrapper ablations and are not presented as the final full-eval score for the current 9B hard-negative head.

| Route / report | Exact precision | Exact recall | Exact F1 | Overlap precision | Overlap recall | Overlap F1 | Type accuracy on overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| OPF-only wrapper baseline | 0.7249 | 0.7715 | 0.7475 | 0.8581 | 0.9081 | 0.8824 | - |
| 9B hybrid legacy / causal_lm ablation | 0.7315 | 0.8807 | 0.7992 | 0.8149 | 0.9756 | 0.8880 | 0.9674 |
| 4B calibrated + OPF bias ablation | 0.7395 | 0.8047 | 0.7707 | 0.8713 | 0.9428 | 0.9056 | 0.9584 |

These results show why the project moved toward a hybrid route: OPF gives recall and stable span boundaries, while Qwen improves semantic type judgment.

## Policy and Post-Processing

`pii-redaction-service/` is the final decision layer. It combines:

- whether OPF detected a span
- Qwen span-head top-1 type and probability
- top-k distribution concentration or ambiguity
- expected surface format for a PII type
- surrounding context
- confidentiality level and false-positive risk
- deterministic rescue rules for stable identifier formats

The policy config is:

```text
pii-redaction-service/configs/policies/hybrid-80class-v2-4b.json
```

The policy's role is to keep generalizable judgment rules outside the model and make behavior explainable. For example:

- Fixed-format, high-confidence fields can be automatically redacted.
- Low-probability spans, type conflicts, insufficient context, or suspicious formats go to review.
- Only clearly non-PII spans are ignored.

Some fields have stable formats, such as UAC IDs, TFNs, Medicare numbers, passport numbers, bank account numbers, credit card numbers, email addresses, and phone numbers. For these fields, deterministic regex/context evidence may be used. If the frontend displays `100%` for deterministic evidence, it means "rule-based high-confidence evidence", not that the neural model literally output 100%.

## Risk and Human Review

Risk is not simply model uncertainty. It is the policy layer's combined estimate of whether a decision needs human review. Common review triggers include:

- low top-1 probability or diffuse top-k distribution
- disagreement between OPF and Qwen types
- PII-like span without enough supporting context
- high false-positive-risk types, such as numeric strings, internal IDs, phone numbers, addresses, IPs, and vehicle plates
- incomplete span boundaries or abnormal cross-field merges

The goal is to preserve uncertain content for review rather than discard it.

## File Input and Multimodal Text Transcription

The service supports text, image, and PDF inputs:

- Text files: decoded directly as text.
- PDFs with a text layer: embedded PDF text is read first.
- Images or scanned PDFs: pages/images are sent to Qwen multimodal text transcription.
- Extracted/transcribed text then enters the same OPF + Qwen span-head + policy pipeline.

Some `ocr` field names remain in code for API compatibility. In the current route, their meaning is closer to "file text extraction / visual text transcription" than a separate OCR-model route.

## Why Hybrid Instead of One Model

Using OPF alone has limited contextual and semantic type judgment. Using Qwen alone as a free-form generative redaction model has higher latency, weaker format stability, and lower controllability.

The hybrid route separates concerns:

- OPF handles fast scanning and high recall.
- Qwen span-head handles semantic type calibration.
- Policy handles thresholds, confidentiality, review/ignore/redact decisions, deterministic rules, and frontend display.

This is more suitable for deployment because speed is controlled, output is structured, errors are easier to diagnose, and the design better matches privacy redaction requirements.

## Running the Service

For local, Docker, and deployment details, see:

```text
pii-redaction-service/README.md
```

The shortest local launch path is:

```bash
cd pii-redaction-service
./scripts/run_server.sh
```

The service provides:

- `GET /api/health`
- `POST /api/redact`
- `POST /api/redact-file`
- `/` web demo
- `/docs` FastAPI docs

## Reproducing Training

The final hybrid training pipeline is kept in `main` so the deployed artifacts
can be reproduced without switching branches:

```text
pii_training_prep_v3_2/
```

That directory now includes the compact reproducibility inputs for the final
route, including the synthetic 19,000-record source dataset and converted
stage-2 teacher rows. It also includes the held-out 1,000-record external test
dataset requested for customer review. Large regenerated artifacts remain
outside Git.

It contains the scripts for:

- converting the 19k source data into stage-1 span distributions;
- canonicalizing labels against the AU PII taxonomy;
- merging augmented and hard-negative teacher rows;
- building OPF and Qwen span-classification splits;
- caching frozen Qwen 3.5 9B span embeddings;
- training and selecting the Qwen 9B hard-negative span-head;
- preparing OPF-format data for `opf train`.

The Qwen backbone is frozen. The trained artifact is only the lightweight
span-head at:

```text
hybrid-pii-model-runtime/runs/qwen9b_hn_spancls_heads/last_linear/head.pt
```

Full commands are documented in:

```text
pii_training_prep_v3_2/README.md
```

## Runtime Artifacts and Large Files

Large model artifacts are intentionally ignored by Git. The latest hybrid config expects:

- `hybrid-pii-model-runtime/runs/opf_hard_79/`
- `hybrid-pii-model-runtime/runs/qwen9b_hn_spancls_heads/last_linear/head.pt`
- Qwen backbone at `/home/admin/model/Qwen3.5-9B-Base`, or set `REDACTION_QWEN_BACKBONE`.

The Qwen span-head is lightweight and may be versioned. The OPF checkpoint and original Qwen backbone are large and should be prepared separately on the deployment machine.

## Legacy Models and Training Code

Earlier model routes, historical reports, generated training data, and old experiment workspaces are preserved in:

```bash
git switch legacy-models-archive
```

That branch includes:

- `Qwen3.5_9b_base_Distill/`: earlier Qwen 9B distillation/redaction experiments, evaluation scripts, and demo API code.
- `Qwen3.5_4b_base_Full_73class/`: earlier Qwen 4B full supervised 73-class route.
- `Qwen3_4b_instruct_Distill/`: earlier Qwen 3 4B instruct distillation route.
- `opf_au_pii/`: standalone OPF training/evaluation route, AU PII label space, taxonomy configs, and OPF training utilities.
- `pii_training_prep_v3_2/`: the full historical backend training workspace, including teacher data, calibration workflows, reports, caches, and earlier variants. The final reproducibility pipeline is also present in `main`.
- `redaction-wrapper/`: previous service name before it was renamed to `pii-redaction-service/`.

## Notes

This repository is a project codebase and experiment-history snapshot. It should not contain private production data or unnecessary large model artifacts. Before public presentation or release, re-check local ignored files, report content, and repository visibility.

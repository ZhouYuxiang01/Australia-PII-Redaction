# Australia PII Redaction

This repository is a code and experiment snapshot for automatic Australian PII identification and redaction. The currently recommended approach is a hybrid `OPF + Qwen span-head + wrapper policy` pipeline: OPF performs high-recall span detection, the Qwen span-head performs type probability estimation, Qwen's multimodal capability transcribes images and scanned PDFs, and the wrapper handles rule-based post-processing, web display, file input, and human-review decisions.

## Repository Structure

- `redaction-wrapper/`: deployable FastAPI wrapper, web demo, post-processing policy, backend configs, and API tests.
- `pii_training_prep_v3_2/`: current main backend training and data preparation code, including OPF/Qwen span-head training, evaluation, teacher data, and calibration workflows.
- `opf_au_pii/`: standalone OPF training/evaluation route, AU PII label space, and taxonomy configs.
- `Qwen3.5_9b_base_Distill/`: earlier Qwen 9B distillation/redaction experiments, evaluation scripts, and demo API code.
- `Qwen3.5_4b_base_Full_73class/`: earlier Qwen 4B full supervised 73-class route.
- `Qwen3_4b_instruct_Distill/`: earlier Qwen 3 4B instruct distillation route.
- `reports/`: project plans, feasibility analyses, and selected experiment records.

## Open-Source Model References

The current mainline mainly builds on the following open-source models:

- OPF / Privacy Filter: [openai/privacy-filter](https://huggingface.co/openai/privacy-filter). In this project it is used as the first-stage token-classification / sequence labeling model and adapted to the AU PII label space.
- Qwen 3.5 9B Base: [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base). In this project it is used as the backbone for the Qwen span-head, which estimates type probabilities for candidate spans. It is also used for multimodal text transcription from images and scanned PDFs.

## Current Mainline Approach

The current deployment route is in `redaction-wrapper/` and uses an OPF + Qwen 9B hybrid backend. It is not a single end-to-end model that directly generates redacted output. Instead, it is a layered PII detection system: first recall as many "PII-like" spans as possible, then use a type model, rules, and risk policy to decide whether each span should be automatically redacted, reviewed by a human, or ignored.

### 1. Overall Objective

The system is designed around a privacy-safe three-way decision:

- `REDACT`: high confidence; redact automatically.
- `REVIEW`: likely PII, but evidence is not strong enough; send to human review.
- `IGNORE`: ignore only when the system is very confident the span is not PII.

The core principle is to prefer additional human review over incorrectly ignoring true PII. Therefore, the system does not simply optimize for automatic redaction rate; it prioritizes avoiding missed real PII.

### 2. Stage 1: OPF High-Recall Candidate Detection

OPF follows the open-source [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) route as the first-layer token-classification / sequence labeling model. Its job is to find candidate spans in text. In this stage, recall is more important than asking OPF to make the final decision by itself.

At the implementation level, OPF tokenizes the input text, outputs a label distribution for each token, and then uses sequence decoding to merge consecutive tokens into entity spans. The training label space is `pii_training_prep_v3_2/pii_schema/opf_label_space_79.json`, which contains:

- `O`: non-entity token.
- 79 AU PII span classes, such as `PERSON`, `DATE_OF_BIRTH`, `EMAIL_ADDRESS`, `MOBILE`, `ADDRESS`, `AU_TFN`, `MEDICARE_NUMBER`, `STUDENT_ID`, `UAC_ID`, and `USI`.

Note that OPF's span label space does not include `NON_PII`. `NON_PII` is used only for candidate-level Qwen span-head classification training. For OPF, ordinary text and hard negatives are learned through the `O` label, meaning "do not produce a span".

In the current hybrid pipeline, OPF is responsible for:

- Locating possible PII spans in the full text.
- Providing the initial type and character offsets.
- Supplying candidate boundaries for the downstream Qwen span-head and wrapper policy.

The data preparation process is implemented in `pii_training_prep_v3_2/src/pii_prep/stage3_dataset_split.py` and `stage3b_opf_prepare.py`. At a high level:

1. Build train/dev/test splits from the augmented AU PII dataset using a group-key hash split to avoid leaking the same template or near-duplicate examples across splits.
2. For the OPF training set, remove `NON_PII` spans and keep only true PII spans. Records with no true PII are kept as empty-span records and used as document-level hard negatives.
3. Validate offsets, values, and label-space membership to ensure every `text[start:end]` matches the annotated value.
4. Write the OPF-required format:
   - `data/train/opf_train_opf_format.jsonl`
   - `data/train/opf_dev_opf_format.jsonl`
   - `data/train/opf_test_opf_format.jsonl`

Current OPF-format data scale from the reports:

- train: 81,298 records, 109,877 PII spans, 64,387 empty-span hard negatives.
- dev: 10,034 records, 13,412 PII spans, 7,977 empty-span hard negatives.
- test: 9,659 records, 13,513 PII spans, 7,557 empty-span hard negatives.
- Data validation: 0 offset mismatches and 0 labels outside the label space.

Training uses OPF's built-in `python -m opf train`. The retraining recipe in this repository is `pii_training_prep_v3_2/scripts/run_opf_hard_v2.sh`; the core parameters are:

```bash
python -m opf train \
  data/train/opf_train_opf_format.jsonl \
  --validation-dataset data/train/opf_dev_opf_format.jsonl \
  --label-space-json pii_schema/opf_label_space_79.json \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --learning-rate 1e-5 \
  --device cuda
```

The current deployment config `redaction-wrapper/configs/backends/hybrid-opf-qwen9b-hn.json` points to `pii_training_prep_v3_2/runs/opf_hard_79` as the OPF checkpoint. This checkpoint is large and is not included in GitHub; the README records the training method and path convention instead.

This design is used because OPF inference is fast, suitable for batch scanning, and relatively stable for fixed-format and common entities. However, OPF can still confuse similar numeric strings across ID types or incorrectly flag ordinary contextual fields as PII. Therefore, OPF output is treated as high-recall candidates rather than final redaction decisions.

### 3. Stage 2: Qwen Span-Head Type Calibration

Qwen uses [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) as the backbone. In the main detection chain, it is not used as a free-form generative LLM that directly outputs JSON. Instead, it is used as a frozen encoder/backbone plus a span-level classification head. The wrapper passes OPF candidate spans to the Qwen span-head, and the span-head outputs top-k type probability distributions for each candidate.

Each candidate sample contains:

- original `text`
- span `start/end`
- original span `value`
- target type distribution `target_distribution`
- top label `top_type`
- sample source and training weight

The Qwen backbone first runs a forward pass over text containing candidate spans, then caches span-related hidden states. The current 9B route uses the cache configuration from `stage3a_qwen_embedding_cache_report.json`:

- backbone: `/home/admin/model/Qwen3.5-9B-Base`
- hidden size: 4096
- max length: 1536
- device: CUDA
- backbone parameters frozen, `qwen_trainable_parameter_count = 0`
- train/dev/test cached span examples: 113,354 / 13,850 / 13,588
- mapping failure: 0; skipped examples: 0

The cache stores multiple span representations:

- `mean_embeddings`: mean pooling over span-token hidden states.
- `first_embeddings`: hidden state of the first span token.
- `last_embeddings`: hidden state of the last span token.

Only a small classification head is trained afterward; the Qwen backbone is not updated. The training code is `pii_training_prep_v3_2/src/pii_prep/qwen_spancls_heads.py`. It compares four head architectures:

- `mean_linear`: mean embedding + linear classifier.
- `first_linear`: first-token embedding + linear classifier.
- `last_linear`: last-token embedding + linear classifier.
- `concat_mlp`: concatenated mean/first/last embeddings followed by an MLP.

The training target is not a normal one-hot label, but a soft target distribution. This allows teacher data and rule-derived data to express that a span primarily belongs to one class while retaining small probabilities for other classes. The loss is soft cross entropy and supports source/label reweighting:

- `candidate_level_negative` weight x3.0 to strengthen hard negatives.
- `qwen_5way_ranking` weight x1.5 to use teacher ranking/distribution signals.
- `NON_PII` label weight x2.0 to improve recognition of non-PII candidates.

Training parameters:

- label count: 80, meaning 79 PII classes plus `NON_PII`
- batch size: 1024
- max epochs: 30
- early stopping patience: 5
- learning rate: 1e-3
- optimizer: AdamW, weight decay 0.01
- calibration: temperature fitting on dev logits

The current 9B hard-negative span-head training result is recorded in `stage3a_qwen9b_hn_head_training_summary.json`. The `last_linear` head is selected by dev NLL:

- best epoch: 11
- best dev NLL: 0.084690
- temperature: 1.004767
- test top-1 accuracy after temperature: 0.987489
- test top-3 accuracy after temperature: 0.999779
- test NLL after temperature: 0.077113

The lightweight artifact published with the repository is:

- `pii_training_prep_v3_2/runs/qwen9b_hn_spancls_heads/last_linear/head.pt`

This file is only about 1.3 MB and contains the Qwen span classification head trained in this project. It does not include the original Qwen 9B backbone. Compared with asking Qwen to generate redaction JSON directly, the span-head route is more stable, lower-latency, more structured, and easier to combine with OPF candidate boundaries. Qwen's generation/multimodal capabilities are mainly used for visual text transcription in file input, not for final free-form PII JSON generation.

### 4. Training and Test Metrics

The following metrics come from the repository's training/evaluation reports. OPF, the Qwen span-head, and wrapper end-to-end evaluation are listed separately because they evaluate different objects:

- OPF metrics measure whether spans can be found.
- Qwen span-head metrics measure whether candidate span types are classified correctly.
- Wrapper metrics measure the final API output after span detection, typing, and policy post-processing.

#### OPF Span Detection

Report sources:

- `pii_training_prep_v3_2/reports/stage3b_opf_hard_eval_summary.json`
- `pii_training_prep_v3_2/reports/final_results_tables.md`

| Split | Records | PII spans | Detection precision | Detection recall | Detection F1 | Span precision | Span recall | Span F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dev | 10,034 | 13,412 | 0.9631 | 0.9817 | 0.9724 | 0.9437 | 0.9782 | 0.9606 |
| Test | 9,659 | 13,513 | 0.9745 | 0.9842 | 0.9793 | 0.9675 | 0.9775 | 0.9725 |

OPF's test detection F1 is 0.9793, showing that it is effective as a first-stage high-recall candidate detector. However, per-label weaknesses still exist, such as `GENDER`, `LATITUDE`, `LONGITUDE`, and `SALARY_WAGE_EXPECTATION`, so the downstream Qwen head and wrapper policy are still needed for calibration and risk control.

#### Qwen 9B HN Span-Head Classification

Report sources:

- `pii_training_prep_v3_2/reports/stage3a_qwen9b_hn_head_eval_last_linear.json`
- `pii_training_prep_v3_2/reports/stage3a_qwen9b_hn_head_prf_report.json`

| Split | Span examples | Micro precision | Micro recall | Micro F1 | Macro F1 | Weighted F1 | Top-1 acc. | Top-3 acc. | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 113,354 | 0.9864 | 0.9864 | 0.9864 | 0.9744 | 0.9860 | 0.9864 | 0.9995 | 0.0733 |
| Dev | 13,850 | 0.9863 | 0.9863 | 0.9863 | 0.9742 | 0.9860 | 0.9863 | 0.9999 | 0.0853 |
| Test | 13,588 | 0.9875 | 0.9875 | 0.9875 | 0.9755 | 0.9872 | 0.9875 | 0.9998 | 0.0771 |

For this single-label multi-class task, micro precision/recall/F1 is equivalent to top-1 accuracy. Macro F1 better reflects smaller-class performance; the test macro F1 is 0.9755, indicating that the 9B HN span-head is overall stable on 80-class candidate classification. `NON_PII` is explicitly included in the label space in this hard-negative version and receives extra training weight to reduce the probability that obvious non-PII candidates are misclassified as PII.

#### Wrapper / Hybrid Endpoint Full Eval

The complete end-to-end F1 reports currently in the GitHub repository mainly come from earlier 9B/4B wrapper ablations. The current published 9B hard-negative head has classification metrics and smoke validation, but a full wrapper full-eval F1 has not yet been regenerated. To avoid confusion, the table below treats existing full-eval results as historical comparisons and does not present them as the final end-to-end metric for the current 9B HN model.

Report sources:

- `redaction-wrapper/reports/stage4_eval_hybrid_legacy_9b.json`
- `redaction-wrapper/reports/ablation_5_9b_causal_lm.json`
- `redaction-wrapper/reports/stage4_eval_opf_only.json`
- `redaction-wrapper/reports/stage4_eval_hybrid_4b_calibrated_with_opf_bias.json`

| Route / report | Exact precision | Exact recall | Exact F1 | Overlap precision | Overlap recall | Overlap F1 | Type accuracy on overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| OPF-only wrapper baseline | 0.7249 | 0.7715 | 0.7475 | 0.8581 | 0.9081 | 0.8824 | - |
| 9B hybrid legacy / causal_lm ablation | 0.7315 | 0.8807 | 0.7992 | 0.8149 | 0.9756 | 0.8880 | 0.9674 |
| 4B calibrated + OPF bias ablation | 0.7395 | 0.8047 | 0.7707 | 0.8713 | 0.9428 | 0.9056 | 0.9584 |

These end-to-end results show that the hybrid route generally improves recall and semantic type judgment. However, final exact F1 is jointly affected by policy, span boundaries, deterministic rescue, and hard-negative suppression. Therefore, the current mainline does not focus only on model top-1 accuracy; it combines model scores, rule evidence, and human review in the final decision.

### 5. Stage 3: Wrapper Policy Post-Processing

`redaction-wrapper` is the final decision layer. It combines:

- whether OPF detected a span
- the Qwen span-head top-1 type and probability
- whether the top-k distribution is concentrated or ambiguous
- whether the span surface form matches the expected format for a PII type
- whether surrounding context supports the type
- the confidentiality level and false-positive risk of the type

The policy's role is not to add endless one-off patches, but to keep generalizable judgment rules outside the model and make system behavior more explainable. For example:

- Fixed-format, high-confidence fields can be automatically redacted.
- Low-probability spans, type conflicts, insufficient context, or suspicious formats go to review.
- Only clearly non-PII spans are ignored.

The system currently avoids aggressively ignoring candidates just to reduce hard negative false positives. In a privacy system, incorrectly ignoring true PII is more costly than sending extra items to human review.

### 6. Relationship Between Deterministic Policy and Model Probability

Some fields have stable formats, such as certain student numbers, UAC IDs, TFNs, Medicare numbers, passport numbers, bank account numbers, credit card numbers, email addresses, and phone numbers. For these fields, the wrapper uses regex/format validation plus context to make deterministic judgments.

If a span is identified by deterministic policy and has no reliable model probability source, the frontend displays that type as `100%`. This does not mean the neural model literally output a 100% probability; it means the result comes from a deterministic fixed-format policy judgment.

If a span comes from the Qwen span-head, the frontend displays the actual model probability and top-k distribution.

### 7. Risk and Human Review

Risk is not simply model uncertainty. It is the wrapper's combined estimate of whether a decision needs human review. Common review triggers include:

- low top-1 probability or a diffuse top-k distribution
- disagreement between OPF and Qwen types
- a span that looks like PII but lacks enough context
- high false-positive-risk types, such as certain numeric strings, internal IDs, phone numbers, addresses, IPs, and vehicle plates
- spans that may contain PII but have incomplete boundaries or abnormal cross-field merges

The goal is to preserve uncertain content for review rather than discard it.

### 8. File Input and Qwen Multimodal Text Transcription

The wrapper supports text, image, and PDF inputs. The current route no longer uses a separate traditional OCR engine, such as PaddleOCR, RapidOCR, or Tesseract, as the main image-recognition path. Instead, it uses Qwen's own multimodal capability to transcribe text from images and scanned PDF pages.

The file input flow is:

- Text files: decoded directly as text.
- PDFs with a text layer: the embedded PDF text layer is read first.
- Images or scanned PDFs: images/pages are sent to the Qwen multimodal model for text transcription.
- Transcribed text: passed into the same OPF + Qwen span-head + wrapper policy pipeline.

This lets web input, text files, images, and PDFs share the same backend logic after entering PII detection. Some `ocr` field names remain in the code for API compatibility, but their current meaning is closer to "file text extraction / visual text transcription" rather than a separate OCR-model route.

### 9. Why Hybrid Instead of a Single-Model Solution

Using OPF alone has limited type judgment and contextual understanding. Using Qwen alone as a generative output model has higher latency, weaker format stability, and lower controllability. The hybrid route separates the tasks:

- OPF handles fast scanning and high recall.
- The Qwen span-head handles semantic type calibration.
- The wrapper policy handles safety thresholds, confidentiality levels, review/ignore/redact decisions, and frontend display.

This makes the system more suitable for deployment: speed is controlled, output is stable, errors are easier to diagnose, and the design better matches PII redaction requirements for low miss rate, explainability, and human review.

## Running the Wrapper Service

```bash
cd redaction-wrapper
export WRAPPER_BACKEND_CONFIG=$PWD/configs/backends/hybrid-opf-qwen9b-hn.json
export WRAPPER_POLICY_CONFIG=$PWD/configs/policies/hybrid-80class-v2-4b.json
./scripts/run_server.sh
```

The default service provides:

- `GET /api/health`: health check
- `POST /api/redact`: text PII detection and redaction
- `POST /api/redact-file`: file input processing
- `/`: web demo

## Training and Evaluation

Mainline training and evaluation code is in `pii_training_prep_v3_2/`:

```bash
cd pii_training_prep_v3_2
pytest tests
```

Wrapper tests:

```bash
cd redaction-wrapper
pytest tests
```

Related reports are located in:

- `pii_training_prep_v3_2/reports/`
- `redaction-wrapper/reports/`

These reports record 4B/9B span-head results, 27B teacher-generated data, hard-negative experiments, calibration, policy tuning, and hybrid API evaluation results.

## Large File Notes

This repository does not include original large model weights, large checkpoints, runs/cache directories, virtual environments, or local training artifacts. These assets need to be prepared separately on the server according to the paths in the backend configs.

Exception: the lightweight head for the current mainline Qwen 9B span-head has been published with the repository:

- `pii_training_prep_v3_2/runs/qwen9b_hn_spancls_heads/last_linear/head.pt`

This file contains only the span-level classification head trained in this project. It does not include the original Qwen 9B backbone. Runtime still requires preparing [Qwen/Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) and the OPF checkpoint separately.

`.gitignore` excludes common large-file directories and model file types, such as:

- `runs/`
- `outputs/`
- `checkpoint*/`
- `*.safetensors`
- `*.pt`
- `*.bin`
- `pii_training_prep_v3_2/data/`

The Qwen span-head artifact above is included in version control through a specific `.gitignore` exception rule.

## Notes

This repository is a project codebase and experiment-history snapshot. It should not contain private production data or large model artifacts. Before using it for public presentation, re-check the data files, report content, and repository visibility.

# Australia PII Redaction

Repository: <https://github.com/ZhouYuxiang01/Australia-PII-Redaction>

This repository contains an Australian personally identifiable information
(PII) redaction workspace. It combines data preparation, model training,
evaluation, and a backend-pluggable redaction service for detecting and
redacting Australian PII spans in text and files.

## What This Project Contains

- OPF token-classifier training and evaluation pipelines for Australian PII.
- Qwen-based span classification and redaction experiments.
- A model-agnostic FastAPI redaction wrapper with shared OCR, post-processing,
  policy, schema, and UI layers.
- Evaluation reports, error analyses, and final experiment summaries.

## Repository Structure

```text
.
├── opf_au_pii/                         OPF AU-PII training, eval, audit, and synthesis tools
├── pii_training_prep_v3_2/             Data preparation and Qwen/OPF training datasets
├── redaction-wrapper/                  Backend-pluggable redaction API, OCR, policy, and UI
├── Qwen3.5_9b_base_Distill/            Qwen 9B LoRA/distillation experiments
├── Qwen3.5_4b_base_Full_73class/       Qwen 4B full supervised 73-class experiments
├── Qwen3_4b_instruct_Distill/          Qwen 3 4B instruct distillation track
└── reports/                            Cross-project notes and inspection reports
```

Large local artifacts such as raw datasets, generated datasets, checkpoints,
model outputs, and run directories are intentionally kept out of git where
possible.

## Main Results

The main result summaries live in:

- `pii_training_prep_v3_2/reports/final_experiment_summary.md`
- `pii_training_prep_v3_2/reports/final_results_tables.md`

Key reported metrics:

| Component | Test Metric | Value |
|---|---:|---:|
| OPF-only hard-label detector | Detection F1 | 0.9793 |
| OPF-only hard-label detector | Span F1 | 0.9725 |
| OPF-only hard-label detector | Token accuracy | 99.15% |
| Qwen span classification head | Top-1 accuracy | 98.53% |

The OPF test predictions and metrics are available at:

- `pii_training_prep_v3_2/reports/stage3b_opf_hard_test_eval.json`
- `pii_training_prep_v3_2/reports/stage3b_opf_hard_test_predictions.jsonl`

## Redaction Wrapper

The `redaction-wrapper/` package provides a shared API and UI around different
redaction backends. It handles normalization, span post-processing, policy
decisions, deterministic redaction, OCR/PDF text extraction, FastAPI routes, and
the browser demo.

Quick start:

```bash
cd redaction-wrapper
export WRAPPER_BACKEND_CONFIG=$PWD/configs/backends/opf-v3.json
export WRAPPER_POLICY_CONFIG=$PWD/configs/policies/opf-v3-default-v1.json
./scripts/run_server.sh
```

Then open:

- Browser demo: `http://127.0.0.1:8090/`
- FastAPI docs: `http://127.0.0.1:8090/docs`

## OPF AU-PII Pipeline

The `opf_au_pii/` directory contains the OPF 73-class Australian PII experiment.
Important entry points include:

- `opf_au_pii/README_PROJECT_STRUCTURE.md`
- `opf_au_pii/scripts/run_opf_pipeline.py`
- `opf_au_pii/scripts/train_eval_v3.sh`
- `opf_au_pii/scripts/eval_char_spans_v2.py`

Example smoke run:

```bash
cd opf_au_pii
python3 scripts/run_opf_pipeline.py \
  --smoke \
  --data-dir data/processed/data_opf \
  --taxonomy configs/taxonomy_v1.1.1.yaml \
  --label-space configs/custom_label_space_73.v1.1.1.json \
  --run-dir runs/ablations/smoke_v1 \
  --eval-on-test \
  --char-eval
```

## Data Preparation

The `pii_training_prep_v3_2/` directory contains the data preparation pipeline
for Australian PII span-distribution training data and downstream model
evaluation.

Useful files:

- Test split: `pii_training_prep_v3_2/data/splits/test.jsonl`
- OPF test format: `pii_training_prep_v3_2/data/train/opf_test_opf_format.jsonl`
- Qwen span-classifier test format: `pii_training_prep_v3_2/data/train/qwen_spancls_test.jsonl`
- Qwen4B token-classifier test format: `pii_training_prep_v3_2/data/train/qwen4b_tokencls_test.jsonl`

Run tests:

```bash
cd pii_training_prep_v3_2
python3 -m unittest discover -s tests -v
```

## Recommended Workflow

1. Prepare or validate the dataset split under `pii_training_prep_v3_2/`.
2. Run OPF smoke training before full training.
3. Train or evaluate the selected OPF checkpoint.
4. Review aggregate metrics and per-label span metrics.
5. Audit misses and false positives, then feed targeted examples into the next
   data or training iteration.
6. Expose the selected checkpoint through `redaction-wrapper/` for API and UI
   testing.

## Notes

- The project is research-oriented and includes multiple experiment tracks.
- Checkpoint paths in reports may refer to local training-server locations.
- Keep new experiment outputs under fresh `runs/` or `reports/` paths so results
  remain reproducible.

# Training Dry-Run and Reproduction Guide

This guide describes how to exercise the data preparation and fine-tuning
workflow without running the full expensive training jobs. It is intended for
handover, smoke testing, and environment validation.

The production route is:

```text
raw/source data
  -> stage-1 canonical dataset
  -> stage-2 augmented and hard-negative data
  -> stage-3 OPF and Qwen span-classification splits
  -> OPF candidate detector training
  -> frozen-Qwen span embedding cache
  -> lightweight Qwen span-head training
  -> model selection/calibration
  -> deployed hybrid runtime
```

The dry-run route should answer a narrower question:

```text
Can the code, inputs, paths, label spaces, and command interfaces run end to end?
```

It should not be used as model-quality evidence.

## Working Directory

Run data preparation commands from:

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2
```

Use:

```bash
PYTHONPATH=src
```

for the data-preparation scripts, and:

```bash
PYTHONPATH=../opf-runtime
```

for the OPF CLI.

## Fastest Sanity Check

The unit tests are the cheapest way to verify that the preparation code and
schemas still agree. They do not load Qwen or OPF checkpoints.

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2
PYTHONPATH=src python3 -m pytest tests
```

Expected result: tests pass without loading large model weights.

## Teacher Dry-Run

This checks the teacher-output conversion and validation path without making
fresh teacher-model calls.

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2

PYTHONPATH=src python3 scripts/run_stage2_teacher_dryrun.py \
  --convert-existing-only \
  --limit 5
```

Use this when the goal is to prove the teacher-data interface can be exercised
on existing checked-in converted outputs. It does not benchmark the 27B teacher
and does not generate new model judgments.

If a vLLM/OpenAI-compatible teacher endpoint is available and you want a tiny
live-call check, use a very low limit:

```bash
PYTHONPATH=src python3 scripts/run_stage2_teacher_dryrun.py \
  --backend vllm_openai \
  --base-url http://localhost:8000/v1 \
  --limit 2 \
  --max-new-tokens 64
```

This is still a smoke test. It only checks that the request/response path works.

## Data Preparation Dry-Run

These commands rebuild the intermediate datasets from tracked inputs. They do
real data processing, but they do not run model training.

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2

PYTHONPATH=src python3 scripts/build_stage1_dataset.py
PYTHONPATH=src python3 scripts/reconcile_taxonomy.py
PYTHONPATH=src python3 scripts/merge_stage2_augmented.py
PYTHONPATH=src python3 scripts/build_stage3_datasets.py
```

Expected outputs include:

```text
data/processed/stage1_v3_2.jsonl
data/processed/stage1_v3_2_canonical.jsonl
data/processed/stage2_v3_2_augmented.jsonl
data/splits/train.jsonl
data/splits/dev.jsonl
data/splits/test.jsonl
data/train/qwen_spancls_train.jsonl
data/train/qwen_spancls_dev.jsonl
data/train/qwen_spancls_test.jsonl
data/train/opf_hard_train.jsonl
data/train/opf_hard_dev.jsonl
data/train/opf_hard_test.jsonl
```

Useful checks after the run:

```bash
wc -l data/train/qwen_spancls_*.jsonl data/train/opf_hard_*.jsonl
python3 -m json.tool reports/stage3_dataset_report.json >/dev/null
```

If any command fails, first check whether the preceding output file exists and
whether the label-space files under `../hybrid-pii-model-runtime/pii_schema/`
are present.

## OPF Fine-Tuning Smoke Test

This is a real OPF training invocation, but it caps the training and validation
sets to tiny sizes and uses one epoch. It validates that the OPF CLI, dataset
format, label space, checkpoint path, optimizer loop, and output writing work.

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2

PYTHONPATH=../opf-runtime python3 -m opf train data/train/opf_hard_train.jsonl \
  --validation-dataset data/train/opf_hard_dev.jsonl \
  --label-space-json ../hybrid-pii-model-runtime/pii_schema/opf_label_space_79.json \
  --checkpoint /home/admin/.opf/privacy_filter \
  --device cpu \
  --epochs 1 \
  --batch-size 1 \
  --max-train-examples 8 \
  --max-validation-examples 4 \
  --output-dir runs/smoke_opf_hard_79 \
  --overwrite-output
```

Expected outputs:

```text
runs/smoke_opf_hard_79/config.json
runs/smoke_opf_hard_79/model.safetensors
runs/smoke_opf_hard_79/finetune_summary.json
runs/smoke_opf_hard_79/USAGE.txt
```

This command is intentionally not comparable to the production OPF model. It is
only a functional training-path check.

## Qwen Span-Head Smoke Options

The Qwen backbone is not fine-tuned. The workflow caches frozen Qwen span
embeddings and trains small classification heads over those cached features.

The full reproduction commands are:

```bash
PYTHONPATH=src python3 scripts/cache_qwen_span_embeddings.py \
  --model-path /home/admin/model/Qwen3.5-9B-Base \
  --batch-size 8 \
  --cache-name-prefix qwen9b_hn_spancls_embeddings

PYTHONPATH=src python3 scripts/train_qwen_spancls_heads.py \
  --cache-name-prefix qwen9b_hn_spancls_embeddings \
  --run-dir-name qwen9b_hn_spancls_heads \
  --report-prefix stage3a_qwen9b_hn_head \
  --source-weight-overrides candidate_level_negative=3,qwen_5way_ranking=1.5 \
  --label-weight-overrides NON_PII=2 \
  --max-epochs 30 \
  --patience 5
```

For dry-run purposes, be careful with the embedding-cache step: it still loads
the Qwen backbone and is therefore a real model/GPU check, not a cheap fake
run. A practical smoke strategy is:

1. Run the data-preparation dry-run first.
2. If a small existing cache is available under `data/cache/`, run only the head
   trainer with low epochs.
3. If no cache exists, treat `cache_qwen_span_embeddings.py` as a GPU/model
   availability check and run it only on a prepared machine.

Example low-epoch head-training smoke command, assuming a matching cache exists:

```bash
PYTHONPATH=src python3 scripts/train_qwen_spancls_heads.py \
  --cache-name-prefix qwen9b_hn_spancls_embeddings \
  --run-dir-name smoke_qwen9b_hn_spancls_heads \
  --report-prefix smoke_stage3a_qwen9b_hn_head \
  --source-weight-overrides candidate_level_negative=3,qwen_5way_ranking=1.5 \
  --label-weight-overrides NON_PII=2 \
  --batch-size 16 \
  --max-epochs 1 \
  --patience 1
```

Expected output, if the cache exists and is compatible:

```text
runs/smoke_qwen9b_hn_spancls_heads/
reports/smoke_stage3a_qwen9b_hn_head*.json
```

## Model Selection Smoke

After a Qwen span-head smoke run, selection can be exercised with matching
prefixes:

```bash
PYTHONPATH=src python3 scripts/select_stage3a_model.py \
  --selection-strategy hard_negative_aware \
  --report-prefix smoke_stage3a_qwen9b_hn_head \
  --run-dir-name smoke_qwen9b_hn_spancls_heads \
  --cache-name-prefix qwen9b_hn_spancls_embeddings \
  --output-prefix smoke_stage3a_qwen9b_hn
```

This checks report discovery, calibration/selection logic, and artifact naming.
It is only meaningful if the preceding smoke head run produced compatible
reports and checkpoints.

## Full Reproduction Commands

Use these only on a prepared training machine with model weights, GPU runtime,
and enough time.

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2

PYTHONPATH=src python3 scripts/merge_stage2_augmented.py
PYTHONPATH=src python3 scripts/build_stage3_datasets.py

PYTHONPATH=src python3 scripts/cache_qwen_span_embeddings.py \
  --model-path /home/admin/model/Qwen3.5-9B-Base \
  --batch-size 8 \
  --cache-name-prefix qwen9b_hn_spancls_embeddings

PYTHONPATH=src python3 scripts/train_qwen_spancls_heads.py \
  --cache-name-prefix qwen9b_hn_spancls_embeddings \
  --run-dir-name qwen9b_hn_spancls_heads \
  --report-prefix stage3a_qwen9b_hn_head \
  --source-weight-overrides candidate_level_negative=3,qwen_5way_ranking=1.5 \
  --label-weight-overrides NON_PII=2 \
  --max-epochs 30 \
  --patience 5

PYTHONPATH=src python3 scripts/select_stage3a_model.py \
  --selection-strategy hard_negative_aware \
  --report-prefix stage3a_qwen9b_hn_head \
  --run-dir-name qwen9b_hn_spancls_heads \
  --cache-name-prefix qwen9b_hn_spancls_embeddings \
  --output-prefix stage3a_qwen9b_hn
```

OPF full training command:

```bash
PYTHONPATH=../opf-runtime python3 -m opf train data/train/opf_hard_train.jsonl \
  --validation-dataset data/train/opf_hard_dev.jsonl \
  --label-space-json ../hybrid-pii-model-runtime/pii_schema/opf_label_space_79.json \
  --checkpoint /home/admin/.opf/privacy_filter \
  --device cuda \
  --epochs 1 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --output-dir runs/opf_hard_79 \
  --overwrite-output
```

## What To Think About During Dry-Run Review

Use the dry-run as a structured checklist, not as a score.

Data questions:

- Do all expected intermediate files exist?
- Did label reconciliation produce labels inside the deployment label spaces?
- Are hard-negative teacher rows included when the optional file exists?
- Are split sizes plausible and non-empty?
- Are OPF and Qwen datasets generated from the same canonical source?

Training-path questions:

- Does the OPF CLI load the intended checkpoint path?
- Does the output directory contain a complete checkpoint and summary?
- Does the Qwen head trainer find the intended cache prefix?
- Are report prefixes and run directory names consistent across training and
  selection?
- Are smoke artifacts clearly separated from production artifacts?

Environment questions:

- Is CUDA required for the command being run, or can CPU be used?
- Is the Qwen backbone path local and readable?
- Is the OPF base checkpoint present at `/home/admin/.opf/privacy_filter`?
- Are `PYTHONPATH` values pointing to this repository's code rather than an old
  installed copy?

Quality questions for the full run:

- Does the hard-negative-aware selection improve false-positive behavior?
- Are probability outputs calibrated enough for policy thresholds?
- Are deterministic rescue rules and model decisions aligned on common fields
  such as email, phone, BSB, account number, and TFN-like values?
- Are final artifacts copied into `hybrid-pii-model-runtime/runs/` with the
  paths expected by `pii-redaction-service/configs/backends/hybrid-opf-qwen9b-hn.json`?

## Common Failure Modes

| Symptom | Likely cause | Action |
|---|---|---|
| Import error for `pii_prep` | Missing `PYTHONPATH=src` | Run from `pii_training_prep_v3_2` with `PYTHONPATH=src`. |
| Import error for `opf` | Missing `PYTHONPATH=../opf-runtime` or uninstalled OPF | Set `PYTHONPATH=../opf-runtime` for OPF CLI commands. |
| Missing `data/train/*.jsonl` | Stage-3 datasets were not generated | Run the data-preparation dry-run first. |
| OPF checkpoint missing | `/home/admin/.opf/privacy_filter` not available | Provide a local checkpoint path with `--checkpoint`. |
| Qwen cache missing | Embedding cache was not generated or prefix mismatch | Check `data/cache/` and use the same `--cache-name-prefix`. |
| CUDA error | Command is loading model weights on a machine without usable GPU | Use CPU where supported, or move the Qwen cache/model step to a GPU machine. |
| Smoke output overwrites production run | Output names reused | Use `smoke_*` run directories and report prefixes. |


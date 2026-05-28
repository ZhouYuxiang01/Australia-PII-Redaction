# Hybrid Training Pipeline

This directory contains the lightweight training and data-preparation code needed
to reproduce the deployed hybrid backend:

```text
OPF candidate detector + frozen Qwen 3.5 9B backbone + trained span classifier head
```

It intentionally does not include old experiments, generated datasets, embedding
caches, checkpoints, or model weights. Those artifacts are large and should be
prepared separately on the training machine.

## Scope

The final service uses two trained artifacts:

- OPF checkpoint: `hybrid-pii-model-runtime/runs/opf_hard_79/`
- Qwen span-head: `hybrid-pii-model-runtime/runs/qwen9b_hn_spancls_heads/last_linear/head.pt`

The Qwen backbone itself is not fine-tuned. Training freezes
`Qwen/Qwen3.5-9B-Base`, caches span-level hidden-state features, and trains a
small classifier head over candidate spans.

## Source Layout

```text
scripts/merge_stage2_augmented.py       merge augmented and hard-negative teacher rows
scripts/build_stage3_datasets.py        build OPF and Qwen span-classification splits
scripts/cache_qwen_span_embeddings.py   cache frozen Qwen span embeddings
scripts/train_qwen_spancls_heads.py     train candidate span classifier heads
scripts/select_stage3a_model.py         select/calibrate the best span head
src/pii_prep/                           implementation modules used by those scripts
tests/                                  lightweight unit tests for the pipeline code
```

## Expected Inputs

The scripts expect the project root to contain the schema files already tracked
in `hybrid-pii-model-runtime/pii_schema/`, plus local data under
`pii_training_prep_v3_2/data/`.

The important generated inputs are:

```text
data/processed/stage2_v3_2_augmented.jsonl
data/generated/stage2_hard_negative_teacher_converted.jsonl
```

If the hard-negative teacher file exists, `merge_stage2_augmented.py` includes it
in the merged stage-2 training data.

## Qwen 9B Span-Head Reproduction

Run from the repository root:

```bash
cd pii_training_prep_v3_2

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

The selected artifact is expected at:

```text
runs/qwen9b_hn_spancls_heads/last_linear/head.pt
```

For deployment, copy or sync that file to:

```text
hybrid-pii-model-runtime/runs/qwen9b_hn_spancls_heads/last_linear/head.pt
```

## OPF Detector Training

`build_stage3_datasets.py` also emits OPF-format data:

```text
data/train/opf_hard_train.jsonl
data/train/opf_hard_dev.jsonl
data/train/opf_hard_test.jsonl
```

Train OPF with the runtime CLI:

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

For deployment, copy or sync the resulting checkpoint directory to:

```text
hybrid-pii-model-runtime/runs/opf_hard_79/
```

## Tests

The lightweight tests do not load Qwen or the large checkpoints:

```bash
PYTHONPATH=src python3 -m pytest tests
```

From the repository root, you can also run:

```bash
PYTHONPATH=$PWD/pii_training_prep_v3_2/src python3 -m pytest pii_training_prep_v3_2/tests
```

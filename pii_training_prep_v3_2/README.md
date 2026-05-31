# Hybrid Training Pipeline

This directory contains the lightweight training and data-preparation code needed
to reproduce the deployed hybrid backend:

```text
OPF candidate detector + frozen Qwen 3.5 9B backbone + trained span classifier head
```

This README is the reproduction guide for data preparation and training. For
customer handover evidence, see `../docs/README.md`; for model summary and
evaluation results, see `../docs/MODEL_CARD.md` and
`../docs/EVALUATION_REPORT.md`.

It intentionally does not include old experiments, generated datasets, embedding
caches, checkpoints, or model weights. Those artifacts are large and should be
prepared separately on the training machine.

It does include the compact reproducibility inputs for the final route:

```text
data/raw/au_pii_19000_final.json
data/external_eval/au_pii_test_1000.json
data/generated/stage2_full_teacher_converted.jsonl
data/generated/stage2_hard_negative_teacher_converted.jsonl
```

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
scripts/build_stage1_dataset.py         convert the 19k source JSON into stage-1 distributions
scripts/reconcile_taxonomy.py           canonicalize labels against the 79-label AU PII schema
scripts/generate_stage2_samples.py      regenerate stage-2 seed examples and teacher prompts
scripts/run_stage2_full_teacher.py      rerun full teacher calls if the 27B teacher is available
scripts/run_stage2_hard_negative_teacher.py rerun hard-negative teacher calls
src/pii_prep/                           implementation modules used by those scripts
tests/                                  lightweight unit tests for the pipeline code
```

## Expected Inputs

The scripts expect local base models to be available on the training machine.
The default paths used during this project were:

```text
Qwen span-head backbone: /home/admin/model/Qwen3.5-9B-Base
Optional teacher model:  /home/admin/model/qwen3.5-27b
OPF base checkpoint:    /home/admin/.opf/privacy_filter
```

The repository tracks the synthetic 19k source dataset, the held-out 1,000-record
test dataset, and the converted teacher outputs that were used for the final
route. It does not track regenerated processed data, train/dev/test splits,
embedding caches, or model checkpoints.

The important generated inputs are:

```text
data/processed/stage2_v3_2_augmented.jsonl
data/generated/stage2_hard_negative_teacher_converted.jsonl
```

If the hard-negative teacher file exists, `merge_stage2_augmented.py` includes it
in the merged stage-2 training data.

## Rebuild Data From Source Inputs

Run from this directory:

```bash
PYTHONPATH=src python3 scripts/build_stage1_dataset.py

PYTHONPATH=src python3 scripts/reconcile_taxonomy.py

PYTHONPATH=src python3 scripts/merge_stage2_augmented.py

PYTHONPATH=src python3 scripts/build_stage3_datasets.py
```

Those commands regenerate:

```text
data/processed/stage1_v3_2.jsonl
data/processed/stage1_v3_2_canonical.jsonl
data/processed/stage2_v3_2_augmented.jsonl
data/splits/{train,dev,test}.jsonl
data/train/qwen_spancls_{train,dev,test}.jsonl
data/train/opf_hard_{train,dev,test}.jsonl
```

The checked-in `stage2_*_teacher_converted.jsonl` files let this path run without
rerunning the teacher model. To regenerate those teacher files instead, first
run:

```bash
PYTHONPATH=src python3 scripts/generate_stage2_samples.py
PYTHONPATH=src python3 scripts/run_stage2_full_teacher.py \
  --base-url http://localhost:8000/v1 \
  --model qwen3.5-27b
PYTHONPATH=src python3 scripts/run_stage2_hard_negative_teacher.py \
  --base-url http://localhost:8000/v1 \
  --model qwen3.5-27b
```

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

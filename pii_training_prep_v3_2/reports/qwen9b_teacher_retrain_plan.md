# Qwen 9B Teacher-Retrain Work Plan

Date: 2026-05-08

## Current Change

This pass adds a targeted 27B-teacher hard-negative path for candidate spans that the current hybrid route can over-redact:

- bank names that look close to bank account fields
- private/internal IPs in documentation
- public organisation phone numbers
- placeholder emails
- course codes that look like student IDs
- order/invoice/reference numbers
- sandbox card numbers
- asset/build codes that look like vehicle IDs
- report periods that look like expiry dates
- public URLs

The generated prompts use `hard_negative_context` and expect `NON_PII` unless the teacher sees strong individual-PII evidence.

## Generated Artifacts

- Prompts: `data/generated/stage2_hard_negative_teacher_prompts.jsonl`
- Plan: `reports/stage2_hard_negative_teacher_plan.json`
- Current prompt count: 60
- Base scenarios: 10
- Self consistency: 3

## Run 27B Teacher

Preferred model: `/home/admin/model/Qwen3.6-27B`.

If an OpenAI-compatible vLLM server is available, run:

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2
PYTHONPATH=src /home/admin/miniconda3/envs/opf/bin/python scripts/run_stage2_hard_negative_teacher.py \
  --model /home/admin/model/Qwen3.6-27B \
  --base-url http://localhost:8000/v1 \
  --examples-per-scenario 2 \
  --self-consistency 3 \
  --concurrency 16 \
  --max-tokens 96
```

If vLLM is not available, use the local Transformers runner in the `opf` environment. This path uses the same Qwen3.6 27B model and records whether `flash-linear-attention` is importable:

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2
PYTHONPATH=src /home/admin/miniconda3/envs/opf/bin/python scripts/run_stage2_hard_negative_teacher.py \
  --backend local \
  --model /home/admin/model/Qwen3.6-27B \
  --examples-per-scenario 2 \
  --self-consistency 3 \
  --max-tokens 96
```

The runner writes:

- `data/generated/stage2_hard_negative_teacher_raw.jsonl`
- `data/generated/stage2_hard_negative_teacher_converted.jsonl`
- `reports/stage2_hard_negative_teacher_generation_report.json`
- `reports/stage2_hard_negative_teacher_quality_report.json`

## Merge And Retrain 9B Head

After reviewing the converted hard-negative quality report, merge those converted rows into the stage2 augmented dataset, rebuild the stage3 split, and regenerate 9B caches. `merge_stage2_augmented.py` now includes `data/generated/stage2_hard_negative_teacher_converted.jsonl` automatically when that file exists.

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2
PYTHONPATH=src /home/admin/miniconda3/envs/opf/bin/python scripts/merge_stage2_augmented.py
PYTHONPATH=src /home/admin/miniconda3/envs/opf/bin/python scripts/build_stage3_datasets.py
PYTHONPATH=src /home/admin/miniconda3/envs/opf/bin/python scripts/cache_qwen_span_embeddings.py \
  --model-path /home/admin/model/Qwen3.5-9B-Base \
  --batch-size 8 \
  --cache-name-prefix qwen9b_hn_spancls_embeddings
```

For the first 9B retrain, use source/label reweighting so the hard negatives are not drowned out by easy Sonnet spans:

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2
PYTHONPATH=src /home/admin/miniconda3/envs/opf/bin/python scripts/train_qwen_spancls_heads.py \
  --cache-name-prefix qwen9b_hn_spancls_embeddings \
  --run-dir-name qwen9b_hn_spancls_heads \
  --report-prefix stage3a_qwen9b_hn_head \
  --source-weight-overrides candidate_level_negative=3,qwen_5way_ranking=1.5 \
  --label-weight-overrides NON_PII=2 \
  --max-epochs 30 \
  --patience 5
```

Select with the hard-negative-aware dev criterion:

```bash
PYTHONPATH=src /home/admin/miniconda3/envs/opf/bin/python scripts/select_stage3a_model.py \
  --selection-strategy hard_negative_aware \
  --report-prefix stage3a_qwen9b_hn_head \
  --run-dir-name qwen9b_hn_spancls_heads \
  --cache-name-prefix qwen9b_hn_spancls_embeddings \
  --output-prefix stage3a_qwen9b_hn
```

## Acceptance Criteria

- `NON_PII` and `candidate_level_negative` dev accuracy improve without a large global NLL regression.
- High-FP categories in wrapper smoke tests produce fewer redactions and more review/ignore decisions.
- End-to-end wrapper eval is used before replacing the current deployed 4B head.

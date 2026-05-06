# PII Training Prep v3.2

Data preparation project for distilling Australian PII span-distribution outputs into Qwen3.5-9B.

## Scope

This project starts the text-only mainline from `pii_training_prep_v3_2.md`:

- normalize the provided data-sensitivity CSV into a training taxonomy;
- convert existing `au_pii_19000_final.json` labels into v3.2 span distributions;
- keep policy decisions out of training data;
- keep vision/PDF work out of the primary deliverable.

## Remote Paths

- Project: `/home/admin/ZYX/pii_training_prep_v3_2`
- Raw dataset source: `/home/admin/ZYX/Qwen3.5_9b_base_Distill/data/raw/au_pii_19000_final.json`
- Student model: `/home/admin/model/Qwen3.5-9B-Base`

## First Commands

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2
python3 -m unittest discover -s tests -v
python3 scripts/build_stage1_dataset.py --limit 100
```

## Current Status

Completed on 2026-04-30:

- source spec copied to `docs/pii_training_prep_v3_2.md`;
- taxonomy CSV copied to `docs/Data Sensitivity.csv`;
- raw data linked from the existing 9B project into `data/raw/au_pii_19000_final.json`;
- stage-1 converter implemented in `src/pii_prep/build_distribution_dataset.py`;
- full stage-1 output written to `data/processed/stage1_v3_2.jsonl`;
- audit written to `reports/stage1_audit.json`.

Full build audit:

```json
{
  "input_records": 19000,
  "positive_records": 19000,
  "hard_negative_records": 75991,
  "span_count": 134732,
  "format_mismatch_count": 47813,
  "offset_mismatch_count": 0,
  "validation_error_count": 0
}
```

## Notes

The first converter intentionally keeps the implementation conservative:

- positive labels are converted to one-hot plus `NON_PII` label smoothing;
- source confidence is stored as `teacher_confidence` and converted into `training_weight`;
- document-level hard negatives are represented as records with empty `spans`;
- policy-layer decisions are not embedded in the training data.

`format_mismatch_count` is expected to be high in this first pass because only the strongest pattern rules are implemented. The next pass should expand the A/C class format rules and produce per-label mismatch reporting before training.

## Stage 2 Synthetic Seed Generation

Stage 2 creates local draft data only. It does not start model training and does not execute teacher calls.

Outputs:

- `data/generated/stage2_seed_examples.jsonl`
- `data/generated/stage2_teacher_prompts_sample.jsonl`
- `reports/stage2_generation_plan.json`
- `reports/zero_example_label_plan.json`

Current counts:

```json
{
  "zero_example_record_count": 80,
  "candidate_level_negative_count": 10,
  "document_level_negative_count": 4,
  "teacher_prompt_sample_count": 20,
  "teacher_calls_executed": 0
}
```

## Stage 2.1 Teacher Dry Run

Stage 2.1 runs only the 20 prompt sample through the Qwen 27B teacher. It does not start model training, does not run the full 6000 teacher calls, and does not merge dry-run rows into the training dataset.

Outputs:

- `data/generated/stage2_teacher_dryrun_20_raw.jsonl`
- `data/generated/stage2_teacher_dryrun_20_converted.jsonl`
- `reports/stage2_teacher_dryrun_report.json`
- `reports/stage2_teacher_output_errors.json`

Dry-run settings:

- model: `/home/admin/model/qwen3.5-27b`
- runtime: `/home/admin/miniconda3/bin/python`
- generation: deterministic `do_sample=False`, thinking disabled, 4-bit loading
- teacher calls executed: 20

Result:

```json
{
  "prompts_attempted": 20,
  "valid_json_outputs": 20,
  "validation_error_count": 0,
  "labels_outside_training_space": {},
  "overconfident_distribution_count": 1,
  "merged_into_training_dataset": false
}
```

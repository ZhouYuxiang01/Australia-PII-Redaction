# OPF AU-PII Project Structure

## Main final model
- `runs/final/opf_73class_v3_full/checkpoint/` (current best, char typed F1 0.7081)
- `runs/final/opf_73class_v2b_full/checkpoint/` (previous, kept for comparison)

## Configuration
- `configs/taxonomy_v1.1.1.yaml`
- `configs/custom_label_space_73.v1.1.1.json`

## Scripts
- `scripts/run_opf_pipeline.py` — original v1/v2 training pipeline
- `scripts/eval_char_spans_v2.py` — char-level eval
- `scripts/audit_diff.py` — pair OPF predictions with gold; classify disagreements (v3)
- `scripts/audit_run.py` — 27B auditor over disagreements (v3)
- `scripts/audit_summary.py` — verdict aggregation (v3)
- `scripts/synth_data.py` — 27B synthesis with audit-derived confusion pairs (v3)
- `scripts/synth_filter.py` — type-form validation (v3)
- `scripts/synth_remap_strict.py` — alias remap and 73-class whitelist (v3)
- `scripts/train_eval_v3.sh` — v3 train + eval orchestration
- `scripts/compare_v2b_v3.py` — v3 vs v2b delta report

## Data
- Raw data: `data/raw/`
- Processed data: `data/processed/`

## Runs
- Final: `runs/final/`
- Baseline: `runs/baselines/`
- Ablations: `runs/ablations/`

## Archive
Old scripts, intermediate data, smoke runs, and zip packages are under `_archive/`.

# OPF AU-PII Project Structure

This folder contains the OPF 73-class Australian PII experiment. The committed
source files are intentionally small; raw data, processed data, checkpoints,
runs, and outputs are local artifacts ignored by git.

## Core Files

```
opf_au_pii/
├── configs/
│   ├── taxonomy_v1.1.1.yaml
│   └── custom_label_space_73.v1.1.1.json
├── privacy-filter/                 vendored OPF source used by the project
├── scripts/                        training, audit, synthesis, and eval tools
├── tests/                          unit tests for local evaluation code
└── README_PROJECT_STRUCTURE.md
```

## Local Artifacts

These directories are expected on the training server but are ignored by git:

```
data/raw/                            generated/source datasets
data/processed/                      OPF JSONL splits and synthetic data
runs/final/opf_73class_v3_full/      current final checkpoint and reports
runs/final/opf_73class_v2b_full/     previous comparison checkpoint
runs/baselines/                      baseline reports
runs/ablations/                      ablation reports
outputs/                             ad hoc evaluation outputs
```

## Main Scripts

- `scripts/train_eval_v3.sh` — v3 train and external eval orchestration.
- `scripts/eval_char_spans_v2.py` — span-level exact, partial, and cost metrics.
- `scripts/run_opf_pipeline.py` — reproducible OPF launcher for v1/v2 style runs.
- `scripts/audit_diff.py` — create gold/model disagreement cases.
- `scripts/audit_run.py` — run the 27B auditor over disagreement cases.
- `scripts/audit_summary.py` — aggregate auditor verdicts.
- `scripts/synth_data.py` — synthesize targeted examples from audit findings.
- `scripts/synth_filter.py` — validate synthetic examples by type/form.
- `scripts/synth_remap_strict.py` — normalize aliases into the 73-class label space.
- `scripts/compare_v2b_v3.py` — compare v2b and v3 reports.

# ZYX Project Workspace

This repository is a multi-track PII redaction workspace containing:

- OPF token-classifier training and evaluation pipelines
- Qwen (LoRA / full SFT) model training and redaction demo services
- A model-agnostic redaction wrapper API (shared OCR/policy/schema surface)
- Supporting experiments, outputs, and archived migration content

## 1) Repository map

- `opf_au_pii/`: OPF AU-PII data pipeline, training/eval scripts, calibration/audit tools
- `Qwen3.5_9b_base_Distill/`: Qwen 9B base + LoRA distill route, demo API/UI docs and scripts
- `Qwen3.5_4b_base_Full_73class/`: Qwen 4B full supervised route for 73-class setup
- `Qwen3_4b_instruct_Distill/`: Qwen 3 4B instruct distill track
- `redaction-wrapper/`: backend-pluggable redaction service (OPF/Qwen backends)
- `LSJ/`: migrated legacy content that was moved out of the original ZYX layout

## 2) OPF small-model finetune -> training workflow (documented)

This section is the practical workflow for OPF in this repo, with the exact docs/scripts to read in order.

### Step A. Understand project structure and current best checkpoints

Read:

- `opf_au_pii/README_PROJECT_STRUCTURE.md`

Key points:

- Current best listed checkpoint: `runs/final/opf_73class_v3_full/checkpoint/`
- Previous baseline checkpoint: `runs/final/opf_73class_v2b_full/checkpoint/`
- Canonical taxonomy and label space paths are under `opf_au_pii/configs/`

### Step B. Understand OPF train command semantics

Read:

- `opf_au_pii/privacy-filter/FINETUNING.md`

Key points:

- Minimal command: `opf train train.jsonl --output-dir <checkpoint_dir>`
- Recommended: provide `--validation-dataset`
- For this project, use custom ontology via `--label-space-json`
- Output checkpoint artifacts include model weights and finetune summary

### Step C. Use the one-command orchestrator for prepare/train/eval

Read:

- `opf_au_pii/scripts/run_opf_pipeline.py`

This script orchestrates:

- Optional dataset preparation from raw JSON (`--raw-json`, `--prepare-script`)
- Label-space validation and run artifact capture
- Safety default: if no mode is provided, it defaults to `--smoke`
- Modes:
  - `--smoke`: tiny prefix subsets for sanity train
  - `--train`: full training
  - `--prepare-only`: schema/prepare stage only
  - `--eval-only`: evaluate an existing checkpoint
- Optional post-train eval:
  - `--eval-on-test`
  - `--char-eval` (calls `scripts/eval_char_spans_v2.py`)

Typical commands:

```bash
cd /home/admin/ZYX/opf_au_pii

# 1) Smoke run (recommended first)
python3 scripts/run_opf_pipeline.py \
  --smoke \
  --data-dir data/processed/data_opf \
  --taxonomy configs/taxonomy_v1.1.1.yaml \
  --label-space configs/custom_label_space_73.v1.1.1.json \
  --run-dir runs/ablations/smoke_v1 \
  --eval-on-test \
  --char-eval

# 2) Full train run
python3 scripts/run_opf_pipeline.py \
  --train \
  --data-dir data/processed/data_opf \
  --taxonomy configs/taxonomy_v1.1.1.yaml \
  --label-space configs/custom_label_space_73.v1.1.1.json \
  --run-dir runs/final/opf_73class_vX \
  --eval-on-test \
  --char-eval

# 3) Evaluate an existing checkpoint only
python3 scripts/run_opf_pipeline.py \
  --eval-only \
  --checkpoint runs/final/opf_73class_v3_full/checkpoint \
  --data-dir data/processed/data_opf \
  --run-dir runs/final/opf_73class_v3_evalonly \
  --eval-on-test \
  --char-eval
```

### Step D. Reproduce the v3 recipe (v2b + synthetic)

Read:

- `opf_au_pii/scripts/train_eval_v3.sh`

What it does:

- Concatenates `v2b` train split + strict synthetic data into `train_v3_full.jsonl`
- Runs OPF training with fixed hyperparameters and custom label space
- Evaluates on `external_1000` positive/hard negatives/trap (if present)
- Runs character-level span metrics on positive split

Run:

```bash
cd /home/admin/ZYX/opf_au_pii
bash scripts/train_eval_v3.sh
```

### Step E. Audit and synth-improvement loop (optional, for iteration)

Start from the script list in:

- `opf_au_pii/README_PROJECT_STRUCTURE.md`

Useful loop components:

- `scripts/audit_diff.py` -> find disagreement cases
- `scripts/audit_run.py` -> LLM-assisted audit
- `scripts/audit_summary.py` -> aggregate verdicts
- `scripts/synth_data.py` / `scripts/synth_filter.py` / `scripts/synth_remap_strict.py` -> build and filter synthetic expansions

## 3) Running the services

### A. Qwen 9B demo API/UI

Read:

- `Qwen3.5_9b_base_Distill/docs/redaction_demo_api.md`

Entry points:

- API health: `/api/health`
- Redaction: `/api/redact`
- File redaction (PDF/image/text): `/api/redact-file`
- Demo page: `/`

### B. Redaction wrapper (backend-pluggable)

Read:

- `redaction-wrapper/README.md`

Quick start:

```bash
cd /home/admin/ZYX/redaction-wrapper
export WRAPPER_BACKEND_CONFIG=$PWD/configs/backends/opf-v3.json
export WRAPPER_POLICY_CONFIG=$PWD/configs/policies/opf-v3-default-v1.json
./scripts/run_server.sh
```

## 4) Suggested working order for new experiments

1. Validate schema/taxonomy and run smoke training first.
2. Run full OPF training with frozen config snapshots under a new `runs/*` path.
3. Evaluate both aggregate metrics and character-level metrics.
4. Audit errors and feed high-quality synthetic data back into next iteration.
5. Expose the selected checkpoint through `redaction-wrapper` for stable API integration.

## 5) Notes

- Large model/checkpoint artifacts are intentionally ignored by git in this workspace.
- Keep all experiment runs under new run directories to preserve reproducibility.
- Prefer recording run-time parameters in `run_config.json` (already done by `run_opf_pipeline.py`).

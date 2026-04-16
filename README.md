# Teacher-Student LoRA Distillation Project

This project follows a clear teacher-student workflow for PII extraction:

1. Raw text is stored in `data/raw/au_pii_19000.json`
2. The teacher model performs span-level annotation
3. Teacher outputs are cleaned into student-trainable data
4. The cleaned labels are converted into BIO token tags
5. The student model is trained with LoRA / BIO distillation

## Recommended Workflow

For the current local environment, the recommended entry point is the one-shot script rather than running `src/train_distill.py` directly.

Why:

- The current teacher is a local `Qwen3.5-27B` GGUF model
- GGUF works well for teacher labeling
- `src/train_distill.py` expects a Hugging Face teacher model and is not designed for direct GGUF use

The safest workflow is therefore to use `scripts/run_train.sh` for the full pipeline.

## Project Structure

- `config/` - configuration files
- `data/raw/` - raw datasets
- `data/teacher/` - teacher labeling outputs generated at runtime
- `data/processed_teacher/` - cleaned teacher supervision data
- `data/processed_validation/` - external validation data prepared for evaluation
- `data/processed_bio/` - BIO token datasets generated for training
- `outputs/` - training outputs
- `src/` - data processing and training scripts
- `scripts/` - one-shot run scripts

## Quick Start

### 1. Small smoke test

Start with a small sample to verify teacher output quality:

```bash
TEACHER_MODEL_PATH=/home/admin/model/Qwen3.5/Qwen3.5-27B-Q4_K_M-GGUF MAX_SAMPLES=10 bash scripts/run_train.sh
```

### 2. Larger run

Once the teacher output looks stable, run a larger experiment:

```bash
TEACHER_MODEL_PATH=/home/admin/model/Qwen3.5/Qwen3.5-27B-Q4_K_M-GGUF MAX_SAMPLES=1000 TEACHER_MAX_NEW_TOKENS=256 bash scripts/run_train.sh
```

If `data/raw/cleaned_test_set.json` exists, the script will automatically use it as an independent validation set instead of splitting validation samples from the teacher-generated training data.

### 3. Remote teacher API option

```bash
TEACHER_API_URL=https://api.example.com TEACHER_API_KEY=xxx bash scripts/run_train.sh
```

## Automatically Generated Directories

These directories contain intermediate artifacts or outputs and can be regenerated if removed:

- `data/teacher/`
- `data/processed_teacher/`
- `data/processed_validation/`
- `data/processed_bio/`
- `outputs/`

## FAQ

### Why not run `src/train_distill.py` directly?

If the teacher is a GGUF file, or a directory that only contains GGUF files, `src/train_distill.py` will fail by design. That path is intended for Hugging Face teacher checkpoints rather than llama.cpp-style inference models.

### What if the script stops midway?

If `scripts/run_train.sh` reports that no valid BIO labels were parsed from the teacher output, it usually means the teacher response still contains reasoning text or malformed JSON. In that case, check:

- `data/teacher/teacher_labels.jsonl`
- whether the teacher is producing JSON only
- whether output remains stable on a small `MAX_SAMPLES` run first

## Key Configuration Files

- Training and LoRA settings: `config/lora_config.yaml`
- Teacher labeling: `src/teacher/teacher_labeling.py`
- BIO dataset building: `src/build_bio_dataset.py`
- BIO distillation training: `src/train_bio_distill.py`
- One-shot pipeline entry point: `scripts/run_train.sh`

## Suggested Next Steps

The repository is already cleaned up and the pipeline is running, so the most sensible next steps are:

1. Run a small smoke test
2. Inspect whether the teacher output is valid JSON
3. Scale up to larger training runs once the labels are stable

This helps avoid expensive full runs before confirming that the teacher output format is consistent.

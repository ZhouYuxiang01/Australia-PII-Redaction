# Qwen3.5 4B Full 73-Class Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new remote project that prepares 73-class AU PII JSON-span SFT data and trains Qwen3.5-4B-Base with full-parameter SFT.

**Architecture:** Reuse OPF's taxonomy as the canonical label source, build JSON-span SFT rows from the 19,000-record raw dataset, and train with TRL SFTTrainer without PEFT. JSON spans are used because full labels contain overlapping spans that inline tags cannot represent.

**Tech Stack:** Python stdlib, PyYAML when available, Hugging Face Datasets, Transformers, TRL, PyTorch, bitsandbytes optimizer profile.

---

### Task 1: Data Builder

**Files:**
- Create: `scripts/build_sft_dataset_73.py`
- Test: `tests/test_build_sft_dataset_73.py`

- [x] **Step 1: Write failing tests** for taxonomy alias mapping, duplicate source synonym dedupe, overlap retention, and SFT message shape.
- [x] **Step 2: Verify RED** with `python -m unittest discover -s Qwen3.5_4b_base_Full_73class\tests -v`; expected missing module.
- [x] **Step 3: Implement builder** with 77-source to 73-class mapping and JSON span assistant output.
- [x] **Step 4: Verify GREEN** with local unittest discovery; expected all tests pass.

### Task 2: Full-Parameter Train Script

**Files:**
- Create: `scripts/train_full_4b.py`
- Test: `tests/test_train_full_4b.py`

- [x] **Step 1: Write failing config tests** for Qwen3.5-4B-Base paths, full finetune mode, no LoRA, and smoke profile.
- [x] **Step 2: Verify RED** with local unittest discovery; expected missing module.
- [x] **Step 3: Implement train script** with heavy ML imports inside `train()`, `peft_config=None`, and profile-driven SFTConfig.
- [x] **Step 4: Verify GREEN** with local unittest discovery; expected all tests pass.

### Task 3: Remote Project Assembly

**Files:**
- Create remote: `/home/admin/ZYX/Qwen3.5_4b_base_Full_73class`
- Copy: `configs/taxonomy_v1.1.1.yaml`
- Copy: `configs/custom_label_space_73.v1.1.1.json`
- Link or copy: `data/raw/au_pii_19000_final.json`

- [x] **Step 1: Sync project files** from local staging to remote.
- [x] **Step 2: Copy OPF taxonomy/configs** into the new remote project.
- [x] **Step 3: Link raw data** from the existing Qwen 9B project.
- [x] **Step 4: Run remote unit tests** with `/home/admin/miniconda3/bin/python -m unittest discover -s tests -v`.
- [x] **Step 5: Run remote data build** with `bash scripts/run_build_dataset.sh`.
- [x] **Step 6: Inspect `data/processed/meta.json`** for 77 raw types, 73 classes, 0 dropped spans.

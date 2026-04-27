# Qwen3.5 4B Full 73-Class Design

## Goal

Create a new project under `/home/admin/ZYX` for full-parameter SFT of `/home/admin/model/Qwen3.5-4B-Base` on the 19,000-record AU PII dataset.

## Label Decision

Use OPF's 73 canonical trainable classes. OPF maps all 77 raw source labels into those 73 classes and drops no raw source type coverage. The four intentional merges are email, phone, payment card number, and vehicle identifier synonyms.

## Output Format

Use JSON span output rather than tagged text. The full 73-class data contains overlapping spans, especially `GEOLOCATION_INFORMATION` with `LATITUDE` and `LONGITUDE`, which cannot be represented reliably with inline tags.

## Components

- `scripts/build_sft_dataset_73.py`: converts raw records and OPF taxonomy into train/dev/test SFT JSONL.
- `scripts/train_full_4b.py`: runs full-parameter SFT without PEFT or LoRA.
- `tests/`: validates label mapping, duplicate merge behavior, overlap retention, and full-training config.
- `README.md`: records commands and expected outputs.

## Verification

Run unit tests locally and remotely. Then run the data builder against the real raw 19,000-record input and confirm 77 raw types, 73 trainable classes, zero dropped spans, and retained overlap count.


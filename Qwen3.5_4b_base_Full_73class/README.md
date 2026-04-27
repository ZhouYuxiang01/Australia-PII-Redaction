# Qwen3.5 4B Base Full 73-Class AU PII Training

This project trains `/home/admin/model/Qwen3.5-4B-Base` with full-parameter SFT on the 19,000-record AU PII dataset.

## Label Policy

The project uses the OPF canonical 73-class taxonomy, not the raw 77 source labels directly. All 77 source labels remain covered through this mapping:

- `EMAIL_ADDRESS` + `WORK_EMAIL` -> `EMAIL`
- `AU_PHONE` + `WORK_PHONE` -> `PHONE`
- `PAYMENT_CARD_NUMBER` + `HASHED_PAYMENT_CARD_NUMBER` -> `PAYMENT_CARD_NUMBER`
- `NUMBER_PLATE` + `VEHICLE_REGO` -> `VEHICLE_ID`

The training output format is JSON spans:

```json
{"spans":[{"start":0,"end":5,"type":"PERSON","value":"Alice"}]}
```

Tagged text is not used for this full-label project because the 73-class data contains overlapping spans, for example `GEOLOCATION_INFORMATION` containing `LATITUDE` and `LONGITUDE`.

## Paths

- Raw data: `data/raw/au_pii_19000_final.json`
- Taxonomy: `configs/taxonomy_v1.1.1.yaml`
- Processed SFT data: `data/processed/qwen_sft_*.jsonl`
- Training output: `outputs/qwen3_5_4b_base_full_73class`

## Build Data

```bash
cd /home/admin/ZYX/Qwen3.5_4b_base_Full_73class/scripts
bash run_build_dataset.sh
```

Expected high-level output:

- `raw_pii_type_count`: 77
- `class_count`: 73
- `split_sizes.train`: about 30,400 rows
- `deduped_after_mapping_count`: 900
- `overlap_count`: non-zero and retained in JSON spans
- `dropped_span_count`: 0

## Train

Smoke test:

```bash
cd /home/admin/ZYX/Qwen3.5_4b_base_Full_73class/scripts
TRAIN_PROFILE=smoke bash run_train_full_4b.sh
```

Full run:

```bash
cd /home/admin/ZYX/Qwen3.5_4b_base_Full_73class/scripts
TRAIN_PROFILE=safe_full bash run_train_full_4b.sh
```

Profiles:

- `smoke`: 20 steps, checks formatting and memory.
- `safe_full`: full-parameter training with `paged_adamw_8bit`, gradient checkpointing, batch 1, grad accumulation 16.
- `adamw_full`: full-parameter training with standard `adamw_torch`; use only if memory allows.

## Tests

```bash
cd /home/admin/ZYX/Qwen3.5_4b_base_Full_73class
python3 -m unittest discover -s tests -v
```


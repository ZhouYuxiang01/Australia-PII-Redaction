#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/home/admin/miniconda3/bin/python}"

"${PYTHON_BIN}" build_sft_dataset_73.py \
  --raw ../data/raw/au_pii_19000_final.json \
  --taxonomy ../configs/taxonomy_v1.1.1.yaml \
  --out-dir ../data/processed \
  --seed 42 \
  --train-ratio 0.8 \
  --dev-ratio 0.1 \
  --test-ratio 0.1 \
  --train-negatives-per-record 1 \
  --eval-negatives all


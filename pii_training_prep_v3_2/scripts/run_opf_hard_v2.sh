#!/usr/bin/env bash
# OPF retrain v2: more epochs + larger effective batch.
#
# vs v1 (runs/opf_hard_79):
#   epochs:           1   -> 3
#   batch-size:       1   -> 2
#   grad-accum-steps: 1   -> 8     (effective batch 16, was 1)
#   learning-rate:    1e-5 (unchanged, proven safe)
#
# Output goes to a NEW dir so v1 stays as rollback.
set -euo pipefail

PROJECT_ROOT="/home/admin/ZYX/pii_training_prep_v3_2"
OPF_ROOT="/home/admin/ZYX/opf_au_pii/privacy-filter"
OUTPUT_DIR="${PROJECT_ROOT}/runs/opf_hard_79_v2"
LOG_PATH="${PROJECT_ROOT}/logs/opf_hard_79_v2_train.log"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"

{
  echo "started_at=$(date -Iseconds)"
  echo "project_root=${PROJECT_ROOT}"
  echo "opf_root=${OPF_ROOT}"
  echo "output_dir=${OUTPUT_DIR}"
  source /home/admin/miniconda3/etc/profile.d/conda.sh
  conda activate opf
  cd "${OPF_ROOT}"
  export PYTHONPATH=.
  export OPF_TRAIN_PROGRESS_INTERVAL_S=60
  python -m opf train \
    "${PROJECT_ROOT}/data/train/opf_train_opf_format.jsonl" \
    --validation-dataset "${PROJECT_ROOT}/data/train/opf_dev_opf_format.jsonl" \
    --label-space-json "${PROJECT_ROOT}/pii_schema/opf_label_space_79.json" \
    --output-dir "${OUTPUT_DIR}" \
    --overwrite-output \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum-steps 8 \
    --learning-rate 1e-5 \
    --device cuda
  echo "finished_at=$(date -Iseconds)"
} >> "${LOG_PATH}" 2>&1

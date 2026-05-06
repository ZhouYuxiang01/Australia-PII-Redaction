#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/admin/ZYX/pii_training_prep_v3_2"
OPF_ROOT="/home/admin/ZYX/opf_au_pii/privacy-filter"
LOG_PATH="${PROJECT_ROOT}/logs/opf_hard_79_train.log"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/runs/opf_hard_79"

{
  echo "started_at=$(date -Iseconds)"
  echo "project_root=${PROJECT_ROOT}"
  echo "opf_root=${OPF_ROOT}"
  echo "output_dir=${PROJECT_ROOT}/runs/opf_hard_79"
  source /home/admin/miniconda3/etc/profile.d/conda.sh
  conda activate opf
  cd "${OPF_ROOT}"
  export PYTHONPATH=.
  export OPF_TRAIN_PROGRESS_INTERVAL_S=60
  python -m opf train \
    "${PROJECT_ROOT}/data/train/opf_train_opf_format.jsonl" \
    --validation-dataset "${PROJECT_ROOT}/data/train/opf_dev_opf_format.jsonl" \
    --label-space-json "${PROJECT_ROOT}/pii_schema/opf_label_space_79.json" \
    --output-dir "${PROJECT_ROOT}/runs/opf_hard_79" \
    --overwrite-output \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum-steps 1 \
    --learning-rate 1e-5 \
    --device cuda
  echo "finished_at=$(date -Iseconds)"
} >> "${LOG_PATH}" 2>&1

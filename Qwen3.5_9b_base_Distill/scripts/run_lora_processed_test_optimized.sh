#!/usr/bin/env bash
set -uo pipefail

cd /home/admin/ZYX/Qwen3.5_9b_base_Distill/scripts

RUN_TAG=20260422_conda3
PYTHON_BIN=/home/admin/miniconda3/bin/python
BASE_MODEL=/home/admin/model/Qwen3.5-9B-Base
BATCH_SIZE=32
OUT_DIR=/home/admin/ZYX/Qwen3.5_9b_base_Distill/outputs/qwen3_5_9b_base_lora_tagged_28_fastretry
LOG_DIR=/home/admin/ZYX/Qwen3.5_9b_base_Distill/scripts/logs
PREDICTIONS_OUT="${OUT_DIR}/processed_test_predictions_optimized_${RUN_TAG}.jsonl"
SUMMARY_OUT="${OUT_DIR}/processed_test_summary_optimized_${RUN_TAG}.json"
LOG_PATH="${LOG_DIR}/05_eval_processed_test_lora_optimized_${RUN_TAG}.log"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

{
  echo "START $(date -Is)"
  echo "PREDICTIONS_OUT=${PREDICTIONS_OUT}"
  echo "SUMMARY_OUT=${SUMMARY_OUT}"
  echo "LOG_PATH=${LOG_PATH}"
} | tee -a "${LOG_PATH}"

"${PYTHON_BIN}" -u 05_eval_processed_test_lora_optimized.py \
  --resume \
  --base-model "${BASE_MODEL}" \
  --batch-size "${BATCH_SIZE}" \
  --predictions-out "${PREDICTIONS_OUT}" \
  --summary-out "${SUMMARY_OUT}" \
  2>&1 | tee -a "${LOG_PATH}"

status=${PIPESTATUS[0]}

{
  echo "EXIT ${status}"
  echo "END $(date -Is)"
} | tee -a "${LOG_PATH}"

exit "${status}"

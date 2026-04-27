#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-/home/admin/miniconda3/bin/python}"
PROFILE="${TRAIN_PROFILE:-safe_full}"
AUTO_RESUME="${AUTO_RESUME:-auto}"
LOG_DIR="../logs"
LOG_PATH="${LOG_DIR}/train_full_4b_${PROFILE}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

{
  echo "START $(date -Is)"
  echo "PROFILE=${PROFILE}"
  echo "AUTO_RESUME=${AUTO_RESUME}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "LOG_PATH=${LOG_PATH}"
} | tee -a "${LOG_PATH}"

"${PYTHON_BIN}" train_full_4b.py \
  --profile "${PROFILE}" \
  --auto-resume "${AUTO_RESUME}" \
  2>&1 | tee -a "${LOG_PATH}"

status=${PIPESTATUS[0]}

{
  echo "EXIT ${status}"
  echo "END $(date -Is)"
} | tee -a "${LOG_PATH}"

exit "${status}"


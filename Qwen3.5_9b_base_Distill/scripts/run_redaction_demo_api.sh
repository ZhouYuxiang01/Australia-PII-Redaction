#!/usr/bin/env bash
set -uo pipefail

cd /home/admin/ZYX/Qwen3.5_9b_base_Distill/scripts

PYTHON_BIN=/home/admin/miniconda3/bin/python
HOST=${DEMO_HOST:-0.0.0.0}
PORT=${DEMO_PORT:-8090}
MODE=${DEMO_MODEL_MODE:-live}
LOG_DIR=/home/admin/ZYX/Qwen3.5_9b_base_Distill/scripts/logs
LOG_PATH="${LOG_DIR}/redaction_demo_api_${MODE}_${PORT}.log"

mkdir -p "${LOG_DIR}"

{
  echo "START $(date -Is)"
  echo "MODE=${MODE}"
  echo "HOST=${HOST}"
  echo "PORT=${PORT}"
  echo "LOG_PATH=${LOG_PATH}"
} | tee -a "${LOG_PATH}"

DEMO_MODEL_MODE="${MODE}" "${PYTHON_BIN}" -m uvicorn redaction_demo_api:app \
  --host "${HOST}" \
  --port "${PORT}" \
  2>&1 | tee -a "${LOG_PATH}"

status=${PIPESTATUS[0]}

{
  echo "EXIT ${status}"
  echo "END $(date -Is)"
} | tee -a "${LOG_PATH}"

exit "${status}"

#!/usr/bin/env bash
# Launch the redaction wrapper FastAPI server.
#
# Required env (or set via -b / -p flags):
#   WRAPPER_BACKEND_CONFIG   path to configs/backends/<x>.json
#   WRAPPER_POLICY_CONFIG    path to configs/policies/<y>.json
#
# Optional:
#   WRAPPER_PORT             default 8090
#   WRAPPER_HOST             default 0.0.0.0
#   WRAPPER_PYTHON           default /home/admin/miniconda3/envs/opf/bin/python
#                            (use the opf conda env when running OPF backend;
#                            qwen backends may want a different env)
#   WRAPPER_OCR_PROVIDER     auto | rapidocr | paddle | tesseract (default auto)
#                            auto uses local RapidOCR, then Tesseract.
#   RAPIDOCR_MIN_CONFIDENCE  default 0.30
#   PADDLEOCR_LANG           default en
#   PADDLEOCR_USE_GPU        default false
#   PADDLEOCR_MIN_CONFIDENCE default 0.30

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_BACKEND="${REPO_ROOT}/configs/backends/opf-v3.json"
DEFAULT_POLICY="${REPO_ROOT}/configs/policies/opf-v3-default-v1.json"

while getopts "b:p:" opt; do
  case $opt in
    b) BACKEND_OVERRIDE="$OPTARG" ;;
    p) POLICY_OVERRIDE="$OPTARG" ;;
  esac
done

export WRAPPER_BACKEND_CONFIG="${BACKEND_OVERRIDE:-${WRAPPER_BACKEND_CONFIG:-$DEFAULT_BACKEND}}"
export WRAPPER_POLICY_CONFIG="${POLICY_OVERRIDE:-${WRAPPER_POLICY_CONFIG:-$DEFAULT_POLICY}}"

PORT="${WRAPPER_PORT:-8090}"
HOST="${WRAPPER_HOST:-0.0.0.0}"
PYTHON_BIN="${WRAPPER_PYTHON:-/home/admin/miniconda3/envs/opf/bin/python}"
LOG_DIR="${REPO_ROOT}/scripts/logs"
mkdir -p "${LOG_DIR}"

BACKEND_NAME="$(python3 -c "import json,sys; print(json.load(open('${WRAPPER_BACKEND_CONFIG}')).get('name','backend'))")"
LOG_PATH="${LOG_DIR}/redaction_wrapper_${BACKEND_NAME}_${PORT}.log"

{
  echo "START $(date -Is)"
  echo "BACKEND_CONFIG=${WRAPPER_BACKEND_CONFIG}"
  echo "POLICY_CONFIG=${WRAPPER_POLICY_CONFIG}"
  echo "BACKEND_NAME=${BACKEND_NAME}"
  echo "HOST=${HOST}  PORT=${PORT}"
  echo "PYTHON=${PYTHON_BIN}"
  echo "LOG=${LOG_PATH}"
  echo "----"
} | tee -a "${LOG_PATH}"

exec "${PYTHON_BIN}" -m uvicorn redaction.api.server:get_app \
  --factory \
  --host "${HOST}" \
  --port "${PORT}" \
  >> "${LOG_PATH}" 2>&1

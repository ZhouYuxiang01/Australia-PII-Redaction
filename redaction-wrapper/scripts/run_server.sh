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
#   WRAPPER_PYTHON           Python interpreter path.
#                            Auto-detected if unset:
#                              1. miniconda3/envs/opf/bin/python (has opf+torch+transformers+fastapi)
#                              2. $(command -v python3)
#   WRAPPER_QWEN_VL_MODEL    default /home/admin/model/Qwen3.5-9B-Base
#   WRAPPER_QWEN_VL_DEVICE   default cuda when available, otherwise cpu
#   WRAPPER_QWEN_VL_DTYPE    default bfloat16

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_BACKEND="${REPO_ROOT}/configs/backends/hybrid-opf-qwen.json"
DEFAULT_POLICY="${REPO_ROOT}/configs/policies/hybrid-80class-v1.json"

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

# ── Python interpreter detection ──────────────────────────────────────────
if [[ -n "${WRAPPER_PYTHON:-}" ]]; then
    PYTHON_BIN="${WRAPPER_PYTHON}"
elif [[ -x "${HOME}/miniconda3/envs/opf/bin/python" ]]; then
    PYTHON_BIN="${HOME}/miniconda3/envs/opf/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

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

# ── Preflight import check ────────────────────────────────────────────────
echo "Preflight: checking critical imports (opf, torch, transformers, fastapi) ..." | tee -a "${LOG_PATH}"
if "${PYTHON_BIN}" -c "import opf, torch, transformers, fastapi" 2>&1 | tee -a "${LOG_PATH}"; then
    echo "Preflight: OK" | tee -a "${LOG_PATH}"
else
    echo "" | tee -a "${LOG_PATH}"
    echo "FATAL: The Python interpreter at ${PYTHON_BIN} is missing required packages." | tee -a "${LOG_PATH}"
    echo "Required: opf, torch, transformers, fastapi" | tee -a "${LOG_PATH}"
    echo "" | tee -a "${LOG_PATH}"
    echo "Fix options:" | tee -a "${LOG_PATH}"
    echo "  1. Set WRAPPER_PYTHON to a conda env that has all four packages" | tee -a "${LOG_PATH}"
    echo "     Example: export WRAPPER_PYTHON=\$HOME/miniconda3/envs/opf/bin/python" | tee -a "${LOG_PATH}"
    echo "  2. Install the missing packages into $(dirname "${PYTHON_BIN}")" | tee -a "${LOG_PATH}"
    exit 1
fi

exec "${PYTHON_BIN}" -m uvicorn redaction.api.server:get_app \
  --factory \
  --host "${HOST}" \
  --port "${PORT}" \
  >> "${LOG_PATH}" 2>&1

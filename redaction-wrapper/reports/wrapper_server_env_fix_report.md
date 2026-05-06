# Server Environment Fix Report

**Date**: 2026-05-02  
**Scope**: `/home/admin/ZYX/redaction-wrapper`  

## Root Cause

`scripts/run_server.sh` defaulted to `$(command -v python3)` which resolves to `/usr/bin/python3` (system Python 3.12). That environment lacks `opf`, `torch`, `transformers`, `fastapi`, and `uvicorn` — all required by the hybrid OPF+Qwen backend.

```
Traceback:
  File "hybrid_opf_qwen.py", line 143:
    from opf import OPF
ModuleNotFoundError: No module named 'opf'
```

## Environment Analysis

| Environment | Python | opf | torch | transformers | fastapi | uvicorn |
|---|---|---|---|---|---|---|
| `opf` conda env | 3.11 | ✓ | ✓ | ✓ | ✓ | ✓ |
| `qwen` conda env | 3.10 | ✗ | ✓ | ✓ | ✗ | ✗ |
| `llm` conda env | — | ✗ | ✓ | ✓ | ✗ | ✗ |
| system `/usr/bin/python3` | 3.12 | ✗ | ✗ | ✗ | ✗ | ✗ |

**Only the `opf` conda env** (`/home/admin/miniconda3/envs/opf/bin/python`) has all required packages.

## Fix Applied

### `scripts/run_server.sh`

**Before**:
```bash
PYTHON_BIN="${WRAPPER_PYTHON:-$(command -v python3)}"
```
This resolves to `/usr/bin/python3` — system Python with no ML packages.

**After**:
```bash
if [[ -n "${WRAPPER_PYTHON:-}" ]]; then
    PYTHON_BIN="${WRAPPER_PYTHON}"
elif [[ -x "${HOME}/miniconda3/envs/opf/bin/python" ]]; then
    PYTHON_BIN="${HOME}/miniconda3/envs/opf/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi
```

Three-tier detection:
1. Explicit `WRAPPER_PYTHON` env var (user override)
2. Auto-detect `opf` conda env (has all required packages)
3. Fallback to system `python3`

### Preflight Import Check

Added before uvicorn launch:
```bash
"${PYTHON_BIN}" -c "import opf, torch, transformers, fastapi" || {
    echo "FATAL: The Python interpreter is missing required packages."
    echo "Required: opf, torch, transformers, fastapi"
    echo "Fix: export WRAPPER_PYTHON=\$HOME/miniconda3/envs/opf/bin/python"
    exit 1
}
```

## Verification

### Startup
```
START 2026-05-02T08:18:41+10:00
PYTHON=/home/admin/miniconda3/envs/opf/bin/python
Preflight: checking critical imports (opf, torch, transformers, fastapi) ...
Preflight: OK
INFO: Uvicorn running on http://0.0.0.0:8090
```

### /redact endpoint (payroll example)
```
POST /api/redact HTTP/1.1 → 200 OK
5 spans detected:
  PERSON          redact  top1=0.760 risk=0.120
  AU_BANK_ACCOUNT redact  top1=0.959 risk=0.041
  AU_BANK_ACCOUNT redact  top1=0.991 risk=0.009
  SALARY          redact  top1=0.895 risk=0.052
  WORK_EMAIL      redact  top1=0.943 risk=0.023
```

### Hard-negative example
```
STUDENT_ID  ignore  top1=0.117 risk=-- reason=low_pii_evidence
```

### Tests
All 47 unit tests pass. Frontend static test passes.

## No Changes To
- Model weights / checkpoints — untouched
- OPF or Qwen inference weights — untouched
- API output format — no new fields in `spans[]` (existing evidence fields from prior fix)
- Training code — untouched

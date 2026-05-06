# Run Qwen3.5-27B Teacher with vLLM

This is for Stage 2 teacher calls only. Do not start student training from these commands.

## 1. Clean Existing Compute Processes

Keep desktop/session processes such as Xorg, GNOME, Firefox, and VS Code. Stop only stale model-serving or inference jobs that consume GPU memory.

```bash
nvidia-smi
ps -eo pid,etime,pcpu,pmem,cmd | grep -E 'vllm|uvicorn|qwen|torch|transformers' | grep -v grep
kill <stale_compute_pid>
```

## 2. Install vLLM Runtime

Use an isolated environment if vLLM is not already installed.

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2
/home/admin/miniconda3/bin/python -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install --upgrade pip
python -m pip install 'vllm' 'openai'
```

If no compatible vLLM wheel exists for this machine, keep using the `transformers_local` backend and record the install error in `reports/stage2_vllm_smoke_report.json`.

## 3. Launch OpenAI-Compatible vLLM Server

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2
source .venv-vllm/bin/activate

python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model /home/admin/model/qwen3.5-27b \
  --served-model-name qwen3.5-27b \
  --trust-remote-code \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

Leave this process running in its terminal.

## 4. Smoke Test the Existing 20 Prompts

In another terminal:

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2
PYTHONPATH=src /home/admin/miniconda3/bin/python scripts/run_stage2_teacher_dryrun.py \
  --backend vllm_openai \
  --base-url http://localhost:8000/v1 \
  --model-path qwen3.5-27b \
  --limit 20 \
  --max-new-tokens 96
```

Expected outputs:

- `data/generated/stage2_vllm_dryrun_20_raw.jsonl`
- `data/generated/stage2_vllm_dryrun_20_converted.jsonl`
- `reports/stage2_vllm_smoke_report.json`
- `reports/stage2_vllm_output_errors.json`

Acceptance gate:

```bash
python - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('reports/stage2_vllm_smoke_report.json').read_text())
assert r['prompts_attempted'] == 20
assert r['valid_json_outputs'] >= 18
assert r['validation_error_count'] == 0
assert r['labels_outside_training_space'] == {}
assert r['merged_into_training_dataset'] is False
print('VLLM_SMOKE_OK')
PY
```


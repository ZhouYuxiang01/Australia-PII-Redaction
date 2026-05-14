#!/usr/bin/env bash
# Wait for dump to complete, run calibration, run stage4 eval with new thresholds.
set -uo pipefail

REPO=/home/admin/ZYX/redaction-wrapper
PYTHON=/home/admin/miniconda3/envs/opf/bin/python
DUMP=$REPO/reports/dev_predictions_for_calibration.jsonl
THRESHOLDS=$REPO/configs/postprocess/per_label_thresholds.json
EVAL_BACKEND=$REPO/configs/backends/hybrid-opf-qwen4b-calibrated.json
LOG=$REPO/logs/calibration_pipeline.log

mkdir -p "$REPO/logs"
exec >> "$LOG" 2>&1
echo "=== started $(date -Iseconds) ==="

while pgrep -f "scripts/dump_dev_predictions.py" > /dev/null; do
  echo "[wait] dump still running ($(wc -l < "$DUMP") lines so far)"
  sleep 30
done
echo "[wait] dump finished, $(wc -l < "$DUMP") lines total"

cd "$REPO"
echo "=== calibrating ==="
"$PYTHON" scripts/calibrate_per_label_thresholds.py
ls -la "$THRESHOLDS"

echo "=== building calibrated backend config ==="
"$PYTHON" -c "
import json
src = json.load(open('$REPO/configs/backends/hybrid-opf-qwen4b.json'))
src['per_label_thresholds_path'] = '$THRESHOLDS'
src['name'] = 'hybrid-opf-qwen4b-calibrated'
src['model_version'] = 'opf-hard-79-qwen4b-spanhead-v1-percalib'
open('$EVAL_BACKEND', 'w').write(json.dumps(src, indent=2, ensure_ascii=False))
"
ls -la "$EVAL_BACKEND"

echo "=== stage4 eval (calibrated) ==="
"$PYTHON" scripts/run_hybrid_offline_eval.py \
  --backend "$EVAL_BACKEND" \
  --policy "$REPO/configs/policies/hybrid-80class-v2-4b.json" \
  --test-set /home/admin/ZYX/pii_training_prep_v3_2/data/train/opf_test_opf_format.jsonl \
  --out-dir "$REPO/reports/" \
  --limit 0

echo "=== done $(date -Iseconds) ==="

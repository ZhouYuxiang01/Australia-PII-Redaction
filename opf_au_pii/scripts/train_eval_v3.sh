#!/bin/bash
# Train OPF v3 = v2b_full + synthetic, then eval on external_1000.
# Same hyperparameters as v2b for fair comparison.

set -euo pipefail

ROOT=/home/admin/ZYX/opf_au_pii
OPF=/home/admin/miniconda3/envs/opf/bin/opf

V2B_TRAIN=${ROOT}/data/processed/data_opf_v2b/train_v2b_full.jsonl
V2B_DEV=${ROOT}/data/processed/data_opf_v2b/dev.jsonl
SYNTH=${ROOT}/data/processed/data_opf_v3/synth_opf_strict.jsonl

V3_DIR=${ROOT}/data/processed/data_opf_v3
V3_TRAIN=${V3_DIR}/train_v3_full.jsonl
RUN_DIR=${ROOT}/runs/final/opf_73class_v3_full
EXTERNAL=${ROOT}/data/processed/data_external_1000

mkdir -p "${V3_DIR}" "${RUN_DIR}"

if [ ! -s "${SYNTH}" ]; then
    echo "ERROR: synth file ${SYNTH} not found or empty"
    exit 1
fi

echo "=== building v3 training set ==="
cat "${V2B_TRAIN}" "${SYNTH}" > "${V3_TRAIN}"
n_v2b=$(wc -l < "${V2B_TRAIN}")
n_synth=$(wc -l < "${SYNTH}")
n_v3=$(wc -l < "${V3_TRAIN}")
echo "v2b: ${n_v2b}  synth: ${n_synth}  v3: ${n_v3}"

# Quick sanity validation on synth: each span offset must match the value
python3 - <<'PY'
import json, sys
ok = bad = 0
with open("/home/admin/ZYX/opf_au_pii/data/processed/data_opf_v3/synth_opf_strict.jsonl") as f:
    for line in f:
        r = json.loads(line)
        text = r["text"]
        for k, ranges in r["spans"].items():
            val = k.split(":", 1)[1].strip()
            for s, e in ranges:
                if text[s:e] == val:
                    ok += 1
                else:
                    bad += 1
print(f"span offset check: ok={ok} bad={bad}")
if bad:
    print("WARN: bad offsets present in synth; inspect before training", file=sys.stderr)
PY

echo "=== training opf v3 ==="
cd "${ROOT}"
"${OPF}" train "${V3_TRAIN}" \
    --validation-dataset "${V2B_DEV}" \
    --label-space-json "${ROOT}/configs/custom_label_space_73.v1.1.1.json" \
    --checkpoint /home/admin/.opf/privacy_filter \
    --device cuda \
    --epochs 1 \
    --batch-size 4 \
    --grad-accum-steps 1 \
    --learning-rate 1e-5 \
    --output-dir "${RUN_DIR}/checkpoint" \
    --overwrite-output \
    --shuffle-seed 42

echo "=== eval on external_1000 ==="
mkdir -p "${RUN_DIR}/external_1000"
"${OPF}" eval "${EXTERNAL}/positive_1000.jsonl" \
    --checkpoint "${RUN_DIR}/checkpoint" \
    --device cuda \
    --predictions-out "${RUN_DIR}/external_1000/positive_predictions.jsonl" \
    --metrics-out "${RUN_DIR}/external_1000/positive_metrics.json"

"${OPF}" eval "${EXTERNAL}/hard_negatives.jsonl" \
    --checkpoint "${RUN_DIR}/checkpoint" \
    --device cuda \
    --predictions-out "${RUN_DIR}/external_1000/hardneg_predictions.jsonl" \
    --metrics-out "${RUN_DIR}/external_1000/hardneg_metrics.json"

if [ -f "${EXTERNAL}/by_difficulty/trap.jsonl" ]; then
    "${OPF}" eval "${EXTERNAL}/by_difficulty/trap.jsonl" \
        --checkpoint "${RUN_DIR}/checkpoint" \
        --device cuda \
        --predictions-out "${RUN_DIR}/external_1000/trap_predictions.jsonl" \
        --metrics-out "${RUN_DIR}/external_1000/trap_metrics.json"
fi

echo "=== char-level eval on external_1000 positive ==="
python3 "${ROOT}/scripts/eval_char_spans_v2.py" \
    --gold "${EXTERNAL}/positive_1000.jsonl" \
    --pred "${RUN_DIR}/external_1000/positive_predictions.jsonl" \
    --out "${RUN_DIR}/external_1000/positive_char_metrics.json"

echo "=== done. checkpoint at ${RUN_DIR}/checkpoint ==="

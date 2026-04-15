#!/bin/bash
set -e

export PYTHONPATH="$PWD:${PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x "$PWD/.venv/bin/python" ]]; then
  PYTHON_BIN="$PWD/.venv/bin/python"
fi

CONFIG_FILE="config/lora_config.yaml"
RAW_INPUT="data/raw/au_pii_19000.json"
TEACHER_DIR="data/teacher"
PROCESSED_TEACHER_DIR="data/processed_teacher"
PROCESSED_BIO_DIR="data/processed_bio"
TRAIN_MODE="${TRAIN_MODE:-bio}"
TEACHER_MAX_NEW_TOKENS="${TEACHER_MAX_NEW_TOKENS:-512}"
MAX_SAMPLES_ARG=()
if [[ -n "$MAX_SAMPLES" ]]; then
  MAX_SAMPLES_ARG=(--max_samples "$MAX_SAMPLES")
fi

IS_GGUF_TEACHER=0
if [[ -n "$TEACHER_MODEL_PATH" ]]; then
  if [[ "$TEACHER_MODEL_PATH" == *.gguf ]]; then
    IS_GGUF_TEACHER=1
  elif [[ -d "$TEACHER_MODEL_PATH" ]] && compgen -G "$TEACHER_MODEL_PATH/*.gguf" > /dev/null; then
    IS_GGUF_TEACHER=1
  fi
fi

if [[ -z "${TEACHER_MODEL_PATH}" && -z "${TEACHER_API_URL}" ]]; then
  echo "请先设置 TEACHER_MODEL_PATH，或者 TEACHER_API_URL 和 TEACHER_API_KEY。"
  exit 1
fi

if [[ -n "$TEACHER_MODEL_PATH" && ! -d "$TEACHER_MODEL_PATH" && ! -f "$TEACHER_MODEL_PATH" ]]; then
  echo "TEACHER_MODEL_PATH 指定的本地模型路径不存在：$TEACHER_MODEL_PATH"
  echo "请确认这是一个真实的本地模型目录，或者改用远程 teacher API。"
  exit 1
fi

LABEL_ARGS=(--input "$RAW_INPUT" --output_dir "$TEACHER_DIR" --max_new_tokens "$TEACHER_MAX_NEW_TOKENS")
if [[ -n "$TEACHER_MODEL_PATH" ]]; then
  LABEL_ARGS+=(--model_path "$TEACHER_MODEL_PATH")
fi
if [[ -n "$TEACHER_API_URL" ]]; then
  LABEL_ARGS+=(--api_url "$TEACHER_API_URL")
fi
if [[ -n "$TEACHER_API_KEY" ]]; then
  LABEL_ARGS+=(--api_key "$TEACHER_API_KEY")
fi

"$PYTHON_BIN" src/teacher/teacher_labeling.py "${LABEL_ARGS[@]}" "${MAX_SAMPLES_ARG[@]}"

"$PYTHON_BIN" src/teacher/prepare_teacher_student_data.py \
  --input "$TEACHER_DIR/teacher_labels.jsonl" \
  --output_dir "$PROCESSED_TEACHER_DIR" \
  --val_ratio 0.05

if [[ "$TRAIN_MODE" == "bio" ]]; then
  "$PYTHON_BIN" src/build_bio_dataset.py \
    --train_input "$PROCESSED_TEACHER_DIR/train.jsonl" \
    --validation_input "$PROCESSED_TEACHER_DIR/val.jsonl" \
    --output_dir "$PROCESSED_BIO_DIR" \
    --model_name "gpt2" \
    --max_length 512

  LABEL_COUNT=$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
p = Path('data/processed_bio/label_map.json')
if not p.exists():
    print(0)
else:
    with p.open('r', encoding='utf-8') as f:
        print(len(json.load(f).get('labels', [])))
PY
)

  if [[ "$LABEL_COUNT" -le 1 ]]; then
    echo "teacher 输出暂未解析出有效 BIO 标签，已停止训练。请先检查 teacher 标注质量后再重试。"
    exit 1
  fi

  echo "启动 BIO token distillation 训练。"
  "$PYTHON_BIN" src/train_bio_distill.py --config "$CONFIG_FILE"
elif [[ "$IS_GGUF_TEACHER" == "1" ]]; then
  echo "检测到 GGUF teacher；当前将使用它完成 teacher 标注，并继续用生成式伪标签训练 student。"
  "$PYTHON_BIN" src/train_lora.py --config "$CONFIG_FILE"
else
  "$PYTHON_BIN" src/train_distill.py --config "$CONFIG_FILE" \
    ${TEACHER_MODEL_PATH:+--teacher_model_name "$TEACHER_MODEL_PATH"}
fi

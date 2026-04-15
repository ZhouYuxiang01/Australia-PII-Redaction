#!/bin/bash
set -e

CONFIG_FILE="config/lora_config.yaml"
RAW_INPUT="data/raw/au_pii_19000.json"
TEACHER_DIR="data/teacher"
PROCESSED_TEACHER_DIR="data/processed_teacher"

if [[ -z "${TEACHER_MODEL_PATH}" && -z "${TEACHER_API_URL}" ]]; then
  echo "请先设置 TEACHER_MODEL_PATH，或者 TEACHER_API_URL 和 TEACHER_API_KEY。"
  exit 1
fi

if [[ -n "$TEACHER_MODEL_PATH" && ! -d "$TEACHER_MODEL_PATH" && ! -f "$TEACHER_MODEL_PATH" ]]; then
  echo "TEACHER_MODEL_PATH 指定的本地模型路径不存在：$TEACHER_MODEL_PATH"
  echo "请确认这是一个真实的本地模型目录，或者改用远程 teacher API。"
  exit 1
fi

LABEL_ARGS=(--input "$RAW_INPUT" --output_dir "$TEACHER_DIR")
if [[ -n "$TEACHER_MODEL_PATH" ]]; then
  LABEL_ARGS+=(--model_path "$TEACHER_MODEL_PATH")
fi
if [[ -n "$TEACHER_API_URL" ]]; then
  LABEL_ARGS+=(--api_url "$TEACHER_API_URL")
fi
if [[ -n "$TEACHER_API_KEY" ]]; then
  LABEL_ARGS+=(--api_key "$TEACHER_API_KEY")
fi

python src/teacher/teacher_labeling.py "${LABEL_ARGS[@]}"

python src/teacher/prepare_teacher_student_data.py \
  --input "$TEACHER_DIR/teacher_labels.jsonl" \
  --output_dir "$PROCESSED_TEACHER_DIR" \
  --val_ratio 0.05

python src/train_distill.py --config "$CONFIG_FILE" \
  ${TEACHER_MODEL_PATH:+--teacher_model_name "$TEACHER_MODEL_PATH"}

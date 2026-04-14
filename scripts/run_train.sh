#!/bin/bash
set -e

CONFIG_FILE="config/lora_config.yaml"
python src/data_preparation.py --input data/raw/dataset.json --output_dir data/processed --text_key input_text --target_key target_text
python src/train_lora.py --config "$CONFIG_FILE"

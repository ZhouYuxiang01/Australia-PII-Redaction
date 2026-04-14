import argparse
import json
import os
import random
from pathlib import Path


def load_raw_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        raise ValueError("数据文件应当是 JSON 数组格式，而不是对象。")
    return data


def make_example(record, text_key, target_key, prompt_template=None):
    text = record.get(text_key, "")
    target = record.get(target_key, "")
    if prompt_template:
        return prompt_template.replace("{input_text}", text).replace("{target_text}", target)
    return json.dumps({"input_text": text, "target_text": target}, ensure_ascii=False)


def prepare_dataset(input_path, output_dir, text_key, target_key, val_ratio, seed, prompt_template=None):
    raw = load_raw_json(input_path)
    random.seed(seed)
    random.shuffle(raw)
    split = int(len(raw) * (1 - val_ratio))
    train = raw[:split]
    val = raw[split:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as f_train:
        for record in train:
            f_train.write(json.dumps({"input_text": record.get(text_key, ""), "target_text": record.get(target_key, "")}, ensure_ascii=False) + "\n")

    with val_path.open("w", encoding="utf-8") as f_val:
        for record in val:
            f_val.write(json.dumps({"input_text": record.get(text_key, ""), "target_text": record.get(target_key, "")}, ensure_ascii=False) + "\n")

    print(f"已生成: {train_path} ({len(train)} 条), {val_path} ({len(val)} 条)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备 LoRA 微调数据集")
    parser.add_argument("--input", required=True, help="原始 JSON 文件路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--text_key", default="input_text", help="输入字段名称")
    parser.add_argument("--target_key", default="target_text", help="目标字段名称")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    prepare_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        text_key=args.text_key,
        target_key=args.target_key,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

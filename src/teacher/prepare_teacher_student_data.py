import argparse
import json
import random
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_last_json_object(text: str) -> str:
    if not text:
        return ""

    if "```json" in text:
        parts = text.split("```json")
        last_block = parts[-1]
        if "```" in last_block:
            last_block = last_block.split("```", 1)[0]
        return last_block.strip()

    # Find the last top-level JSON object in the text.
    last_open = text.rfind("{")
    if last_open == -1:
        return text.strip()

    stack = 0
    escaped = False
    in_string = False
    for idx, ch in enumerate(text[last_open:], start=last_open):
        if ch == "\\" and not escaped:
            escaped = True
            continue
        if ch == '"' and not escaped:
            in_string = not in_string
        if in_string:
            escaped = False
            continue
        if ch == "{" and not in_string:
            stack += 1
        elif ch == "}" and not in_string:
            stack -= 1
            if stack == 0:
                candidate = text[last_open : idx + 1].strip()
                return candidate
        escaped = False

    return text.strip()


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_teacher_student_dataset(input_path, output_dir, val_ratio, seed):
    raw_records = load_jsonl(input_path)
    random.seed(seed)
    random.shuffle(raw_records)

    cleaned_records = []
    for record in raw_records:
        input_text = record.get("input_text", "")
        teacher_output = record.get("teacher_output", "")
        target_text = extract_last_json_object(teacher_output)
        if not input_text or not target_text:
            continue
        cleaned_records.append({
            "input_text": input_text,
            "target_text": target_text,
        })

    split = int(len(cleaned_records) * (1 - val_ratio))
    train_records = cleaned_records[:split]
    val_records = cleaned_records[split:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    print(f"Saved {len(train_records)} train records to {train_path}")
    print(f"Saved {len(val_records)} validation records to {val_path}")
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(description="从 teacher labels 生成 student 训练数据")
    parser.add_argument("--input", required=True, help="teacher labels jsonl 文件路径")
    parser.add_argument("--output_dir", default="data/processed_teacher", help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    prepare_teacher_student_dataset(
        args.input,
        args.output_dir,
        args.val_ratio,
        args.seed,
    )


if __name__ == "__main__":
    main()

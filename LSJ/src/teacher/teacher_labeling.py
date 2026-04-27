import argparse
import json
from pathlib import Path
from typing import Callable, Optional


def load_raw_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
        return data["records"]
    if isinstance(data, list):
        return data
    raise ValueError("输入文件必须是 JSON 数组，或包含 records 列表的对象。")


def build_teacher_prompt(input_text: str) -> str:
    return (
        "Extract all PII spans from the following text.\n"
        "Return ONLY one valid JSON object.\n"
        "Do NOT output explanations, markdown fences, notes, or think tags.\n"
        "Use the exact span text copied from the input.\n"
        "If multiple spans share the same entity type, use a JSON array.\n"
        "If no entity exists, return {}.\n\n"
        "Input:\n"
        f"{input_text}\n\n"
        "JSON output:"
    )


def resolve_model_path(model_path: str) -> tuple[str, bool]:
    path = Path(model_path)
    if path.is_file():
        return str(path), path.suffix.lower() == ".gguf"
    if path.is_dir():
        gguf_files = sorted(path.glob("*.gguf"))
        if len(gguf_files) == 1:
            return str(gguf_files[0]), True
        return str(path), False
    return model_path, model_path.lower().endswith(".gguf")


def load_teacher_backend(
    model_path: Optional[str] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_new_tokens: int = 256,
) -> Callable[[str], str]:
    if model_path:
        resolved_path, is_gguf = resolve_model_path(model_path)

        if is_gguf:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise ImportError("请先安装 llama-cpp-python，以便加载 GGUF teacher 模型") from exc

            llm = Llama(
                model_path=resolved_path,
                n_ctx=4096,
                n_gpu_layers=-1,
                verbose=False,
            )

            def infer(prompt: str) -> str:
                if hasattr(llm, "create_chat_completion"):
                    try:
                        result = llm.create_chat_completion(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You extract PII spans. Reply with JSON only. No reasoning, no markdown, no notes.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.0,
                            max_tokens=max_new_tokens,
                            response_format={"type": "json_object"},
                        )
                    except TypeError:
                        result = llm.create_chat_completion(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You extract PII spans. Reply with JSON only. No reasoning, no markdown, no notes.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.0,
                            max_tokens=max_new_tokens,
                        )
                    return result["choices"][0]["message"]["content"].strip()

                result = llm(
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                    echo=False,
                    stop=["\n\nNote:", "</think>", "```"],
                )
                return result["choices"][0]["text"].strip()

            return infer

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as exc:
            raise ImportError("请先安装 transformers 和 torch，以便本地运行 teacher 模型") from exc

        tokenizer = AutoTokenizer.from_pretrained(resolved_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        def infer(prompt: str) -> str:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return infer

    if api_url and api_key:
        try:
            import requests
        except ImportError as exc:
            raise ImportError("请先安装 requests，用于调用远程 teacher API") from exc

        def infer(prompt: str) -> str:
            response = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "qwen-3.5-9b", "prompt": prompt, "max_tokens": max_new_tokens},
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("output") or data.get("text") or json.dumps(data)

        return infer

    raise ValueError("请提供本地模型路径 model_path，或远程 api_url/api_key。")


def write_teacher_labels(records, output_path: str, infer_fn: Callable[[str], str], max_samples: Optional[int] = None):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "teacher_labels.jsonl"

    with out_file.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            if max_samples is not None and idx >= max_samples:
                break
            input_text = record.get("input", {}).get("text") or record.get("input_text", "")
            if not input_text:
                continue
            prompt = build_teacher_prompt(input_text)
            output_text = infer_fn(prompt)
            item = {
                "id": record.get("id"),
                "input_text": input_text,
                "teacher_output": output_text.strip(),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} records...")
    print(f"已生成 teacher labels: {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="生成 teacher labelling 数据")
    parser.add_argument("--input", required=True, help="原始 JSON 文件路径")
    parser.add_argument("--output_dir", default="data/teacher", help="teacher 输出目录")
    parser.add_argument("--model_path", default=None, help="本地模型路径，支持 HF 目录或单个 GGUF 文件")
    parser.add_argument("--api_url", default=None, help="远程 qwen API URL")
    parser.add_argument("--api_key", default=None, help="远程 qwen API Key")
    parser.add_argument("--max_samples", type=int, default=None, help="仅处理前 N 条记录，用于测试")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="teacher 最大生成长度")
    args = parser.parse_args()

    records = load_raw_json(args.input)
    infer_fn = load_teacher_backend(
        model_path=args.model_path,
        api_url=args.api_url,
        api_key=args.api_key,
        max_new_tokens=args.max_new_tokens,
    )
    write_teacher_labels(records, args.output_dir, infer_fn, max_samples=args.max_samples)


if __name__ == "__main__":
    main()

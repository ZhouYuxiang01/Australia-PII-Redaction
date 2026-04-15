import argparse
import inspect
import sys
from pathlib import Path
from typing import Optional

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from src.config import load_config


def build_prompt(input_text: str, target_text: str, template: Optional[str]):
    if template:
        return template.replace("{input_text}", input_text).replace("{target_text}", target_text)
    return f"{input_text}\n{target_text}"


def tokenize_fn(examples, tokenizer, cfg):
    template = cfg.get("prompt", {}).get("template")
    prompts = [
        build_prompt(input_text, target_text, template)
        for input_text, target_text in zip(examples["input_text"], examples["target_text"])
    ]
    output = tokenizer(prompts, truncation=True, max_length=cfg["max_length"], padding="max_length")
    output["labels"] = output["input_ids"].copy()
    return output


def file_has_content(path: str) -> bool:
    file_path = Path(path)
    return file_path.exists() and file_path.stat().st_size > 0


def main(config_path: str):
    cfg = load_config(config_path)
    model_name = cfg["model_name"]
    train_file = cfg["train_file"]
    validation_file = cfg["validation_file"]
    has_validation = file_has_content(validation_file)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": train_file}
    if has_validation:
        data_files["validation"] = validation_file

    dataset = load_dataset(
        "json",
        data_files=data_files,
        field=None,
    )

    dataset = dataset.map(
        lambda examples: tokenize_fn(examples, tokenizer, cfg),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=int(cfg["lora"]["r"]),
        lora_alpha=float(cfg["lora"]["lora_alpha"]),
        lora_dropout=float(cfg["lora"]["lora_dropout"]),
        target_modules=cfg["lora"]["target_modules"],
    )

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = get_peft_model(model, lora_cfg)

    eval_arg_name = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    training_kwargs = {
        "output_dir": cfg["output_dir"],
        "per_device_train_batch_size": int(cfg["training"]["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(cfg["training"]["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(cfg["training"]["gradient_accumulation_steps"]),
        "learning_rate": float(cfg["training"]["learning_rate"]),
        "weight_decay": float(cfg["training"]["weight_decay"]),
        "num_train_epochs": float(cfg["training"]["num_train_epochs"]),
        "logging_steps": int(cfg["training"]["logging_steps"]),
        "save_steps": int(cfg["training"]["save_steps"]),
        "eval_steps": int(cfg["training"]["eval_steps"]),
        "save_total_limit": int(cfg["training"]["save_total_limit"]),
        "fp16": bool(cfg["training"].get("fp16", False)),
        "report_to": "none",
        "load_best_model_at_end": has_validation,
    }
    training_kwargs[eval_arg_name] = cfg["training"]["evaluation_strategy"] if has_validation else "no"
    if has_validation:
        training_kwargs["metric_for_best_model"] = "loss"

    training_args = TrainingArguments(**training_kwargs)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if has_validation else None,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 LoRA 微调")
    parser.add_argument("--config", default="config/lora_config.yaml", help="配置文件路径")
    args = parser.parse_args()
    main(args.config)

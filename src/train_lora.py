import argparse
from typing import Optional

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


def main(config_path: str):
    cfg = load_config(config_path)
    model_name = cfg["model_name"]
    train_file = cfg["train_file"]
    validation_file = cfg["validation_file"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={"train": train_file, "validation": validation_file},
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
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
    )

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = get_peft_model(model, lora_cfg)

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        evaluation_strategy=cfg["training"]["evaluation_strategy"],
        eval_steps=cfg["training"]["eval_steps"],
        save_total_limit=cfg["training"]["save_total_limit"],
        fp16=cfg["training"].get("fp16", False),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 LoRA 微调")
    parser.add_argument("--config", default="config/lora_config.yaml", help="配置文件路径")
    args = parser.parse_args()
    main(args.config)

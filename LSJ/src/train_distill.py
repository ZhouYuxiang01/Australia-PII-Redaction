import argparse
import inspect
import sys
from pathlib import Path
from typing import Optional

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class DistillationTrainer(Trainer):
    def __init__(
        self,
        *args,
        teacher_model=None,
        temperature: float = 1.0,
        alpha_distill: float = 0.5,
        alpha_ce: float = 0.5,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha_distill = alpha_distill
        self.alpha_ce = alpha_ce
        self.pad_token_id = pad_token_id

        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        teacher_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        distill_loss = torch.tensor(0.0, device=student_logits.device)
        if self.teacher_model is not None and self.alpha_distill > 0:
            if self.teacher_model.device != model.device:
                self.teacher_model.to(model.device)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

            if self.pad_token_id is not None and labels is not None:
                mask = labels.ne(self.pad_token_id)
                mask = mask.unsqueeze(-1).expand_as(student_log_probs)
                student_log_probs = student_log_probs.masked_select(mask).view(-1, student_log_probs.size(-1))
                teacher_probs = teacher_probs.masked_select(mask).view(-1, teacher_probs.size(-1))

            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (self.temperature ** 2)

        ce_loss = torch.tensor(0.0, device=student_logits.device)
        if labels is not None and self.alpha_ce > 0:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            ce_loss = loss_fct(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

        loss = self.alpha_distill * distill_loss + self.alpha_ce * ce_loss
        return (loss, student_outputs) if return_outputs else loss


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


def is_gguf_model_path(path_str: str) -> bool:
    path = Path(path_str)
    if path.suffix.lower() == ".gguf":
        return True
    return path.is_dir() and any(path.glob("*.gguf"))


def main(config_path: str, teacher_model_name_override: Optional[str] = None):
    cfg = load_config(config_path)
    model_name = cfg["model_name"]
    teacher_model_name = cfg.get("distillation", {}).get("teacher_model_name", model_name)
    if teacher_model_name_override:
        teacher_model_name = teacher_model_name_override
    if teacher_model_name and is_gguf_model_path(str(teacher_model_name)):
        raise ValueError(
            "当前 train_distill.py 需要 Hugging Face 格式的 teacher 模型；单个 GGUF 文件或仅含 GGUF 的目录目前仅支持 teacher labelling 阶段。"
        )
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

    student_model = AutoModelForCausalLM.from_pretrained(model_name)
    student_model = get_peft_model(student_model, lora_cfg)

    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    teacher_model.to(student_model.device)

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

    distill_cfg = cfg.get("distillation", {})
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if has_validation else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        temperature=distill_cfg.get("temperature", 1.0),
        alpha_distill=distill_cfg.get("alpha_distill", 0.5),
        alpha_ce=distill_cfg.get("alpha_ce", 0.5),
        pad_token_id=tokenizer.pad_token_id,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 LoRA 蒸馏训练")
    parser.add_argument("--config", default="config/lora_config.yaml", help="配置文件路径")
    parser.add_argument("--teacher_model_name", default=None, help="覆盖 config 中的 teacher 模型名称或路径")
    args = parser.parse_args()
    main(args.config, teacher_model_name_override=args.teacher_model_name)

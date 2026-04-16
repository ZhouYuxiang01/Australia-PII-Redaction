import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DefaultDataCollator,
    Qwen2Config,
    Trainer,
    TrainingArguments,
)

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config


def file_has_content(path: str) -> bool:
    file_path = Path(path)
    return file_path.exists() and file_path.stat().st_size > 0


def build_model_config(model_name: str, num_labels: int, id2label: dict[int, str], label2id: dict[str, int]):
    raw_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    raw_config.id2label = {int(k): v for k, v in id2label.items()}
    raw_config.label2id = label2id
    raw_config.num_labels = num_labels

    if getattr(raw_config, "model_type", None) != "qwen2_5_vl":
        return raw_config, raw_config

    text_cfg = raw_config.text_config
    model_config = Qwen2Config(
        vocab_size=text_cfg.vocab_size,
        hidden_size=text_cfg.hidden_size,
        intermediate_size=text_cfg.intermediate_size,
        num_hidden_layers=text_cfg.num_hidden_layers,
        num_attention_heads=text_cfg.num_attention_heads,
        num_key_value_heads=text_cfg.num_key_value_heads,
        max_position_embeddings=text_cfg.max_position_embeddings,
        rms_norm_eps=text_cfg.rms_norm_eps,
        hidden_act=text_cfg.hidden_act,
        bos_token_id=text_cfg.bos_token_id,
        eos_token_id=text_cfg.eos_token_id,
        num_labels=num_labels,
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id,
    )
    rope_theta = getattr(text_cfg, "rope_theta", None)
    if rope_theta is not None:
        model_config.rope_theta = rope_theta
    tie_word_embeddings = getattr(text_cfg, "tie_word_embeddings", None)
    if tie_word_embeddings is not None:
        model_config.tie_word_embeddings = tie_word_embeddings
    torch_dtype = getattr(raw_config, "torch_dtype", None) or getattr(text_cfg, "torch_dtype", None)
    if torch_dtype is not None:
        model_config.torch_dtype = torch_dtype
    return raw_config, model_config


def resolve_target_modules(model_type: str, configured_modules: list[str]) -> list[str]:
    if model_type == "qwen2_5_vl" and any(module.startswith("c_") for module in configured_modules):
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return configured_modules


def bio_spans(tags):
    spans = set()
    current_type = None
    start_idx = None

    for idx, tag in enumerate(tags):
        if tag == "O":
            if current_type is not None:
                spans.add((current_type, start_idx, idx - 1))
                current_type = None
                start_idx = None
            continue

        if tag.startswith("B-"):
            if current_type is not None:
                spans.add((current_type, start_idx, idx - 1))
            current_type = tag[2:]
            start_idx = idx
            continue

        if tag.startswith("I-"):
            entity = tag[2:]
            if current_type == entity:
                continue
            if current_type is not None:
                spans.add((current_type, start_idx, idx - 1))
            current_type = entity
            start_idx = idx

    if current_type is not None:
        spans.add((current_type, start_idx, len(tags) - 1))
    return spans


def compute_span_metrics(eval_pred, id2label):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    gold_total = 0
    pred_total = 0
    true_positive = 0
    token_total = 0
    token_correct = 0

    for pred_row, label_row in zip(preds, labels):
        pred_tags = []
        gold_tags = []
        for pred_id, gold_id in zip(pred_row, label_row):
            if gold_id == -100:
                continue
            pred_tags.append(id2label[int(pred_id)])
            gold_tags.append(id2label[int(gold_id)])
            token_total += 1
            token_correct += int(int(pred_id) == int(gold_id))

        pred_spans = bio_spans(pred_tags)
        gold_spans = bio_spans(gold_tags)
        pred_total += len(pred_spans)
        gold_total += len(gold_spans)
        true_positive += len(pred_spans & gold_spans)

    precision = true_positive / pred_total if pred_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = token_correct / token_total if token_total else 0.0

    return {
        "token_accuracy": accuracy,
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
    }


class BioDistillTrainer(Trainer):
    def __init__(self, *args, temperature=2.0, alpha_distill=0.5, alpha_ce=0.5, teacher_confidence=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.alpha_distill = alpha_distill
        self.alpha_ce = alpha_ce
        self.teacher_confidence = teacher_confidence
        self.num_labels = self.model.config.num_labels

    def build_soft_targets(self, labels: torch.Tensor, label_confidences: torch.Tensor | None = None) -> torch.Tensor:
        valid_mask = labels.ne(-100)
        base_conf = torch.full(labels.shape, self.teacher_confidence, device=labels.device, dtype=torch.float32)
        if label_confidences is not None:
            label_confidences = label_confidences.to(labels.device, dtype=torch.float32).clamp(0.0, 1.0)
            base_conf = torch.where(valid_mask, label_confidences, torch.zeros_like(base_conf))
            base_conf = torch.where((base_conf <= 0) & valid_mask, torch.full_like(base_conf, self.teacher_confidence), base_conf)

        probs = torch.full(
            (*labels.shape, self.num_labels),
            fill_value=0.0,
            device=labels.device,
            dtype=torch.float32,
        )

        if self.num_labels > 1:
            other_probs = ((1.0 - base_conf) / (self.num_labels - 1)).unsqueeze(-1)
            probs += other_probs

        safe_labels = labels.masked_fill(~valid_mask, 0)
        probs.scatter_(2, safe_labels.unsqueeze(-1), base_conf.unsqueeze(-1))
        probs = probs * valid_mask.unsqueeze(-1)
        return probs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        label_confidences = inputs.pop("label_confidences", None)
        outputs = model(**inputs)
        logits = outputs.logits
        ce_loss = outputs.loss

        teacher_probs = self.build_soft_targets(labels, label_confidences=label_confidences).to(logits.device)
        valid_mask = labels.ne(-100)
        student_log_probs = F.log_softmax(logits / self.temperature, dim=-1)

        if valid_mask.any():
            expanded_mask = valid_mask.unsqueeze(-1).expand_as(student_log_probs)
            student_log_probs = student_log_probs.masked_select(expanded_mask).view(-1, logits.size(-1))
            teacher_probs = teacher_probs.masked_select(expanded_mask).view(-1, logits.size(-1))
            teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (self.temperature ** 2)
        else:
            distill_loss = torch.tensor(0.0, device=logits.device)

        loss = self.alpha_ce * ce_loss + self.alpha_distill * distill_loss
        return (loss, outputs) if return_outputs else loss


def main(config_path: str):
    cfg = load_config(config_path)
    model_name = cfg["model_name"]
    train_file = cfg.get("bio_train_file", "data/processed_bio/train.jsonl")
    validation_file = cfg.get("bio_validation_file", "data/processed_bio/val.jsonl")
    label_map_file = cfg.get("bio_label_map", "data/processed_bio/label_map.json")
    has_validation = file_has_content(validation_file)

    with open(label_map_file, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    label_names = label_map["labels"]
    label_to_id = {k: int(v) for k, v in label_map["label_to_id"].items()}
    id2label = {idx: label for label, idx in label_to_id.items()}

    raw_config, model_config = build_model_config(model_name, len(label_names), id2label, label_to_id)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    data_files = {"train": train_file}
    if has_validation:
        data_files["validation"] = validation_file
    dataset = load_dataset("json", data_files=data_files, field=None)
    keep_columns = {"input_ids", "attention_mask", "labels", "label_confidences"}
    for split_name in list(dataset.keys()):
        drop_columns = [col for col in dataset[split_name].column_names if col not in keep_columns]
        if drop_columns:
            dataset[split_name] = dataset[split_name].remove_columns(drop_columns)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=model_config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_cfg = LoraConfig(
        task_type="TOKEN_CLS",
        inference_mode=False,
        r=int(cfg["lora"]["r"]),
        lora_alpha=float(cfg["lora"]["lora_alpha"]),
        lora_dropout=float(cfg["lora"]["lora_dropout"]),
        target_modules=resolve_target_modules(getattr(raw_config, "model_type", ""), cfg["lora"]["target_modules"]),
    )
    model = get_peft_model(model, lora_cfg)

    eval_arg_name = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"
    training_kwargs = {
        "output_dir": cfg.get("bio_output_dir", "outputs/bio_distill"),
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
        "bf16": bool(cfg["training"].get("bf16", False)),
        "report_to": "none",
        "load_best_model_at_end": has_validation,
        "remove_unused_columns": False,
    }
    training_kwargs[eval_arg_name] = cfg["training"]["evaluation_strategy"] if has_validation else "no"
    if has_validation:
        training_kwargs["metric_for_best_model"] = "span_f1"
        training_kwargs["greater_is_better"] = True

    training_args = TrainingArguments(**training_kwargs)

    distill_cfg = cfg.get("distillation", {})
    trainer = BioDistillTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if has_validation else None,
        data_collator=DefaultDataCollator(),
        temperature=float(distill_cfg.get("temperature", 2.0)),
        alpha_distill=float(distill_cfg.get("alpha_distill", 0.5)),
        alpha_ce=float(distill_cfg.get("alpha_ce", 0.5)),
        teacher_confidence=float(distill_cfg.get("teacher_confidence", 0.9)),
        compute_metrics=(lambda eval_pred: compute_span_metrics(eval_pred, id2label)) if has_validation else None,
    )

    trainer.train()
    trainer.save_model(training_kwargs["output_dir"])
    if has_validation:
        metrics = trainer.evaluate()
        print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 BIO token distillation 训练")
    parser.add_argument("--config", default="config/lora_config.yaml", help="配置文件路径")
    args = parser.parse_args()
    main(args.config)

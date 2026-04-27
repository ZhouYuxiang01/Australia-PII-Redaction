#!/usr/bin/env python3
"""Full-parameter SFT for Qwen3.5-4B-Base on 73-class AU PII JSON spans."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    profile: str
    train_path: str
    dev_path: str
    meta_path: str
    base_model: str
    output_dir: str
    max_length: int
    num_train_epochs: float
    max_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    gradient_checkpointing: bool
    eval_steps: int
    save_steps: int
    logging_steps: int
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    optim: str
    seed: int
    full_finetune: bool = True
    peft_config: None = None


PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "smoke": {
        "max_length": 1280,
        "num_train_epochs": 1,
        "max_steps": 20,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-5,
        "warmup_steps": 5,
        "gradient_checkpointing": True,
        "eval_steps": 10,
        "save_steps": 10,
        "logging_steps": 1,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "optim": "paged_adamw_8bit",
    },
    "safe_full": {
        "max_length": 1280,
        "num_train_epochs": 1,
        "max_steps": -1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2e-5,
        "warmup_steps": 150,
        "gradient_checkpointing": True,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 20,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": False,
        "optim": "paged_adamw_8bit",
    },
    "adamw_full": {
        "max_length": 1280,
        "num_train_epochs": 1,
        "max_steps": -1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-5,
        "warmup_steps": 150,
        "gradient_checkpointing": True,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 20,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": False,
        "optim": "adamw_torch",
    },
}


def build_run_config(
    profile: str = "safe_full",
    train_path: str = "../data/processed/qwen_sft_train.jsonl",
    dev_path: str = "../data/processed/qwen_sft_dev.jsonl",
    meta_path: str = "../data/processed/meta.json",
    base_model: str = "/home/admin/model/Qwen3.5-4B-Base",
    output_dir: str = "../outputs/qwen3_5_4b_base_full_73class",
    seed: int = 42,
) -> RunConfig:
    if profile not in PROFILE_CONFIGS:
        raise ValueError(f"unknown profile {profile!r}; choose one of {sorted(PROFILE_CONFIGS)}")
    cfg = dict(PROFILE_CONFIGS[profile])
    return RunConfig(
        profile=profile,
        train_path=train_path,
        dev_path=dev_path,
        meta_path=meta_path,
        base_model=base_model,
        output_dir=output_dir,
        seed=seed,
        **cfg,
    )


def find_latest_checkpoint(output_dir: str) -> str | None:
    path = Path(output_dir)
    if not path.exists():
        return None
    checkpoints = [p for p in path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1])


def validate_inputs(cfg: RunConfig) -> dict[str, Any]:
    for path in [cfg.train_path, cfg.dev_path, cfg.meta_path, cfg.base_model]:
        if not Path(path).exists():
            raise FileNotFoundError(path)
    meta = json.loads(Path(cfg.meta_path).read_text(encoding="utf-8"))
    if meta.get("schema") != "json-spans":
        raise ValueError(f"expected meta schema json-spans, got {meta.get('schema')!r}")
    if int(meta.get("class_count", 0)) != 73:
        raise ValueError(f"expected 73 classes, got {meta.get('class_count')!r}")
    return meta


def estimate_steps(train_size: int, cfg: RunConfig) -> int:
    if cfg.max_steps > 0:
        return cfg.max_steps
    effective_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    return math.ceil(train_size / effective_batch) * int(cfg.num_train_epochs)


def train(cfg: RunConfig, auto_resume: str | bool = "auto") -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    for name in ["model", "trainer"]:
        if name in globals():
            del globals()[name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    random.seed(cfg.seed)
    set_seed(cfg.seed)
    meta = validate_inputs(cfg)
    dataset = load_dataset("json", data_files={"train": cfg.train_path, "validation": cfg.dev_path})
    train_size = len(dataset["train"])
    planned_steps = estimate_steps(train_size, cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        from trl.chat_template_utils import qwen3_training_chat_template
    except ImportError as exc:
        raise ImportError("trl.chat_template_utils.qwen3_training_chat_template is required") from exc
    tokenizer.chat_template = qwen3_training_chat_template

    rendered = tokenizer.apply_chat_template(
        dataset["train"][0]["messages"],
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    if int(sum(rendered["assistant_masks"])) <= 0:
        raise RuntimeError("assistant token mask is empty; assistant_only_loss would train on nothing")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        max_length=cfg.max_length,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type="cosine",
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=cfg.logging_steps,
        report_to="none",
        assistant_only_loss=True,
        packing=False,
        optim=cfg.optim,
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        remove_unused_columns=False,
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=None,
        callbacks=[],
    )

    latest_checkpoint = find_latest_checkpoint(cfg.output_dir)
    if auto_resume == "auto":
        resume_from_checkpoint = latest_checkpoint
    elif auto_resume is True:
        if latest_checkpoint is None:
            raise FileNotFoundError("AUTO_RESUME=True but no checkpoint exists")
        resume_from_checkpoint = latest_checkpoint
    else:
        resume_from_checkpoint = None

    run_summary = {
        **asdict(cfg),
        "train_size": train_size,
        "dev_size": len(dataset["validation"]),
        "planned_steps": planned_steps,
        "target_labels": meta["target_labels"],
        "resume_from_checkpoint": resume_from_checkpoint,
    }
    Path(cfg.output_dir, "run_summary.pretrain.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    start = time.time()
    result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed = time.time() - start

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    Path(cfg.output_dir, "trained_chat_template.txt").write_text(tokenizer.chat_template or "", encoding="utf-8")
    final_summary = {
        **run_summary,
        "train_result": result.metrics,
        "elapsed_seconds": elapsed,
    }
    Path(cfg.output_dir, "run_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_CONFIGS), default=os.environ.get("TRAIN_PROFILE", "safe_full"))
    parser.add_argument("--train-path", default="../data/processed/qwen_sft_train.jsonl")
    parser.add_argument("--dev-path", default="../data/processed/qwen_sft_dev.jsonl")
    parser.add_argument("--meta-path", default="../data/processed/meta.json")
    parser.add_argument("--base-model", default="/home/admin/model/Qwen3.5-4B-Base")
    parser.add_argument("--output-dir", default="../outputs/qwen3_5_4b_base_full_73class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--auto-resume", choices=["auto", "true", "false"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_run_config(
        profile=args.profile,
        train_path=args.train_path,
        dev_path=args.dev_path,
        meta_path=args.meta_path,
        base_model=args.base_model,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    auto_resume: str | bool
    if args.auto_resume == "true":
        auto_resume = True
    elif args.auto_resume == "false":
        auto_resume = False
    else:
        auto_resume = "auto"
    print(json.dumps({"config": asdict(cfg), "auto_resume": auto_resume}, ensure_ascii=False, indent=2))
    summary = train(cfg, auto_resume=auto_resume)
    print(json.dumps({"output_dir": cfg.output_dir, "train_result": summary["train_result"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

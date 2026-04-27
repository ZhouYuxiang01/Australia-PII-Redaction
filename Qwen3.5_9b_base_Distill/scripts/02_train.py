import gc, os

for name in ["model", "trainer", "infer_tokenizer"]:
    if name in globals():
        del globals()[name]
gc.collect()

try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU allocated after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
except Exception as e:
    print("cleanup warning:", e)



import os
import re
import gc
import json
import math
import time
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from redaction_utils import parse_annotated



TRAIN_PATH = "../data/processed/qwen_sft_train.jsonl"
DEV_PATH   = "../data/processed/qwen_sft_dev.jsonl"
META_PATH  = "../data/processed/meta.json"

BASE_MODEL = "../../model/Qwen3.5-9B-Base"
# BASE_MODEL = "../../model/Qwen3-4B-Instruct-2507"   # 如需回退可改这里

OUTPUT_DIR = "../outputs/qwen3_5_9b_base_lora_tagged_28_fastretry"
AUTO_RESUME = "auto"   # "auto" / True / False

for p in [TRAIN_PATH, DEV_PATH, META_PATH]:
    assert os.path.exists(p), f"找不到 {p}"
assert os.path.exists(BASE_MODEL), f"找不到模型 {BASE_MODEL}"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("TRAIN_PATH:", TRAIN_PATH)
print("DEV_PATH:  ", DEV_PATH)
print("META_PATH: ", META_PATH)
print("BASE_MODEL:", BASE_MODEL)
print("OUTPUT_DIR:", OUTPUT_DIR)



dataset = load_dataset("json", data_files={"train": TRAIN_PATH, "validation": DEV_PATH})

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

assert meta.get("schema") == "span-tagged", "这份 02 期望 01 输出 span-tagged 数据"
target_labels = meta.get("target_labels", [])

print(dataset)
print("train:", len(dataset["train"]))
print("validation:", len(dataset["validation"]))
print("target label count:", len(target_labels))
print("first 10 labels:", target_labels[:10])

sample = dataset["train"][0]
assert "messages" in sample and len(sample["messages"]) == 3
print("roles:", [m["role"] for m in sample["messages"]])
print("\n=== first user ===\n", sample["messages"][1]["content"][:400])
print("\n=== first assistant ===\n", sample["messages"][2]["content"][:600])



SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_seed(SEED)
print("seed:", SEED)



tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    from trl.chat_template_utils import qwen3_training_chat_template
except ImportError:
    import trl.chat_template_utils as tct
    raise ImportError(f"找不到 qwen3_training_chat_template，可用候选: {[x for x in dir(tct) if 'qwen' in x.lower()]}")

tokenizer.chat_template = qwen3_training_chat_template

print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id)
print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)
print("已切换到 TRL qwen3_training_chat_template")



enc = tokenizer.apply_chat_template(
    dataset["train"][0]["messages"],
    tokenize=True,
    return_dict=True,
    return_assistant_tokens_mask=True,
)
assistant_tokens = int(sum(enc["assistant_masks"]))
total_tokens = len(enc["input_ids"])
ratio = assistant_tokens / max(1, total_tokens)
print(f"assistant tokens: {assistant_tokens} / {total_tokens} ({ratio*100:.1f}%)")
assert assistant_tokens > 0, "assistant_masks 全是 0，assistant_only_loss 不会生效"



sample_size = min(500, len(dataset["train"]))
sample_idx = random.sample(range(len(dataset["train"])), k=sample_size)

lens = []
for i in sample_idx:
    rendered = tokenizer.apply_chat_template(
        dataset["train"][i]["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    lens.append(len(tokenizer(rendered, add_special_tokens=False).input_ids))

lens.sort()
n = len(lens)
def pct(v):
    idx = min(n - 1, int(n * v))
    return lens[idx]

print(f"sampled={n}")
print(f"mean={sum(lens)/n:.1f}")
print(f"p50={pct(0.50)}  p90={pct(0.90)}  p95={pct(0.95)}  p99={pct(0.99)}  max={lens[-1]}")



PROFILE = "safe_full"   # "smoke" / "safe_full" / "fast_full"
MAX_LENGTH = 1024

PROFILE_CONFIG = {
    "smoke": {
        "epochs": 0.2,
        "train_bs": 1,
        "eval_bs": 1,
        "grad_accum": 16,
        "learning_rate": 8e-5,
        "warmup_steps": 40,
        "gradient_checkpointing": False,
        "eval_steps": 40,
        "save_steps": 40,
        "logging_steps": 5,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
    },
    "safe_full": {
        "epochs": 1,
        "train_bs": 1,
        "eval_bs": 1,
        "grad_accum": 16,
        "learning_rate": 8e-5,
        "warmup_steps": 120,
        "gradient_checkpointing": False,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 20,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
    },
    "fast_full": {
        "epochs": 1,
        "train_bs": 1,
        "eval_bs": 1,
        "grad_accum": 4,
        "learning_rate": 1e-4,
        "warmup_steps": 120,
        "gradient_checkpointing": False,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 20,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
    },
}

cfg = PROFILE_CONFIG[PROFILE]
cfg



n_train = len(dataset["train"])
effective_batch = cfg["train_bs"] * cfg["grad_accum"]
steps_per_epoch = math.ceil(n_train / effective_batch)

if PROFILE == "smoke":
    approx_steps = max(30, int(steps_per_epoch * cfg["epochs"]))
    planned_epochs = cfg["epochs"]
else:
    approx_steps = steps_per_epoch * int(cfg["epochs"])
    planned_epochs = int(cfg["epochs"])

print("=" * 60)
print("训练计划")
print("=" * 60)
print(f"PROFILE               = {PROFILE}")
print(f"train samples         = {n_train}")
print(f"effective batch       = {effective_batch}")
print(f"steps/epoch           = ~{steps_per_epoch}")
print(f"planned epochs        = {planned_epochs}")
print(f"approx total steps    = ~{approx_steps}")
print(f"max_length            = {MAX_LENGTH}")
print(f"gradient_checkpointing= {cfg['gradient_checkpointing']}")
print("=" * 60)



model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.config.use_cache = False

print(f"model type: {type(model).__name__}")
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
if torch.cuda.is_available():
    print(f"GPU allocated after load: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")



if cfg["gradient_checkpointing"]:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    print("✓ gradient checkpointing 已启用")
else:
    print("✓ gradient checkpointing 已关闭")



peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
print(peft_config)



if PROFILE == "smoke":
    num_train_epochs = 1
    max_steps = approx_steps
else:
    num_train_epochs = int(cfg["epochs"])
    max_steps = -1

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=MAX_LENGTH,

    per_device_train_batch_size=cfg["train_bs"],
    per_device_eval_batch_size=cfg["eval_bs"],
    gradient_accumulation_steps=cfg["grad_accum"],

    learning_rate=cfg["learning_rate"],
    warmup_steps=cfg["warmup_steps"],
    lr_scheduler_type="cosine",
    num_train_epochs=num_train_epochs,
    max_steps=max_steps,

    weight_decay=0.0,
    max_grad_norm=1.0,

    bf16=True,
    fp16=False,
    gradient_checkpointing=cfg["gradient_checkpointing"],

    eval_strategy="no",
    eval_steps=cfg["eval_steps"],
    save_strategy="steps",
    save_steps=cfg["save_steps"],
    save_total_limit=2,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_steps=cfg["logging_steps"],
    report_to="none",

    assistant_only_loss=True,
    packing=False,

    dataloader_num_workers=cfg["dataloader_num_workers"],
    dataloader_pin_memory=cfg["dataloader_pin_memory"],
    remove_unused_columns=False,

    seed=SEED,
)
training_args



trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    peft_config=peft_config,
    callbacks=[],
)
trainer.model.print_trainable_parameters()



def find_latest_checkpoint(output_dir: str):
    p = Path(output_dir)
    if not p.exists():
        return None
    ckpts = [x for x in p.iterdir() if x.is_dir() and x.name.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts = sorted(ckpts, key=lambda x: int(x.name.split("-")[-1]))
    return str(ckpts[-1])

def build_run_signature():
    return {
        "base_model": BASE_MODEL,
        "profile": PROFILE,
        "max_length": MAX_LENGTH,
        "train_size": len(dataset["train"]),
        "dev_size": len(dataset["validation"]),
        "learning_rate": cfg["learning_rate"],
        "warmup_steps": cfg["warmup_steps"],
        "train_bs": cfg["train_bs"],
        "eval_bs": cfg["eval_bs"],
        "grad_accum": cfg["grad_accum"],
        "gradient_checkpointing": cfg["gradient_checkpointing"],
        "dataloader_num_workers": cfg["dataloader_num_workers"],
        "dataloader_pin_memory": cfg["dataloader_pin_memory"],
        "lora_r": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "lora_dropout": peft_config.lora_dropout,
        "target_modules": list(peft_config.target_modules),
        "target_labels": target_labels,
        "seed": SEED,
    }

latest_ckpt = find_latest_checkpoint(OUTPUT_DIR)
print("latest checkpoint:", latest_ckpt)

current_run_signature = build_run_signature()
run_summary_path = os.path.join(OUTPUT_DIR, "run_summary.json")
existing_run_summary = None
resume_mismatches = []

if os.path.exists(run_summary_path):
    with open(run_summary_path, "r", encoding="utf-8") as f:
        existing_run_summary = json.load(f)
    for key, value in current_run_signature.items():
        if existing_run_summary.get(key) != value:
            resume_mismatches.append(key)
elif latest_ckpt is not None:
    print("run_summary.json not found; auto resume disabled for safety")

resume_guard_passed = latest_ckpt is not None and existing_run_summary is not None and not resume_mismatches

if AUTO_RESUME == "auto":
    resume_from_checkpoint = latest_ckpt if resume_guard_passed else None
elif AUTO_RESUME is True:
    assert latest_ckpt is not None, "AUTO_RESUME=True but no checkpoint was found"
    assert existing_run_summary is not None, "AUTO_RESUME=True but run_summary.json is missing; refusing to resume blindly"
    assert not resume_mismatches, f"AUTO_RESUME=True but run config mismatches: {resume_mismatches}"
    resume_from_checkpoint = latest_ckpt
else:
    resume_from_checkpoint = None

print("resume_guard_passed:", resume_guard_passed)
if resume_mismatches:
    print("resume config mismatches:", resume_mismatches)
print("resume_from_checkpoint:", resume_from_checkpoint)



start = time.time()
train_result = trainer.train(resume_from_checkpoint="../outputs/qwen3_5_9b_base_lora_tagged_28_fastretry/checkpoint-900")
elapsed = time.time() - start

print("\n" + "=" * 70)
print("训练完成")
print("=" * 70)
print(f"global_step : {train_result.global_step}")
print(f"train_loss  : {train_result.metrics.get('train_loss', 'N/A')}")
print(f"runtime     : {train_result.metrics.get('train_runtime', elapsed)/3600:.2f} 小时")
print(f"samples/sec : {train_result.metrics.get('train_samples_per_second', 'N/A')}")
print(f"steps/sec   : {train_result.metrics.get('train_steps_per_second', 'N/A')}")



trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "trained_chat_template.txt"), "w", encoding="utf-8") as f:
    f.write(tokenizer.chat_template or "")

with open(os.path.join(OUTPUT_DIR, "target_labels.json"), "w", encoding="utf-8") as f:
    json.dump(target_labels, f, ensure_ascii=False, indent=2)

run_summary = {
    "profile": PROFILE,
    "base_model": BASE_MODEL,
    "output_dir": OUTPUT_DIR,
    "max_length": MAX_LENGTH,
    "train_size": len(dataset["train"]),
    "dev_size": len(dataset["validation"]),
    "learning_rate": cfg["learning_rate"],
    "warmup_steps": cfg["warmup_steps"],
    "train_bs": cfg["train_bs"],
    "eval_bs": cfg["eval_bs"],
    "grad_accum": cfg["grad_accum"],
    "gradient_checkpointing": cfg["gradient_checkpointing"],
    "dataloader_num_workers": cfg["dataloader_num_workers"],
    "dataloader_pin_memory": cfg["dataloader_pin_memory"],
    "lora_r": peft_config.r,
    "lora_alpha": peft_config.lora_alpha,
    "lora_dropout": peft_config.lora_dropout,
    "target_modules": list(peft_config.target_modules),
    "target_labels": target_labels,
    "seed": SEED,
    "resume_guard_passed": resume_guard_passed,
    "resume_mismatches": resume_mismatches,
    "resume_from_checkpoint": resume_from_checkpoint,
}
with open(os.path.join(OUTPUT_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(run_summary, f, ensure_ascii=False, indent=2)

print("saved to:", OUTPUT_DIR)
for name in sorted(os.listdir(OUTPUT_DIR)):
    p = os.path.join(OUTPUT_DIR, name)
    if os.path.isfile(p):
        print(f"  {name:<30s} {os.path.getsize(p)/1024**2:8.1f} MB")
    else:
        print(f"  {name}/")



SYSTEM_PROMPT = (
    "You are a PII annotator for Australian context.\n"
    "Return the SAME text with supported PII wrapped as <pii type=\"TYPE\">VALUE</pii>.\n"
    "Preserve every character exactly. Do not paraphrase, summarize, or explain.\n"
    "Wrap every occurrence of supported PII. Do not deduplicate.\n"
    "If no supported PII is present, return the input unchanged.\n"
    "Supported types:\n- " + "\n- ".join(target_labels)
)

test_text = (
    "Name: Alice Wong\n"
    "DOB: 21/05/1998\n"
    "Email: alice.wong@gmail.com\n"
    "Phone: 0412 345 678\n"
    "TFN: 123 456 789\n"
    "Address: 25 George St, Sydney NSW 2000\n"
    "Student ID: 512345678\n"
    "Work email: alice.wong@uts.edu.au"
)

expected_spans = {
    ("PERSON", "Alice Wong"),
    ("DATE_OF_BIRTH", "21/05/1998"),
    ("EMAIL_ADDRESS", "alice.wong@gmail.com"),
    ("AU_PHONE", "0412 345 678"),
    ("AU_TFN", "123 456 789"),
    ("ADDRESS", "25 George St, Sydney NSW 2000"),
    ("STUDENT_ID", "512345678"),
    ("WORK_EMAIL", "alice.wong@uts.edu.au"),
}

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": test_text},
]

infer_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True, use_fast=False)
if infer_tokenizer.pad_token is None:
    infer_tokenizer.pad_token = infer_tokenizer.eos_token
infer_tokenizer.chat_template = tokenizer.chat_template

prompt = infer_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = infer_tokenizer(prompt, return_tensors="pt").to(trainer.model.device)

trainer.model.eval()
with torch.no_grad():
    outputs = trainer.model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=infer_tokenizer.pad_token_id,
        eos_token_id=infer_tokenizer.eos_token_id,
    )

generated = infer_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

generated = strip_think_blocks(generated).strip()
print("=== GENERATED ===")
print(generated)

plain_text, spans = parse_annotated(generated, strict=False)
predicted_spans = {(s["type"], s["value"]) for s in spans}
missing_expected = sorted(expected_spans - predicted_spans)
unsupported_span_types = sorted({s["type"] for s in spans if s["type"] not in target_labels})

print("\n=== PARSED SPANS ===")
print(spans)
print("\nmissing_expected =", missing_expected)
print("unsupported_span_types =", unsupported_span_types)
print("round_trip_ok =", plain_text == test_text)

assert plain_text == test_text, "round-trip failed: generated text does not preserve the original input"
assert not unsupported_span_types, f"generated unsupported span types: {unsupported_span_types}"
assert not missing_expected, f"sanity check missed expected spans: {missing_expected}"


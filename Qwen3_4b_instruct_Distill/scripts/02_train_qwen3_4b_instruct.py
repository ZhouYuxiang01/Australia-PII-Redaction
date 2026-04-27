#!/usr/bin/env python
# coding: utf-8

# # 02_train_qwen3_4b_instruct_tagged_span (GB10 加速版)
# 
# 这份 notebook 用 **Qwen3-4B-Instruct-2507 + LoRA** 训练一个 **tagged-text PII annotator**：
# 
# - 输入：原始文本
# - 输出：与输入完全相同的文本，只在 PII span 外层加  
#   `<pii type="TYPE">VALUE</pii>`
# - 后续由 `redaction_utils.parse_annotated()` 解析回 spans，再交给 `apply_redaction()`
# 
# ## 这版相对旧版的关键改动
# 
# - 保留 **assistant-only loss**
# - 使用 **TRL 的 `qwen3_training_chat_template`**
# - 默认关闭 **gradient checkpointing**（你这台 128GB 机器通常不需要它）
# - 默认增大 batch，减少 accumulation
# - 降低 eval/save 频率
# - 支持 **自动恢复 checkpoint**
# - 默认先跑 **1 epoch fast_full**，确认稳定后再拉到 3 epoch
# 

# ## 0. 环境说明
# 
# 建议环境：
# 
# - `transformers >= 4.56`
# - `trl >= 0.23`
# - `peft >= 0.17`
# - `datasets`
# - `accelerate`
# 
# 如果缺包，先执行：
# 
# ```bash
# pip install -U transformers trl peft datasets accelerate sentencepiece
# ```
# 

# In[1]:


import gc
import os

for name in ["model", "trainer", "infer_model", "infer_tokenizer"]:
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


# ## 1. 导入依赖
# 

# In[2]:


import os
import json
import math
import time
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

import transformers, trl, peft, datasets

print("transformers:", transformers.__version__)
print("trl:         ", trl.__version__)
print("peft:        ", peft.__version__)
print("datasets:    ", datasets.__version__)
print("torch:       ", torch.__version__)
print("cuda:        ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:      ", torch.cuda.get_device_name(0))
    print("bf16:        ", torch.cuda.is_bf16_supported())


# ## 2. 路径配置
# 

# In[3]:


# === 按你的目录改这里 ===
TRAIN_PATH = "../data/processed/qwen_sft_train.jsonl"
DEV_PATH   = "../data/processed/qwen_sft_dev.jsonl"
META_PATH  = "../data/processed/meta.json"

BASE_MODEL = "../../model/Qwen3-4B-Instruct-2507"
OUTPUT_DIR = "../outputs/qwen3_4b_pii_lora_tagged"

# auto / True / False
AUTO_RESUME = "auto"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for p in [TRAIN_PATH, DEV_PATH, META_PATH]:
    assert os.path.exists(p), f"找不到 {p}"
assert os.path.exists(BASE_MODEL), f"找不到模型 {BASE_MODEL}"

print("TRAIN_PATH:", TRAIN_PATH)
print("DEV_PATH:  ", DEV_PATH)
print("META_PATH: ", META_PATH)
print("BASE_MODEL:", BASE_MODEL)
print("OUTPUT_DIR:", OUTPUT_DIR)


# ## 3. 加载数据
# 

# In[4]:


dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_PATH, "validation": DEV_PATH},
)

print(dataset)

sample = dataset["train"][0]
assert "messages" in sample, "数据缺少 messages 字段"
assert len(sample["messages"]) == 3, "每条样本必须是 system / user / assistant 三条消息"
print("first roles:", [m["role"] for m in sample["messages"]])
print()
print("=== first user ===")
print(sample["messages"][1]["content"][:400])
print()
print("=== first assistant ===")
print(sample["messages"][2]["content"][:600])


# ## 4. 读取 meta，确认任务 schema
# 

# In[5]:


with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

print(json.dumps({
    "schema": meta.get("schema"),
    "tag_format": meta.get("tag_format"),
    "train_size": meta.get("train_size"),
    "dev_size": meta.get("dev_size"),
    "target_labels": meta.get("target_labels", [])[:8],
}, ensure_ascii=False, indent=2))

assert meta.get("schema") == "span-tagged", "这份 02 notebook 期望 01 生成的是 span-tagged 数据"


# ## 5. 加载 tokenizer
# 

# In[6]:


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("eos_token:", tokenizer.eos_token, "id=", tokenizer.eos_token_id)
print("pad_token:", tokenizer.pad_token, "id=", tokenizer.pad_token_id)


# ## 6. 修复 Qwen3 chat template（assistant-only loss 必须）
# 
# Qwen3 官方模板没有 TRL 需要的 `{% generation %}` 段标记。  
# 这里切到 TRL 自带的训练模板。
# 

# In[7]:


try:
    from trl.chat_template_utils import qwen3_training_chat_template
except ImportError:
    import trl.chat_template_utils as tct
    candidates = [x for x in dir(tct) if "qwen" in x.lower()]
    raise ImportError(f"找不到 qwen3_training_chat_template，可用候选: {candidates}")

tokenizer.chat_template = qwen3_training_chat_template
print("已切换到 TRL 的 qwen3_training_chat_template")


# ## 7. 验证 assistant token mask
# 

# In[8]:


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
print("✓ assistant_only_loss 所需 mask 正常")


# ## 8. 统计 token 长度
# 

# In[9]:


random.seed(42)
sample_size = min(500, len(dataset["train"]))
sample_idx = random.sample(range(len(dataset["train"])), k=sample_size)

lens = []
for i in sample_idx:
    rendered = tokenizer.apply_chat_template(
        dataset["train"][i]["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    n_tokens = len(tokenizer(rendered, add_special_tokens=False).input_ids)
    lens.append(n_tokens)

lens.sort()
n = len(lens)

def pct(v):
    idx = min(n - 1, int(n * v))
    return lens[idx]

print(f"sampled={n}")
print(f"mean={sum(lens)/n:.1f}")
print(f"p50={pct(0.50)}  p90={pct(0.90)}  p95={pct(0.95)}  p99={pct(0.99)}  max={lens[-1]}")


# ## 9. 训练档位
# 
# 推荐：
# 
# - **`PROFILE = "smoke"`**：先测能否稳定收敛、格式是否正常
# - **`PROFILE = "fast_full"`**：默认推荐
# - **`PROFILE = "safe_full"`**：如果你发现 batch 太大或系统抖动，再退回
# 
# 你这台机器显存空间很大，所以默认不再走“极度保守”的旧配置。
# 

# In[2]:


PROFILE = "safe_full"   # "smoke" / "fast_full" / "safe_full"

MAX_LENGTH = 1280

# 如果 p99 很低，可以考虑 768；否则先 1024
PROFILE_CONFIG = {
    "smoke": {
        "epochs": 0.2,   # 只快速冒烟；后面会自动换算成 max_steps
        "train_bs": 8,
        "eval_bs": 8,
        "grad_accum": 2,
        "gradient_checkpointing": False,
        "eval_steps": 100,
        "save_steps": 100,
        "logging_steps": 10,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "warmup_steps": 50,
    },
    "fast_full": {
        "epochs": 1,
        "train_bs": 8,
        "eval_bs": 8,
        "grad_accum": 2,
        "gradient_checkpointing": False,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 20,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "warmup_steps": 150,
    },
    "safe_full": {
        "epochs": 1,
        "train_bs": 4,
        "eval_bs": 4,
        "grad_accum": 4,
        "gradient_checkpointing": True,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 20,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "warmup_steps": 150,
    },
}

cfg = PROFILE_CONFIG[PROFILE]
cfg


# ## 10. 估算训练步数
# 

# In[3]:


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
print(f"PROFILE              = {PROFILE}")
print(f"train samples         = {n_train}")
print(f"per_device_batch      = {cfg['train_bs']}")
print(f"grad_accum            = {cfg['grad_accum']}")
print(f"effective batch       = {effective_batch}")
print(f"steps/epoch           = ~{steps_per_epoch}")
print(f"planned epochs        = {planned_epochs}")
print(f"approx total steps    = ~{approx_steps}")
print(f"max_length            = {MAX_LENGTH}")
print(f"gradient_checkpointing= {cfg['gradient_checkpointing']}")
print("=" * 60)


# ## 11. 加载基座模型
# 
# 说明：
# 
# - 默认使用 `torch.bfloat16`
# - `use_cache=False` 训练时必须关
# - 不强绑 `attn_implementation="eager"`，让当前版本自己走默认实现  
#   如果你遇到特定 attention 实现问题，再手动改
# 

# In[12]:


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


# ## 12. gradient checkpointing 配置
# 
# 只有在 `cfg["gradient_checkpointing"] == True` 时启用。  
# 关闭它通常会明显提速，但会多占显存。
# 

# In[13]:


if cfg["gradient_checkpointing"]:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    print("✓ gradient checkpointing 已启用")
else:
    print("✓ gradient checkpointing 已关闭（加速优先）")


# ## 13. LoRA 配置
# 

# In[14]:


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
print(peft_config)


# ## 14. 构建训练参数
# 

# In[15]:


from trl import SFTConfig

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

    learning_rate=2e-4,
    warmup_steps=cfg["warmup_steps"],      # 用 profile 的 150
    lr_scheduler_type="cosine",

    num_train_epochs=num_train_epochs,
    max_steps=max_steps,                   # 用上面算的 -1,不是 30

    weight_decay=0.0,
    max_grad_norm=1.0,

    bf16=True,
    fp16=False,
    gradient_checkpointing=cfg["gradient_checkpointing"],

    eval_strategy="steps",                 # 从 "no" 改
    eval_steps=cfg["eval_steps"],
    save_strategy="steps",                 # 从 "no" 改
    save_steps=cfg["save_steps"],
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_steps=cfg["logging_steps"],
    report_to="none",

    assistant_only_loss=True,
    packing=False,

    seed=42,
    dataloader_num_workers=2,              # 保守点,从 4 降到 2
    dataloader_pin_memory=False,           # 统一内存不需要
    remove_unused_columns=False,
)
training_args


# ## 15. 构建 SFTTrainer
# 

# In[16]:


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


# ## 16. 最终配置确认
# 

# In[17]:


print("=" * 70)
print("FINAL TRAIN CONFIG")
print("=" * 70)
print(f"PROFILE                   : {PROFILE}")
print(f"max_steps                 : {trainer.args.max_steps}")
print(f"num_train_epochs          : {trainer.args.num_train_epochs}")
print(f"per_device_train_batch    : {trainer.args.per_device_train_batch_size}")
print(f"per_device_eval_batch     : {trainer.args.per_device_eval_batch_size}")
print(f"gradient_accumulation     : {trainer.args.gradient_accumulation_steps}")
print(f"effective_batch           : {trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps}")
print(f"gradient_checkpointing    : {trainer.args.gradient_checkpointing}")
print(f"dataloader_num_workers    : {trainer.args.dataloader_num_workers}")
print(f"dataloader_pin_memory     : {trainer.args.dataloader_pin_memory}")
print(f"assistant_only_loss       : {trainer.args.assistant_only_loss}")
print(f"packing                   : {trainer.args.packing}")
print(f"max_length                : {trainer.args.max_length}")
print(f"eval_steps                : {trainer.args.eval_steps}")
print(f"save_steps                : {trainer.args.save_steps}")
print("=" * 70)


# ## 17. 自动恢复 checkpoint（可选）
# 
# 规则：
# 
# - `AUTO_RESUME = "auto"`：如果 `OUTPUT_DIR/checkpoint-*` 存在，就从最新 checkpoint 继续
# - `AUTO_RESUME = True`：强制找 checkpoint
# - `AUTO_RESUME = False`：从头开始
# 

# In[18]:


def find_latest_checkpoint(output_dir: str):
    p = Path(output_dir)
    if not p.exists():
        return None
    ckpts = [x for x in p.iterdir() if x.is_dir() and x.name.startswith("checkpoint-")]
    if not ckpts:
        return None
    ckpts = sorted(ckpts, key=lambda x: int(x.name.split("-")[-1]))
    return str(ckpts[-1])

latest_ckpt = find_latest_checkpoint(OUTPUT_DIR)
print("latest checkpoint:", latest_ckpt)

if AUTO_RESUME == "auto":
    resume_from_checkpoint = latest_ckpt
elif AUTO_RESUME is True:
    assert latest_ckpt is not None, "AUTO_RESUME=True 但没有找到 checkpoint"
    resume_from_checkpoint = latest_ckpt
else:
    resume_from_checkpoint = None

print("resume_from_checkpoint:", resume_from_checkpoint)


# ## 18. 开始训练
# 
# 另开一个 SSH 窗口监控：
# 
# ```bash
# watch -n 2 'echo "=== free ==="; free -h; echo; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv'
# ```
# 
# 更细一点的监控：
# 
# ```bash
# nvidia-smi dmon -s u -d 1
# ```
# 

# In[19]:


start = time.time()

train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

elapsed = time.time() - start
print("\n" + "=" * 70)
print("训练完成")
print("=" * 70)
print(f"global_step : {train_result.global_step}")
print(f"train_loss  : {train_result.metrics.get('train_loss', 'N/A')}")
print(f"runtime     : {train_result.metrics.get('train_runtime', elapsed)/3600:.2f} 小时")
print(f"samples/sec : {train_result.metrics.get('train_samples_per_second', 'N/A')}")
print(f"steps/sec   : {train_result.metrics.get('train_steps_per_second', 'N/A')}")


# ## 19. 保存 LoRA adapter
# 

# In[ ]:


trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "trained_chat_template.txt"), "w", encoding="utf-8") as f:
    f.write(tokenizer.chat_template or "")

summary = {
    "profile": PROFILE,
    "base_model": BASE_MODEL,
    "output_dir": OUTPUT_DIR,
    "max_length": MAX_LENGTH,
    "train_size": len(dataset["train"]),
    "dev_size": len(dataset["validation"]),
}
with open(os.path.join(OUTPUT_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("saved to:", OUTPUT_DIR)
for name in sorted(os.listdir(OUTPUT_DIR)):
    p = os.path.join(OUTPUT_DIR, name)
    if os.path.isfile(p):
        print(f"  {name:<30s} {os.path.getsize(p)/1024**2:8.1f} MB")
    else:
        print(f"  {name}/")


# ## 20. 快速推理冒烟测试
# 
# 这里直接用 **基座 + LoRA adapter** 做几条样例测试。  
# 推理 tokenizer 重新从基座加载，不沿用训练时那个带 `{% generation %}` 的模板。
# 

# In[ ]:


with open(META_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = json.load(f)["system_prompt"]

infer_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if infer_tokenizer.pad_token is None:
    infer_tokenizer.pad_token = infer_tokenizer.eos_token

test_cases = [
    "Layla Williams lives at Unit 7, 465 Collins Ave, Devonport TAS 7000 and her email is layla_williams98@tpg.com.au.",
    "The meter replacement went through as work order 718723804; invoice to follow.",
    "Please transfer the salary of $85,000 to BSB 062-000, account 12345678 for Mark Chen, DOB 15/03/1988.",
    "Our IT team noticed unusual traffic from 192.168.1.47 hitting the login endpoint.",
]

trainer.model.eval()

for text in test_cases:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Annotate PII in the following text:\n\n{text}"},
    ]
    prompt = infer_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = infer_tokenizer(prompt, return_tensors="pt").to(trainer.model.device)

    with torch.no_grad():
        out = trainer.model.generate(
            **model_inputs,
            max_new_tokens=min(MAX_LENGTH, len(model_inputs["input_ids"][0]) + 256),
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=infer_tokenizer.pad_token_id,
            eos_token_id=infer_tokenizer.eos_token_id,
        )

    gen = infer_tokenizer.decode(
        out[0][len(model_inputs["input_ids"][0]):],
        skip_special_tokens=True,
    )

    print("=" * 80)
    print("INPUT:")
    print(text)
    print()
    print("OUTPUT:")
    print(gen)
    print()


# ## 21. 建议的使用方式
# 
# ### 第一次跑
# 先用：
# 
# ```python
# PROFILE = "fast_full"
# ```
# 
# 把 1 epoch 跑完。  
# 看：
# 
# - loss 是否正常下降
# - `03_evaluate.ipynb` 里的 parse / round-trip 是否正常
# - exact / partial span F1 是否可接受
# 
# ### 机器不稳或速度还是慢
# 改成：
# 
# - 速度优先：把 `train_bs` 从 8 提到 16，再把 `grad_accum` 降到 1
# - 稳定优先：切到 `PROFILE = "safe_full"`
# 
# ### 正式拉满
# 确认效果靠谱后，把 `fast_full` 里的：
# 
# ```python
# "epochs": 1
# ```
# 
# 改成：
# 
# ```python
# "epochs": 3
# ```
# 
# 然后重新构造 trainer 再训练。
# 

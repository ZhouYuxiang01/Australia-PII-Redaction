"""Throughput test for Qwen3.6-27B under realistic audit prompts.

Measures three configs in one model load:
  A. thinking on,  bs=1
  B. thinking off, bs=1
  C. thinking off, bs=8
"""
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/admin/model/Qwen3.6-27B"

AUDIT_SYS = (
    "You are an auditor for a PII span annotation task. "
    "You will be shown a text snippet, the gold span and a model-predicted span "
    "that disagree. Output a JSON object with keys verdict and reason. "
    "verdict must be one of: gold_correct, model_correct, both_wrong, ambiguous."
)
AUDIT_USER = (
    "TEXT: 'D.O.B: 21/05/1968\\nTel no.: 0414 904 749\\nT.F.N: 224 751 441'\n"
    "GOLD: AU_TFN, value='224 751 441'\n"
    "PRED: PHONE,  value='224 751 441'\n"
    "Which is correct?"
)

def build_prompt(tok, enable_thinking):
    msgs = [{"role": "system", "content": AUDIT_SYS},
            {"role": "user",   "content": AUDIT_USER}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                   tokenize=False, enable_thinking=enable_thinking)

def run(model, tok, enable_thinking, batch, max_new):
    text = build_prompt(tok, enable_thinking)
    enc = tok([text]*batch, return_tensors="pt", padding=True).to("cuda:0")
    in_len = enc["input_ids"].shape[1]
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    dt = time.time() - t0
    new_tokens = out.shape[1] - in_len
    total_new = new_tokens * batch
    return dt, new_tokens, total_new, out, in_len

print("loading...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
model.eval()
print(f"loaded, vram={torch.cuda.memory_allocated()/1e9:.1f}GB")

for label, thinking, bs, mxn in [
    ("A think+bs1", True,  1, 200),
    ("B nothink+bs1", False, 1, 120),
    ("C nothink+bs8", False, 8, 120),
]:
    # warmup small
    _ = run(model, tok, thinking, bs, 8)
    dt, ntok, total, out, in_len = run(model, tok, thinking, bs, mxn)
    print(f"{label:20s}: {dt:6.1f}s  new={ntok}/seq  total_new={total}  "
          f"throughput={total/dt:5.1f} tok/s")
    if bs == 1:
        sample = tok.decode(out[0, in_len:], skip_special_tokens=True)
        print("  sample:", sample[:200].replace("\n"," | "))

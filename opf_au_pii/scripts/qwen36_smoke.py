"""Minimal smoke test: load Qwen3.6-27B and generate a short reply."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

MODEL_PATH = "/home/admin/model/Qwen3.6-27B"

t0 = time.time()
print("loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"tokenizer loaded in {time.time()-t0:.1f}s, vocab={tok.vocab_size}")

t0 = time.time()
print("loading model (this can take a minute)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
)
model.eval()
print(f"model loaded in {time.time()-t0:.1f}s")
print(f"vram allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB")

prompt = "Reply with a single short sentence: what is an Australian Tax File Number?"
messages = [{"role": "user", "content": prompt}]
text_in = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
enc = tok(text_in, return_tensors="pt").to("cuda:0")
input_ids = enc["input_ids"]

t0 = time.time()
with torch.inference_mode():
    out = model.generate(
        **enc,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
new = out[0, input_ids.shape[1]:]
text = tok.decode(new, skip_special_tokens=True)
elapsed = time.time() - t0
n_tok = new.shape[0]
print(f"generated {n_tok} tokens in {elapsed:.1f}s ({n_tok/elapsed:.1f} tok/s)")
print("---")
print(text)

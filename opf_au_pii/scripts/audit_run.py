"""Run Qwen3.6-27B as an auditor over OPF v2b-full disagreements.

Streams results JSONL so the run is resumable: if it crashes, just rerun and it
will skip already-judged case_ids.

Output JSONL schema (per case):
{
  "case_id": str,
  "kind": str,
  "verdict": "gold_correct" | "model_correct" | "both_wrong" | "ambiguous"
              | "model_should_detect" | "model_correct_to_skip"  (for FN/FP),
  "reason": str,
  "raw": str (raw model output, for debugging)
}
"""
import argparse, json, os, re, sys, time, yaml
from typing import Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_CASES = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v2b_full/external_1000/audit_disagreements.jsonl"
DEFAULT_OUT   = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v2b_full/external_1000/audit_verdicts.jsonl"
DEFAULT_TAX   = "/home/admin/ZYX/opf_au_pii/configs/taxonomy_v1.1.1.yaml"
DEFAULT_MODEL = "/home/admin/model/Qwen3.6-27B"


def load_taxonomy(path: str) -> Dict[str, str]:
    """code -> description, including AU-specific notes."""
    with open(path) as f:
        tax = yaml.safe_load(f)
    out = {}
    for cls in tax.get("classes", []):
        code = cls.get("code")
        desc = cls.get("description", "").strip()
        if code:
            out[code] = desc
    return out


SYSTEM_PROMPT = """You are an expert auditor of PII span annotations for Australian privacy-filter training data.

You will be shown a text snippet and a disagreement between a gold annotation and a model prediction. Decide which is correct.

Allowed verdicts:
- "gold_correct"  : the gold label/span is right; the model is wrong.
- "model_correct" : the model label/span is right; the gold has an annotation error.
- "both_wrong"    : neither is correct (e.g. correct type would be a third label, or the span boundary is wrong on both sides).
- "ambiguous"     : the dispute is a genuine taxonomy ambiguity that a reasonable annotator could go either way on.

For "false_negative" cases (gold has a span the model missed), use:
- "gold_correct" if the gold span is a real PII instance the model should have detected,
- "model_correct" if the gold span is over-annotation (e.g. not actually PII under this taxonomy),
- "ambiguous" if it could go either way.

For "false_positive" cases (model predicted a span the gold doesn't have), use:
- "gold_correct" if the model's prediction is a hallucination / over-detection,
- "model_correct" if the gold is missing a real PII span the model correctly found,
- "ambiguous" if reasonable.

Respond ONLY with a single line of compact JSON:
{"verdict": "<one of the four>", "reason": "<one short sentence under 25 words>"}
No explanation outside the JSON. No markdown."""


def fmt_span(s):
    if not s:
        return "(none)"
    s = s[0]
    return f"type={s['type']} chars=[{s['start']},{s['end']}] value={s['value']!r}"


def relevant_types(case) -> List[str]:
    out = []
    if case.get("gold"):
        out.append(case["gold"][0]["type"])
    if case.get("pred"):
        out.append(case["pred"][0]["type"])
    return list(dict.fromkeys(out))


def case_to_user_prompt(case, tax_desc: Dict[str, str]) -> str:
    types = relevant_types(case)
    type_lines = []
    for t in types:
        d = tax_desc.get(t, "(no description)")
        type_lines.append(f"- {t}: {d}")
    type_block = "\n".join(type_lines) if type_lines else "(no types)"

    kind_help = {
        "type_mismatch":  "Same character span, but gold and prediction disagree on the TYPE.",
        "boundary":       "Overlapping spans with the same or different type, but different boundaries.",
        "false_negative": "GOLD has a span here; the model predicted nothing overlapping it.",
        "false_positive": "MODEL predicted a span here; the gold has nothing overlapping it.",
    }[case["kind"]]

    return f"""KIND: {case['kind']}
EXPLANATION: {kind_help}

TYPE DEFINITIONS (relevant to this case):
{type_block}

CONTEXT (snippet around the disputed span; ... means truncated):
{case['context']!r}

GOLD: {fmt_span(case.get('gold'))}
PRED: {fmt_span(case.get('pred'))}

Output your verdict JSON now."""


VERDICT_RE = re.compile(r'\{[^}]*"verdict"\s*:\s*"([a-z_]+)"[^}]*\}', re.S)
REASON_RE  = re.compile(r'"reason"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', re.S)


def parse_output(text: str) -> Dict[str, str]:
    text = text.strip()
    # Try strict JSON first
    try:
        # find first { and last } and try
        i = text.find("{"); j = text.rfind("}")
        if i >= 0 and j > i:
            obj = json.loads(text[i:j+1])
            if isinstance(obj, dict) and "verdict" in obj:
                return {"verdict": obj["verdict"], "reason": obj.get("reason", "")}
    except Exception:
        pass
    m = VERDICT_RE.search(text)
    rm = REASON_RE.search(text)
    if m:
        return {"verdict": m.group(1), "reason": rm.group(1) if rm else ""}
    return {"verdict": "parse_error", "reason": text[:200]}


def already_done(out_path: str) -> set:
    seen = set()
    if not os.path.exists(out_path):
        return seen
    with open(out_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                seen.add(obj["case_id"])
            except Exception:
                continue
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=DEFAULT_CASES)
    ap.add_argument("--out",   default=DEFAULT_OUT)
    ap.add_argument("--taxonomy", default=DEFAULT_TAX)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=80)
    ap.add_argument("--limit", type=int, default=0, help="0=no limit")
    ap.add_argument("--kinds", default="", help="comma-separated; empty=all")
    args = ap.parse_args()

    tax = load_taxonomy(args.taxonomy)

    cases = []
    with open(args.cases) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if args.kinds and c["kind"] not in args.kinds.split(","):
                continue
            cases.append(c)

    seen = already_done(args.out)
    todo = [c for c in cases if c["case_id"] not in seen]
    if args.limit:
        todo = todo[:args.limit]
    print(f"total cases: {len(cases)}  already done: {len(seen)}  todo: {len(todo)}")
    if not todo:
        return

    print("loading model...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s, vram={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    def build_prompt(case):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": case_to_user_prompt(case, tax)},
        ]
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)

    out_f = open(args.out, "a", buffering=1)
    n_done = 0
    n_parse_err = 0
    t_start = time.time()

    for i in range(0, len(todo), args.batch):
        batch_cases = todo[i:i+args.batch]
        prompts = [build_prompt(c) for c in batch_cases]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")
        in_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        new = out[:, in_len:]
        texts = tok.batch_decode(new, skip_special_tokens=True)

        for case, raw in zip(batch_cases, texts):
            parsed = parse_output(raw)
            rec = {
                "case_id": case["case_id"],
                "kind": case["kind"],
                "verdict": parsed["verdict"],
                "reason": parsed["reason"],
                "raw": raw.strip()[:300],
            }
            if parsed["verdict"] == "parse_error":
                n_parse_err += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += 1

        elapsed = time.time() - t_start
        rate = n_done / max(elapsed, 1e-6)
        eta = (len(todo) - n_done) / max(rate, 1e-6)
        print(f"  [{n_done}/{len(todo)}] rate={rate:.2f} cases/s "
              f"parse_err={n_parse_err} elapsed={elapsed:.0f}s eta={eta:.0f}s",
              flush=True)

    out_f.close()
    print(f"DONE. {n_done} cases written. parse_err={n_parse_err}.")


if __name__ == "__main__":
    main()

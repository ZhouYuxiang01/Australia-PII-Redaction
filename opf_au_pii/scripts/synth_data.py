"""Generate synthetic training documents with Qwen3.6-27B for OPF v3.

Strategy:
- Each generated document is a realistic AU document (email/form/JSON/chat/case-note).
- The generator is told which TYPES to include, weighted toward weak classes.
- Model outputs JSON: {"text": str, "spans": [{"type": str, "value": str}, ...]}
- Post-processor finds char offsets for each value in text (skipping unfound values).
- Final output is OPF training format JSONL with offsets verified.

Resumable: streaming JSONL output; skipping already-emitted indices.
"""
import argparse, json, os, random, re, sys, time, yaml
from collections import Counter
from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TAX = "/home/admin/ZYX/opf_au_pii/configs/taxonomy_v1.1.1.yaml"
DEFAULT_PERLABEL = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v2b_full/external_1000/positive_char_metrics.json"
DEFAULT_OUT = "/home/admin/ZYX/opf_au_pii/data/processed/data_opf_v3/synth_raw.jsonl"
DEFAULT_OPF_OUT = "/home/admin/ZYX/opf_au_pii/data/processed/data_opf_v3/synth_opf.jsonl"
DEFAULT_MODEL = "/home/admin/model/Qwen3.6-27B"

DOC_TYPES = [
    "forwarded internal email thread between case workers",
    "structured JSON payload from a CRM/API request",
    "intake/registration form with labelled fields",
    "customer-service chat transcript (Agent / User turns)",
    "HR record extract with multiple sections",
    "medical referral note from a clinician",
    "student enrolment record with academic history",
    "case management notes with timestamped entries",
    "tax-related correspondence with ATO references",
    "Centrelink claim file with payment history",
    "scholarship application review letter",
    "bank statement remarks with transaction notes",
    "employer onboarding confirmation document",
    "educational verification letter with multiple identifiers",
]

# Confusion pairs derived from audit. Higher-priority pairs are listed multiple
# times so weighted random.choice samples them more often.
CONFUSION_PAIRS = (
    # === audit top 10 real OPF errors (replicated 3x for emphasis) ===
    [("NEXT_OF_KIN", "PERSON")] * 3
    + [("SALARY", "SALARY_WAGE_EXPECTATION")] * 3
    + [("MEDICARE_EXPIRY", "CREDIT_CARD_EXPIRY")] * 3
    + [("PASSPORT_START_DATE", "DATE_OF_BIRTH")] * 3
    + [("PASSPORT_EXPIRY", "PASSPORT_START_DATE")] * 3
    + [("WEBSITE_HISTORY", "SOCIAL_MEDIA_HISTORY")] * 3
    + [("FINGERPRINT", "AU_DRIVERS_LICENCE")] * 3
    + [("PERSONAL_DEBT", "SALARY_WAGE_EXPECTATION")] * 3
    + [("RACIAL_ETHNIC_ORIGIN", "NATIONALITY")] * 3
    + [("AU_BANK_ACCOUNT", "AU_DRIVERS_LICENCE")] * 3
    # === audit 11-25 real OPF errors (replicated 2x) ===
    + [("AUDIO_INFORMATION", "COUNSELLING_RECORDS")] * 2
    + [("FINGERPRINT", "NATIONAL_IDENTITY_CARD")] * 2
    + [("SANCTIONS", "DISABILITY_OR_SPECIFIC_CONDITION")] * 2
    + [("SOCIO_ECONOMIC_STATUS", "MEDICAL_INFORMATION")] * 2
    + [("MEDICARE_NUMBER", "PAYMENT_CARD_NUMBER")] * 2
    + [("LONGITUDE", "LATITUDE")] * 2
    + [("SANCTIONS", "MEDICAL_INFORMATION")] * 2
    + [("RACIAL_ETHNIC_ORIGIN", "ABORIGINALITY")] * 2
    + [("PASSPORT_EXPIRY", "MEDICARE_EXPIRY")] * 2
    + [("CAMERA_FOOTAGE_AUDIO", "COUNSELLING_RECORDS")] * 2
    + [("FACIAL_RECOGNITION", "VOICE_RECOGNITION")] * 2
    + [("FACIAL_RECOGNITION", "PERSONNEL_NUMBER")] * 2
    + [("SOCIO_ECONOMIC_STATUS", "COUNSELLING_RECORDS")] * 2
    + [("PASSPORT_EXPIRY", "CREDIT_CARD_EXPIRY")] * 2
    + [("AU_TFN", "PHONE")] * 2
    # === remaining sensible discriminations (1x) ===
    + [
        ("AU_TFN", "AU_DRIVERS_LICENCE"),
        ("AU_TFN", "STUDENT_ID"),
        ("AU_TFN", "AU_BANK_ACCOUNT"),
        ("STUDENT_ID", "AU_DRIVERS_LICENCE"),
        ("SOCIAL_MEDIA_ID", "SOCIAL_MEDIA_ACCOUNT"),
        ("WEBSITE_HISTORY", "COOKIE_INFORMATION"),
        ("FINGERPRINT", "FACIAL_RECOGNITION"),
        ("AUDIO_INFORMATION", "CAMERA_FOOTAGE_AUDIO"),
        ("SANCTIONS", "SPECIAL_CONSIDERATION"),
        ("SANCTIONS", "CRIMINAL_RECORDS"),
        ("MEDICAL_INFORMATION", "MEDICAL_CERTIFICATE"),
        ("MEDICAL_INFORMATION", "DISABILITY_OR_SPECIFIC_CONDITION"),
        ("COUNSELLING_RECORDS", "MEDICAL_INFORMATION"),
        ("SALARY", "PERSONAL_DEBT"),
        ("PERSONAL_DEBT", "SOCIO_ECONOMIC_STATUS"),
        ("DEVICE_ID", "IP_ADDRESS"),
        ("USERNAME", "SOCIAL_MEDIA_ACCOUNT"),
        ("EMPLOYEE_NUMBER", "PERSONNEL_NUMBER"),
        ("EMPLOYEE_NUMBER", "STUDENT_ID"),
        ("PRONOUN", "GENDER"),
    ]
)


def load_taxonomy(path: str):
    with open(path) as f:
        tax = yaml.safe_load(f)
    out = {}
    for cls in tax.get("classes", []):
        c = cls.get("code")
        d = cls.get("description", "").strip()
        if c:
            out[c] = d
    return out


def load_per_label_f1(path: str) -> Dict[str, float]:
    m = json.load(open(path))
    typed = m["typed"]
    tp = typed["tp_by_label"]; fp = typed.get("fp_by_label", {}); fn = typed.get("fn_by_label", {})
    out = {}
    for L in set(tp) | set(fp) | set(fn):
        t = tp.get(L, 0); f = fp.get(L, 0); n = fn.get(L, 0)
        p = t/(t+f) if t+f else 0
        r = t/(t+n) if t+n else 0
        out[L] = 2*p*r/(p+r) if p+r else 0
    return out


def class_weight(f1: float) -> float:
    if f1 < 0.20: return 8.0
    if f1 < 0.40: return 5.0
    if f1 < 0.60: return 3.0
    if f1 < 0.80: return 1.5
    return 0.5


def sample_types(rng: random.Random, weights: Dict[str, float],
                 confusion_pairs: List[Tuple[str, str]],
                 n_main: int = 9) -> List[str]:
    """Sample ~9 types: 1 confusion pair + remaining weighted draws (no dup)."""
    chosen: set = set()
    pair = rng.choice(confusion_pairs)
    chosen.update(pair)
    pool = [t for t in weights if t not in chosen]
    pool_w = [weights[t] for t in pool]
    while len(chosen) < n_main and pool:
        # Weighted draw without replacement
        total = sum(pool_w)
        x = rng.random() * total
        acc = 0.0
        idx = 0
        for i, w in enumerate(pool_w):
            acc += w
            if x <= acc:
                idx = i
                break
        chosen.add(pool[idx])
        del pool[idx]; del pool_w[idx]
    return list(chosen)


SYSTEM = """You generate realistic Australian PII-rich documents for training a privacy filter.
Strict rules:
1. Output ONLY a single JSON object on one line. No markdown, no code fences.
2. The JSON has keys: "text" (the document) and "spans" (a list of {"type", "value"}).
3. Every value in "spans" MUST appear verbatim (case-sensitive) at least once in "text".
4. Each value should be UNIQUE in the text (avoid identical strings) so offsets are unambiguous.
5. Use Australian conventions: AU phones (04xx xxx xxx for mobile, 0x xxxx xxxx for landline),
   TFN format "xxx xxx xxx" or "xxx-xxx-xxx" (9 digits), Medicare "xxxx xxxxx x" (10-11 digits),
   AU passports "Letter+digits" (e.g. N1234567), AU driver licence varies (5-10 digit/alpha).
6. Make label boundaries precise: for ADDRESS include suburb+state+postcode where natural;
   for PERSON use full name (no leading title in span); for IDs include the digits only,
   not the prefix label.
7. Cover the requested types; you may include up to 2 EXTRA types if natural to the doc.
8. KEEP THE DOCUMENT COMPACT: 100-220 words. Use form-style or list-style layouts
   that pack many labelled fields per line (e.g. "TFN: xxx xxx xxx | DL: ABC1234").
   Do NOT pad with prose. Density of PII spans matters more than narrative.
9. CRITICAL: never produce a value that exists by accident multiple times -- vary digits/text.
10. Output must fit comfortably below 900 tokens including the JSON wrapper.
"""

EXAMPLE = """{"text": "Subject: Enrolment update -- A. Wong\\nFrom: registrar@example.edu.au\\nDate: 12 Mar 2026\\n\\nHi Amelia,\\n\\nYour student record has been updated. Confirming details below for your records.\\n\\nName: Amelia Wong\\nDate of birth: 14/06/2002\\nStudent ID: 449221\\nTFN: 832 119 477\\nAddress: 18 Wattle Cres, Marrickville NSW 2204\\nMobile: 0421 905 663\\nEmail: amelia.wong22@example.edu.au\\nMedicare: 4567 81234 1\\nScholarship: awarded the Indigenous Excellence Scholarship\\nAboriginality: identifies as Aboriginal Australian\\n\\nIf any of the above is incorrect, please reply by 15 Mar.\\n\\nKind regards,\\nRegistrar's Office", "spans": [{"type": "PERSON", "value": "Amelia Wong"}, {"type": "DATE_OF_BIRTH", "value": "14/06/2002"}, {"type": "STUDENT_ID", "value": "449221"}, {"type": "AU_TFN", "value": "832 119 477"}, {"type": "ADDRESS", "value": "18 Wattle Cres, Marrickville NSW 2204"}, {"type": "PHONE", "value": "0421 905 663"}, {"type": "EMAIL", "value": "amelia.wong22@example.edu.au"}, {"type": "MEDICARE_NUMBER", "value": "4567 81234 1"}, {"type": "SCHOLARSHIP", "value": "Indigenous Excellence Scholarship"}, {"type": "ABORIGINALITY", "value": "Aboriginal Australian"}]}"""


def build_user(types: List[str], descriptions: Dict[str, str], doc_type: str) -> str:
    type_block = "\n".join(f'- {t}: {descriptions.get(t, "(no description)")}' for t in types)
    return f"""Document type: {doc_type}

Required PII types to include (with definitions):
{type_block}

Example output (for a different scenario):
{EXAMPLE}

Now produce ONE new document covering the requested types in the document type above.
Output a single JSON object on one line. JSON only, no commentary."""


def parse_response(raw: str) -> dict:
    """Parse model JSON; salvage truncated outputs by recovering text + complete spans."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    # Strict balanced-{...} first.
    i = raw.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(raw)):
        ch = raw[j]
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[i:j+1])
                except Exception:
                    break  # fall through to salvage
    # Salvage: extract "text": "..." and any complete {"type":"X","value":"Y"} objects.
    text_m = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if not text_m:
        return None
    try:
        text_val = json.loads('"' + text_m.group(1) + '"')
    except Exception:
        return None
    span_objs = []
    for sm in re.finditer(
        r'\{\s*"type"\s*:\s*"([^"\\]+)"\s*,\s*"value"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        raw,
    ):
        try:
            v = json.loads('"' + sm.group(2) + '"')
        except Exception:
            continue
        span_objs.append({"type": sm.group(1), "value": v})
    if not span_objs:
        return None
    return {"text": text_val, "spans": span_objs, "_salvaged": True}


def find_offsets(text: str, value: str) -> List[Tuple[int, int]]:
    out = []
    if not value:
        return out
    start = 0
    while True:
        k = text.find(value, start)
        if k < 0:
            break
        out.append((k, k + len(value)))
        start = k + 1
    return out


def to_opf(record: dict, eid: str) -> Tuple[dict, dict]:
    """Convert {text, spans:[{type,value}]} -> OPF schema. Returns (record, audit)."""
    text = record.get("text", "")
    spans = record.get("spans", [])
    grouped: Dict[str, List[List[int]]] = {}
    audit = {"requested": len(spans), "found": 0, "not_found": [], "ambiguous_multi": []}
    seen_offsets = set()
    for sp in spans:
        if not isinstance(sp, dict):
            continue
        t = str(sp.get("type", "")).strip()
        v = sp.get("value", "")
        # The model occasionally emits a number for value; coerce to string.
        if v is None:
            continue
        v = str(v).strip()
        if not t or not v:
            continue
        positions = find_offsets(text, v)
        if not positions:
            audit["not_found"].append({"type": t, "value": v})
            continue
        # If value appears multiple times, take only the first to avoid noisy duplicates
        if len(positions) > 1:
            audit["ambiguous_multi"].append({"type": t, "value": v, "n": len(positions)})
        s, e = positions[0]
        if (s, e, t) in seen_offsets:
            continue
        seen_offsets.add((s, e, t))
        key = f"{t}: {v}"
        grouped.setdefault(key, []).append([s, e])
        audit["found"] += 1
    return ({"example_id": eid, "text": text, "spans": grouped}, audit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="number of documents to generate")
    ap.add_argument("--out_raw", default=DEFAULT_OUT)
    ap.add_argument("--out_opf", default=DEFAULT_OPF_OUT)
    ap.add_argument("--taxonomy", default=DEFAULT_TAX)
    ap.add_argument("--per_label", default=DEFAULT_PERLABEL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=900)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--prefix", default="SYN")
    args = ap.parse_args()

    desc = load_taxonomy(args.taxonomy)
    f1 = load_per_label_f1(args.per_label)
    weights = {L: class_weight(f1.get(L, 0.5)) for L in desc.keys()}

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(args.out_raw), exist_ok=True)

    # Resume: count existing
    existing_n = 0
    if os.path.exists(args.out_opf):
        existing_n = sum(1 for _ in open(args.out_opf))
    todo_n = max(0, args.n - existing_n)
    if todo_n == 0:
        print(f"already have {existing_n} >= {args.n}")
        return
    print(f"need {todo_n} more docs (have {existing_n})")

    print("loading model...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model.eval()
    print(f"  loaded {time.time()-t0:.1f}s vram={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    raw_f = open(args.out_raw, "a", buffering=1)
    opf_f = open(args.out_opf, "a", buffering=1)
    n_emitted = existing_n
    n_parse_err = 0
    n_no_spans = 0
    span_type_counts = Counter()
    not_found_counts = Counter()
    t_start = time.time()

    while n_emitted < args.n:
        bs = min(args.batch, args.n - n_emitted)
        batch_meta = []
        prompts = []
        for _ in range(bs):
            doc_type = rng.choice(DOC_TYPES)
            types = sample_types(rng, weights, CONFUSION_PAIRS, n_main=9)
            user = build_user(types, desc, doc_type)
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user}]
            text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
            prompts.append(text)
            batch_meta.append({"types": types, "doc_type": doc_type})

        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")
        in_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                                 temperature=0.9, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        new = out[:, in_len:]
        outs = tok.batch_decode(new, skip_special_tokens=True)

        for raw, meta in zip(outs, batch_meta):
            n_emitted += 1
            eid = f"{args.prefix}-{n_emitted:06d}"
            try:
                obj = parse_response(raw)
                if obj is None or "text" not in obj or "spans" not in obj:
                    n_parse_err += 1
                    raw_f.write(json.dumps({"id": eid, "raw": raw[:600], "error": "parse"},
                                           ensure_ascii=False) + "\n")
                    continue
                if not isinstance(obj["text"], str) or not isinstance(obj["spans"], list):
                    n_parse_err += 1
                    raw_f.write(json.dumps({"id": eid, "error": "bad_shape",
                                            "raw": raw[:300]}, ensure_ascii=False) + "\n")
                    continue
                opf_rec, audit = to_opf(obj, eid)
                if not opf_rec["spans"]:
                    n_no_spans += 1
                    raw_f.write(json.dumps({"id": eid, "raw_text_head": obj["text"][:200],
                                            "audit": audit, "error": "no_spans"},
                                           ensure_ascii=False) + "\n")
                    continue
                for k in opf_rec["spans"]:
                    t = k.split(":", 1)[0].strip()
                    span_type_counts[t] += len(opf_rec["spans"][k])
                for nf in audit["not_found"]:
                    not_found_counts[nf["type"]] += 1
                opf_f.write(json.dumps(opf_rec, ensure_ascii=False) + "\n")
                raw_f.write(json.dumps({"id": eid, "meta": meta, "audit": audit},
                                       ensure_ascii=False) + "\n")
            except Exception as ex:
                n_parse_err += 1
                raw_f.write(json.dumps({"id": eid, "error": "exception",
                                        "exc": str(ex)[:300],
                                        "raw": raw[:300]},
                                       ensure_ascii=False) + "\n")
                continue

        elapsed = time.time() - t_start
        rate = (n_emitted - existing_n) / max(elapsed, 1e-6)
        eta = (args.n - n_emitted) / max(rate, 1e-6)
        print(f"  [{n_emitted}/{args.n}] rate={rate:.2f}/s parse_err={n_parse_err} "
              f"no_spans={n_no_spans} elapsed={elapsed:.0f}s eta={eta:.0f}s",
              flush=True)

    raw_f.close()
    opf_f.close()
    print("\n--- summary ---")
    print(f"emitted: {n_emitted}  parse_err: {n_parse_err}  no_spans: {n_no_spans}")
    print(f"top span types: {span_type_counts.most_common(15)}")
    print(f"top not_found types: {not_found_counts.most_common(10)}")


if __name__ == "__main__":
    main()

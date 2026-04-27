"""Aggregate audit verdicts: by kind, by verdict, top confusion patterns."""
import json, sys
from collections import Counter, defaultdict

VERDICTS = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v2b_full/external_1000/audit_verdicts.jsonl"
CASES    = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v2b_full/external_1000/audit_disagreements.jsonl"
OUT      = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v2b_full/external_1000/audit_summary.json"

verdicts = {}
with open(VERDICTS) as f:
    for line in f:
        r = json.loads(line)
        verdicts[r["case_id"]] = r

cases = {}
with open(CASES) as f:
    for line in f:
        c = json.loads(line)
        cases[c["case_id"]] = c

# Cross-tab: kind x verdict
xtab = defaultdict(Counter)
total_by_kind = Counter()
for cid, v in verdicts.items():
    c = cases[cid]
    xtab[c["kind"]][v["verdict"]] += 1
    total_by_kind[c["kind"]] += 1

# Per-type confusion: gold_type -> pred_type -> verdict counts (only type_mismatch)
confusion = defaultdict(lambda: defaultdict(Counter))
for cid, v in verdicts.items():
    c = cases[cid]
    if c["kind"] != "type_mismatch":
        continue
    g = c["gold"][0]["type"] if c["gold"] else "-"
    p = c["pred"][0]["type"] if c["pred"] else "-"
    confusion[g][p][v["verdict"]] += 1

# False-negative analysis: which gold types are most often agreed-real-FN (model should detect)?
fn_by_gold_type = defaultdict(Counter)
for cid, v in verdicts.items():
    c = cases[cid]
    if c["kind"] != "false_negative":
        continue
    g = c["gold"][0]["type"]
    fn_by_gold_type[g][v["verdict"]] += 1

# False-positive analysis
fp_by_pred_type = defaultdict(Counter)
for cid, v in verdicts.items():
    c = cases[cid]
    if c["kind"] != "false_positive":
        continue
    p = c["pred"][0]["type"]
    fp_by_pred_type[p][v["verdict"]] += 1

# Boundary
bd_by_type = defaultdict(Counter)
for cid, v in verdicts.items():
    c = cases[cid]
    if c["kind"] != "boundary":
        continue
    g = c["gold"][0]["type"] if c["gold"] else "-"
    bd_by_type[g][v["verdict"]] += 1

# ---- print ----
print("=== verdict cross-tab (kind × verdict) ===")
all_verdicts = sorted({v for k in xtab for v in xtab[k]})
header = f"{'kind':18s}  {'total':>6s}  " + "  ".join(f"{v:>15s}" for v in all_verdicts)
print(header)
for kind in ("type_mismatch", "boundary", "false_negative", "false_positive"):
    total = total_by_kind[kind]
    parts = []
    for v in all_verdicts:
        c = xtab[kind][v]
        pct = 100*c/total if total else 0
        parts.append(f"{c:5d} ({pct:4.1f}%)")
    print(f"{kind:18s}  {total:6d}  " + "  ".join(f"{p:>15s}" for p in parts))

# Aggregate: total real-OPF-errors vs real-gold-noise vs ambiguous/parse_err
real_opf_error = 0
real_gold_noise = 0
ambiguous = 0
both_wrong = 0
parse_err = 0
for v in verdicts.values():
    if v["verdict"] == "gold_correct":
        real_opf_error += 1
    elif v["verdict"] == "model_correct":
        real_gold_noise += 1
    elif v["verdict"] == "ambiguous":
        ambiguous += 1
    elif v["verdict"] == "both_wrong":
        both_wrong += 1
    elif v["verdict"] == "parse_error":
        parse_err += 1
total = len(verdicts)
print(f"\n=== overall ({total} cases) ===")
print(f"  real OPF error  (gold_correct):  {real_opf_error:5d}  ({100*real_opf_error/total:.1f}%)")
print(f"  real gold noise (model_correct): {real_gold_noise:5d}  ({100*real_gold_noise/total:.1f}%)")
print(f"  ambiguous:                       {ambiguous:5d}  ({100*ambiguous/total:.1f}%)")
print(f"  both_wrong:                      {both_wrong:5d}  ({100*both_wrong/total:.1f}%)")
print(f"  parse_error:                     {parse_err:5d}  ({100*parse_err/total:.1f}%)")

# Top type-mismatch confusions where OPF is genuinely wrong (to target with synthesis)
print(f"\n=== top type_mismatch confusions where OPF was wrong (gold_correct) ===")
pairs = Counter()
for g, preds in confusion.items():
    for p, vc in preds.items():
        pairs[(g, p)] += vc.get("gold_correct", 0)
for (g, p), n in pairs.most_common(25):
    print(f"  {g:32s} -> {p:32s}  {n:4d}")

# Top false_negatives (real gold present, model missed) by type
print(f"\n=== top false_negatives where model SHOULD have detected (gold_correct) ===")
fn_real = Counter()
for g, vc in fn_by_gold_type.items():
    fn_real[g] = vc.get("gold_correct", 0)
for g, n in fn_real.most_common(20):
    print(f"  {g:32s}  {n:4d}")

# Top false_positives where model was hallucinating (gold_correct)
print(f"\n=== top false_positives where model hallucinated (gold_correct) ===")
fp_real = Counter()
for p, vc in fp_by_pred_type.items():
    fp_real[p] = vc.get("gold_correct", 0)
for p, n in fp_real.most_common(20):
    print(f"  {p:32s}  {n:4d}")

# Save aggregated json
out = {
    "total": total,
    "by_verdict": {
        "gold_correct": real_opf_error,
        "model_correct": real_gold_noise,
        "ambiguous": ambiguous,
        "both_wrong": both_wrong,
        "parse_error": parse_err,
    },
    "by_kind_verdict": {k: dict(v) for k, v in xtab.items()},
    "type_mismatch_real_opf_errors": [
        {"gold": g, "pred": p, "n": n} for (g, p), n in pairs.most_common(50)
    ],
    "false_negative_real": dict(fn_real.most_common(50)),
    "false_positive_real": dict(fp_real.most_common(50)),
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {OUT}")

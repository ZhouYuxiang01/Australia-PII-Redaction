"""Pair OPF v2b predictions with gold on external_1000 and emit a disagreement file.

Output schema (one JSON per line):
{
  "row_id": "<text-hash>",
  "case_id": "<row_id>:<idx>",
  "kind": "type_mismatch" | "boundary" | "false_negative" | "false_positive",
  "text": str,                # full row text
  "context": str,             # +/-80 char window around the disputed span(s)
  "gold": [{"type": str, "value": str, "start": int, "end": int}, ...] | null,
  "pred": [{"type": str, "value": str, "start": int, "end": int}, ...] | null,
}

Pairing strategy: by exact text match (the two files share text content).
"""
import argparse, hashlib, json, os
from collections import Counter, defaultdict

GOLD_PATH = "/home/admin/ZYX/Qwen3.5_4b_base_Full_73class/data/eval_external_1000/positive_1000.jsonl"
PRED_PATH = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v2b_full/external_1000/positive_predictions.jsonl"
OUT_PATH  = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v2b_full/external_1000/audit_disagreements.jsonl"
STATS_PATH = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v2b_full/external_1000/audit_diff_stats.json"

CTX_WINDOW = 80


def text_hash(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def explode_spans(spans_dict):
    """Convert the OPF format {'TYPE: value': [[s,e], ...]} into a flat list."""
    out = []
    if not spans_dict:
        return out
    for key, ranges in spans_dict.items():
        if ":" in key:
            typ, val = key.split(":", 1)
            typ = typ.strip()
            val = val.strip()
        else:
            typ, val = key, ""
        for s, e in ranges:
            out.append({"type": typ, "value": val, "start": int(s), "end": int(e)})
    return out


def context(text, start, end, w=CTX_WINDOW):
    a = max(0, start - w)
    b = min(len(text), end + w)
    prefix = "..." if a > 0 else ""
    suffix = "..." if b < len(text) else ""
    return prefix + text[a:b] + suffix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=GOLD_PATH)
    ap.add_argument("--pred", default=PRED_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--stats", default=STATS_PATH)
    args = ap.parse_args()

    gold_rows = load_jsonl(args.gold)
    pred_rows = load_jsonl(args.pred)

    gold_by_text = {r["text"]: r for r in gold_rows}
    pred_by_text = {r["text"]: r for r in pred_rows}

    common_texts = set(gold_by_text) & set(pred_by_text)
    only_gold = set(gold_by_text) - common_texts
    only_pred = set(pred_by_text) - common_texts

    cases = []
    counts = Counter()

    for text in sorted(common_texts):
        rid = text_hash(text)
        gold_spans = explode_spans(gold_by_text[text].get("spans", {}))
        pred_spans = explode_spans(pred_by_text[text].get("predicted_spans", {}))

        # Index by (start, end) for matching
        gold_by_se = defaultdict(list)
        for s in gold_spans:
            gold_by_se[(s["start"], s["end"])].append(s)
        pred_by_se = defaultdict(list)
        for s in pred_spans:
            pred_by_se[(s["start"], s["end"])].append(s)

        used_gold_se = set()
        used_pred_se = set()

        # 1) type_mismatch: same exact span boundaries, different type
        for se, gs in gold_by_se.items():
            if se in pred_by_se:
                for g in gs:
                    for p in pred_by_se[se]:
                        if g["type"] != p["type"]:
                            cases.append({
                                "row_id": rid, "case_id": f"{rid}:tm:{len(cases)}",
                                "kind": "type_mismatch",
                                "text": text,
                                "context": context(text, se[0], se[1]),
                                "gold": [g], "pred": [p],
                            })
                            counts["type_mismatch"] += 1
                # Mark consumed even if same type
                used_gold_se.add(se)
                used_pred_se.add(se)

        # 2) boundary: overlapping but not equal (any overlap)
        unused_gold = [se for se in gold_by_se if se not in used_gold_se]
        unused_pred = [se for se in pred_by_se if se not in used_pred_se]
        for gse in unused_gold:
            for pse in unused_pred:
                if pse in used_pred_se or gse in used_gold_se:
                    continue
                if gse[0] < pse[1] and pse[0] < gse[1]:  # overlap
                    g = gold_by_se[gse][0]
                    p = pred_by_se[pse][0]
                    cases.append({
                        "row_id": rid, "case_id": f"{rid}:bd:{len(cases)}",
                        "kind": "boundary",
                        "text": text,
                        "context": context(text, min(gse[0], pse[0]), max(gse[1], pse[1])),
                        "gold": [g], "pred": [p],
                    })
                    counts["boundary"] += 1
                    used_gold_se.add(gse)
                    used_pred_se.add(pse)
                    break

        # 3) false_negative: gold spans never matched (gold has it, model missed)
        for gse in gold_by_se:
            if gse in used_gold_se:
                continue
            g = gold_by_se[gse][0]
            cases.append({
                "row_id": rid, "case_id": f"{rid}:fn:{len(cases)}",
                "kind": "false_negative",
                "text": text,
                "context": context(text, gse[0], gse[1]),
                "gold": [g], "pred": None,
            })
            counts["false_negative"] += 1

        # 4) false_positive: pred spans never matched (model has it, gold missed)
        for pse in pred_by_se:
            if pse in used_pred_se:
                continue
            p = pred_by_se[pse][0]
            cases.append({
                "row_id": rid, "case_id": f"{rid}:fp:{len(cases)}",
                "kind": "false_positive",
                "text": text,
                "context": context(text, pse[0], pse[1]),
                "gold": None, "pred": [p],
            })
            counts["false_positive"] += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    stats = {
        "rows_total_gold": len(gold_rows),
        "rows_total_pred": len(pred_rows),
        "rows_paired_by_text": len(common_texts),
        "rows_only_gold": len(only_gold),
        "rows_only_pred": len(only_pred),
        "disagreement_total": len(cases),
        "by_kind": dict(counts),
    }
    # Per-type counts within each kind
    by_kind_type = defaultdict(Counter)
    for c in cases:
        kind = c["kind"]
        if c["gold"]:
            by_kind_type[kind + ".gold_type"][c["gold"][0]["type"]] += 1
        if c["pred"]:
            by_kind_type[kind + ".pred_type"][c["pred"][0]["type"]] += 1
    stats["by_kind_type"] = {k: dict(v.most_common(20)) for k, v in by_kind_type.items()}

    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()

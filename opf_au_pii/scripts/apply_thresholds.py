"""Apply thresholds (derived on dev) to a confidence-augmented predictions file
and report achieved P/R/F1 globally and per type, in BLOCK and REVIEW modes.

This is the methodologically clean reporting:
  thresholds derived on dev (in-distribution) -> applied to external_1000 (test).
"""
import argparse, json, os
from collections import Counter, defaultdict

DEFAULT_PRED = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v3_full/external_1000_with_logprobs/predictions_with_confidence.jsonl"
DEFAULT_GOLD = "/home/admin/ZYX/Qwen3.5_4b_base_Full_73class/data/eval_external_1000/positive_1000.jsonl"
DEFAULT_THRESH = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v3_full/dev_with_logprobs/thresholds_dev.json"
DEFAULT_REPORT = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v3_full/external_1000_with_logprobs/dev_thresholds_applied_report.json"
DEFAULT_MD = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v3_full/external_1000_with_logprobs/dev_thresholds_applied.md"


def load_gold_count(p):
    counts = Counter()
    total = 0
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            for k, ranges in r.get("spans", {}).items():
                if ":" not in k: continue
                t = k.split(":", 1)[0].strip()
                counts[t] += len(ranges)
                total += len(ranges)
    return counts, total


def load_preds(p):
    """Yield (type, confidence, is_tp) flatly."""
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            for sp in r.get("preds", []):
                yield sp["type"], sp["confidence"], (sp["match"] == "tp")


def evaluate_mode(preds_iter, gold_counts, gold_total, thresholds, mode_key):
    """For each pred, decide if it survives the type's threshold for the given mode.
    Compute global + per-type P/R/F1.
    """
    by_type_tp = Counter()
    by_type_fp = Counter()
    g_tp = 0; g_fp = 0
    n_kept = 0

    global_thr = thresholds.get(f"global_{mode_key}_threshold")
    per_type = thresholds.get("per_type", {})

    for typ, conf, is_tp in preds_iter:
        ti = per_type.get(typ, {})
        thr = ti.get(f"{mode_key}_threshold")
        if thr is None:
            thr = global_thr
        if thr is None:
            continue  # mode unreachable
        if conf >= thr:
            n_kept += 1
            if is_tp:
                by_type_tp[typ] += 1
                g_tp += 1
            else:
                by_type_fp[typ] += 1
                g_fp += 1

    g_fn = gold_total - g_tp
    g_P = g_tp / max(g_tp + g_fp, 1)
    g_R = g_tp / max(gold_total, 1)
    g_F1 = 2 * g_P * g_R / (g_P + g_R) if (g_P + g_R) else 0.0

    per_type_out = {}
    for t in set(list(gold_counts) + list(by_type_tp) + list(by_type_fp)):
        tp = by_type_tp.get(t, 0); fp = by_type_fp.get(t, 0)
        n_g = gold_counts.get(t, 0); fn = n_g - tp
        P = tp / max(tp + fp, 1)
        R = tp / max(n_g, 1)
        F1 = 2 * P * R / (P + R) if (P + R) else 0.0
        per_type_out[t] = {"tp": tp, "fp": fp, "fn": fn, "n_gold": n_g,
                           "P": P, "R": R, "F1": F1}
    return {
        "global": {"tp": g_tp, "fp": g_fp, "fn": g_fn, "n_gold": gold_total,
                   "P": g_P, "R": g_R, "F1": g_F1},
        "per_type": per_type_out,
        "n_kept_predictions": n_kept,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=DEFAULT_PRED)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--thresholds", default=DEFAULT_THRESH)
    ap.add_argument("--report_out", default=DEFAULT_REPORT)
    ap.add_argument("--md_out", default=DEFAULT_MD)
    args = ap.parse_args()

    gold_counts, gold_total = load_gold_count(args.gold)
    thresholds = json.load(open(args.thresholds))

    block = evaluate_mode(load_preds(args.pred), gold_counts, gold_total,
                          thresholds, "block")
    review = evaluate_mode(load_preds(args.pred), gold_counts, gold_total,
                           thresholds, "review")
    raw = evaluate_mode(load_preds(args.pred), gold_counts, gold_total,
                        {"global_block_threshold": 0.0,
                         "per_type": {t: {"block_threshold": 0.0}
                                      for t in gold_counts}},
                        "block")  # baseline at thr=0 (no filter)

    report = {
        "thresholds_source": args.thresholds,
        "test_set": args.gold,
        "raw_no_threshold": raw,
        "block_mode": block,
        "review_mode": review,
    }
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)

    md = []
    md.append("# Dev-derived thresholds applied to external_1000\n")
    md.append(f"Thresholds from: `{os.path.basename(args.thresholds)}`")
    md.append(f"Test set: `{os.path.basename(args.gold)}`  ({gold_total} gold spans)\n")
    md.append("## Global metrics by mode\n")
    md.append("| mode | thr (global) | tp | fp | fn | P | R | F1 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for label, key, mode in [("RAW (no thr)", None, raw), ("BLOCK", "block", block), ("REVIEW", "review", review)]:
        thr = thresholds.get(f"global_{key}_threshold") if key else 0.0
        g = mode["global"]
        thr_str = f"{thr:.4f}" if thr is not None else "—"
        md.append(f"| {label} | {thr_str} | {g['tp']} | {g['fp']} | {g['fn']} | "
                  f"{g['P']:.3f} | {g['R']:.3f} | {g['F1']:.3f} |")

    # Per-type comparison: focus types
    focus = ["AU_TFN","AU_PASSPORT","AU_DRIVERS_LICENCE","MEDICARE_NUMBER",
             "PAYMENT_CARD_NUMBER","CREDIT_CARD_CVV","BSB","AU_BANK_ACCOUNT",
             "PERSON","PHONE","EMAIL","ADDRESS","DATE_OF_BIRTH","NEXT_OF_KIN"]
    md.append(f"\n## High-importance types: BLOCK vs REVIEW\n")
    md.append("| type | n_gold | block_thr | block_P | block_R | block_F1 | review_thr | review_P | review_R | review_F1 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    pt_thr = thresholds.get("per_type", {})
    for t in focus:
        n_g = gold_counts.get(t, 0)
        b = block["per_type"].get(t, {"P":0,"R":0,"F1":0})
        r = review["per_type"].get(t, {"P":0,"R":0,"F1":0})
        bt = pt_thr.get(t, {}).get("block_threshold")
        rt = pt_thr.get(t, {}).get("review_threshold")
        bt_str = f"{bt:.4f}" if bt is not None else "—"
        rt_str = f"{rt:.4f}" if rt is not None else "—"
        md.append(f"| {t} | {n_g} | {bt_str} | {b['P']:.3f} | {b['R']:.3f} | {b['F1']:.3f} | "
                  f"{rt_str} | {r['P']:.3f} | {r['R']:.3f} | {r['F1']:.3f} |")

    with open(args.md_out, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"wrote {args.report_out}")
    print(f"wrote {args.md_out}")
    print()
    print(f"=== external_1000, dev-derived thresholds applied ===")
    print(f"  RAW    (no thr): P={raw['global']['P']:.3f}  R={raw['global']['R']:.3f}  F1={raw['global']['F1']:.3f}")
    print(f"  BLOCK  (P-first): P={block['global']['P']:.3f}  R={block['global']['R']:.3f}  F1={block['global']['F1']:.3f}")
    print(f"  REVIEW (R-first): P={review['global']['P']:.3f}  R={review['global']['R']:.3f}  F1={review['global']['F1']:.3f}")


if __name__ == "__main__":
    main()

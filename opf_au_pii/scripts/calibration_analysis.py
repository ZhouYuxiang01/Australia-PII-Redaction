"""Calibration + threshold-search for OPF v3 confidence scores.

Reads predictions_with_confidence.jsonl and gold spans, outputs:
  - calibration_report.json: overall + per-type ECE, reliability bins
  - thresholds.json: per-type block (P >= block_floor) and review (R >= review_floor)
                     thresholds. Falls back to global thresholds for sparse types.
  - calibration_summary.md: readable summary.
"""
import argparse, json, math, os, sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

DEFAULT_PRED = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v3_full/external_1000_with_logprobs/predictions_with_confidence.jsonl"
DEFAULT_GOLD = "/home/admin/ZYX/Qwen3.5_4b_base_Full_73class/data/eval_external_1000/positive_1000.jsonl"
DEFAULT_REPORT = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v3_full/external_1000_with_logprobs/calibration_report.json"
DEFAULT_THRESH = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v3_full/external_1000_with_logprobs/thresholds.json"
DEFAULT_MD = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v3_full/external_1000_with_logprobs/calibration_summary.md"

# Default operating-point floors come from taxonomy_v1.1.1.yaml policy_modes.
BLOCK_PRECISION_FLOOR  = 0.98
REVIEW_RECALL_FLOOR    = 0.95
N_BINS = 10
MIN_TYPE_POSITIVES = 30  # types with fewer gold positives use the global threshold


def load_predictions(p):
    out = []
    with open(p) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def load_gold(p):
    """Return list of (text, list_of_gold_spans)."""
    rows = []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            spans = []
            for k, ranges in r.get("spans", {}).items():
                if ":" not in k: continue
                t, v = k.split(":", 1)
                t = t.strip(); v = v.strip()
                for s, e in ranges:
                    spans.append({"type": t, "value": v, "start": int(s), "end": int(e)})
            rows.append((r["text"], spans))
    return rows


def reliability_bins(items, n_bins=N_BINS):
    """items: list of (confidence, is_tp). Returns bin info + ECE."""
    bins = [[] for _ in range(n_bins)]
    for conf, is_tp in items:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, is_tp))
    out = []
    total = max(len(items), 1)
    ece = 0.0
    for i, b in enumerate(bins):
        if not b:
            out.append({"bin_lo": i / n_bins, "bin_hi": (i + 1) / n_bins,
                        "n": 0, "mean_conf": None, "accuracy": None, "gap": None})
            continue
        n = len(b)
        mean_conf = sum(c for c, _ in b) / n
        acc = sum(1 for _, t in b if t) / n
        gap = abs(mean_conf - acc)
        ece += (n / total) * gap
        out.append({"bin_lo": i / n_bins, "bin_hi": (i + 1) / n_bins,
                    "n": n, "mean_conf": mean_conf, "accuracy": acc, "gap": gap})
    return out, ece


def threshold_sweep(preds_with_label, total_positives):
    """Return list of (thr, P, R, F1, n_tp, n_fp). preds_with_label = list of (conf, is_tp)."""
    sorted_preds = sorted(preds_with_label, key=lambda x: -x[0])  # high conf first
    n_tp = 0; n_fp = 0
    out = []
    # Add point at threshold = max+epsilon (zero predictions)
    out.append({"thr": 1.000001, "tp": 0, "fp": 0, "fn": total_positives,
                "P": 1.0 if total_positives == 0 else 0.0,
                "R": 0.0, "F1": 0.0})
    last_conf = None
    for conf, is_tp in sorted_preds:
        if is_tp:
            n_tp += 1
        else:
            n_fp += 1
        # Only emit a point at unique confidence values (last item per group)
        if last_conf is None or conf < last_conf:
            P = n_tp / max(n_tp + n_fp, 1)
            R = n_tp / max(total_positives, 1)
            F1 = 2 * P * R / (P + R) if (P + R) else 0.0
            out.append({"thr": conf, "tp": n_tp, "fp": n_fp,
                        "fn": total_positives - n_tp, "P": P, "R": R, "F1": F1})
            last_conf = conf
    return out


def find_thresholds(sweep, p_floor, r_floor):
    """Find:
      - block threshold: highest threshold where R is maximized subject to P>=p_floor
      - review threshold: lowest threshold where P is maximized subject to R>=r_floor
    Returns (block_thr, block_P, block_R, review_thr, review_P, review_R) or None for unreachable.
    """
    # Block: precision-first
    block = None
    for pt in sweep:
        if pt["P"] >= p_floor:
            # Pick the one with max recall among precision-passing points
            if block is None or pt["R"] > block["R"]:
                block = pt
    # Review: recall-first
    review = None
    for pt in sweep:
        if pt["R"] >= r_floor:
            if review is None or pt["P"] > review["P"]:
                review = pt
    return block, review


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=DEFAULT_PRED)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--report_out", default=DEFAULT_REPORT)
    ap.add_argument("--thresh_out", default=DEFAULT_THRESH)
    ap.add_argument("--md_out", default=DEFAULT_MD)
    ap.add_argument("--block_precision_floor", type=float, default=BLOCK_PRECISION_FLOOR)
    ap.add_argument("--review_recall_floor", type=float, default=REVIEW_RECALL_FLOOR)
    args = ap.parse_args()

    preds_rows = load_predictions(args.pred)
    gold_rows  = load_gold(args.gold)
    gold_by_text = {t: spans for t, spans in gold_rows}

    # Aggregate per-type and global lists of (conf, is_tp).
    by_type_pred = defaultdict(list)   # type -> [(conf, is_tp), ...]
    by_type_gold_count = Counter()     # type -> number of gold positives (for recall denom)
    global_pred = []
    global_gold = 0

    for r in preds_rows:
        text = r["text"]
        gold = gold_by_text.get(text, [])
        # Count gold positives by type
        for g in gold:
            by_type_gold_count[g["type"]] += 1
            global_gold += 1
        # Predictions with TP/FP labels (already classified)
        for p in r["preds"]:
            is_tp = (p["match"] == "tp")
            by_type_pred[p["type"]].append((p["confidence"], is_tp))
            global_pred.append((p["confidence"], is_tp))

    # ----- global calibration -----
    global_bins, global_ece = reliability_bins(global_pred)
    global_sweep = threshold_sweep(global_pred, global_gold)
    g_block, g_review = find_thresholds(global_sweep, args.block_precision_floor,
                                        args.review_recall_floor)

    # ----- per-type -----
    per_type = {}
    for t, items in by_type_pred.items():
        n_pred = len(items)
        n_gold = by_type_gold_count.get(t, 0)
        bins, ece = reliability_bins(items)
        sweep = threshold_sweep(items, n_gold)
        block, review = find_thresholds(sweep,
                                        args.block_precision_floor,
                                        args.review_recall_floor)
        per_type[t] = {
            "n_pred": n_pred,
            "n_gold": n_gold,
            "ece": ece,
            "bins": bins,
            "block": block,
            "review": review,
        }

    # ----- write report -----
    report = {
        "config": {
            "block_precision_floor": args.block_precision_floor,
            "review_recall_floor": args.review_recall_floor,
            "n_bins": N_BINS,
            "min_type_positives": MIN_TYPE_POSITIVES,
        },
        "global": {
            "n_pred": len(global_pred),
            "n_gold": global_gold,
            "ece": global_ece,
            "bins": global_bins,
            "block": g_block,
            "review": g_review,
        },
        "per_type": per_type,
    }
    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(report, f, indent=2)

    # ----- write thresholds -----
    thresholds = {
        "version": "v3-calibration-1",
        "global_block_threshold": (g_block or {}).get("thr"),
        "global_review_threshold": (g_review or {}).get("thr"),
        "global_block_P": (g_block or {}).get("P"),
        "global_block_R": (g_block or {}).get("R"),
        "global_review_P": (g_review or {}).get("P"),
        "global_review_R": (g_review or {}).get("R"),
        "block_precision_floor": args.block_precision_floor,
        "review_recall_floor": args.review_recall_floor,
        "per_type": {},
    }
    for t, info in per_type.items():
        b = info["block"]; r = info["review"]
        if info["n_gold"] >= MIN_TYPE_POSITIVES and b is not None:
            block_thr = b["thr"]
        else:
            block_thr = thresholds["global_block_threshold"]
        if info["n_gold"] >= MIN_TYPE_POSITIVES and r is not None:
            review_thr = r["thr"]
        else:
            review_thr = thresholds["global_review_threshold"]
        thresholds["per_type"][t] = {
            "block_threshold": block_thr,
            "review_threshold": review_thr,
            "block_P": (b or {}).get("P"),
            "block_R": (b or {}).get("R"),
            "review_P": (r or {}).get("P"),
            "review_R": (r or {}).get("R"),
            "ece": info["ece"],
            "n_gold": info["n_gold"],
            "n_pred": info["n_pred"],
            "uses_global_fallback": info["n_gold"] < MIN_TYPE_POSITIVES,
        }
    with open(args.thresh_out, "w") as f:
        json.dump(thresholds, f, indent=2)

    # ----- write markdown summary -----
    md = []
    md.append("# OPF v3 confidence calibration\n")
    md.append(f"Eval set: external_1000 positive (1,000 docs / {global_gold} gold spans)\n")
    md.append(f"\n## Global metrics\n")
    md.append(f"- ECE: **{global_ece:.4f}**")
    md.append(f"- Predictions: {len(global_pred)}")
    md.append(f"- Block (precision >= {args.block_precision_floor}):  ")
    if g_block:
        md.append(f"  thr=**{g_block['thr']:.4f}**  P={g_block['P']:.4f}  R={g_block['R']:.4f}  F1={g_block['F1']:.4f}  (tp={g_block['tp']}, fp={g_block['fp']})")
    else:
        md.append("  unreachable at any threshold")
    md.append(f"- Review (recall >= {args.review_recall_floor}):  ")
    if g_review:
        md.append(f"  thr=**{g_review['thr']:.4f}**  P={g_review['P']:.4f}  R={g_review['R']:.4f}  F1={g_review['F1']:.4f}  (tp={g_review['tp']}, fp={g_review['fp']})")
    else:
        md.append("  unreachable at any threshold")

    md.append(f"\n## Reliability bins (global)\n")
    md.append("| bin | n | mean_conf | accuracy | gap |")
    md.append("|---|---|---|---|---|")
    for b in global_bins:
        if b["n"] == 0:
            md.append(f"| {b['bin_lo']:.1f}–{b['bin_hi']:.1f} | 0 | – | – | – |")
        else:
            md.append(f"| {b['bin_lo']:.1f}–{b['bin_hi']:.1f} | {b['n']} | {b['mean_conf']:.3f} | {b['accuracy']:.3f} | {b['gap']:.3f} |")

    # Per-type table — show types with enough data
    md.append(f"\n## Per-type thresholds (only types with >= {MIN_TYPE_POSITIVES} gold positives)\n")
    md.append("| type | n_gold | n_pred | ECE | block_thr | block_P | block_R | review_thr | review_P | review_R |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    typed = sorted(per_type.items(), key=lambda kv: -kv[1]["n_gold"])
    for t, info in typed:
        if info["n_gold"] < MIN_TYPE_POSITIVES:
            continue
        b = info["block"]; r = info["review"]
        b_thr = f"{b['thr']:.4f}" if b else "N/A"
        b_p   = f"{b['P']:.3f}" if b else "—"
        b_r   = f"{b['R']:.3f}" if b else "—"
        r_thr = f"{r['thr']:.4f}" if r else "N/A"
        r_p   = f"{r['P']:.3f}" if r else "—"
        r_r   = f"{r['R']:.3f}" if r else "—"
        md.append(
            f"| {t} | {info['n_gold']} | {info['n_pred']} | {info['ece']:.3f} | "
            f"{b_thr} | {b_p} | {b_r} | {r_thr} | {r_p} | {r_r} |"
        )

    md.append(f"\n## Sparse types (use global fallback thresholds; <{MIN_TYPE_POSITIVES} gold positives)\n")
    sparse = [(t, info) for t, info in typed if info["n_gold"] < MIN_TYPE_POSITIVES]
    if sparse:
        md.append("| type | n_gold | n_pred | ECE |")
        md.append("|---|---|---|---|")
        for t, info in sparse:
            md.append(f"| {t} | {info['n_gold']} | {info['n_pred']} | {info['ece']:.3f} |")

    with open(args.md_out, "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"wrote {args.report_out}")
    print(f"wrote {args.thresh_out}")
    print(f"wrote {args.md_out}")
    print()
    print(f"=== global ===")
    print(f"  ECE = {global_ece:.4f}")
    if g_block:
        print(f"  BLOCK  thr={g_block['thr']:.4f}  P={g_block['P']:.4f}  R={g_block['R']:.4f}  F1={g_block['F1']:.4f}")
    else:
        print(f"  BLOCK  unreachable at P>={args.block_precision_floor}")
    if g_review:
        print(f"  REVIEW thr={g_review['thr']:.4f}  P={g_review['P']:.4f}  R={g_review['R']:.4f}  F1={g_review['F1']:.4f}")
    else:
        print(f"  REVIEW unreachable at R>={args.review_recall_floor}")


if __name__ == "__main__":
    main()

"""Sweep per-label top1_prob thresholds on dev predictions, output per_label_thresholds.json.

Inputs:
  reports/dev_predictions_for_calibration.jsonl  (from dump_dev_predictions.py)

Outputs:
  configs/postprocess/per_label_thresholds.json
  reports/per_label_calibration_report.md
  reports/per_label_calibration_curves.json  (full sweep data for inspection)

Method:
  For each label L, all predictions with predicted_type=L form the candidate pool.
  We sweep a threshold τ on top1_prob; predictions with top1_prob >= τ are kept,
  others would be dropped. Compute label-level F1 at each τ:
      TP(τ) = # spans (predicted_type=L, is_correct=True, top1>=τ)
      FP(τ) = # spans (predicted_type=L, is_correct=False, top1>=τ)
      FN(τ) = (TP_max - TP(τ)) + missed_gold[L]
      F1(τ) = 2 P R / (P + R)
  missed_gold[L] is constant (gold-of-type-L not covered by any prediction-of-type-L)
  and acts as a recall floor.

  We pick the τ maximizing F1. To avoid overfitting tiny labels, we also enforce:
    * minimum sample size: only calibrate L if num positives >= MIN_POSITIVES
    * thresholds bounded to [0.05, 0.85] in steps of 0.01
    * if calibrated F1 is not at least DEFAULT_F1 + 0.005, fall back to default

Defaults (from hybrid_opf_qwen.py):
  MIN_TOP1_PII = 0.20  → no per-label entry means use this default
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TOP1_THRESHOLD = 0.20
SWEEP_LO = 0.05
SWEEP_HI = 0.85
SWEEP_STEP = 0.01
MIN_POSITIVES = 10  # don't calibrate labels with too few correct predictions
MIN_F1_GAIN = 0.005  # only override default if we beat it by this much


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def sweep_label(
    pos_top1s: list[float],
    neg_top1s: list[float],
    missed: int,
) -> tuple[float, dict[str, float], list[dict[str, float]]]:
    """Return (best_threshold, best_metrics, full_curve)."""
    pos_sorted = sorted(pos_top1s, reverse=True)
    neg_sorted = sorted(neg_top1s, reverse=True)
    tp_total = len(pos_top1s)

    curve: list[dict[str, float]] = []
    best_f1 = -1.0
    best_threshold = DEFAULT_TOP1_THRESHOLD
    best_metrics: dict[str, float] = {}

    tau = SWEEP_LO
    while tau <= SWEEP_HI + 1e-9:
        tp = sum(1 for x in pos_top1s if x >= tau)
        fp = sum(1 for x in neg_top1s if x >= tau)
        fn = (tp_total - tp) + missed
        p, r, f = f1_from_counts(tp, fp, fn)
        curve.append({"threshold": round(tau, 4), "tp": tp, "fp": fp, "fn": fn,
                       "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)})
        if f > best_f1 + 1e-12:
            best_f1 = f
            best_threshold = round(tau, 4)
            best_metrics = {"precision": round(p, 4), "recall": round(r, 4),
                            "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}
        tau += SWEEP_STEP

    return best_threshold, best_metrics, curve


def evaluate_at_default(
    pos_top1s: list[float], neg_top1s: list[float], missed: int,
) -> dict[str, float]:
    tp = sum(1 for x in pos_top1s if x >= DEFAULT_TOP1_THRESHOLD)
    fp = sum(1 for x in neg_top1s if x >= DEFAULT_TOP1_THRESHOLD)
    fn = (len(pos_top1s) - tp) + missed
    p, r, f = f1_from_counts(tp, fp, fn)
    return {"threshold": DEFAULT_TOP1_THRESHOLD, "precision": round(p, 4),
            "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions",
                    default=str(REPO_ROOT / "reports" / "dev_predictions_for_calibration.jsonl"))
    ap.add_argument("--out-thresholds",
                    default=str(REPO_ROOT / "configs" / "postprocess" / "per_label_thresholds.json"))
    ap.add_argument("--out-report",
                    default=str(REPO_ROOT / "reports" / "per_label_calibration_report.md"))
    ap.add_argument("--out-curves",
                    default=str(REPO_ROOT / "reports" / "per_label_calibration_curves.json"))
    args = ap.parse_args()

    pos_by_label: dict[str, list[float]] = defaultdict(list)
    neg_by_label: dict[str, list[float]] = defaultdict(list)
    missed_by_label: Counter[str] = Counter()
    total_lines = 0
    pred_lines = 0
    miss_lines = 0

    for line in Path(args.predictions).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total_lines += 1
        rec = json.loads(line)
        if rec.get("miss"):
            label = str(rec.get("gold_type", ""))
            if label:
                missed_by_label[label] += 1
            miss_lines += 1
            continue
        pred_lines += 1
        label = str(rec.get("predicted_type", ""))
        if not label:
            continue
        top1 = float(rec.get("top1_prob") or 0.0)
        if rec.get("is_correct"):
            pos_by_label[label].append(top1)
        else:
            neg_by_label[label].append(top1)

    print(f"[calib] {total_lines} lines  pred={pred_lines}  miss={miss_lines}")
    print(f"[calib] {len(pos_by_label)} labels with positive predictions")

    thresholds: dict[str, float] = {}
    report_rows: list[dict[str, Any]] = []
    curves: dict[str, list[dict[str, float]]] = {}

    all_labels = sorted(set(pos_by_label) | set(neg_by_label) | set(missed_by_label))
    for label in all_labels:
        pos = pos_by_label.get(label, [])
        neg = neg_by_label.get(label, [])
        missed = missed_by_label.get(label, 0)
        n_pos, n_neg = len(pos), len(neg)

        default_metrics = evaluate_at_default(pos, neg, missed)
        if n_pos < MIN_POSITIVES:
            row = {
                "label": label,
                "n_pos": n_pos, "n_neg": n_neg, "missed": missed,
                "calibrated_threshold": None,
                "default_metrics": default_metrics,
                "best_metrics": default_metrics,
                "f1_gain": 0.0,
                "decision": "skip_low_sample",
            }
            report_rows.append(row)
            continue

        best_tau, best_metrics, curve = sweep_label(pos, neg, missed)
        f1_gain = round(best_metrics["f1"] - default_metrics["f1"], 4)
        if f1_gain >= MIN_F1_GAIN:
            thresholds[label] = best_tau
            decision = "override"
        else:
            decision = f"keep_default_(gain={f1_gain:+.4f})"
        report_rows.append({
            "label": label,
            "n_pos": n_pos, "n_neg": n_neg, "missed": missed,
            "calibrated_threshold": best_tau,
            "default_metrics": default_metrics,
            "best_metrics": best_metrics,
            "f1_gain": f1_gain,
            "decision": decision,
        })
        curves[label] = curve

    Path(args.out_thresholds).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "default_top1_threshold": DEFAULT_TOP1_THRESHOLD,
            "sweep_range": [SWEEP_LO, SWEEP_HI],
            "sweep_step": SWEEP_STEP,
            "min_positives": MIN_POSITIVES,
            "min_f1_gain": MIN_F1_GAIN,
            "labels_overridden": len(thresholds),
            "labels_total": len(all_labels),
        },
        "top1_prob_min": thresholds,
    }
    Path(args.out_thresholds).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[calib] wrote {len(thresholds)} per-label overrides → {args.out_thresholds}")

    Path(args.out_curves).write_text(json.dumps(curves, indent=2))

    lines = ["# Per-Label Top1 Threshold Calibration", "",
             f"Total labels: {len(all_labels)}  |  Overridden: {len(thresholds)}",
             f"Default threshold: {DEFAULT_TOP1_THRESHOLD}  |  "
             f"Min F1 gain to override: {MIN_F1_GAIN}", ""]
    lines.append("| Label | N_pos | N_neg | Missed | Default F1 | Best τ | Best F1 | Gain | Decision |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in sorted(report_rows, key=lambda r: -r.get("f1_gain", 0)):
        lines.append(
            "| {label} | {n_pos} | {n_neg} | {missed} | {def_f1} | {tau} | {best_f1} | {gain:+.4f} | {dec} |"
            .format(
                label=row["label"], n_pos=row["n_pos"], n_neg=row["n_neg"], missed=row["missed"],
                def_f1=row["default_metrics"]["f1"],
                tau=row["calibrated_threshold"] if row["calibrated_threshold"] is not None else "—",
                best_f1=row["best_metrics"]["f1"],
                gain=row.get("f1_gain", 0.0),
                dec=row["decision"],
            )
        )
    Path(args.out_report).write_text("\n".join(lines))
    print(f"[calib] wrote report → {args.out_report}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

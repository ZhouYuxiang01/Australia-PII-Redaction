"""End-to-end offline evaluation of the redaction wrapper on the OPF test set.

Loads the hybrid backend through the wrapper (env-substituted config), iterates
the full ``opf_test_opf_format.jsonl`` dataset, and computes the metrics the
brief calls for:

  * span-level P / R / F1 (exact match + partial overlap)
  * type accuracy on overlapping spans
  * per-label P / R / F1
  * decision distribution (AUTO_REDACT / REVIEW / PASS)
  * over-redaction cost (chars masked that are not gold PII)
  * under-redaction cost (gold PII chars not masked, weighted)
  * latency p50 / p95 / mean (per text)

The OPF dataset uses raw model labels (EMAIL_ADDRESS, FIRST_NAME, MOBILE...);
the wrapper emits canonical schema labels after alias normalisation. We apply
the same alias map to the gold labels before comparison so the two sides agree.

Usage::

    REDACTION_PII_PROJECT_ROOT=/path/to/pii_training_prep_v3_2 \
    REDACTION_QWEN_BACKBONE=/path/to/Qwen3.5-9B-Base \
    /home/admin/miniconda3/envs/opf/bin/python scripts/run_hybrid_offline_eval.py \
        --backend configs/backends/hybrid-opf-qwen.json \
        --policy configs/policies/hybrid-80class-v1.json \
        --test-set $REDACTION_PII_PROJECT_ROOT/data/train/opf_test_opf_format.jsonl \
        --out-dir reports/ \
        --limit 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from redaction.backends import build_backend_from_path  # noqa: E402
from redaction.core import (  # noqa: E402
    apply_policy,
    build_response,
    load_json,
    normalize_text,
    safe_postprocess_spans,
)
from redaction.core.span import Span  # noqa: E402


HIGH_RISK_LABELS = {"AU_TFN", "AU_PASSPORT", "AU_DRIVERS_LICENCE",
                    "MEDICARE_NUMBER", "PAYMENT_CARD_NUMBER", "CREDIT_CARD_CVV",
                    "AU_BANK_ACCOUNT", "BSB", "PASSPORT_NUMBER",
                    "DRIVERS_LICENCE", "BANK_ACCOUNT_NUMBER", "TFN"}


def load_alias_map(repo_root: Path) -> dict[str, str]:
    reg = json.loads(
        (repo_root / "configs" / "postprocess" / "postprocess_rule_registry.json")
        .read_text(encoding="utf-8")
    )
    return dict(reg.get("label_alias_normalization", {}))


def parse_gold(rec: dict[str, Any], alias: dict[str, str]) -> list[tuple[int, int, str]]:
    """Return list of (start, end, canonical_label) from a test record."""
    spans_raw = rec.get("spans", {})
    out: list[tuple[int, int, str]] = []
    if isinstance(spans_raw, dict):
        for raw_label, positions in spans_raw.items():
            label = alias.get(raw_label, raw_label)
            for pos in positions:
                if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                    out.append((int(pos[0]), int(pos[1]), label))
    elif isinstance(spans_raw, list):
        for s in spans_raw:
            label = alias.get(s.get("label", ""), s.get("label", ""))
            out.append((int(s["start"]), int(s["end"]), label))
    return out


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def overlap_chars(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def f_score(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, type=Path)
    ap.add_argument("--policy", required=True, type=Path)
    ap.add_argument("--test-set", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap test examples (0 = all)")
    ap.add_argument("--max-text-chars", type=int, default=8000,
                    help="Skip examples longer than this to keep inference bounded")
    ap.add_argument("--report-name", default="wrapper_hybrid_full_eval",
                    help="Filename stem for the JSON report")
    args = ap.parse_args()

    alias = load_alias_map(REPO_ROOT)

    print(f"[init] backend config: {args.backend}", flush=True)
    print(f"[init] policy config:  {args.policy}", flush=True)
    backend = build_backend_from_path(args.backend)
    policy = load_json(args.policy)
    print(f"[init] loading backend ({backend.name}, {backend.model_version})", flush=True)
    t0 = time.time()
    backend.load()
    print(f"[init] backend ready in {time.time() - t0:.1f}s", flush=True)

    records = []
    for line in args.test_set.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[init] loaded {len(records)} test records", flush=True)

    # Counters for span-level metrics
    exact_tp = exact_fp = exact_fn = 0
    overlap_tp = overlap_fp = overlap_fn = 0
    per_label_tp: Counter = Counter()
    per_label_fp: Counter = Counter()
    per_label_fn: Counter = Counter()

    # Type accuracy on overlapping pred/gold
    type_correct = type_total = 0

    decision_counts: Counter = Counter()
    over_redaction_chars = 0
    under_redaction_chars = 0
    under_redaction_chars_high_risk = 0
    total_pred_redacted_chars = 0
    total_gold_chars = 0
    total_gold_chars_high_risk = 0

    latencies: list[float] = []
    skipped_long = 0
    errors = 0
    error_examples: list[dict[str, Any]] = []

    t_run = time.time()
    for idx, rec in enumerate(records):
        text_raw = rec.get("text", "")
        if len(text_raw) > args.max_text_chars:
            skipped_long += 1
            continue

        gold = parse_gold(rec, alias)

        try:
            text = normalize_text(text_raw)
            t1 = time.time()
            spans, diag = backend.detect_spans(text)
            spans, post_warn = safe_postprocess_spans(text, spans, policy)
            spans = apply_policy(spans, policy)
            payload = build_response(
                text=text, spans=spans, policy=policy,
                raw_offset_mapping_applied=diag.get("raw_offset_mapping_applied", False),
                warnings=[*diag.get("warnings", []), *post_warn],
            )
            latencies.append(time.time() - t1)
        except Exception as e:
            errors += 1
            if len(error_examples) < 20:
                error_examples.append({"id": rec.get("id"), "error": str(e)[:300]})
            continue

        pred_spans_redact = [
            (s["start"], s["end"], s["type"]) for s in payload["spans"]
            if s.get("decision") in ("AUTO_REDACT", "redact")
        ]
        pred_all = [(s["start"], s["end"], s["type"]) for s in payload["spans"]]

        for s in payload["spans"]:
            decision_counts[s.get("decision", "UNKNOWN")] += 1

        # Exact span+type match
        gold_set = set(gold)
        pred_set = set(pred_spans_redact)
        exact_tp += len(gold_set & pred_set)
        exact_fp += len(pred_set - gold_set)
        exact_fn += len(gold_set - pred_set)

        # Overlap-based detection match (label-agnostic)
        pred_used = [False] * len(pred_spans_redact)
        gold_used = [False] * len(gold)
        for gi, g in enumerate(gold):
            for pi, p in enumerate(pred_spans_redact):
                if pred_used[pi]:
                    continue
                if overlaps((g[0], g[1]), (p[0], p[1])):
                    overlap_tp += 1
                    pred_used[pi] = True
                    gold_used[gi] = True
                    type_total += 1
                    if g[2] == p[2]:
                        type_correct += 1
                    if g[2] == p[2]:
                        per_label_tp[g[2]] += 1
                    else:
                        per_label_fp[p[2]] += 1
                        per_label_fn[g[2]] += 1
                    break
            if not gold_used[gi]:
                overlap_fn += 1
                per_label_fn[g[2]] += 1
        for pi, used in enumerate(pred_used):
            if not used:
                overlap_fp += 1
                per_label_fp[pred_spans_redact[pi][2]] += 1

        # Cost: char-level redaction
        for s in pred_spans_redact:
            total_pred_redacted_chars += s[1] - s[0]
        for g in gold:
            total_gold_chars += g[1] - g[0]
            if g[2] in HIGH_RISK_LABELS:
                total_gold_chars_high_risk += g[1] - g[0]

        # Over-redaction: chars in pred not covered by any gold span
        # Under-redaction: chars in gold not covered by any pred (redact-decision) span
        for p in pred_spans_redact:
            covered = sum(overlap_chars((p[0], p[1]), (g[0], g[1])) for g in gold)
            over_redaction_chars += max(0, (p[1] - p[0]) - covered)
        for g in gold:
            covered = sum(overlap_chars((g[0], g[1]), (p[0], p[1])) for p in pred_spans_redact)
            missed = max(0, (g[1] - g[0]) - covered)
            under_redaction_chars += missed
            if g[2] in HIGH_RISK_LABELS:
                under_redaction_chars_high_risk += missed

        if (idx + 1) % 250 == 0:
            elapsed = time.time() - t_run
            rate = (idx + 1) / max(elapsed, 1e-6)
            eta = (len(records) - idx - 1) / max(rate, 1e-6)
            print(
                f"[run] {idx+1}/{len(records)} "
                f"elapsed={elapsed:.0f}s rate={rate:.2f} ex/s eta={eta:.0f}s "
                f"exact_tp={exact_tp} overlap_tp={overlap_tp} fp={overlap_fp} fn={overlap_fn}",
                flush=True,
            )

    n_evaluated = len(records) - skipped_long - errors
    elapsed_total = time.time() - t_run

    def safe_div(a: float, b: float) -> float:
        return a / b if b else 0.0

    exact_p = safe_div(exact_tp, exact_tp + exact_fp)
    exact_r = safe_div(exact_tp, exact_tp + exact_fn)
    overlap_p = safe_div(overlap_tp, overlap_tp + overlap_fp)
    overlap_r = safe_div(overlap_tp, overlap_tp + overlap_fn)

    per_label = {}
    all_labels = set(per_label_tp) | set(per_label_fp) | set(per_label_fn)
    for label in sorted(all_labels):
        tp = per_label_tp[label]
        fp = per_label_fp[label]
        fn = per_label_fn[label]
        p = safe_div(tp, tp + fp)
        r = safe_div(tp, tp + fn)
        per_label[label] = {
            "support": tp + fn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f_score(p, r), 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    report = {
        "task": "wrapper_hybrid_full_offline_eval",
        "backend_config": str(args.backend),
        "policy_config": str(args.policy),
        "policy_id": policy.get("policy_id"),
        "schema_version": policy.get("schema_version"),
        "test_set": str(args.test_set),
        "total_records": len(records),
        "evaluated": n_evaluated,
        "skipped_long": skipped_long,
        "errors": errors,
        "elapsed_seconds": round(elapsed_total, 2),
        "throughput_examples_per_sec": round(safe_div(n_evaluated, elapsed_total), 3),
        "metrics": {
            "exact": {
                "precision": round(exact_p, 4),
                "recall": round(exact_r, 4),
                "f1": round(f_score(exact_p, exact_r), 4),
                "tp": exact_tp, "fp": exact_fp, "fn": exact_fn,
            },
            "overlap": {
                "precision": round(overlap_p, 4),
                "recall": round(overlap_r, 4),
                "f1": round(f_score(overlap_p, overlap_r), 4),
                "f2": round(f_score(overlap_p, overlap_r, beta=2.0), 4),
                "tp": overlap_tp, "fp": overlap_fp, "fn": overlap_fn,
            },
            "type_accuracy_on_overlap": round(safe_div(type_correct, type_total), 4),
            "type_correct": type_correct,
            "type_total": type_total,
        },
        "per_label": per_label,
        "decision_distribution": dict(decision_counts),
        "redaction_cost": {
            "total_gold_chars": total_gold_chars,
            "total_pred_redacted_chars": total_pred_redacted_chars,
            "over_redaction_chars": over_redaction_chars,
            "under_redaction_chars": under_redaction_chars,
            "over_redaction_rate_vs_pred": round(safe_div(over_redaction_chars, total_pred_redacted_chars), 4),
            "under_redaction_rate_vs_gold": round(safe_div(under_redaction_chars, total_gold_chars), 4),
            "high_risk_under_redaction_chars": under_redaction_chars_high_risk,
            "high_risk_under_redaction_rate": round(
                safe_div(under_redaction_chars_high_risk, total_gold_chars_high_risk), 4),
        },
        "latency_seconds": {
            "count": len(latencies),
            "mean": round(sum(latencies) / max(len(latencies), 1), 4),
            "p50": round(percentile(latencies, 0.5), 4),
            "p95": round(percentile(latencies, 0.95), 4),
            "p99": round(percentile(latencies, 0.99), 4),
            "max": round(max(latencies) if latencies else 0.0, 4),
        },
        "error_examples": error_examples,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.report_name}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[done] wrote {out_path}", flush=True)
    print(f"[done] overlap P/R/F1 = "
          f"{report['metrics']['overlap']['precision']}/"
          f"{report['metrics']['overlap']['recall']}/"
          f"{report['metrics']['overlap']['f1']}", flush=True)
    print(f"[done] exact   P/R/F1 = "
          f"{report['metrics']['exact']['precision']}/"
          f"{report['metrics']['exact']['recall']}/"
          f"{report['metrics']['exact']['f1']}", flush=True)
    print(f"[done] type accuracy on overlap = {report['metrics']['type_accuracy_on_overlap']}", flush=True)
    print(f"[done] decisions = {report['decision_distribution']}", flush=True)
    print(f"[done] latency p50/p95 = {report['latency_seconds']['p50']}/{report['latency_seconds']['p95']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Coordinate-descent tuning for OPF Viterbi transition biases.

Runs the FULL hybrid pipeline (OPF -> Qwen head -> verifier -> postprocess -> policy)
on a dev sample, sweeping the 6 transition biases one at a time and accepting any
move that improves overlap F1 by at least MIN_GAIN.

Output:
  runs/opf_hard_79/viterbi_calibration.json     (winning calibration)
  reports/opf_bias_tuning_history.json          (full search log)
  reports/opf_bias_tuning_report.md             (human summary)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from redaction.backends.registry import build_backend_from_path  # noqa: E402
from redaction.core import (  # noqa: E402
    apply_policy, load_json, normalize_text, safe_postprocess_spans,
)
from redaction.core.postprocess import _label_alias_map  # noqa: E402

BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)

# Sweep values per dimension (interpreted as logit-space additions).
SWEEP_VALUES = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
MIN_GAIN = 0.001  # min F1 improvement to accept a coord-descent move


def parse_gold(rec: dict[str, Any], alias: dict[str, str]) -> list[tuple[int, int, str]]:
    """opf_*_opf_format.jsonl uses spans = {LABEL: [[start, end], ...]}."""
    out: list[tuple[int, int, str]] = []
    raw = rec.get("spans") or {}
    if isinstance(raw, dict):
        for label, pairs in raw.items():
            label = alias.get(str(label), str(label))
            if not label or label == "O":
                continue
            for pair in pairs or []:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                s, e = int(pair[0]), int(pair[1])
                if 0 <= s < e:
                    out.append((s, e, label))
    return out


def overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def write_calibration_file(biases: dict[str, float], path: str) -> None:
    payload = {"operating_points": {"default": {"biases": dict(biases)}}}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2))


class Evaluator:
    def __init__(self, backend, policy, alias, records, calib_path):
        self.backend = backend
        self.policy = policy
        self.alias = alias
        self.records = records
        self.calib_path = calib_path
        self.cache: dict[tuple, dict] = {}
        self.calls = 0

    def __call__(self, biases: dict[str, float]) -> dict:
        key = tuple(round(biases[k], 6) for k in BIAS_KEYS)
        if key in self.cache:
            return self.cache[key]
        write_calibration_file(biases, self.calib_path)
        self.backend._opf.set_viterbi_decoder(calibration_path=self.calib_path)

        tp_o = fp_o = fn_o = 0
        tp_x = fp_x = fn_x = 0
        n_visible = 0
        t0 = time.time()
        for rec in self.records:
            text_raw = rec.get("text", "")
            if not text_raw:
                continue
            text = normalize_text(text_raw)
            gold = parse_gold(rec, self.alias)

            spans, _ = self.backend.detect_spans(text)
            spans, _ = safe_postprocess_spans(text, spans, self.policy)
            spans = apply_policy(spans, self.policy)

            pred = []
            for sp in spans:
                if (sp.decision or "").lower() in ("ignore", "pass"):
                    continue
                t = self.alias.get(sp.type, sp.type)
                pred.append((sp.start, sp.end, t))
            n_visible += len(pred)

            gold_used = [False] * len(gold)
            pred_used = [False] * len(pred)
            for pi, (ps, pe, pt) in enumerate(pred):
                for gi, (gs, ge, gt) in enumerate(gold):
                    if gold_used[gi] or pred_used[pi]:
                        continue
                    if overlap((ps, pe), (gs, ge)) and pt == gt:
                        tp_o += 1
                        gold_used[gi] = True
                        pred_used[pi] = True
                        if (ps, pe) == (gs, ge):
                            tp_x += 1
                        break
            for pi, used in enumerate(pred_used):
                if not used:
                    fp_o += 1
                    fp_x += 1
            for gi, used in enumerate(gold_used):
                if not used:
                    fn_o += 1
                    fn_x += 1
            for pi, (ps, pe, pt) in enumerate(pred):
                if not pred_used[pi]:
                    continue
                # exact-match credit only if exact offsets matched
                pass

        def _f1(tp, fp, fn):
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return 2 * p * r / (p + r) if (p + r) > 0 else 0.0, p, r

        f1_o, p_o, r_o = _f1(tp_o, fp_o, fn_o)
        f1_x, p_x, r_x = _f1(tp_x, fp_x, fn_x)
        elapsed = time.time() - t0
        result = {
            "biases": dict(biases),
            "overlap_f1": round(f1_o, 4),
            "overlap_p": round(p_o, 4),
            "overlap_r": round(r_o, 4),
            "exact_f1": round(f1_x, 4),
            "tp_o": tp_o, "fp_o": fp_o, "fn_o": fn_o,
            "n_predictions_visible": n_visible,
            "elapsed_s": round(elapsed, 1),
        }
        self.cache[key] = result
        self.calls += 1
        return result


def coordinate_descent(eval_fn: Evaluator, init: dict[str, float], rounds: int = 2) -> tuple[dict, list]:
    """Sweep one coord at a time; accept any move with gain >= MIN_GAIN."""
    history = []
    current = dict(init)
    base = eval_fn(current)
    print(f"[init] biases={current} F1={base['overlap_f1']:.4f}", flush=True)
    history.append({"phase": "init", **base})
    best_f1 = base["overlap_f1"]

    for r in range(1, rounds + 1):
        improved_this_round = False
        for k in BIAS_KEYS:
            best_v = current[k]
            best_local_f1 = best_f1
            best_local_metrics = None
            for v in SWEEP_VALUES:
                if abs(v - current[k]) < 1e-9:
                    continue
                cand = dict(current); cand[k] = v
                m = eval_fn(cand)
                history.append({"phase": f"r{r}.{k}", **m})
                if m["overlap_f1"] >= best_local_f1 + MIN_GAIN:
                    best_local_f1 = m["overlap_f1"]
                    best_v = v
                    best_local_metrics = m
            if best_v != current[k]:
                print(
                    f"[r{r} {k}] {current[k]:+.2f} -> {best_v:+.2f}  F1: {best_f1:.4f} -> {best_local_f1:.4f}",
                    flush=True,
                )
                current[k] = best_v
                best_f1 = best_local_f1
                improved_this_round = True
        if not improved_this_round:
            print(f"[r{r}] no improvement, stopping early", flush=True)
            break

    return current, history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend",
        default=str(REPO_ROOT / "configs" / "backends" / "hybrid-opf-qwen4b-calibrated.json"),
    )
    ap.add_argument(
        "--policy",
        default=str(REPO_ROOT / "configs" / "policies" / "hybrid-80class-v2-4b.json"),
    )
    ap.add_argument(
        "--dev",
        default="/home/admin/ZYX/pii_training_prep_v3_2/data/train/opf_dev_opf_format.jsonl",
    )
    ap.add_argument("--limit", type=int, default=300, help="dev subsample size for tuning")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument(
        "--output-calibration",
        default="/home/admin/ZYX/pii_training_prep_v3_2/runs/opf_hard_79/viterbi_calibration.json",
        help="final calibration file path; OPF auto-discovers it from checkpoint dir",
    )
    ap.add_argument(
        "--report-history",
        default=str(REPO_ROOT / "reports" / "opf_bias_tuning_history.json"),
    )
    ap.add_argument(
        "--report-summary",
        default=str(REPO_ROOT / "reports" / "opf_bias_tuning_report.md"),
    )
    args = ap.parse_args()

    print(f"[init] backend={args.backend}", flush=True)
    print(f"[init] policy ={args.policy}", flush=True)
    print(f"[init] dev    ={args.dev} limit={args.limit}", flush=True)

    backend = build_backend_from_path(args.backend)
    backend.load()
    policy = load_json(args.policy)
    alias = _label_alias_map()

    records: list[dict[str, Any]] = []
    for line in Path(args.dev).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[init] {len(records)} dev records loaded", flush=True)

    tmp_calib = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    eval_fn = Evaluator(backend, policy, alias, records, tmp_calib)

    init_biases = {k: 0.0 for k in BIAS_KEYS}
    print("[init] starting coord-descent...", flush=True)
    started = time.time()
    best, history = coordinate_descent(eval_fn, init_biases, rounds=args.rounds)
    elapsed = time.time() - started

    final_metrics = eval_fn(best)
    print(f"\n[done] best F1 = {final_metrics['overlap_f1']:.4f}  ({elapsed:.0f}s, {eval_fn.calls} evals)", flush=True)
    print(f"[done] best biases:", flush=True)
    for k, v in best.items():
        print(f"   {k}: {v:+.2f}", flush=True)

    Path(args.output_calibration).parent.mkdir(parents=True, exist_ok=True)
    write_calibration_file(best, args.output_calibration)
    print(f"[done] wrote calibration -> {args.output_calibration}", flush=True)

    Path(args.report_history).write_text(json.dumps(history, indent=2))

    init_metrics = next(h for h in history if h.get("phase") == "init")
    lines = [
        "# OPF Viterbi Transition Bias Tuning",
        "",
        f"Dev sample: {args.limit} docs   |   Coord-descent rounds: {args.rounds}   |   Total evals: {eval_fn.calls}   |   Time: {elapsed:.0f}s",
        "",
        f"Init F1 (all biases = 0): **{init_metrics['overlap_f1']:.4f}** (P={init_metrics['overlap_p']:.4f} R={init_metrics['overlap_r']:.4f})",
        f"Final F1 (tuned):        **{final_metrics['overlap_f1']:.4f}** (P={final_metrics['overlap_p']:.4f} R={final_metrics['overlap_r']:.4f})",
        f"Δ F1 = **{final_metrics['overlap_f1'] - init_metrics['overlap_f1']:+.4f}**",
        "",
        "## Best biases",
        "",
        "| Key | Value | Effect |",
        "|---|---:|---|",
    ]
    intent = {
        "transition_bias_background_stay": "+ stay in O / − leave O",
        "transition_bias_background_to_start": "+ enter span / − stay in O",
        "transition_bias_inside_to_continue": "+ extend span / − close span",
        "transition_bias_inside_to_end": "+ close span / − extend span",
        "transition_bias_end_to_background": "+ return to O after span",
        "transition_bias_end_to_start": "+ back-to-back spans",
    }
    for k in BIAS_KEYS:
        lines.append(f"| {k} | {best[k]:+.2f} | {intent[k]} |")
    lines.extend([
        "",
        f"**Calibration written to**: `{args.output_calibration}`",
        "",
        "OPF auto-discovers this file when the checkpoint dir is loaded. To pick it up:",
        "1. Restart the wrapper server (or any process holding `OPF(model=...runs/opf_hard_79)`)",
        "2. Run stage4 eval on the test set to validate end-to-end gain",
        "",
        "## Search history",
        "",
        f"Full per-step log: `{args.report_history}`",
    ])
    Path(args.report_summary).write_text("\n".join(lines))
    print(f"[done] wrote report -> {args.report_summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

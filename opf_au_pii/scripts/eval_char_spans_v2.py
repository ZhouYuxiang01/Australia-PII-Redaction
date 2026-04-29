#!/usr/bin/env python3
"""
eval_char_spans_v2.py
======================

Character-span evaluation for OPF predictions, robust to OPF changing or
omitting example_id values in --predictions-out.

Gold JSONL expected from prepare_dataset_v2.py:
  {"example_id": "...", "text": "...", "spans": {"EMAIL: x@y.com": [[s,e]]}}

Prediction JSONL expected from `opf eval --predictions-out`:
  {"example_id": "...", "text": "...", "predicted_spans": {"EMAIL: x@y.com": [[s,e]]}}

Matching strategy:
  1. Try exact example_id matching.
  2. If id overlap is poor, match by text hash + occurrence index.
  3. If text matching is poor, optionally fall back to line-order matching.

Metrics:
   - typed exact char-span P/R/F1
   - untyped exact char-span P/R/F1
   - family/parent_group exact char-span P/R/F1
   - partial-overlap P/R/F1 for boundary drift
   - over-redaction cost (false-positive character count)
   - under-redaction cost (missed span character count weighted by taxonomy cost)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pip install pyyaml") from exc


Span = tuple[str, int, int]


@dataclass
class Row:
    row_index: int
    example_id: str
    text: str
    spans: list[Span]


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_parent_groups(taxonomy: Path | None) -> dict[str, str]:
    if taxonomy is None:
        return {}
    doc = yaml.safe_load(taxonomy.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for entry in doc.get("classes", []):
        code = entry["code"]
        out[code] = entry.get("parent_group", code)
    return out


def load_cost_weights(taxonomy: Path | None) -> dict[str, int]:
    if taxonomy is None:
        return {}
    doc = yaml.safe_load(taxonomy.read_text(encoding="utf-8"))
    weights = doc.get("under_redaction_weights", {})
    out: dict[str, int] = {}
    for entry in doc.get("classes", []):
        code = entry["code"]
        weight_name = entry.get("cost_weight")
        out[code] = int(weights.get(weight_name, 1))
    return out


def parse_span_dict(spans: Any) -> list[Span]:
    out: list[Span] = []
    if not spans:
        return out
    if isinstance(spans, dict):
        for key, offsets in spans.items():
            label = key.split(":", 1)[0].strip()
            for start, end in offsets:
                out.append((label, int(start), int(end)))
    elif isinstance(spans, list):
        for item in spans:
            label = item.get("label") or item.get("class") or item.get("type")
            out.append((str(label), int(item["start"]), int(item["end"])))
    else:
        raise TypeError(f"Unsupported spans form: {type(spans)!r}")
    return out


def read_rows(path: Path, *, prediction: bool) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            if prediction:
                spans_obj = obj.get("predicted_spans") or obj.get("spans") or obj.get("label") or {}
            else:
                spans_obj = obj.get("spans") or obj.get("label") or obj.get("predicted_spans") or {}
            rows.append(Row(
                row_index=i,
                example_id=str(obj.get("example_id", i)),
                text=str(obj.get("text", "")),
                spans=parse_span_dict(spans_obj),
            ))
    return rows


def to_by_id(rows: list[Row]) -> dict[str, Row]:
    out: dict[str, Row] = {}
    dup: Counter[str] = Counter()
    for r in rows:
        if r.example_id in out:
            dup[r.example_id] += 1
            # keep IDs unique without throwing away rows
            out[f"{r.example_id}#dup{dup[r.example_id]}"] = r
        else:
            out[r.example_id] = r
    return out


def pair_by_text(gold_rows: list[Row], pred_rows: list[Row]) -> tuple[dict[str, list[Span]], dict[str, list[Span]], dict[str, Any]]:
    """Pair rows by text hash plus occurrence order for duplicate texts."""
    gold_buckets: dict[str, deque[Row]] = defaultdict(deque)
    pred_buckets: dict[str, deque[Row]] = defaultdict(deque)
    for r in gold_rows:
        gold_buckets[text_key(r.text)].append(r)
    for r in pred_rows:
        pred_buckets[text_key(r.text)].append(r)

    all_hashes = sorted(set(gold_buckets) | set(pred_buckets))
    gold: dict[str, list[Span]] = {}
    pred: dict[str, list[Span]] = {}
    paired = 0
    gold_unmatched = 0
    pred_unmatched = 0

    for h in all_hashes:
        gb = gold_buckets.get(h, deque())
        pb = pred_buckets.get(h, deque())
        n = min(len(gb), len(pb))
        for j in range(n):
            gr = gb.popleft()
            pr = pb.popleft()
            key = f"text:{h}:{j}:{gr.row_index}:{pr.row_index}"
            gold[key] = gr.spans
            pred[key] = pr.spans
            paired += 1
        # unmatched rows remain as pure FN or pure FP
        while gb:
            gr = gb.popleft()
            gold[f"gold_unmatched:{gr.row_index}:{h}"] = gr.spans
            gold_unmatched += 1
        while pb:
            pr = pb.popleft()
            pred[f"pred_unmatched:{pr.row_index}:{h}"] = pr.spans
            pred_unmatched += 1

    diag = {
        "strategy": "text_hash_occurrence",
        "paired_rows": paired,
        "gold_unmatched_rows": gold_unmatched,
        "pred_unmatched_rows": pred_unmatched,
    }
    return gold, pred, diag


def pair_by_order(gold_rows: list[Row], pred_rows: list[Row]) -> tuple[dict[str, list[Span]], dict[str, list[Span]], dict[str, Any]]:
    gold: dict[str, list[Span]] = {}
    pred: dict[str, list[Span]] = {}
    n = min(len(gold_rows), len(pred_rows))
    text_mismatches = 0
    for i in range(n):
        key = f"order:{i}"
        gold[key] = gold_rows[i].spans
        pred[key] = pred_rows[i].spans
        if gold_rows[i].text != pred_rows[i].text:
            text_mismatches += 1
    for i in range(n, len(gold_rows)):
        gold[f"gold_unmatched_order:{i}"] = gold_rows[i].spans
    for i in range(n, len(pred_rows)):
        pred[f"pred_unmatched_order:{i}"] = pred_rows[i].spans
    diag = {
        "strategy": "line_order",
        "paired_rows": n,
        "gold_unmatched_rows": max(0, len(gold_rows) - n),
        "pred_unmatched_rows": max(0, len(pred_rows) - n),
        "paired_text_mismatches": text_mismatches,
    }
    return gold, pred, diag


def load_paired(gold_path: Path, pred_path: Path, match: str) -> tuple[dict[str, list[Span]], dict[str, list[Span]], dict[str, Any]]:
    gold_rows = read_rows(gold_path, prediction=False)
    pred_rows = read_rows(pred_path, prediction=True)
    gold_by_id = to_by_id(gold_rows)
    pred_by_id = to_by_id(pred_rows)

    id_overlap = len(set(gold_by_id) & set(pred_by_id))
    id_overlap_ratio = id_overlap / max(1, min(len(gold_by_id), len(pred_by_id)))

    diag: dict[str, Any] = {
        "gold_rows": len(gold_rows),
        "pred_rows": len(pred_rows),
        "gold_unique_ids": len(gold_by_id),
        "pred_unique_ids": len(pred_by_id),
        "id_overlap": id_overlap,
        "id_overlap_ratio": id_overlap_ratio,
    }

    if match == "id" or (match == "auto" and id_overlap_ratio >= 0.80):
        diag["strategy"] = "example_id"
        ids = sorted(set(gold_by_id) | set(pred_by_id))
        gold = {eid: gold_by_id[eid].spans for eid in ids if eid in gold_by_id}
        pred = {eid: pred_by_id[eid].spans for eid in ids if eid in pred_by_id}
        diag["paired_rows"] = id_overlap
        diag["gold_unmatched_rows"] = len(set(gold_by_id) - set(pred_by_id))
        diag["pred_unmatched_rows"] = len(set(pred_by_id) - set(gold_by_id))
        return gold, pred, diag

    if match == "text" or match == "auto":
        gold, pred, d2 = pair_by_text(gold_rows, pred_rows)
        diag.update(d2)
        # If text pairing found almost nothing, order is safer than zero-match metrics.
        if match == "auto" and d2["paired_rows"] < 0.80 * min(len(gold_rows), len(pred_rows)):
            gold, pred, d3 = pair_by_order(gold_rows, pred_rows)
            diag.update(d3)
        return gold, pred, diag

    if match == "order":
        gold, pred, d2 = pair_by_order(gold_rows, pred_rows)
        diag.update(d2)
        return gold, pred, diag

    raise ValueError(match)


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def normalize(spans: list[Span], mode: str, parent: dict[str, str]) -> set[Span]:
    if mode == "typed":
        return set(spans)
    if mode == "untyped":
        return {("PII", s, e) for _, s, e in spans}
    if mode == "family":
        return {(parent.get(label, label), s, e) for label, s, e in spans}
    raise ValueError(mode)


def overlap(a: Span, b: Span, same_label: bool = True) -> bool:
    la, sa, ea = a
    lb, sb, eb = b
    if same_label and la != lb:
        return False
    return max(sa, sb) < min(ea, eb)


def span_len(span: Span) -> int:
    _, start, end = span
    return max(0, end - start)


def partial_prf(gold_spans: Iterable[Span], pred_spans: Iterable[Span]) -> dict[str, float]:
    gold_list = list(gold_spans)
    pred_list = list(pred_spans)
    matched_gold: set[int] = set()
    tp = 0
    for ps in pred_list:
        for gi, gs in enumerate(gold_list):
            if gi in matched_gold:
                continue
            if overlap(ps, gs, same_label=True):
                matched_gold.add(gi)
                tp += 1
                break
    fp = len(pred_list) - tp
    fn = len(gold_list) - tp
    return prf(tp, fp, fn)


def evaluate(
    gold: dict[str, list[Span]],
    pred: dict[str, list[Span]],
    parent: dict[str, str],
    diag: dict[str, Any],
    cost_weights: dict[str, int] | None = None,
) -> dict[str, Any]:
    ids = sorted(set(gold) | set(pred))
    result: dict[str, Any] = {"examples": len(ids), "matching": diag}
    cost_weights = cost_weights or {}

    for mode in ("typed", "untyped", "family"):
        if mode == "family" and not parent:
            continue
        tp = fp = fn = 0
        partial_tp = partial_fp = partial_fn = 0
        over_redaction_cost = 0
        under_redaction_cost = 0
        fp_by_label: Counter[str] = Counter()
        fn_by_label: Counter[str] = Counter()
        tp_by_label: Counter[str] = Counter()
        overlap_tp = 0

        for eid in ids:
            g = normalize(gold.get(eid, []), mode, parent)
            p = normalize(pred.get(eid, []), mode, parent)
            inter = g & p
            fp_set = p - g
            fn_set = g - p
            tp += len(inter)
            fp += len(fp_set)
            fn += len(fn_set)
            partial = partial_prf(g, p)
            partial_tp += int(partial["tp"])
            partial_fp += int(partial["fp"])
            partial_fn += int(partial["fn"])
            over_redaction_cost += sum(span_len(s) for s in fp_set)
            under_redaction_cost += sum(
                span_len(s) * int(cost_weights.get(s[0], 1)) for s in fn_set
            )
            for label, _, _ in inter:
                tp_by_label[label] += 1
            for label, _, _ in fp_set:
                fp_by_label[label] += 1
            for label, _, _ in fn_set:
                fn_by_label[label] += 1
            for ps in fp_set:
                if any(overlap(ps, gs, same_label=(mode != "untyped")) for gs in fn_set):
                    overlap_tp += 1

        mode_result = prf(tp, fp, fn)
        mode_result["partial"] = prf(partial_tp, partial_fp, partial_fn)
        mode_result["overlapping_non_exact_predictions"] = overlap_tp
        mode_result["over_redaction_cost"] = over_redaction_cost
        mode_result["under_redaction_cost"] = under_redaction_cost
        mode_result["tp_by_label"] = dict(tp_by_label)
        mode_result["fp_by_label"] = dict(fp_by_label)
        mode_result["fn_by_label"] = dict(fn_by_label)
        result[mode] = mode_result

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True, help="gold JSONL, usually test.jsonl")
    p.add_argument("--pred", required=True, help="prediction JSONL from opf eval --predictions-out")
    p.add_argument("--taxonomy", default=None, help="taxonomy YAML for parent_group family metrics")
    p.add_argument("--out", default="char_eval_metrics.json")
    p.add_argument("--match", choices=("auto", "id", "text", "order"), default="auto",
                   help="row matching strategy; auto tries id then text-hash then order")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    taxonomy = Path(args.taxonomy) if args.taxonomy else None
    parent = load_parent_groups(taxonomy) if taxonomy else {}
    cost_weights = load_cost_weights(taxonomy) if taxonomy else {}
    gold, pred, diag = load_paired(Path(args.gold), Path(args.pred), args.match)
    metrics = evaluate(gold, pred, parent, diag, cost_weights=cost_weights)
    Path(args.out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    preview = {
        "matching": metrics["matching"],
        "typed": {k: metrics["typed"][k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
        "untyped": {k: metrics["untyped"][k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
    }
    if "family" in metrics:
        preview["family"] = {k: metrics["family"][k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")}
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    print(f"[write] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

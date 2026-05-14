"""Run dev set through the live server and dump per-span predictions with gold alignment.

Outputs reports/dev_predictions_for_calibration.jsonl with one line per predicted span:
  {
    "doc_id": str, "doc_idx": int,
    "start": int, "end": int, "value": str,
    "predicted_type": str,
    "top1_prob": float, "top3_sum": float, "non_pii_prob": float,
    "risk_score": float, "uncertainty": float,
    "decision": str, "decision_reason": str, "source": str,
    "is_correct": bool,           # overlap with gold of same canonical type
    "gold_overlap_type": str|null # whatever gold type we overlap with (for confusion)
  }

Plus a separate "missed" record for gold spans nothing predicted overlapped:
  {"doc_id": ..., "miss": true, "gold_type": str, "gold_start": int, "gold_end": int}

Server: assumes wrapper is already running on http://localhost:8090.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from redaction.core.postprocess import _label_alias_map  # noqa: E402


def normalize_type(label: str, alias_map: dict[str, str]) -> str:
    return alias_map.get(label, label)


def call_server(url: str, text: str, timeout: float) -> dict[str, Any]:
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_gold(rec: dict[str, Any], alias: dict[str, str]) -> list[tuple[int, int, str]]:
    """opf_*_opf_format.jsonl uses spans = {LABEL: [[start, end], ...]}."""
    out: list[tuple[int, int, str]] = []
    raw = rec.get("spans") or rec.get("entities") or {}
    if isinstance(raw, dict):
        for label, pairs in raw.items():
            label = normalize_type(str(label), alias)
            if not label or label == "O":
                continue
            for pair in pairs or []:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                s, e = int(pair[0]), int(pair[1])
                if 0 <= s < e:
                    out.append((s, e, label))
    elif isinstance(raw, list):
        for span in raw:
            if not isinstance(span, dict):
                continue
            s = int(span.get("start", -1))
            e = int(span.get("end", -1))
            t = normalize_type(str(span.get("label") or span.get("type") or ""), alias)
            if 0 <= s < e and t and t != "O":
                out.append((s, e, t))
    return out


def overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8090/redact")
    ap.add_argument(
        "--dev",
        default="/home/admin/ZYX/pii_training_prep_v3_2/data/train/opf_dev_opf_format.jsonl",
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "reports" / "dev_predictions_for_calibration.jsonl"),
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-text-chars", type=int, default=12000)
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    alias = _label_alias_map()

    records: list[dict[str, Any]] = []
    for line in Path(args.dev).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    if args.limit > 0:
        records = records[: args.limit]
    print(f"[dump] {len(records)} dev records", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_spans = 0
    n_misses = 0
    skipped_long = 0
    errors = 0
    type_counts: Counter[str] = Counter()
    started = time.time()

    with open(args.out, "w", encoding="utf-8") as f:
        for idx, rec in enumerate(records):
            text = rec.get("text", "")
            if not text:
                continue
            if len(text) > args.max_text_chars:
                skipped_long += 1
                continue
            doc_id = str(rec.get("id") or rec.get("record_id") or f"dev_{idx}")
            gold = parse_gold(rec, alias)

            try:
                result = call_server(args.server, text, args.timeout)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"[err] doc {idx}: {e}", flush=True)
                continue

            spans = result.get("spans", [])
            gold_matched = [False] * len(gold)
            for sp in spans:
                ps = int(sp.get("start", 0))
                pe = int(sp.get("end", 0))
                ptype = normalize_type(str(sp.get("type", "")), alias)
                gold_overlap_type = None
                is_correct = False
                for gi, (gs, ge, gt) in enumerate(gold):
                    if overlap((ps, pe), (gs, ge)):
                        gold_overlap_type = gt
                        gold_matched[gi] = True
                        if gt == ptype:
                            is_correct = True
                        break

                rec_out = {
                    "doc_id": doc_id,
                    "doc_idx": idx,
                    "start": ps,
                    "end": pe,
                    "value": sp.get("value", text[ps:pe] if 0 <= ps < pe <= len(text) else ""),
                    "predicted_type": ptype,
                    "top1_prob": float(sp.get("top1_prob") or 0.0),
                    "top3_sum": float(sp.get("top3_sum") or 0.0),
                    "non_pii_prob": float(sp.get("non_pii_prob") or 0.0),
                    "risk_score": float(sp.get("risk_score") or 0.0),
                    "uncertainty": float(sp.get("uncertainty") or 0.0),
                    "decision": str(sp.get("decision", "")),
                    "decision_reason": str(sp.get("decision_reason", "")),
                    "source": str(sp.get("source", "")),
                    "deterministic_evidence": bool(sp.get("deterministic_evidence", False)),
                    "is_correct": is_correct,
                    "gold_overlap_type": gold_overlap_type,
                }
                f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                n_spans += 1
                type_counts[ptype] += 1

            for gi, matched in enumerate(gold_matched):
                if matched:
                    continue
                gs, ge, gt = gold[gi]
                miss = {
                    "doc_id": doc_id,
                    "doc_idx": idx,
                    "miss": True,
                    "gold_type": gt,
                    "gold_start": gs,
                    "gold_end": ge,
                }
                f.write(json.dumps(miss, ensure_ascii=False) + "\n")
                n_misses += 1

            if (idx + 1) % args.progress_every == 0:
                elapsed = time.time() - started
                rate = (idx + 1) / max(elapsed, 1)
                print(
                    f"[dump] {idx + 1}/{len(records)} docs, "
                    f"{n_spans} spans, {n_misses} misses, {rate:.1f} doc/s",
                    flush=True,
                )

    print(f"[done] {n_spans} predictions + {n_misses} misses → {args.out}", flush=True)
    print(f"[done] skipped_long={skipped_long}  errors={errors}", flush=True)
    print(f"[done] top types: {type_counts.most_common(20)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

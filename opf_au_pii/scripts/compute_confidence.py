"""Augment OPF v3 predictions with span-level confidence + TP/FP/FN labels.

Span confidence = exp(mean_token_logprob) over the tokens whose char-range
overlaps the span's [start, end]. We use mean (not min) because mean is the
standard NER convention; min is too pessimistic for long spans.

For each predicted span, we mark it tp / fp_type (boundary matched but type
wrong) / fp_boundary (no overlapping gold) using:
  - tp:           any gold span has the same type AND any char overlap
  - fp_type:      a gold span overlaps but type differs
  - fp_boundary:  no gold span has any char overlap

Output: predictions_with_confidence.jsonl with spans like
  {type, start, end, value, confidence, match}

And gold-side summary: list of unmatched gold spans (FN) for recall analysis.
"""
import argparse, json, math, os
from collections import defaultdict
from typing import Dict, List, Tuple

import tiktoken

DEFAULT_PRED = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v3_full/external_1000_with_logprobs/positive_predictions.jsonl"
DEFAULT_GOLD = "/home/admin/ZYX/Qwen3.5_4b_base_Full_73class/data/eval_external_1000/positive_1000.jsonl"
DEFAULT_OUT  = "/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v3_full/external_1000_with_logprobs/predictions_with_confidence.jsonl"

ENC = tiktoken.get_encoding("o200k_base")


def token_char_spans(text: str, encoded: List[int]) -> List[Tuple[int, int]]:
    """For each token id, return (char_start, char_end) in text via cumulative decode."""
    spans = []
    prev_len = 0
    for i in range(len(encoded)):
        cur = ENC.decode(encoded[: i + 1])
        spans.append((prev_len, len(cur)))
        prev_len = len(cur)
    return spans


def explode_gold(spans_dict):
    out = []
    for k, ranges in spans_dict.items():
        if ":" not in k:
            continue
        typ, val = k.split(":", 1)
        typ = typ.strip(); val = val.strip()
        for s, e in ranges:
            out.append({"type": typ, "value": val, "start": int(s), "end": int(e)})
    return out


def explode_pred(pred_spans_dict):
    out = []
    for k, ranges in pred_spans_dict.items():
        if ":" not in k:
            continue
        typ, val = k.split(":", 1)
        typ = typ.strip(); val = val.strip()
        for s, e in ranges:
            out.append({"type": typ, "value": val, "start": int(s), "end": int(e)})
    return out


def overlap(a_s, a_e, b_s, b_e):
    return a_s < b_e and b_s < a_e


def span_confidence(pred, tok_logs, tok_chars):
    """Mean exp(logprob) of pred_label over tokens whose char-range overlaps the span."""
    s, e = pred["start"], pred["end"]
    lps = []
    for tok, (cs, ce) in zip(tok_logs, tok_chars):
        if cs >= e:
            break
        if ce <= s:
            continue
        # Token overlaps the span. Take its predicted-label logprob.
        # The first topk entry is the chosen label.
        topk = tok.get("topk_logprobs", [])
        if not topk:
            continue
        # Find the predicted label's logprob (should be topk[0] if argmax decode,
        # but viterbi may pick a non-argmax label; look it up explicitly).
        pred_label_id = tok.get("pred_label_id")
        chosen = None
        for ent in topk:
            if ent.get("label_id") == pred_label_id:
                chosen = ent["logprob"]; break
        if chosen is None:
            chosen = topk[0]["logprob"]
        lps.append(chosen)
    if not lps:
        return 0.0, 0
    mean_lp = sum(lps) / len(lps)
    return math.exp(mean_lp), len(lps)


def classify_pred(pred, gold_spans):
    """Return ('tp' | 'fp_type' | 'fp_boundary', best_gold_or_None)."""
    overlaps_any = False
    for g in gold_spans:
        if overlap(pred["start"], pred["end"], g["start"], g["end"]):
            overlaps_any = True
            if g["type"] == pred["type"]:
                return "tp", g
    if overlaps_any:
        return "fp_type", None
    return "fp_boundary", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=DEFAULT_PRED)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--out",  default=DEFAULT_OUT)
    args = ap.parse_args()

    gold_by_text = {}
    with open(args.gold) as f:
        for line in f:
            r = json.loads(line)
            gold_by_text[r["text"]] = explode_gold(r.get("spans", {}))

    n_rows = 0
    n_pred_spans = 0
    n_with_conf = 0
    out_f = open(args.out, "w")

    with open(args.pred) as f:
        for line in f:
            r = json.loads(line)
            text = r["text"]
            preds = explode_pred(r.get("predicted_spans", {}))
            gold = gold_by_text.get(text, [])
            tok_logs = r.get("token_logprobs_topk", [])
            # Recompute the same encoding the model used.
            encoded = ENC.encode(text)
            tok_chars = token_char_spans(text, encoded)
            # Sanity: tok_logs should have same length as encoded (or be a windowed slice)
            n_tok = min(len(tok_logs), len(tok_chars))
            tok_logs = tok_logs[:n_tok]
            tok_chars = tok_chars[:n_tok]

            row_preds = []
            for p in preds:
                conf, ntok = span_confidence(p, tok_logs, tok_chars)
                match, _ = classify_pred(p, gold)
                row_preds.append({
                    "type": p["type"],
                    "start": p["start"],
                    "end": p["end"],
                    "value": p["value"],
                    "confidence": round(conf, 6),
                    "n_tokens": ntok,
                    "match": match,
                })
                n_pred_spans += 1
                if ntok > 0:
                    n_with_conf += 1

            # Compute FN: gold spans that no pred overlaps with same type
            fn = []
            for g in gold:
                covered = False
                for p in preds:
                    if overlap(g["start"], g["end"], p["start"], p["end"]) and p["type"] == g["type"]:
                        covered = True; break
                if not covered:
                    fn.append({"type": g["type"], "start": g["start"],
                               "end": g["end"], "value": g["value"]})

            out_f.write(json.dumps({
                "example_id": r["example_id"],
                "text": text,
                "preds": row_preds,
                "fn_spans": fn,
            }, ensure_ascii=False) + "\n")
            n_rows += 1

    out_f.close()
    print(f"rows: {n_rows}  pred_spans: {n_pred_spans}  with_conf: {n_with_conf}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

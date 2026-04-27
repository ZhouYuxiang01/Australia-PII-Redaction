#!/usr/bin/env python3
"""Evaluate a JSON dataset through the deployed redaction API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SUPPORTED_LABELS = [
    "PERSON",
    "ADDRESS",
    "DATE_OF_BIRTH",
    "EMAIL_ADDRESS",
    "AU_PHONE",
    "AU_TFN",
    "AU_PASSPORT",
    "AU_DRIVERS_LICENCE",
    "STUDENT_ID",
    "MEDICARE_NUMBER",
    "AU_BANK_ACCOUNT",
    "BSB",
    "PAYMENT_CARD_NUMBER",
    "IP_ADDRESS",
    "VEHICLE_REGO",
    "SALARY",
    "WORK_EMAIL",
    "WORK_PHONE",
    "EMPLOYEE_NUMBER",
    "PERSONNEL_NUMBER",
    "MEDICARE_EXPIRY",
    "PASSPORT_EXPIRY",
    "UAC_ID",
    "USI",
    "CENTRELINK_REFERENCE_NUMBER",
    "CREDIT_CARD_EXPIRY",
    "CREDIT_CARD_CVV",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--api-url", default="http://127.0.0.1:8090/api/redact")
    parser.add_argument("--predictions-out", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--supported-labels", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=10.0)
    return parser.parse_args()


def load_supported_labels(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_SUPPORTED_LABELS
    return list(json.loads(path.read_text(encoding="utf-8")))


def load_dataset(path: Path, limit: int | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    rows: list[dict[str, Any]] = []
    for record in records:
        text = (record.get("input") or {}).get("text") or (record.get("positive_sample") or {}).get("text")
        if not text:
            continue
        labels = list((record.get("positive_sample") or {}).get("labels") or [])
        rows.append(
            {
                "id": record.get("id"),
                "text": text,
                "gold_labels": labels,
                "test_metadata": record.get("test_metadata") or {},
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def read_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            completed[str(item["id"])] = item
    return completed


def call_api(api_url: str, text: str, timeout_seconds: float) -> dict[str, Any]:
    body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def call_api_with_retry(args: argparse.Namespace, text: str) -> tuple[dict[str, Any] | None, str | None, float]:
    started = time.perf_counter()
    last_error: str | None = None
    for attempt in range(args.retries + 1):
        try:
            payload = call_api(args.api_url, text, args.timeout_seconds)
            return payload, None, (time.perf_counter() - started) * 1000.0
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = repr(exc)
            if attempt < args.retries:
                time.sleep(args.retry_sleep_seconds)
    return None, last_error, (time.perf_counter() - started) * 1000.0


def canonical_value(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def label_value(label: dict[str, Any], text: str) -> str:
    value = label.get("value")
    if value is not None:
        return str(value)
    start = label.get("start")
    end = label.get("end")
    if isinstance(start, int) and isinstance(end, int):
        return text[start:end]
    return ""


def counter_from_labels(
    labels: list[dict[str, Any]],
    text: str,
    *,
    supported: set[str] | None,
    strict_span: bool,
) -> Counter[tuple[Any, ...]]:
    counter: Counter[tuple[Any, ...]] = Counter()
    for label in labels:
        label_type = label.get("type")
        if not label_type:
            continue
        if supported is not None and label_type not in supported:
            continue
        value = label_value(label, text)
        if strict_span:
            counter[(label_type, label.get("start"), label.get("end"), value)] += 1
        else:
            counter[(label_type, canonical_value(value))] += 1
    return counter


def spans_from_api(item: dict[str, Any]) -> list[dict[str, Any]]:
    payload = item.get("api_response") or {}
    return list(payload.get("spans") or [])


def compute_metric(gold: Counter[tuple[Any, ...]], pred: Counter[tuple[Any, ...]]) -> dict[str, Any]:
    tp = sum((gold & pred).values())
    fp = sum((pred - gold).values())
    fn = sum((gold - pred).values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def compute_summary(rows: list[dict[str, Any]], predictions: list[dict[str, Any]], supported_labels: list[str]) -> dict[str, Any]:
    by_id = {str(item["id"]): item for item in predictions}
    supported = set(supported_labels)
    unsupported_gold_counts: Counter[str] = Counter()
    gold_label_counts: Counter[str] = Counter()
    pred_label_counts: Counter[str] = Counter()
    failed = 0

    counters: dict[str, Counter[tuple[Any, ...]]] = {
        "full_gold_strict": Counter(),
        "full_pred_strict": Counter(),
        "full_gold_value": Counter(),
        "full_pred_value": Counter(),
        "supported_gold_strict": Counter(),
        "supported_pred_strict": Counter(),
        "supported_gold_value": Counter(),
        "supported_pred_value": Counter(),
    }
    per_type_gold: dict[str, Counter[tuple[Any, ...]]] = defaultdict(Counter)
    per_type_pred: dict[str, Counter[tuple[Any, ...]]] = defaultdict(Counter)

    for row in rows:
        item = by_id.get(str(row["id"]))
        spans = spans_from_api(item) if item else []
        if not item or item.get("error"):
            failed += 1
        for label in row["gold_labels"]:
            label_type = label.get("type")
            if label_type:
                gold_label_counts[label_type] += 1
                if label_type not in supported:
                    unsupported_gold_counts[label_type] += 1
        for span in spans:
            if span.get("type"):
                pred_label_counts[span["type"]] += 1

        counters["full_gold_strict"] += counter_from_labels(row["gold_labels"], row["text"], supported=None, strict_span=True)
        counters["full_pred_strict"] += counter_from_labels(spans, row["text"], supported=None, strict_span=True)
        counters["full_gold_value"] += counter_from_labels(row["gold_labels"], row["text"], supported=None, strict_span=False)
        counters["full_pred_value"] += counter_from_labels(spans, row["text"], supported=None, strict_span=False)
        counters["supported_gold_strict"] += counter_from_labels(row["gold_labels"], row["text"], supported=supported, strict_span=True)
        counters["supported_pred_strict"] += counter_from_labels(spans, row["text"], supported=supported, strict_span=True)
        supported_gold_value = counter_from_labels(row["gold_labels"], row["text"], supported=supported, strict_span=False)
        supported_pred_value = counter_from_labels(spans, row["text"], supported=supported, strict_span=False)
        counters["supported_gold_value"] += supported_gold_value
        counters["supported_pred_value"] += supported_pred_value
        for key, count in supported_gold_value.items():
            per_type_gold[str(key[0])][key] += count
        for key, count in supported_pred_value.items():
            per_type_pred[str(key[0])][key] += count

    per_type = {
        label_type: compute_metric(per_type_gold[label_type], per_type_pred[label_type])
        for label_type in sorted(set(per_type_gold) | set(per_type_pred))
    }
    return {
        "rows_total": len(rows),
        "rows_completed": len(predictions),
        "rows_failed": failed,
        "supported_labels": supported_labels,
        "gold_label_counts": dict(gold_label_counts.most_common()),
        "pred_label_counts": dict(pred_label_counts.most_common()),
        "unsupported_gold_counts": dict(unsupported_gold_counts.most_common()),
        "full_label_strict_span": compute_metric(counters["full_gold_strict"], counters["full_pred_strict"]),
        "full_label_value_level": compute_metric(counters["full_gold_value"], counters["full_pred_value"]),
        "supported_label_strict_span": compute_metric(counters["supported_gold_strict"], counters["supported_pred_strict"]),
        "supported_label_value_level": compute_metric(counters["supported_gold_value"], counters["supported_pred_value"]),
        "supported_label_value_level_by_type": per_type,
    }


def write_summary(summary_path: Path, rows: list[dict[str, Any]], predictions: list[dict[str, Any]], labels: list[str]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = compute_summary(rows, predictions, labels)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_dataset(args.dataset, args.limit)
    supported_labels = load_supported_labels(args.supported_labels)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(args.predictions_out) if args.resume else {}

    print(f"rows: {len(rows)}")
    print(f"already done: {len(completed)}")
    print(f"pending: {len(rows) - len(completed)}")
    print(f"api_url: {args.api_url}")
    print(f"predictions_out: {args.predictions_out}")
    print(f"summary_out: {args.summary_out}")

    mode = "a" if args.resume else "w"
    with args.predictions_out.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows, start=1):
            row_id = str(row["id"])
            if row_id in completed:
                continue
            payload, error, latency_ms = call_api_with_retry(args, row["text"])
            item = {
                "id": row["id"],
                "text": row["text"],
                "gold_labels": row["gold_labels"],
                "test_metadata": row["test_metadata"],
                "api_response": payload,
                "error": error,
                "latency_ms": round(latency_ms, 1),
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            completed[row_id] = item
            status = "error" if error else "ok"
            print(f"[{len(completed)}/{len(rows)}] {row_id} {status} {latency_ms / 1000.0:.1f}s", flush=True)
            if index % 10 == 0:
                write_summary(args.summary_out, rows, list(completed.values()), supported_labels)

    write_summary(args.summary_out, rows, list(completed.values()), supported_labels)
    print("done")


if __name__ == "__main__":
    main()

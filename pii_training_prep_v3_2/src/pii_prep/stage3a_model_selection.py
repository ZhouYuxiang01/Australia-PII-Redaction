from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pii_prep.qwen_spancls_heads import (
    build_head,
    load_cache,
    load_labels,
    run_logits,
    select_features,
)


EXPERIMENTS = ["mean_linear", "first_linear", "last_linear", "concat_mlp"]
ZERO_EXAMPLE_RECOVERED_LABELS = ["BANK_ACCOUNT_INFORMATION", "FIRST_NAME", "HOME_PHONE", "LAST_NAME"]
PATTERN_BASED_LABELS = {
    "BANK_ACCOUNT_NUMBER",
    "CENTRELINK_REFERENCE_NUMBER",
    "CREDIT_CARD_EXPIRY",
    "DATE_OF_BIRTH",
    "DEVICE_ID",
    "DRIVERS_LICENCE",
    "EMAIL_ADDRESS",
    "EMPLOYEE_NUMBER",
    "HOME_PHONE",
    "IHI",
    "IP_ADDRESS",
    "LATITUDE",
    "LONGITUDE",
    "MEDICARE_EXPIRY",
    "MEDICARE_NUMBER",
    "MOBILE",
    "NUMBER_PLATE",
    "PASSPORT_EXPIRY",
    "PASSPORT_NUMBER",
    "PASSPORT_START_DATE",
    "PAYMENT_CARD_NUMBER",
    "PENSION_CARD_NUMBER",
    "PERSONNEL_NUMBER",
    "STUDENT_ID",
    "AU_TFN",
    "UAC_ID",
    "USERNAME",
    "USI",
    "VEHICLE_REGO",
    "WORK_EMAIL",
    "WORK_PHONE",
}
SEMANTIC_CONTEXT_LABELS = {
    "ABORIGINALITY",
    "AUDIO_INFORMATION",
    "BANK_ACCOUNT_INFORMATION",
    "CAMERA_FOOTAGE_AUDIO",
    "CARING_RESPONSIBILITIES",
    "CITIZENSHIP_STATUS",
    "CONTRACT_TYPE",
    "COOKIE_INFORMATION",
    "CRIMINAL_RECORDS",
    "DISABILITY_OR_SPECIFIC_CONDITION",
    "EMPLOYMENT_INFORMATION",
    "FACIAL_RECOGNITION",
    "FINGERPRINT",
    "FIRST_NAME",
    "ADDRESS",
    "PERSON",
    "GENDER",
    "GEOLOCATION_INFORMATION",
    "HASHED_PAYMENT_CARD_NUMBER",
    "LAST_NAME",
    "MARITAL_STATUS",
    "MEDICAL_INFORMATION",
    "MILITARY_VETERAN_STATUS",
    "NATIONAL_IDENTITY_CARD",
    "NATIONALITY",
    "NEXT_OF_KIN",
    "PRONOUN",
    "RACIAL_ETHNIC_ORIGIN",
    "RELIGION_BELIEF",
    "SALARY",
    "SALARY_WAGE_EXPECTATION",
    "SEXUAL_ORIENTATION",
    "SIGNATURE",
    "SOCIAL_MEDIA_ACCOUNT",
    "SOCIAL_MEDIA_HISTORY",
    "SOCIAL_MEDIA_ID",
    "SOCIO_ECONOMIC_STATUS",
    "VOICE_RECOGNITION",
    "WEBSITE_HISTORY",
    "WORKERS_COMPENSATION_CLAIM",
    "COUNSELLING_RECORDS",
    "MEDICAL_CERTIFICATE",
    "SPECIAL_CONSIDERATION",
    "SCHOLARSHIP",
    "WAM_SCORE",
    "SUBJECT_RESULTS",
    "SANCTIONS",
    "PERSONAL_DEBT",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def choose_best_model(
    eval_reports: dict[str, dict[str, Any]],
    selection_strategy: str = "dev_nll",
    *,
    run_dir_name: str = "qwen_spancls_heads",
) -> dict[str, Any]:
    candidates = []
    for name, report in eval_reports.items():
        dev = report["metrics"]["after_temperature"]["dev"]
        sort_key = selection_sort_key(dev, selection_strategy)
        candidates.append((sort_key, name, report))
    sort_key, name, report = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    return {
        "selected_model": name,
        "selected_checkpoint": f"runs/{run_dir_name}/{name}/head.pt",
        "selected_temperature": report["temperature"],
        "selection_sort_key": [round(value, 6) for value in sort_key],
        "selection_strategy": selection_strategy,
        "reason": selection_reason(selection_strategy),
    }


def selection_sort_key(dev: dict[str, Any], selection_strategy: str) -> list[float]:
    nll_key = [float(dev["nll"]), float(dev["ece"]), -float(dev["top3_accuracy"])]
    if selection_strategy == "dev_nll":
        return nll_key
    if selection_strategy != "hard_negative_aware":
        raise ValueError(f"unknown selection_strategy: {selection_strategy}")
    hard_negative_values = []
    per_source = dev.get("per_source_accuracy") or {}
    if "candidate_level_negative" in per_source:
        hard_negative_values.append(float(per_source["candidate_level_negative"]))
    if dev.get("non_pii_accuracy") is not None:
        hard_negative_values.append(float(dev["non_pii_accuracy"]))
    if not hard_negative_values:
        return nll_key
    hard_negative_floor = min(hard_negative_values)
    return [1.0 - hard_negative_floor, *nll_key]


def selection_reason(selection_strategy: str) -> str:
    if selection_strategy == "hard_negative_aware":
        return (
            "Selected using calibrated dev metrics only: highest conservative hard-negative/NON_PII recall first, "
            "then lowest dev NLL, lowest dev ECE, and highest dev top3 accuracy."
        )
    return "Selected using calibrated dev metrics only: lowest dev NLL, then lowest dev ECE, then highest dev top3 accuracy."


def template_group(row: dict[str, Any]) -> str:
    record_id = str(row.get("record_id") or row.get("id") or "")
    if record_id.startswith("STAGE2-STAGE2-FULL-BASE-"):
        return record_id.rsplit("-SC", 1)[0]
    if record_id.startswith("AU-PII-"):
        parts = record_id.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:3])
    return record_id


def cross_split_overlap(index: dict[Any, set[str]]) -> tuple[int, list[dict[str, Any]]]:
    examples = []
    count = 0
    for key, splits in index.items():
        if len(splits) > 1:
            count += 1
            if len(examples) < 20:
                examples.append({"key": key, "splits": sorted(splits)})
    return count, examples


def leakage_summary(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    text_index: dict[str, set[str]] = defaultdict(set)
    normalized_text_index: dict[str, set[str]] = defaultdict(set)
    tuple_index: dict[tuple[str, int, int, str], set[str]] = defaultdict(set)
    record_id_index: dict[str, set[str]] = defaultdict(set)
    template_index: dict[str, set[str]] = defaultdict(set)
    id_index: dict[str, set[str]] = defaultdict(set)
    for split, rows in rows_by_split.items():
        for row in rows:
            text = str(row.get("text", ""))
            value = str(row.get("value", ""))
            start = int(row.get("start", -1))
            end = int(row.get("end", -1))
            text_index[text].add(split)
            normalized_text_index[normalize_text(text)].add(split)
            tuple_index[(text, start, end, value)].add(split)
            record_id_index[str(row.get("record_id") or row.get("id"))].add(split)
            template_index[template_group(row)].add(split)
            id_index[str(row.get("id"))].add(split)
    exact_text_count, exact_text_examples = cross_split_overlap(text_index)
    normalized_text_count, normalized_text_examples = cross_split_overlap(normalized_text_index)
    tuple_count, tuple_examples = cross_split_overlap(tuple_index)
    record_count, record_examples = cross_split_overlap(record_id_index)
    template_count, template_examples = cross_split_overlap(template_index)
    duplicate_id_count, duplicate_id_examples = cross_split_overlap(id_index)
    total_rows = sum(len(rows) for rows in rows_by_split.values())
    exact_tuple_rate = tuple_count / max(1, total_rows)
    severe = any([record_count, duplicate_id_count, template_count]) or exact_tuple_rate > 0.01
    return {
        "split_record_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "exact_duplicate_text_cross_split_count": exact_text_count,
        "exact_duplicate_text_examples": exact_text_examples,
        "exact_duplicate_text_start_end_value_cross_split_count": tuple_count,
        "exact_duplicate_text_start_end_value_cross_split_rate": round(exact_tuple_rate, 6),
        "exact_duplicate_text_start_end_value_examples": tuple_examples,
        "duplicate_record_id_cross_split_count": record_count,
        "duplicate_record_id_examples": record_examples,
        "duplicate_example_id_cross_split_count": duplicate_id_count,
        "duplicate_example_id_examples": duplicate_id_examples,
        "template_group_cross_split_count": template_count,
        "template_group_examples": template_examples,
        "normalized_text_overlap_count": normalized_text_count,
        "normalized_text_overlap_examples": normalized_text_examples,
        "near_duplicate_warning": normalized_text_count > exact_text_count,
        "exact_duplicate_warning": exact_text_count > 0 or tuple_count > 0,
        "severity_policy": "Severe if record_id, example_id, or template group crosses splits, or if exact (text,start,end,value) overlap exceeds 1% of examples. Low-rate synthetic bare-span/template value repeats are reported as warnings.",
        "severe_leakage_detected": severe,
        "lora_started": False,
        "opf_started": False,
    }


def high_entropy_mask(entropies: list[float], quantile: float = 0.9) -> list[bool]:
    if not entropies:
        return []
    sorted_values = sorted(entropies)
    index = min(len(sorted_values) - 1, max(0, math.ceil(len(sorted_values) * quantile) - 1))
    threshold = sorted_values[index]
    return [value > threshold for value in entropies]


def targets_from_records(records: list[dict[str, Any]], labels: list[str]) -> torch.Tensor:
    rows = []
    for record in records:
        values = torch.tensor([float(record.get("target_distribution", {}).get(label, 0.0)) for label in labels], dtype=torch.float32)
        rows.append(values / values.sum().clamp_min(1e-12))
    return torch.stack(rows, dim=0)


def target_entropy(targets: torch.Tensor) -> torch.Tensor:
    return -(targets * targets.clamp_min(1e-12).log()).sum(dim=-1)


def subset_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    records: list[dict[str, Any]],
    labels: list[str],
    indices: list[int],
) -> dict[str, Any]:
    if not indices:
        return {"example_count": 0}
    idx = torch.tensor(indices, dtype=torch.long)
    subset_logits = logits[idx]
    subset_targets = targets[idx]
    probs = F.softmax(subset_logits, dim=-1)
    pred = probs.argmax(dim=-1)
    truth = subset_targets.argmax(dim=-1)
    top3 = probs.topk(k=min(3, probs.shape[-1]), dim=-1).indices
    correct = pred.eq(truth)
    nll = -torch.log(probs[torch.arange(len(idx)), truth].clamp_min(1e-12)).mean().item()
    brier = ((probs - subset_targets) ** 2).sum(dim=-1).mean().item()
    ece = expected_calibration_error(probs.max(dim=-1).values, correct.float())
    return {
        "example_count": int(len(indices)),
        "top1_accuracy": round(float(correct.float().mean().item()), 6),
        "top3_accuracy": round(float((top3 == truth.unsqueeze(1)).any(dim=1).float().mean().item()), 6),
        "nll": round(float(nll), 6),
        "brier_score": round(float(brier), 6),
        "ece": round(float(ece), 6),
    }


def expected_calibration_error(confidences: torch.Tensor, correct: torch.Tensor, bins: int = 10) -> float:
    ece = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += float(mask.float().mean() * (confidences[mask].mean() - correct[mask].mean()).abs())
    return ece


def per_group_metrics(logits: torch.Tensor, targets: torch.Tensor, records: list[dict[str, Any]], labels: list[str], group_values: list[str], field: str) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    truth = targets.argmax(dim=-1)
    for i, record in enumerate(records):
        if field == "gold_label":
            key = labels[int(truth[i])]
        else:
            key = str(record.get(field, "unknown"))
        if key in group_values or field != "gold_label":
            grouped[key].append(i)
    return {key: subset_metrics(logits, targets, records, labels, indices) for key, indices in sorted(grouped.items())}


def load_selected_model(root: Path, selected_model: str, checkpoint_path: Path) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_head(selected_model, input_dim=int(checkpoint["input_dim"]), num_labels=int(checkpoint["num_labels"]))
    model.load_state_dict(checkpoint["head_state_dict"])
    model.eval()
    return model, checkpoint


def selected_model_predictions(
    root: Path,
    selected_model: str,
    temperature: float,
    labels: list[str],
    *,
    run_dir_name: str = "qwen_spancls_heads",
    cache_name_prefix: str = "qwen_spancls_embeddings",
) -> dict[str, dict[str, Any]]:
    checkpoint_path = root / "runs" / run_dir_name / selected_model / "head.pt"
    model, _checkpoint = load_selected_model(root, selected_model, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    outputs = {}
    for split in ["train", "dev", "test"]:
        cache = load_cache(root, split, cache_name_prefix)
        features = select_features(cache, selected_model)
        logits = run_logits(model, features, batch_size=2048, device=device) / float(temperature)
        targets = targets_from_records(cache["records"], labels)
        outputs[split] = {"logits": logits, "targets": targets, "records": cache["records"]}
    return outputs


def build_breakdown(outputs: dict[str, dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    breakdown = {}
    for split, data in outputs.items():
        logits = data["logits"]
        targets = data["targets"]
        records = data["records"]
        entropy_values = target_entropy(targets).tolist()
        high_mask = high_entropy_mask(entropy_values, quantile=0.9)
        truth = targets.argmax(dim=-1)
        source_values = sorted({str(record.get("source", "unknown")) for record in records})
        zero_indices = {
            label: [i for i in range(len(records)) if labels[int(truth[i])] == label]
            for label in ZERO_EXAMPLE_RECOVERED_LABELS
        }
        breakdown[split] = {
            "per_source": per_group_metrics(logits, targets, records, labels, source_values, "source"),
            "per_label": per_group_metrics(logits, targets, records, labels, labels, "gold_label"),
            "high_entropy_samples": subset_metrics(logits, targets, records, labels, [i for i, value in enumerate(high_mask) if value]),
            "non_pii": subset_metrics(logits, targets, records, labels, [i for i in range(len(records)) if labels[int(truth[i])] == "NON_PII"]),
            "zero_example_recovered_labels": {
                label: subset_metrics(logits, targets, records, labels, indices)
                for label, indices in zero_indices.items()
            },
            "semantic_context_based_labels": subset_metrics(
                logits,
                targets,
                records,
                labels,
                [i for i in range(len(records)) if labels[int(truth[i])] in SEMANTIC_CONTEXT_LABELS],
            ),
            "pattern_based_labels": subset_metrics(
                logits,
                targets,
                records,
                labels,
                [i for i in range(len(records)) if labels[int(truth[i])] in PATTERN_BASED_LABELS],
            ),
        }
    return breakdown


def prediction_examples(outputs: dict[str, dict[str, Any]], labels: list[str], limit: int = 25) -> list[dict[str, Any]]:
    rows = []
    confusion_pairs: Counter[tuple[str, str]] = Counter()
    candidates = {
        "correct_high_confidence": [],
        "wrong_high_confidence": [],
        "high_entropy": [],
    }
    for split, data in outputs.items():
        logits = data["logits"]
        targets = data["targets"]
        records = data["records"]
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        truth = targets.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        entropy_values = target_entropy(targets).tolist()
        high_mask = high_entropy_mask(entropy_values, quantile=0.9)
        for i, record in enumerate(records):
            gold = labels[int(truth[i])]
            predicted = labels[int(pred[i])]
            item = {
                "category": "",
                "split": split,
                "example_id": record.get("example_id"),
                "source": record.get("source"),
                "value": record.get("value"),
                "gold": gold,
                "predicted": predicted,
                "confidence": round(float(confidence[i]), 6),
                "target_entropy": round(float(entropy_values[i]), 6),
                "top_predictions": [
                    {"label": labels[int(index)], "probability": round(float(value), 6)}
                    for value, index in zip(*probs[i].topk(k=min(5, len(labels))))
                ],
            }
            if gold != predicted:
                confusion_pairs[(gold, predicted)] += 1
            if gold == predicted and confidence[i] >= 0.95:
                candidates["correct_high_confidence"].append(item)
            if gold != predicted and confidence[i] >= 0.80:
                candidates["wrong_high_confidence"].append(item)
            if high_mask[i]:
                candidates["high_entropy"].append(item)
    for category, items in candidates.items():
        if category == "wrong_high_confidence":
            items = sorted(items, key=lambda item: item["confidence"], reverse=True)
        elif category == "high_entropy":
            items = sorted(items, key=lambda item: item["target_entropy"], reverse=True)
        else:
            items = sorted(items, key=lambda item: item["confidence"], reverse=True)
        for item in items[:limit]:
            copied = dict(item)
            copied["category"] = category
            rows.append(copied)
    for (gold, predicted), count in confusion_pairs.most_common(limit):
        rows.append({"category": "top_confusion", "gold": gold, "predicted": predicted, "count": count})
    return rows


def run_model_selection(
    root: Path | str = ".",
    *,
    selection_strategy: str = "dev_nll",
    report_prefix: str = "stage3a_head",
    run_dir_name: str = "qwen_spancls_heads",
    cache_name_prefix: str = "qwen_spancls_embeddings",
    output_prefix: str = "stage3a",
) -> dict[str, Any]:
    root = Path(root)
    reports = {
        experiment: json.loads((root / "reports" / f"{report_prefix}_eval_{experiment}.json").read_text(encoding="utf-8"))
        for experiment in EXPERIMENTS
    }
    summary = json.loads((root / "reports" / f"{report_prefix}_training_summary.json").read_text(encoding="utf-8"))
    selection = choose_best_model(reports, selection_strategy=selection_strategy, run_dir_name=run_dir_name)
    selected = selection["selected_model"]
    selection_report = {
        **selection,
        "dev_metrics_before_temperature": reports[selected]["metrics"]["before_temperature"]["dev"],
        "dev_metrics_after_temperature": reports[selected]["metrics"]["after_temperature"]["dev"],
        "test_metrics_final_reporting_only": reports[selected]["metrics"]["after_temperature"]["test"],
        "all_model_dev_selection_metrics": {
            name: {
                "calibrated_dev_nll": report["metrics"]["after_temperature"]["dev"]["nll"],
                "calibrated_dev_ece": report["metrics"]["after_temperature"]["dev"]["ece"],
                "calibrated_dev_top3_accuracy": report["metrics"]["after_temperature"]["dev"]["top3_accuracy"],
            }
            for name, report in reports.items()
        },
        "selection_used_test_metrics": False,
        "report_prefix": report_prefix,
        "run_dir_name": run_dir_name,
        "cache_name_prefix": cache_name_prefix,
        "qwen_model_loaded": False,
        "lora_started": False,
        "opf_started": False,
        "source_summary_best_by_uncalibrated_dev_nll": summary.get("best_experiment_by_dev_nll"),
    }
    rows_by_split = {
        split: read_jsonl(root / "data" / "train" / f"qwen_spancls_{split}.jsonl")
        for split in ["train", "dev", "test"]
    }
    leakage = leakage_summary(rows_by_split)
    labels = load_labels(root)
    outputs = selected_model_predictions(
        root,
        selected,
        float(selection["selected_temperature"]),
        labels,
        run_dir_name=run_dir_name,
        cache_name_prefix=cache_name_prefix,
    )
    breakdown = {
        "selected_model": selected,
        "selected_checkpoint": selection["selected_checkpoint"],
        "selected_temperature": selection["selected_temperature"],
        "breakdown": build_breakdown(outputs, labels),
        "lora_started": False,
        "opf_started": False,
    }
    examples = prediction_examples(outputs, labels)
    reports_dir = root / "reports"
    write_json(reports_dir / f"{output_prefix}_model_selection_report.json", selection_report)
    write_json(reports_dir / f"{output_prefix}_leakage_check_report.json", leakage)
    write_json(reports_dir / f"{output_prefix}_selected_model_breakdown.json", breakdown)
    write_jsonl(reports_dir / f"{output_prefix}_selected_model_error_examples.jsonl", examples)
    return selection_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--selection-strategy", choices=["dev_nll", "hard_negative_aware"], default="dev_nll")
    parser.add_argument("--report-prefix", default="stage3a_head")
    parser.add_argument("--run-dir-name", default="qwen_spancls_heads")
    parser.add_argument("--cache-name-prefix", default="qwen_spancls_embeddings")
    parser.add_argument("--output-prefix", default="stage3a")
    args = parser.parse_args(argv)
    report = run_model_selection(
        args.root,
        selection_strategy=args.selection_strategy,
        report_prefix=args.report_prefix,
        run_dir_name=args.run_dir_name,
        cache_name_prefix=args.cache_name_prefix,
        output_prefix=args.output_prefix,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

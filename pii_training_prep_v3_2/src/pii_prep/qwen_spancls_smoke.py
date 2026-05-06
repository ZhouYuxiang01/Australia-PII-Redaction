from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n", encoding="utf-8")


class SpanClsJsonlDataset(Dataset):
    def __init__(self, path: Path, labels: list[str], limit: int | None = None):
        self.rows = load_jsonl(path, limit=limit)
        self.labels = labels
        self.label_set = set(labels)
        self.labels_outside: Counter[str] = Counter()
        for row in self.rows:
            labels = set(row.get("target_distribution", {})) | {row.get("top_type")}
            for label in labels:
                if label not in self.label_set:
                    self.labels_outside[str(label)] += 1

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def char_span_to_token_span(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    token_indices = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= tok_start:
            continue
        if tok_end <= start:
            continue
        if tok_start >= end:
            break
        if tok_start < end and tok_end > start:
            token_indices.append(idx)
    if not token_indices:
        return None
    return token_indices[0], token_indices[-1] + 1


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    span_token_ranges: list[tuple[int, int]]
    targets: torch.Tensor
    weights: torch.Tensor
    rows: list[dict[str, Any]]
    mapping_errors: list[dict[str, Any]]


class SpanCollator:
    def __init__(self, tokenizer: Any, labels: list[str], max_length: int):
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, Any]]) -> Batch:
        texts = [str(row["text"]) for row in rows]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets_batch = encoded.pop("offset_mapping").tolist()
        valid_rows: list[dict[str, Any]] = []
        valid_indices: list[int] = []
        span_ranges: list[tuple[int, int]] = []
        mapping_errors: list[dict[str, Any]] = []
        targets = []
        weights = []
        for i, row in enumerate(rows):
            mapped = char_span_to_token_span([(int(a), int(b)) for a, b in offsets_batch[i]], int(row["start"]), int(row["end"]))
            if mapped is None:
                mapping_errors.append(
                    {
                        "id": row.get("id"),
                        "start": row.get("start"),
                        "end": row.get("end"),
                        "value": row.get("value"),
                        "reason": "char_span_not_mapped_to_tokens",
                    }
                )
                continue
            valid_rows.append(row)
            valid_indices.append(i)
            span_ranges.append(mapped)
            targets.append([float(row["target_distribution"].get(label, 0.0)) for label in self.labels])
            weights.append(float(row.get("training_weight", 1.0)))
        input_ids = encoded["input_ids"][valid_indices] if valid_indices else encoded["input_ids"][:0]
        attention_mask = encoded["attention_mask"][valid_indices] if valid_indices else encoded["attention_mask"][:0]
        return Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_token_ranges=span_ranges,
            targets=torch.tensor(targets, dtype=torch.float32),
            weights=torch.tensor(weights, dtype=torch.float32),
            rows=valid_rows,
            mapping_errors=mapping_errors,
        )


class FrozenQwenSpanClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int):
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, span_token_ranges: list[tuple[int, int]]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self._forward_backbone(input_ids, attention_mask, output_hidden_states=False)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                outputs = self._forward_backbone(input_ids, attention_mask, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
        pooled = []
        for batch_index, (start, end) in enumerate(span_token_ranges):
            span_hidden = hidden[batch_index, start:end]
            mean_pool = span_hidden.mean(dim=0)
            first_pool = span_hidden[0]
            last_pool = span_hidden[-1]
            pooled.append((mean_pool + first_pool + last_pool) / 3.0)
        if not pooled:
            return torch.empty((0, self.classifier.out_features), device=input_ids.device)
        return self.classifier(torch.stack(pooled, dim=0).float())

    def _forward_backbone(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, output_hidden_states: bool):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        try:
            return self.backbone(**kwargs, use_cache=False)
        except TypeError:
            return self.backbone(**kwargs)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    losses = -(targets.to(logits.device) * log_probs).sum(dim=-1)
    weights = weights.to(logits.device)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor, labels: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    probs = F.softmax(logits, dim=-1).detach().cpu()
    targets_cpu = targets.detach().cpu()
    pred = probs.argmax(dim=-1)
    truth = targets_cpu.argmax(dim=-1)
    top3 = probs.topk(k=min(3, probs.shape[-1]), dim=-1).indices
    correct = pred.eq(truth)
    n = max(1, len(rows))
    nll = -torch.log(probs[torch.arange(len(rows)), truth].clamp_min(1e-12)).mean().item() if rows else 0.0
    brier = ((probs - targets_cpu) ** 2).sum(dim=-1).mean().item() if rows else 0.0
    source_totals: Counter[str] = Counter()
    source_correct: Counter[str] = Counter()
    non_pii_total = 0
    non_pii_correct = 0
    non_pii_idx = labels.index("NON_PII") if "NON_PII" in labels else -1
    for i, row in enumerate(rows):
        source = str(row.get("source", "unknown"))
        source_totals[source] += 1
        if bool(correct[i]):
            source_correct[source] += 1
        if int(truth[i]) == non_pii_idx:
            non_pii_total += 1
            non_pii_correct += int(bool(correct[i]))
    confidences = probs.max(dim=-1).values
    ece = expected_calibration_error(confidences, correct.float()) if rows else 0.0
    return {
        "top1_accuracy": round(float(correct.float().mean().item()), 6) if rows else 0.0,
        "top3_accuracy": round(float((top3 == truth.unsqueeze(1)).any(dim=1).float().mean().item()), 6) if rows else 0.0,
        "nll": round(float(nll), 6),
        "brier_score": round(float(brier), 6),
        "ece": round(float(ece), 6),
        "per_source_accuracy": {
            source: round(source_correct[source] / total, 6)
            for source, total in sorted(source_totals.items())
        },
        "non_pii_accuracy": round(non_pii_correct / non_pii_total, 6) if non_pii_total else None,
        "example_count": n,
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


def import_transformers():
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    return AutoModel, AutoModelForCausalLM, AutoTokenizer


def load_backbone_and_tokenizer(model_path: str, device: torch.device, use_bf16: bool):
    AutoModel, AutoModelForCausalLM, AutoTokenizer = import_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if device.type == "cuda" else torch.float32
    causal = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, torch_dtype=dtype)
    inner = None
    for attr in ("model", "language_model"):
        cand = getattr(causal, attr, None)
        if cand is not None and hasattr(cand, "embed_tokens"):
            inner = cand
            break
    backbone = inner if inner is not None else causal
    backbone.to(device)
    backbone.eval()
    config = causal.config
    if hasattr(config, "hidden_size") and config.hidden_size is not None:
        hidden_size = int(config.hidden_size)
    elif hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        hidden_size = int(config.text_config.hidden_size)
    else:
        raise ValueError("could not determine Qwen hidden_size from model config")
    return backbone, tokenizer, hidden_size


def train_one_epoch(
    model: FrozenQwenSpanClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mapping_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    model.train()
    model.backbone.eval()
    total_loss = 0.0
    steps = 0
    valid_examples = 0
    for batch in loader:
        mapping_errors.extend(batch.mapping_errors)
        if not batch.rows:
            continue
        logits = model(batch.input_ids.to(device), batch.attention_mask.to(device), batch.span_token_ranges)
        loss = soft_cross_entropy(logits, batch.targets.to(device), batch.weights.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1
        valid_examples += len(batch.rows)
    return {"train_loss": round(total_loss / max(1, steps), 6), "train_steps": steps, "train_valid_examples": valid_examples}


def evaluate(
    model: FrozenQwenSpanClassifier,
    loader: DataLoader,
    device: torch.device,
    labels: list[str],
    mapping_errors: list[dict[str, Any]],
    prediction_limit: int = 50,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    all_logits = []
    all_targets = []
    all_rows = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            mapping_errors.extend(batch.mapping_errors)
            if not batch.rows:
                continue
            logits = model(batch.input_ids.to(device), batch.attention_mask.to(device), batch.span_token_ranges)
            loss = soft_cross_entropy(logits, batch.targets.to(device), batch.weights.to(device))
            losses.append(float(loss.item()))
            all_logits.append(logits.cpu())
            all_targets.append(batch.targets)
            all_rows.extend(batch.rows)
    logits_cat = torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, len(labels)))
    targets_cat = torch.cat(all_targets, dim=0) if all_targets else torch.empty((0, len(labels)))
    metrics = classification_metrics(logits_cat, targets_cat, labels, all_rows)
    metrics["loss"] = round(sum(losses) / max(1, len(losses)), 6)
    predictions = []
    probs = F.softmax(logits_cat, dim=-1) if len(all_rows) else torch.empty((0, len(labels)))
    for row, prob in zip(all_rows[:prediction_limit], probs[:prediction_limit]):
        top_values, top_indices = prob.topk(k=min(5, len(labels)))
        predictions.append(
            {
                "id": row.get("id"),
                "value": row.get("value"),
                "gold_top_type": row.get("top_type"),
                "source": row.get("source"),
                "predictions": [
                    {"label": labels[int(idx)], "probability": round(float(value), 6)}
                    for value, idx in zip(top_values, top_indices)
                ],
            }
        )
    return metrics, predictions


def trainable_parameter_report(model: FrozenQwenSpanClassifier) -> dict[str, Any]:
    backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    head_trainable = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    return {
        "qwen_trainable_parameter_count": backbone_trainable,
        "classification_head_trainable_parameter_count": head_trainable,
        "only_classification_head_trainable": backbone_trainable == 0 and head_trainable > 0,
    }


def run_smoke(
    root: Path | str = ".",
    *,
    model_path: str = "/home/admin/model/Qwen3.5-9B-Base",
    train_limit: int = 1000,
    dev_limit: int = 200,
    epochs: int = 1,
    batch_size: int = 1,
    max_length: int = 1536,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    root = Path(root)
    random.seed(7)
    torch.manual_seed(7)
    labels = json.loads((root / "pii_schema" / "training_label_space_80.json").read_text(encoding="utf-8"))
    train_dataset = SpanClsJsonlDataset(root / "data" / "train" / "qwen_spancls_train.jsonl", labels, limit=train_limit)
    dev_dataset = SpanClsJsonlDataset(root / "data" / "train" / "qwen_spancls_dev.jsonl", labels, limit=dev_limit)
    labels_outside = dict(train_dataset.labels_outside + dev_dataset.labels_outside)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(device.type == "cuda" and torch.cuda.is_bf16_supported())
    backbone, tokenizer, hidden_size = load_backbone_and_tokenizer(model_path, device, use_bf16)
    collator = SpanCollator(tokenizer, labels, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model = FrozenQwenSpanClassifier(backbone, hidden_size=hidden_size, num_labels=len(labels)).to(device)
    parameter_report = trainable_parameter_report(model)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    mapping_errors: list[dict[str, Any]] = []
    started = time.time()
    train_metrics = {}
    for _epoch in range(epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, mapping_errors)
    dev_metrics, predictions = evaluate(model, dev_loader, device, labels, mapping_errors)
    runs_dir = root / "runs" / "qwen_spancls_smoke"
    reports_dir = root / "reports"
    runs_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.classifier.state_dict(), runs_dir / "classification_head.pt")
    report = {
        "stage": "3A.1",
        "model_path": model_path,
        "model_loaded_successfully": True,
        "label_count": len(labels),
        "train_sample_limit": train_limit,
        "dev_sample_limit": dev_limit,
        "epochs": epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "bf16": use_bf16,
        "device": str(device),
        "hidden_size": hidden_size,
        "logits_shape_last_eval": [batch_size, len(labels)],
        "labels_outside_training_space": labels_outside,
        "mapping_failure_count": len(mapping_errors),
        "training": train_metrics,
        "dev": dev_metrics,
        "parameter_report": parameter_report,
        "risk_score_calibration_placeholder": None,
        "student_full_training_started": False,
        "lora_started": False,
        "opf_started": False,
        "wall_time_seconds": round(time.time() - started, 3),
    }
    write_json(reports_dir / "stage3a_qwen_spancls_smoke_report.json", report)
    write_json(reports_dir / "stage3a_qwen_spancls_mapping_errors.json", {"mapping_failure_count": len(mapping_errors), "errors": mapping_errors[:200]})
    write_jsonl(reports_dir / "stage3a_qwen_spancls_dev_predictions_sample.jsonl", predictions)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--model-path", default="/home/admin/model/Qwen3.5-9B-Base")
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--dev-limit", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args(argv)
    report = run_smoke(
        args.root,
        model_path=args.model_path,
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["model_loaded_successfully"]:
        raise SystemExit("model failed to load")
    if not report["parameter_report"]["only_classification_head_trainable"]:
        raise SystemExit("unexpected trainable Qwen parameters")
    if report["labels_outside_training_space"]:
        raise SystemExit("labels outside training space found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

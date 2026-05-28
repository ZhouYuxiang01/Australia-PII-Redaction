from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENTS = ["mean_linear", "first_linear", "last_linear", "concat_mlp"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_features(cache: dict[str, Any], experiment: str) -> torch.Tensor:
    if experiment == "mean_linear":
        return cache["mean_embeddings"]
    if experiment == "first_linear":
        return cache["first_embeddings"]
    if experiment == "last_linear":
        return cache["last_embeddings"]
    if experiment == "concat_mlp":
        return torch.cat([cache["mean_embeddings"], cache["first_embeddings"], cache["last_embeddings"]], dim=1)
    raise ValueError(f"unknown experiment: {experiment}")


def build_head(experiment: str, *, input_dim: int, num_labels: int) -> nn.Module:
    if experiment in {"mean_linear", "first_linear", "last_linear"}:
        return nn.Linear(input_dim, num_labels)
    if experiment == "concat_mlp":
        hidden = 1024
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels),
        )
    raise ValueError(f"unknown experiment: {experiment}")


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    losses = -(targets.to(logits.device) * log_probs).sum(dim=-1)
    weights = weights.to(logits.device)
    return (losses * weights).sum() / weights.sum().clamp_min(1e-6)


def expected_calibration_error(confidences: torch.Tensor, correct: torch.Tensor, bins: int = 10) -> float:
    ece = 0.0
    for i in range(bins):
        lo = i / bins
        hi = (i + 1) / bins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += float(mask.float().mean() * (confidences[mask].mean() - correct[mask].mean()).abs())
    return ece


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor, labels: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    probs = F.softmax(logits, dim=-1).detach().cpu()
    targets_cpu = targets.detach().cpu()
    pred = probs.argmax(dim=-1)
    truth = targets_cpu.argmax(dim=-1)
    top3 = probs.topk(k=min(3, probs.shape[-1]), dim=-1).indices
    correct = pred.eq(truth)
    n = len(rows)
    if n == 0:
        return {
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "nll": 0.0,
            "brier_score": 0.0,
            "ece": 0.0,
            "per_source_accuracy": {},
            "non_pii_accuracy": None,
            "per_label_top1_accuracy": {},
            "confusion_top_pairs": [],
            "example_count": 0,
        }
    nll = -torch.log(probs[torch.arange(n), truth].clamp_min(1e-12)).mean().item()
    brier = ((probs - targets_cpu) ** 2).sum(dim=-1).mean().item()
    source_totals: Counter[str] = Counter()
    source_correct: Counter[str] = Counter()
    label_totals: Counter[str] = Counter()
    label_correct: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    non_pii_total = 0
    non_pii_correct = 0
    non_pii_idx = labels.index("NON_PII") if "NON_PII" in labels else -1
    for i, row in enumerate(rows):
        source = str(row.get("source", "unknown"))
        gold_label = labels[int(truth[i])]
        pred_label = labels[int(pred[i])]
        is_correct = bool(correct[i])
        source_totals[source] += 1
        label_totals[gold_label] += 1
        if is_correct:
            source_correct[source] += 1
            label_correct[gold_label] += 1
        else:
            confusion[(gold_label, pred_label)] += 1
        if int(truth[i]) == non_pii_idx:
            non_pii_total += 1
            non_pii_correct += int(is_correct)
    confidences = probs.max(dim=-1).values
    return {
        "top1_accuracy": round(float(correct.float().mean().item()), 6),
        "top3_accuracy": round(float((top3 == truth.unsqueeze(1)).any(dim=1).float().mean().item()), 6),
        "nll": round(float(nll), 6),
        "brier_score": round(float(brier), 6),
        "ece": round(float(expected_calibration_error(confidences, correct.float())), 6),
        "per_source_accuracy": {source: round(source_correct[source] / total, 6) for source, total in sorted(source_totals.items())},
        "non_pii_accuracy": round(non_pii_correct / non_pii_total, 6) if non_pii_total else None,
        "per_label_top1_accuracy": {label: round(label_correct[label] / total, 6) for label, total in sorted(label_totals.items())},
        "confusion_top_pairs": [
            {"gold": gold, "predicted": predicted, "count": count}
            for (gold, predicted), count in confusion.most_common(50)
        ],
        "example_count": n,
    }


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, max_iter: int = 200) -> float:
    logits = logits.detach().float()
    targets = targets.detach().float()
    weights = weights.detach().float()
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = soft_cross_entropy(logits / temperature, targets, weights)
        loss.backward()
        return loss

    optimizer.step(closure)
    return round(float(log_temperature.detach().exp().clamp(0.05, 20.0).item()), 6)


def load_labels(root: Path) -> list[str]:
    return json.loads((root / "pii_schema" / "training_label_space_80.json").read_text(encoding="utf-8"))


def load_cache(root: Path, split: str, cache_name_prefix: str = "qwen_spancls_embeddings") -> dict[str, Any]:
    return torch.load(root / "data" / "cache" / f"{cache_name_prefix}_{split}.pt", map_location="cpu", weights_only=False)


def build_targets(
    records: list[dict[str, Any]],
    labels: list[str],
    *,
    source_weight_overrides: dict[str, float] | None = None,
    label_weight_overrides: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    label_set = set(labels)
    labels_outside: Counter[str] = Counter()
    targets = []
    weights = []
    source_weight_overrides = source_weight_overrides or {}
    label_weight_overrides = label_weight_overrides or {}
    for row in records:
        distribution = row.get("target_distribution", {})
        for label in distribution:
            if label not in label_set:
                labels_outside[str(label)] += 1
        if row.get("top_type") not in label_set:
            labels_outside[str(row.get("top_type"))] += 1
        values = torch.tensor([float(distribution.get(label, 0.0)) for label in labels], dtype=torch.float32)
        total = values.sum().clamp_min(1e-12)
        targets.append(values / total)
        top_type = str(row.get("top_type"))
        source = str(row.get("source", "unknown"))
        weight = float(row.get("training_weight", 1.0))
        weight *= float(source_weight_overrides.get(source, 1.0))
        weight *= float(label_weight_overrides.get(top_type, 1.0))
        weights.append(weight)
    return torch.stack(targets, dim=0), torch.tensor(weights, dtype=torch.float32), dict(labels_outside)


def parse_weight_overrides(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    overrides: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"weight override must be NAME=FLOAT, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"weight override has empty name: {item!r}")
        multiplier = float(value)
        if multiplier <= 0:
            raise ValueError(f"weight override must be positive for {key}: {multiplier}")
        overrides[key] = multiplier
    return overrides


def make_loader(features: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(features, targets, weights, torch.arange(features.shape[0]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_logits(model: nn.Module, features: torch.Tensor, *, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    logits_parts = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = features[start : start + batch_size].to(device=device, dtype=torch.float32)
            logits_parts.append(model(batch).detach().cpu())
    return torch.cat(logits_parts, dim=0)


def evaluate_split(
    *,
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    records: list[dict[str, Any]],
    labels: list[str],
    batch_size: int,
    device: torch.device,
    temperature: float | None = None,
) -> dict[str, Any]:
    logits = run_logits(model, features, batch_size=batch_size, device=device)
    scaled_logits = logits / float(temperature) if temperature else logits
    metrics = classification_metrics(scaled_logits, targets, labels, records)
    metrics["loss"] = round(float(soft_cross_entropy(scaled_logits, targets, weights).item()), 6)
    return metrics


def train_model(
    *,
    model: nn.Module,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    train_weights: torch.Tensor,
    dev_features: torch.Tensor,
    dev_targets: torch.Tensor,
    dev_weights: torch.Tensor,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    train_loader = make_loader(train_features, train_targets, train_weights, batch_size=batch_size, shuffle=True)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_dev_nll = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        steps = 0
        for features, targets, weights, _indices in train_loader:
            logits = model(features.to(device=device, dtype=torch.float32))
            loss = soft_cross_entropy(logits, targets.to(device), weights.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            steps += 1
        dev_logits = run_logits(model, dev_features, batch_size=batch_size, device=device)
        dev_loss = float(soft_cross_entropy(dev_logits, dev_targets, dev_weights).item())
        dev_truth = dev_targets.argmax(dim=-1)
        dev_nll = float(-torch.log(F.softmax(dev_logits, dim=-1)[torch.arange(len(dev_truth)), dev_truth].clamp_min(1e-12)).mean().item())
        history.append({"epoch": epoch, "train_loss": round(loss_sum / max(1, steps), 6), "dev_loss": round(dev_loss, 6), "dev_nll": round(dev_nll, 6)})
        if dev_nll < best_dev_nll - 1e-5:
            best_dev_nll = dev_nll
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    best_dev_logits = run_logits(model, dev_features, batch_size=batch_size, device=device)
    return (
        {
            "best_epoch": best_epoch,
            "best_dev_nll": round(best_dev_nll, 6),
            "epochs_ran": len(history),
            "history": history,
            "wall_time_seconds": round(time.time() - started, 3),
        },
        best_state,
        best_dev_logits,
    )


def train_experiment(
    *,
    root: Path,
    experiment: str,
    labels: list[str],
    caches: dict[str, dict[str, Any]],
    targets: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
    labels_outside: dict[str, dict[str, int]],
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    device: torch.device,
    run_dir_name: str = "qwen_spancls_heads",
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    features = {split: select_features(cache, experiment) for split, cache in caches.items()}
    input_dim = int(features["train"].shape[1])
    model = build_head(experiment, input_dim=input_dim, num_labels=len(labels))
    logits_shape = list(model(features["dev"][: min(2, features["dev"].shape[0])].float()).shape)
    training, best_state, dev_logits = train_model(
        model=model,
        train_features=features["train"],
        train_targets=targets["train"],
        train_weights=weights["train"],
        dev_features=features["dev"],
        dev_targets=targets["dev"],
        dev_weights=weights["dev"],
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=learning_rate,
        device=device,
    )
    temperature = fit_temperature(dev_logits, targets["dev"], weights["dev"])
    evals = {}
    evals_calibrated = {}
    for split in ["train", "dev", "test"]:
        evals[split] = evaluate_split(
            model=model,
            features=features[split],
            targets=targets[split],
            weights=weights[split],
            records=caches[split]["records"],
            labels=labels,
            batch_size=batch_size,
            device=device,
        )
        evals_calibrated[split] = evaluate_split(
            model=model,
            features=features[split],
            targets=targets[split],
            weights=weights[split],
            records=caches[split]["records"],
            labels=labels,
            batch_size=batch_size,
            device=device,
            temperature=temperature,
        )
    run_dir = root / "runs" / run_dir_name / experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "experiment": experiment,
            "head_state_dict": best_state,
            "input_dim": input_dim,
            "num_labels": len(labels),
            "labels": labels,
            "temperature": temperature,
            "training": training,
        },
        run_dir / "head.pt",
    )
    report = {
        "experiment": experiment,
        "status": "completed",
        "input_dim": input_dim,
        "logits_shape": logits_shape,
        "training": training,
        "temperature": temperature,
        "labels_outside_training_space": labels_outside,
        "metrics": {"before_temperature": evals, "after_temperature": evals_calibrated},
        "risk_score_calibration_placeholder": None,
        "qwen_model_loaded": False,
        "lora_started": False,
        "opf_started": False,
    }
    confusion = [
        {
            "experiment": experiment,
            "split": split,
            "pairs": evals_calibrated[split]["confusion_top_pairs"][:20],
        }
        for split in ["train", "dev", "test"]
    ]
    calibration = {
        "experiment": experiment,
        "temperature": temperature,
        "dev_before": {"nll": evals["dev"]["nll"], "ece": evals["dev"]["ece"], "loss": evals["dev"]["loss"]},
        "dev_after": {"nll": evals_calibrated["dev"]["nll"], "ece": evals_calibrated["dev"]["ece"], "loss": evals_calibrated["dev"]["loss"]},
    }
    return report, calibration, confusion


def run_head_training(
    root: Path | str = ".",
    *,
    experiments: list[str] | None = None,
    batch_size: int = 1024,
    max_epochs: int = 30,
    patience: int = 5,
    learning_rate: float = 1e-3,
    cache_name_prefix: str = "qwen_spancls_embeddings",
    run_dir_name: str = "qwen_spancls_heads",
    report_prefix: str = "stage3a_head",
    source_weight_overrides: dict[str, float] | None = None,
    label_weight_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    experiments = experiments or EXPERIMENTS
    labels = load_labels(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caches = {split: load_cache(root, split, cache_name_prefix) for split in ["train", "dev", "test"]}
    targets: dict[str, torch.Tensor] = {}
    weights: dict[str, torch.Tensor] = {}
    labels_outside: dict[str, dict[str, int]] = {}
    for split, cache in caches.items():
        targets[split], weights[split], labels_outside[split] = build_targets(
            cache["records"],
            labels,
            source_weight_overrides=source_weight_overrides,
            label_weight_overrides=label_weight_overrides,
        )
    reports_dir = root / "reports"
    summary: dict[str, Any] = {
        "stage": "3A.3",
        "label_count": len(labels),
        "experiments": {},
        "device": str(device),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "source_weight_overrides": source_weight_overrides or {},
        "label_weight_overrides": label_weight_overrides or {},
        "labels_outside_training_space": labels_outside,
        "qwen_model_loaded": False,
        "lora_started": False,
        "opf_started": False,
        "classifier_full_training_started": False,
    }
    calibration_reports = []
    confusion_reports = []
    started = time.time()
    for experiment in experiments:
        print(f"starting experiment: {experiment}", flush=True)
        try:
            report, calibration, confusion = train_experiment(
                root=root,
                experiment=experiment,
                labels=labels,
                caches=caches,
                targets=targets,
                weights=weights,
                labels_outside=labels_outside,
                batch_size=batch_size,
                max_epochs=max_epochs,
                patience=patience,
                learning_rate=learning_rate,
                device=device,
                run_dir_name=run_dir_name,
            )
            write_json(reports_dir / f"{report_prefix}_eval_{experiment}.json", report)
            summary["experiments"][experiment] = {
                "status": "completed",
                "best_epoch": report["training"]["best_epoch"],
                "best_dev_nll": report["training"]["best_dev_nll"],
                "dev_nll_after_temperature": report["metrics"]["after_temperature"]["dev"]["nll"],
                "test_top1_after_temperature": report["metrics"]["after_temperature"]["test"]["top1_accuracy"],
                "test_nll_after_temperature": report["metrics"]["after_temperature"]["test"]["nll"],
                "checkpoint": str(root / "runs" / run_dir_name / experiment / "head.pt"),
            }
            calibration_reports.append(calibration)
            confusion_reports.extend(confusion)
            print(f"completed experiment: {experiment}", flush=True)
        except Exception as exc:
            summary["experiments"][experiment] = {"status": "failed", "failure_reason": repr(exc)}
            print(f"failed experiment: {experiment}: {exc!r}", flush=True)
    completed = {name: data for name, data in summary["experiments"].items() if data["status"] == "completed"}
    if completed:
        best_name, best_data = min(completed.items(), key=lambda item: item[1]["best_dev_nll"])
        summary["best_experiment_by_dev_nll"] = {"experiment": best_name, "best_dev_nll": best_data["best_dev_nll"]}
    else:
        summary["best_experiment_by_dev_nll"] = None
    summary["wall_time_seconds"] = round(time.time() - started, 3)
    write_json(reports_dir / f"{report_prefix}_training_summary.json", summary)
    write_json(reports_dir / f"{report_prefix}_calibration_report.json", {"experiments": calibration_reports})
    write_json(reports_dir / f"{report_prefix}_confusion_examples.json", {"confusion_top_pairs": confusion_reports})
    if any(labels_outside.values()):
        raise SystemExit("labels outside training space found")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--cache-name-prefix", default="qwen_spancls_embeddings",
                        help="cache file prefix under data/cache, e.g. qwen4b_spancls_embeddings")
    parser.add_argument("--run-dir-name", default="qwen_spancls_heads",
                        help="checkpoint dir under runs/, e.g. qwen4b_spancls_heads")
    parser.add_argument("--report-prefix", default="stage3a_head",
                        help="report filename prefix under reports/, e.g. stage3a_qwen4b_head")
    parser.add_argument("--source-weight-overrides", default="",
                        help="comma-separated source multipliers, e.g. candidate_level_negative=3,qwen_5way_ranking=1.5")
    parser.add_argument("--label-weight-overrides", default="",
                        help="comma-separated top_type multipliers, e.g. NON_PII=2")
    args = parser.parse_args(argv)
    summary = run_head_training(
        args.root,
        experiments=[item.strip() for item in args.experiments.split(",") if item.strip()],
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        cache_name_prefix=args.cache_name_prefix,
        run_dir_name=args.run_dir_name,
        report_prefix=args.report_prefix,
        source_weight_overrides=parse_weight_overrides(args.source_weight_overrides),
        label_weight_overrides=parse_weight_overrides(args.label_weight_overrides),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failed = {name: data for name, data in summary["experiments"].items() if data["status"] != "completed"}
    if failed:
        raise SystemExit(f"one or more experiments failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

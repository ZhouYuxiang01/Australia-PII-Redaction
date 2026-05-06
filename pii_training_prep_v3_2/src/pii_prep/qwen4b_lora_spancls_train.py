"""Training loop for Qwen4B LoRA SpanCls.

Handles:
- Data loading from qwen_spancls_{split}.jsonl
- Tokenization with span position tracking
- LoRA + head training with CE + optional KL loss
- Evaluation with NLL, ECE, Brier, per-label accuracy
"""
from __future__ import annotations

import json, math, time, sys
from pathlib import Path
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# ── Dataset ────────────────────────────────────────────────────────────

class SpanClsDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, labels: list[str], max_seq_len: int = 512, max_samples: int = 0):
        self.tokenizer = tokenizer
        self.labels = labels
        self.label_to_id = {l: i for i, l in enumerate(labels)}
        self.max_seq_len = max_seq_len

        records = []
        for line in open(jsonl_path):
            records.append(json.loads(line))

        if max_samples > 0:
            records = records[:max_samples]

        self.samples = []
        for rec in records:
            text = rec["text"]
            start = rec["start"]
            end = rec["end"]
            top_type = rec["top_type"]
            hard_label = self.label_to_id.get(top_type, self.label_to_id.get("NON_PII", 0))

            # Teacher distribution
            teacher = torch.zeros(len(labels))
            td = rec.get("target_distribution", {})
            for label, prob in td.items():
                lid = self.label_to_id.get(label)
                if lid is not None:
                    teacher[lid] = prob
            if teacher.sum() < 0.1:
                teacher[hard_label] = 1.0

            self.samples.append({
                "text": text,
                "start": start,
                "end": end,
                "hard_label": hard_label,
                "teacher": teacher,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = s["text"]

        # Tokenize and track token offsets
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        offset_mapping = enc["offset_mapping"]

        # Convert char offsets to token offsets
        char_start = s["start"]
        char_end = s["end"]
        token_start = 0
        token_end = len(input_ids)

        for ti, (cs, ce) in enumerate(offset_mapping):
            if cs <= char_start < ce:
                token_start = ti
            if cs < char_end <= ce:
                token_end = ti + 1
                break

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "span_start": token_start,
            "span_end": token_end,
            "hard_label": s["hard_label"],
            "teacher": s["teacher"],
        }


def collate_fn(batch):
    # Pad input_ids and attention_mask
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.stack([
        F.pad(b["input_ids"], (0, max_len - b["input_ids"].size(0)), value=0)
        for b in batch
    ])
    attention_mask = torch.stack([
        F.pad(b["attention_mask"], (0, max_len - b["attention_mask"].size(0)), value=0)
        for b in batch
    ])
    span_starts = torch.tensor([b["span_start"] for b in batch], dtype=torch.long)
    span_ends = torch.tensor([b["span_end"] for b in batch], dtype=torch.long)
    hard_labels = torch.tensor([b["hard_label"] for b in batch], dtype=torch.long)
    teachers = torch.stack([b["teacher"] for b in batch])

    return input_ids, attention_mask, span_starts, span_ends, hard_labels, teachers


# ── Metrics ────────────────────────────────────────────────────────────

def compute_metrics(logits, hard_labels, teachers=None):
    probs = F.softmax(logits, dim=-1)
    confs, preds = probs.max(dim=-1)
    correct = (preds == hard_labels).float()

    top1 = correct.mean().item()
    nll = F.cross_entropy(logits, hard_labels).item()

    # ECE
    ece = _compute_ece(confs, correct)

    # Brier
    one_hot = F.one_hot(hard_labels, num_classes=logits.size(-1)).float()
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean().item()

    return {"top1": top1, "nll": nll, "ece": ece, "brier": brier}


def _compute_ece(confs, correct, bins=10):
    if correct.numel() == 0:
        return 0.0
    confs = confs.detach().cpu()
    correct = correct.detach().cpu()
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confs[mask].mean()
            ece += mask.sum().item() / correct.numel() * abs(bin_acc - bin_conf)
    return ece


def evaluate(model, dataloader, device, max_batches=0):
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    per_label_correct = defaultdict(int)
    per_label_total = defaultdict(int)

    with torch.no_grad():
        for bi, (input_ids, attn, starts, ends, labels, teachers) in enumerate(dataloader):
            if max_batches > 0 and bi >= max_batches:
                break
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            starts = starts.to(device)
            ends = ends.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attn, starts, ends)
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())

            for i in range(len(labels)):
                lid = labels[i].item()
                per_label_total[lid] += 1
                if preds[i].item() == lid:
                    per_label_correct[lid] += 1

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    metrics = compute_metrics(all_logits, all_labels)

    per_label_acc = {}
    for lid in sorted(per_label_total.keys()):
        if per_label_total[lid] > 0:
            per_label_acc[lid] = per_label_correct[lid] / per_label_total[lid]

    metrics["per_label_acc"] = per_label_acc
    return metrics


# ── Training ───────────────────────────────────────────────────────────

def train_model(
    model,
    train_loader,
    dev_loader,
    labels: list[str],
    epochs: int = 3,
    lr_lora: float = 2e-4,
    lr_head: float = 1e-3,
    kl_weight: float = 0.1,
    warmup_ratio: float = 0.1,
    device: torch.device = torch.device("cuda"),
    output_dir: str = "runs/qwen4b_lora_spancls",
    dev_eval_batches: int = 250,
):
    model = model.to(device)
    model.train()

    lora_params = list(model.get_lora_params())
    head_params = list(model.get_head_params())

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": lr_lora},
        {"params": head_params, "lr": lr_head},
    ])

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    best_dev_nll = float("inf")
    best_epoch = 0
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(output_dir) / "checkpoints"
    best_dir = Path(output_dir) / "best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[train] Starting {epochs} epochs, {len(train_loader)} batches/epoch", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        print(f"[train] Epoch {epoch}/{epochs}...", flush=True)

        for batch_idx, (input_ids, attn, starts, ends, labels, teachers) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            starts = starts.to(device)
            ends = ends.to(device)
            labels = labels.to(device)
            teachers = teachers.to(device)

            logits = model(input_ids, attn, starts, ends)

            ce = F.cross_entropy(logits, labels)
            kl = F.kl_div(
                F.log_softmax(logits, dim=-1),
                teachers,
                reduction="batchmean",
            )
            loss = ce + kl_weight * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * input_ids.size(0)
            train_n += input_ids.size(0)

            if (batch_idx + 1) % 200 == 0:
                print(f"  [e{epoch} b{batch_idx+1}/{len(train_loader)}] loss={loss.item():.4f} ce={ce.item():.4f}", flush=True)

            if (batch_idx + 1) % 1000 == 0:
                dev_metrics = evaluate(model, dev_loader, device, max_batches=dev_eval_batches)
                print(f"  [e{epoch} b{batch_idx+1}] EVAL (subset) dev_nll={dev_metrics['nll']:.4f} dev_top1={dev_metrics['top1']:.4f}",
                      flush=True)

        train_loss /= max(train_n, 1)
        dev_metrics = evaluate(model, dev_loader, device)  # full eval

        elapsed = time.time() - t0
        print(f"[e{epoch}] train_loss={train_loss:.4f} "
              f"dev_nll={dev_metrics['nll']:.4f} dev_top1={dev_metrics['top1']:.4f} "
              f"dev_ece={dev_metrics['ece']:.4f} dev_brier={dev_metrics['brier']:.4f} "
              f"elapsed={elapsed:.0f}s", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_nll": dev_metrics["nll"],
            "dev_top1": dev_metrics["top1"],
            "dev_ece": dev_metrics["ece"],
            "dev_brier": dev_metrics["brier"],
        })

        # Save checkpoint
        epoch_ckpt = ckpt_dir / f"epoch_{epoch}"
        model.save_full(str(epoch_ckpt), str(epoch_ckpt / "head.pt"))

        if dev_metrics["nll"] < best_dev_nll:
            best_dev_nll = dev_metrics["nll"]
            best_epoch = epoch
            model.save_full(str(best_dir), str(best_dir / "head.pt"))
            print(f"  → new best (dev_nll={best_dev_nll:.4f})", flush=True)

    return {"history": history, "best_epoch": best_epoch, "best_dev_nll": best_dev_nll, "wall_time": time.time() - t0}

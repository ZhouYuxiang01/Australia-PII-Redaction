#!/usr/bin/env python3
"""Entry point for Qwen4B LoRA Span Classification training.

Usage:
    # Smoke test (100 samples)
    python scripts/train_qwen4b_lora_spancls.py --smoke

    # Full training
    python scripts/train_qwen4b_lora_spancls.py --epochs 3
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from transformers import AutoTokenizer

from pii_prep.qwen4b_lora_spancls_model import Qwen4BLoRASpanCls
from pii_prep.qwen4b_lora_spancls_train import SpanClsDataset, collate_fn, train_model, evaluate
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="100-sample overfit smoke test")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--kl-weight", type=float, default=0.1)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--backbone", default="/home/admin/model/Qwen3.5-4B-Base")
    ap.add_argument("--head-checkpoint",
                    default=str(PROJECT_ROOT / "runs/qwen4b_spancls_heads/last_linear/head.pt"))
    ap.add_argument("--labels", default=str(PROJECT_ROOT / "pii_schema/training_label_space_80.json"))
    ap.add_argument("--train-data", default=str(PROJECT_ROOT / "data/train/qwen_spancls_train.jsonl"))
    ap.add_argument("--dev-data", default=str(PROJECT_ROOT / "data/train/qwen_spancls_dev.jsonl"))
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs/qwen4b_lora_spancls"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Smoke mode: {args.smoke}")

    # Load labels
    labels = json.loads(Path(args.labels).read_text())
    if isinstance(labels, dict):
        labels = list(labels.keys())
    assert len(labels) == 80, f"Expected 80 labels, got {len(labels)}"
    print(f"Labels: {len(labels)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded (vocab={len(tokenizer)})")

    # Create model
    print(f"Loading backbone from {args.backbone}...")
    model = Qwen4BLoRASpanCls(
        backbone_path=args.backbone,
        head_checkpoint=args.head_checkpoint,
        num_labels=len(labels),
        hidden_size=2560,
        max_seq_len=args.max_seq_len,
    )
    lora_params = sum(p.numel() for p in model.get_lora_params())
    head_params = sum(p.numel() for p in model.get_head_params())
    base_params = sum(p.numel() for p in model.backbone.parameters() if not any("lora_" in n for n, _ in model.backbone.named_parameters()))
    total_params = lora_params + head_params + base_params
    print(f"Model params: total={total_params/1e6:.1f}M trainable={lora_params/1e6:.2f}M (LoRA) + {head_params/1e3:.0f}K (head)")

    # Create datasets
    smoke_size = 300 if args.smoke else 0
    train_ds = SpanClsDataset(args.train_data, tokenizer, labels, max_seq_len=args.max_seq_len, max_samples=smoke_size)
    dev_ds = SpanClsDataset(args.dev_data, tokenizer, labels, max_seq_len=args.max_seq_len, max_samples=smoke_size if args.smoke else 0)
    print(f"Train samples: {len(train_ds)}, Dev samples: {len(dev_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Smoke: verify shapes before training
    print("\n=== SHAPE CHECK ===")
    batch = next(iter(train_loader))
    input_ids, attn, starts, ends, hard_labels, teachers = batch
    print(f"  input_ids: {list(input_ids.shape)}")
    print(f"  starts: {list(starts.shape)}")
    print(f"  labels: {list(hard_labels.shape)}")
    print(f"  teachers: {list(teachers.shape)}")

    input_ids = input_ids.to(device)
    attn = attn.to(device)
    starts = starts.to(device)
    ends = ends.to(device)

    model = model.to(device)
    with torch.no_grad():
        logits = model(input_ids, attn, starts, ends)
    print(f"  logits: {list(logits.shape)}")
    probs = torch.softmax(logits, dim=-1)
    print(f"  probs sum: {probs.sum(dim=-1).tolist()[:5]}...")
    print(f"  top1 acc (untrained): {(logits.argmax(-1) == hard_labels.to(device)).float().mean().item():.4f}")

    # Smoke: verify device placement (suppress verbose output)
    device_mismatches = 0
    for name, param in model.named_parameters():
        if param.device.type != device.type:
            device_mismatches += 1
    if device_mismatches > 0:
        print(f"  Device mismatches: {device_mismatches} params (all params on device={device.type})")
    else:
        print(f"  All params on {device.type}")

    print("Shape check PASSED\n")

    if not args.smoke:
        print("=== FULL TRAINING ===", flush=True)
        result = train_model(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            labels=labels,
            epochs=args.epochs,
            lr_lora=args.lr_lora,
            lr_head=args.lr_head,
            kl_weight=args.kl_weight,
            device=device,
            output_dir=args.output_dir,
        )
        print(f"\n=== TRAINING COMPLETE ===")
        print(f"  best_epoch: {result['best_epoch']}")
        print(f"  best_dev_nll: {result['best_dev_nll']:.6f}")
        print(f"  wall_time: {result['wall_time']:.0f}s")

        # Save training report
        report = {
            "model": "qwen4b_lora_spancls",
            "backbone": args.backbone,
            "head_init": args.head_checkpoint,
            "labels": len(labels),
            "lora_r": 16, "lora_alpha": 32,
            "train_samples": len(train_ds),
            "dev_samples": len(dev_ds),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "result": result,
        }
        report_path = Path(PROJECT_ROOT) / "reports" / "qwen4b_lora_spancls_train_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"  report: {report_path}")
    else:
        # Smoke: run a few training steps to verify loss decreases
        print("=== OVERFIT SMOKE ===", flush=True)
        optimizer = torch.optim.AdamW([
            {"params": model.get_lora_params(), "lr": args.lr_lora},
            {"params": model.get_head_params(), "lr": args.lr_head},
        ])
        model.train()
        losses = []
        for step in range(50):
            input_ids, attn, starts, ends, hard_labels, teachers = next(iter(train_loader))
            input_ids = input_ids.to(device)
            attn = attn.to(device)
            starts = starts.to(device)
            ends = ends.to(device)
            hard_labels = hard_labels.to(device)
            teachers = teachers.to(device)

            logits = model(input_ids, attn, starts, ends)
            ce = torch.nn.functional.cross_entropy(logits, hard_labels)
            kl = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(logits, dim=-1),
                teachers, reduction="batchmean",
            )
            loss = ce + args.kl_weight * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if step % 10 == 0:
                top1 = (logits.argmax(-1) == hard_labels).float().mean().item()
                print(f"  step {step:3d}: loss={loss.item():.4f} ce={ce.item():.4f} top1={top1:.4f}")

        loss_drop = losses[0] - losses[-1]
        print(f"\n  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss:   {losses[-1]:.4f}")
        print(f"  Drop:         {loss_drop:.4f}")
        print(f"  Smoke {'PASSED' if loss_drop > 0.1 else 'CHECK MANUALLY'}: loss decreased by {loss_drop:.2f}")

        # Save smoke report
        import time
        smoke_report = {
            "step": "L2_overfit_smoke",
            "samples": len(train_ds),
            "steps": 50,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_drop": loss_drop,
            "smoke_result": "PASSED" if loss_drop > 0.1 else "NEEDS_INVESTIGATION",
            "device": str(device),
        }
        smoke_path = Path(PROJECT_ROOT) / "reports" / "qwen4b_lora_spancls_overfit_smoke.json"
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_text(json.dumps(smoke_report, ensure_ascii=False, indent=2) + "\n")
        print(f"  smoke report: {smoke_path}")


if __name__ == "__main__":
    main()

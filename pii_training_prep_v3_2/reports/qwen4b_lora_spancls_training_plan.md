# Qwen4B LoRA SpanCls — Training Plan (Stage L0 Audit)

**Date**: 2026-05-05
**Status**: L0 AUDIT COMPLETE → Proceeding to L1

---

## Audit Findings

### Data
| Split | Records | Source |
|-------|---------|--------|
| train | 113,318 | `qwen_spancls_train.jsonl` |
| dev | 13,838 | `qwen_spancls_dev.jsonl` |
| test | 13,576 | `qwen_spancls_test.jsonl` |

Each record fields:
- `text`: full document text
- `start` / `end`: span character offsets
- `value`: span text
- `target_distribution`: dict of {label → probability} — teacher soft labels FROM 9B model
- `top_type`: hard label (highest probability class)
- `training_weight`: float, typically 0.8
- `source`: "sonnet_high_conf"
- `split`: train/dev/test

### Label Space
- 80 classes: 79 PII labels + `NON_PII`
- Label IDs are positional in the list
- Source: `pii_schema/training_label_space_80.json`
- Teacher distributions available → can use CE + KL loss

### Head Checkpoint
```
head.pt: Linear(2560 → 80)
  weight: [80, 2560]
  bias: [80]
  temperature: 1.046
  best_dev_nll: 0.0922 (epoch 2)
  input_dim: 2560
```

### Environment
| Component | Status |
|-----------|--------|
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| BF16 | Supported |
| PEFT | 0.19.1 (installed) |
| accelerate | Installed |
| datasets | Installed |
| bitsandbytes | Not installed (not needed) |

### Span Length Stats (sample)
- Span: median ~14 chars, max ~200
- Text: mean ~1100, p95 ~2500
- → `max_seq_len = 512` should work; can try 1024

---

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | Qwen3.5-4B-Base (`/home/admin/model/Qwen3.5-4B-Base`) |
| Loader | `AutoModelForCausalLM.from_pretrained()` → `.model` |
| r (rank) | 16 |
| alpha | 32 |
| dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Optional modules | `gate_proj`, `up_proj`, `down_proj` (add if VRAM allows) |
| max_seq_len | 512 (conservative; 1024 after smoke) |
| dtype | bf16 |
| Head init | From frozen baseline `last_linear/head.pt` (weight + bias) |
| Head type | `nn.Linear(2560, 80)` |

## Training Configuration (Initial)

| Parameter | Value |
|-----------|-------|
| Loss | `CE(hard_label) + 0.1 * KL(teacher_dist)` |
| Optimizer | AdamW, lr=2e-4 (LoRA), lr=1e-3 (head) |
| Scheduler | Cosine with warmup (10% of steps) |
| Batch size | 16 (effective, via grad_accum if needed) |
| Epochs | 3-4 |
| Early stopping | dev NLL, patience=2 |
| Metrics | NLL, ECE, Brier, top1, per-label acc |

---

## Output Paths

| Item | Path |
|------|------|
| Model code | `src/pii_prep/qwen4b_lora_spancls_model.py` |
| Training code | `src/pii_prep/qwen4b_lora_spancls_train.py` |
| Entry script | `scripts/train_qwen4b_lora_spancls.py` |
| Checkpoints | `runs/qwen4b_lora_spancls/checkpoints/` |
| Best model | `runs/qwen4b_lora_spancls/best/` |
| Smoke report | `reports/qwen4b_lora_spancls_overfit_smoke.md` |
| Train report | `reports/qwen4b_lora_spancls_train_report.md` |
| Backend config | `configs/backends/hybrid-opf-qwen4b-lora.json` |

---

## Expected Resources

| Resource | Estimate |
|----------|----------|
| GPU VRAM | ~8-10GB (Qwen4B BF16 + LoRA + activations) |
| Train time (3 epochs) | ~2-3 hours on DGX Spark |
| Smoke (100 samples) | ~5 minutes |

---

## Stage Gates

- [x] L0: Audit complete
- [ ] L1: Build model + training code
- [ ] L2: 100-sample overfit smoke → **APPROVAL REQUIRED before L3**
- [ ] L3: Full training → **APPROVAL REQUIRED**
- [ ] L4: Temperature calibration
- [ ] L5: Wrapper integration
- [ ] L6: Final evaluation

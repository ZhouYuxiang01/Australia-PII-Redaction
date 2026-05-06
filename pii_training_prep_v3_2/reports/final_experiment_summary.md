# PII Redaction Training — Final Experiment Summary

**Project**: `pii_training_prep_v3_2`
**Date**: 2026-05-01
**Status**: Stage 1–4 complete. Hybrid pipeline smoke-verified.

---

## Pipeline Overview

```
Raw Text
    │
    ▼
┌──────────────────────────────────┐
│  Stage 1–2: Data Preparation     │
│  - CSV → JSONL conversion        │
│  - Teacher LLM generation        │
│  - Taxonomy reconciliation       │
│  - Augmented dataset merge       │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│  Stage 3A: Qwen Span Classifier  │
│  - Cached Qwen3.5-9B embeddings  │
│  - 4 head architectures tested   │
│  - Selected: last_linear (best)  │
│  - Temperature: 1.035854         │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│  Stage 3B: OPF Hard-Label Model  │
│  - Fine-tuned on 79-class schema │
│  - 81,298 training examples      │
│  - 1 epoch, ~2.3h training       │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│  Stage 4: Integration Pipeline   │
│  OPF spans → Qwen head → Policy  │
│  - Wrappers implemented          │
│  - Smoke test passed (10 ex)     │
│  - CLI demo script running       │
└──────────────────────────────────┘
```

## Model Inventory

| Model | Path | Type | Labels |
|-------|------|------|--------|
| OPF Hard-Label | `runs/opf_hard_79` | Span detector | 79 PII |
| Qwen Head | `runs/qwen_spancls_heads/last_linear/head.pt` | Span classifier | 80 (incl. NON_PII) |
| Qwen Backbone | `/home/admin/model/Qwen3.5-9B-Base` | LLM backbone | Frozen |

## Key Results

| Route | Metric | Value |
|-------|--------|-------|
| OPF-only | Test Detection F1 | 0.9793 |
| OPF-only | Test Span F1 | 0.9725 |
| OPF-only | Token Accuracy | 99.15% |
| Qwen Head | Test Top-1 Accuracy | 98.53% |
| Qwen Head | NON_PII Accuracy | 57.14% |
| Qwen Head | Temperature | 1.035854 |
| Hybrid Smoke | Spans detected | 7/10 examples |
| Hybrid Smoke | Redactions at default | 0 (threshold too high) |

## Known Limitations

1. **Policy thresholds conservative**: Default redact≥0.60 produces 0 redactions in smoke.
2. **OPF misses financial/health IDs**: BSB, Medicare numbers not detected despite being in label space.
3. **Gender/location weak**: GENDER F1=0.36, LONGITUDE F1=0.31 in OPF test.
4. **Hybrid eval incomplete**: Subprocess-per-text OPF too slow for full 9,659-example test set.
5. **NON_PII classification weak**: Qwen head NON_PII accuracy only 57%.

## Recommended Final Architecture

```
Input Text
    │
    ▼
OPF Detector (batched) ─── extracts candidate PII spans
    │
    ▼
Qwen Span Classifier ─── 80-class softmax per span (temp=1.036)
    │
    ▼
Policy Layer ─── risk_score = Σ(prob × weight), CSV Data Classification
    │
    ├── redact (≥0.50) ──► Redact spans in output
    ├── review (≥0.20) ──► Flag for human review
    └── ignore ──────────► Leave unchanged
```

## Future Work

1. **Batched OPF inference** — Python API instead of subprocess-per-text; enables full test evaluation
2. **Soft-label OPF training** — Use Qwen head distributions as soft targets for OPF fine-tuning
3. **Policy threshold calibration** — Sweep on dev data; target redact≥0.40, review≥0.20
4. **Data augmentation** — More examples for GENDER, LONGITUDE, LATITUDE, SALARY_WAGE_EXPECTATION
5. **Qwen LoRA fine-tuning** — Full backbone fine-tuning for better NON_PII discrimination
6. **End-to-end redaction quality** — Human evaluation of final redacted outputs
7. **Performance optimization** — Quantize OPF model, batch Qwen inference

## Reports Index

| Stage | Report |
|-------|--------|
| 3A | `reports/stage3a_model_selection_report.json` |
| 3A | `reports/stage3a_selected_model_breakdown.json` |
| 3B | `reports/stage3b_opf_hard_test_eval.json` |
| 3B | `reports/stage3b_opf_hard_status.json` |
| 4 | `reports/stage4_final_comparison_summary.json` |
| 4 | `reports/stage4_integration_smoke_report.json` |
| 4 | `reports/stage4_opf_only_eval.json` |
| 4 | `reports/stage4_qwen_head_only_eval.json` |
| Final | `reports/final_experiment_summary.md` |
| Final | `reports/final_results_tables.md` |
| Final | `reports/final_demo_examples.md` |

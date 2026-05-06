# Final Demo Backend Decision — After Ablation

**Date**: 2026-05-05
**Author**: Ablation-driven verification

---

## Summary of Ablation Findings

Five controlled experiments (9659 records each) compared backend model (9B vs 4B), policy (v1 vs v2), and loader mode (automodel_legacy vs causal_lm).

### Critical Result: 9B loader is NOT broken

The experiment that forced 9B head with `causal_lm` loader mode produced **identical results** to the `automodel_legacy` loader (Overlap F1 0.8880, Exact F1 0.7992 in both cases). This refutes the prior hypothesis that the legacy 9B head was trained on broken/random embeddings. The corrected assessment is:

> The 9B head was trained with `AutoModel.from_pretrained()`, which loaded correct Qwen9B weights. The head is fully compatible with both loader modes. The legacy 9B backend represents a valid, non-broken Qwen9B semantic baseline.

### Root Cause of 4B v2 Underperformance

The 4B+v2 configuration (Overlap F1 0.8474) underperforms for TWO reasons:

1. **v2 policy cost (~2.4 F1 points)**: The v2 policy's aggressive rescue/postprocess pipeline (deterministic rescue, registry rescue, contextual rescue) creates false positives that reduce precision on BOTH backends. 9B+v2 loses 2.7 F1 points vs 9B+v1. 4B+v2 loses 2.4 F1 points vs 4B+v1.

2. **Model quality gap (~1.7 F1 points)**: With the same v1 policy, 9B (0.8880) outperforms 4B (0.8715). The 9B head is genuinely more accurate.

---

## Recommended Configuration

### Demo Backend: **4B + v1 policy**

```
WRAPPER_BACKEND_CONFIG=configs/backends/hybrid-opf-qwen4b.json
WRAPPER_POLICY_CONFIG=configs/policies/hybrid-80class-v1.json
```

| Metric | 4B+v1 | 4B+v2 | 9B+v1 |
|--------|-------|-------|-------|
| Overlap F1 | **0.8715** | 0.8474 | 0.8880 |
| Exact F1 | **0.7793** | 0.7201 | 0.7992 |
| Overlap Recall | **0.9741** | 0.9487 | 0.9756 |
| Under-redaction | **6.08%** | 8.75% | 4.28% |
| High-risk under | **7.06%** | 9.25% | 7.00% |
| p50 latency | **133ms** | 133ms | 155ms |
| 80-class prob dist | ✅ | ✅ | ✅ v1 |

**Rationale**: 4B+v1 achieves an overlap F1 of 0.8715 — only 1.7 points behind the 9B flagship, while being 15% faster. It provides a full 80-class calibrated probability distribution from a properly loaded Qwen4B backbone (causal_lm). The v1 policy avoids the precision penalties of v2 while retaining all OPF detection quality.

### Flagship (highest accuracy): **9B + v1 policy**

```
WRAPPER_BACKEND_CONFIG=configs/backends/hybrid-opf-qwen.json
WRAPPER_POLICY_CONFIG=configs/policies/hybrid-80class-v1.json
```

Best overlap F1 (0.8880), best recall (0.9756), lowest under-redaction (4.28%), lowest high-risk under-redaction (7.00%). Pairs corrected Qwen9B embeddings with the 9B span classification head.

### Fast Baseline: **OPF-only**

```
WRAPPER_BACKEND_CONFIG=configs/backends/opf-v3.json
WRAPPER_POLICY_CONFIG=configs/policies/opf-v3-default-v1.json
```

Fastest (26ms p50), strong overlap F1 (0.8824). Limitation: no 80-class probability distribution.

### Experimental: **qwen4b-tokencls**

Retained as experimental single-model BIOES route. Not currently part of the demo pipeline.

---

## What Changed From Previous Decision

| Aspect | Before Ablation | After Ablation |
|--------|----------------|----------------|
| Demo backend | 4B + v2 | **4B + v1** (or 9B+v1 for max accuracy) |
| 9B status | "Broken loader, random features" | **Fully valid — identical to causal_lm** |
| v2 policy | "Rescue improves safety" | **Rescue creates precision cost; v1 is safer** |
| F1 gap cause | Unknown | **60% policy, 40% model** |

---

## Policy v2 vs v1 Trade-off

The v2 policy (`hybrid-80class-v2-4b`) adds:
- Registry contextual rescue
- Contextual identifier rescue
- Hard negative drop
- Label alias normalization (also in v1)
- Deterministic rescue patterns

**These features do improve specific low-recall labels**: UAC_ID recall rises from 0.907→1.000 (9B+v2). CENTRELINK_REFERENCE recall rises from 0.919→1.000 (9B+v2). USI recall rises from 0.988→1.000.

**But the aggregate cost is too high**: Overlap F1 drops by 2.4-2.7 points. The v2 policy should be refined — rescue should be more selective, and false positives from rescue should be suppressed before policy application.

---

## Regression Gaps (Unchanged)

Three hard-negative policy gaps persist (independent of v1/v2 choice):

1. **Example plates in training context** — `XYZ123` in "example plate ... training slide" is redacted
2. **Office phone numbers** — `02 9000 1111` described as "placement office phone ... general office number" is redacted
3. **Credit card with test-looking context** — `4111 9090 3333 1200` after "test-looking string, but this one" is IGNOREd on server path (offline eval shows REVIEW — code path difference suspected)

These are hard-negative suppression gaps that would require targeted context-aware rule additions.

---

## Current Server Status

```
Host: aitopatom-5c4b (NVIDIA DGX Spark)
Port: 8090
Backend: hybrid-opf-qwen4b (configs/backends/hybrid-opf-qwen4b.json)
Policy:  hybrid-80class-v2-4b (configs/policies/hybrid-80class-v2-4b.json)
Loader: causal_lm
Head:   last_linear/head.pt
Temp:   1.046
```

**Recommendation**: Switch policy to v1 for improved demo metrics.

---

## Report Index

| Report | Description |
|--------|-------------|
| `stage4_ablation_backend_policy_loader.json` | Structured ablation data |
| `stage4_ablation_backend_policy_loader.md` | Full ablation analysis |
| `final_demo_backend_decision_after_ablation.md` | This document |
| `ablation_2_9b_legacy_v2.json` | 9B+v2 raw eval |
| `ablation_3_4b_v1.json` | 4B+v1 raw eval |
| `ablation_5_9b_causal_lm.json` | 9B+causal_lm raw eval |
| `stage4_eval_hybrid_qwen4b_v2_after_negative_fix.json` | 4B+v2 raw eval |
| `stage4_eval_hybrid_legacy_9b.json` | 9B+v1 raw eval |
| `stage4_eval_opf_only.json` | OPF-only raw eval |

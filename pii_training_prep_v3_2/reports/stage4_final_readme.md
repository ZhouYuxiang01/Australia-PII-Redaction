# Stage 4: Multi-Route PII Detection Evaluation

**Date**: 2026-05-01
**Status**: Smoke-verified. Full hybrid evaluation pending batched OPF inference.

---

## Models

### 1. OPF Hard-Label Detector
- **Checkpoint**: `runs/opf_hard_79`
- **Type**: End-to-end span detector (79 PII classes)
- **Training**: Fine-tuned OPF on 79-class AU PII schema, 81,298 training examples

| Metric | Dev | Test |
|--------|-----|------|
| Detection F1 | 0.9724 | 0.9793 |
| Span F1 | 0.9606 | 0.9725 |
| Precision | 0.9631 | 0.9745 |
| Recall | 0.9817 | 0.9842 |
| Token Accuracy | 98.94% | 99.15% |
| Examples | 10,034 | 9,659 |

**Problematic labels (span F1 < 0.50 on test)**:
- GENDER: F1=0.36 (recall 26%)
- LONGITUDE: F1=0.31 (recall 19%)

### 2. Qwen Span Classification Head
- **Checkpoint**: `runs/qwen_spancls_heads/last_linear/head.pt`
- **Type**: Given-span classifier (80 classes incl. NON_PII)
- **Backbone**: Qwen3.5-9B-Base (frozen)
- **Temperature**: 1.035854

| Metric | Test |
|--------|------|
| Top-1 Accuracy | 98.53% |
| Top-3 Accuracy | — |
| NLL | — |
| ECE | — |
| Brier Score | — |
| NON_PII Accuracy | 57.14% |

**Low-accuracy labels (< 80%)**: 4 labels

### 3. Hybrid Pipeline
- **Status**: Smoke-tested (10 examples), NOT full-evaluated
- **Architecture**: OPF candidate spans → Qwen head re-scoring → Policy layer → Redaction

**Smoke results**: 7/10 examples had spans detected (6 review, 1 ignore, 0 redact).
0 redactions because default threshold 0.60 is conservative (highest risk was 0.48 for DATE_OF_BIRTH).

---

## Known Issues

1. **Default thresholds too conservative**: Redact >= 0.60, Review >= 0.25. No smoke example reached redact threshold. Consider 0.40-0.50.

2. **OPF misses BSB/bank, Medicare, ambiguous dates**: These labels exist in OPF's label space but the model fails to detect them. False negatives for financial and health identifiers.

3. **Hybrid needs batched OPF inference**: Current subprocess-per-text OPF call takes ~0.9s. Full test set (9,659 examples) would take ~2.4 hours. Batched API needed.

4. **GENDER detection near-zero**: F1=0.36 on test in OPF. Qwen head also <80% accuracy on GENDER.

5. **Coordinate (LAT/LONG) detection weak**: F1=0.31-0.52 on test in OPF.

---

## Reports

| Report | Path |
|--------|------|
| OPF-Only Eval | `reports/stage4_opf_only_eval.json` |
| OPF Error Examples | `reports/stage4_opf_only_error_examples.jsonl` |
| Qwen-Head-Only Eval | `reports/stage4_qwen_head_only_eval.json` |
| Qwen Error Examples | `reports/stage4_qwen_head_only_error_examples.jsonl` |
| Hybrid Smoke Report | `reports/stage4_integration_smoke_report.json` |
| Hybrid Examples | `reports/stage4_integration_examples.jsonl` |
| Smoke Miss Investigation | `reports/stage4_smoke_miss_investigation.json` |
| **Final Comparison (this)** | `reports/stage4_final_comparison_summary.json` |

---

## Recommended Next Steps

1. Implement batched OPF inference for full hybrid evaluation
2. Sweep policy thresholds on dev data (target: redact 0.40-0.50, review 0.20-0.25)
3. Augment training data for weak labels (GENDER, LONGITUDE, MEDICARE, BSB)
4. Add confidence thresholding to OPF spans before Qwen re-scoring
5. Evaluate end-to-end redaction quality with human review

# Final Demo Backend Decision: PII Redaction System

**Date**: 2026-05-05
**Author**: Verification run — Sisyphus

---

## Executive Recommendation

**Recommended demo backend**: `hybrid-opf-qwen4b` + `hybrid-80class-v2-4b`

This is the most suitable demonstration candidate because it provides:
1. A full 80-class probability distribution from a properly loaded Qwen4B backbone (`causal_lm` mode)
2. Deterministic rescue covering BSB/account, email, AU mobile/phone, vehicle rego, SID, UAC, USI, passport, IHI, CRN, driver licence
3. Risk-based policy decisions (REDACT / REVIEW / IGNORE) with calibrated confidence
4. Post-processing pipeline with registry/contextual rescue and hard-negative suppression
5. Repaired model loading that avoids the legacy AutoModel broken feature space

---

## Backend Comparison (Full Stage4 — 9659 records)

### Overall Performance

| Metric | OPF-only | 4B v2 (Recommended) | Legacy 9B (Rollback) |
|--------|----------|---------------------|----------------------|
| Exact P | 0.7249 | 0.6490 | 0.7315 |
| Exact R | 0.7715 | 0.8087 | 0.8807 |
| **Exact F1** | 0.7475 | **0.7201** | **0.7992** |
| Overlap P | 0.8581 | 0.7656 | 0.8149 |
| Overlap R | 0.9081 | **0.9487** | **0.9756** |
| **Overlap F1** | 0.8824 | **0.8474** | **0.8880** |
| Overlap F2 | 0.8976 | **0.9054** | **0.9386** |
| Type Accuracy | 0.9590 | 0.9572 | 0.9674 |
| p50 Latency | 25.9ms | 132.6ms | 154.7ms |
| p95 Latency | 112.8ms | 259.8ms | 306.2ms |
| Throughput | 25.5 ex/s | 6.6 ex/s | 5.6 ex/s |

### Redaction Safety

| Metric | OPF-only | 4B v2 | Legacy 9B |
|--------|----------|-------|-----------|
| Over-redaction rate | 17.06% | 21.49% | 15.38% |
| Under-redaction rate | 11.10% | **8.75%** | **4.28%** |
| High-risk under-redaction | 16.45% | **9.25%** | **7.00%** |

### Decision Distribution

| Decision | OPF-only | 4B v2 | Legacy 9B |
|----------|----------|-------|-----------|
| AUTO_REDACT / redact | 14,301 | 16,744 | 16,177 |
| REVIEW / review | 625 | 2,692 | 2,176 |
| IGNORE / ignore | 0 | 3,522 | 5,039 |

---

## Key Label Recall (User-Requested Focus)

| Label | OPF-only | 4B v2 | Legacy 9B |
|-------|----------|-------|-----------|
| ADDRESS | 0.9395 | **0.9892** | **0.9973** |
| VEHICLE_REGO/VEHICLE_ID | 0.0427 | 0.5366 | 0.5366 |
| STUDENT_ID | 0.9725 | **0.9969** | **0.9969** |
| USI | 0.9884 | **1.0000** | 0.9884 |
| UAC_ID | 0.7442 | 0.7326 | **0.9419** |
| PASSPORT_NUMBER/AU_PASSPORT | 0.9787 | **0.9817** | **0.9939** |
| IHI | 0.9881 | **1.0000** | 0.9940 |
| CRN/CENTRELINK_REF | 0.9767 | 0.9419 | **0.9884** |
| DRIVERS_LICENCE | 0.9754 | 0.9590 | **1.0000** |
| DATE_OF_BIRTH | 0.9905 | **1.0000** | 0.9971 |
| EMAIL_ADDRESS/EMAIL | 1.0000 | 0.9744 | 0.9744 |
| PHONE/MOBILE | 0.9769 | **0.9964** | **1.0000** |
| PAYMENT_CARD_NUMBER | 0.6581 | **0.9485** | **0.9559** |
| CREDIT_CARD_EXPIRY | 0.9753 | **0.9938** | **0.9938** |
| BANK_ACCOUNT/AU_BANK_ACCOUNT | 0.9216 | **1.0000** | 0.9972 |
| GENDER | 0.3000 | 0.2600 | 0.2600 |

---

## Backend Classification

### Recommended: hybrid-opf-qwen4b + hybrid-80class-v2-4b

**Strengths**:
- 80-class calibrated probability distribution with repaired Qwen4B loader (`causal_lm`)
- Deterministic rescue covers all critical AU PII types (BSB, account, email, mobile, rego, SID, UAC, USI, passport, IHI, CRN, driver licence)
- Risk-based policy separates REDACT / REVIEW / IGNORE
- Registry/contextual rescue recovers identifiers from surrounding text
- Hard-negative suppression for placeholder/example/fake data (with noted gaps)
- Sanity checks: expected hidden_size=2560, expected loader_mode=causal_lm
- Overlap recall 0.9487 — strong coverage
- High-risk under-redaction rate 9.25% — acceptable for demo

**Known Limitations**:
- Exact F1 (0.7201) trails legacy 9B due to lower precision on broad-coverage labels (EMAIL, PHONE, IP_ADDRESS, PASSPORT_EXPIRY, VEHICLE_ID)
- Over-redaction rate 21.49% — higher than both alternatives, primarily from pattern-matching false positives on EMAIL, PHONE, IP_ADDRESS, PASSPORT_EXPIRY
- Two hard-negative gaps: office phone numbers and example plates in training context not suppressed
- VEHICLE_ID precision 0.1709 — rescue over-captures non-rego patterns

### Fast Baseline: OPF-only

**Strengths**:
- Fastest: 25.9ms p50, 25 ex/s throughput
- No Qwen dependency — simpler deployment, lower GPU memory
- Competitive overlap F1 (0.8824) despite simpler architecture
- Lowest over-redaction rate (17.06%)
- 73-class coverage with good precision on structured identifiers

**Known Limitations**:
- No 80-class probability distribution — only OPF's 73 classes
- Higher under-redaction rate (11.10%) — misses more PII
- High-risk under-redaction 16.45% — substantially more missed high-risk PII
- No rescue/registry pipeline — no deterministic recovery
- Misses more PAYMENT_CARD_NUMBER (recall 0.6581 vs 0.9485)
- VEHICLE_ID recall near zero (0.0427)

### Rollback: hybrid-opf-qwen legacy + hybrid-80class-v1

**Strengths**:
- Highest exact F1 (0.7992) and overlap F1 (0.8880)
- Best under-redaction rate (4.28%)
- Best high-risk under-redaction rate (7.00%)
- Best type accuracy (0.9674)
- Preserved as operational fallback

**Important Caveat**:
The legacy 9B head was trained and operates in `automodel_legacy` feature space using `AutoModel.from_pretrained()`, which loads a broken/random feature representation from the Qwen backbone. While this backend achieves strong metrics as a rollback, **it does not represent a corrected Qwen9B semantic baseline**. Its performance is measured in the same broken feature space it was trained in, making it internally consistent but not transferable to a properly loaded backbone.

---

## Experimental: qwen4b-tokencls

Token classification approach retained as experimental. **Not the current recommended backend.** This route remains available for future exploration but is not part of the demo pipeline.

---

## Regression Gap Assessment

Two hard-negative policy gaps were identified in the 9-case regression test:

1. **Example plates in training context** (Case 6): `XYZ123` in "example plate ... training slide" is not caught by the hard-negative logic. Impact: low — requires both a number-plate-like pattern AND explicit example/training keywords in close proximity.

2. **Office phone numbers** (Case 8): Numbers explicitly described as "office", "general office", or "placement office" are not distinguished from personal phones. Impact: moderate — office phone numbers represent a distinct operational category.

These are policy-layer issues, not model issues. Both could be addressed through expanded hard-negative trigger lists and context-aware disambiguation rules without retraining.

---

## Configuration Reference

### Demo Server Start
```bash
cd /home/admin/ZYX/redaction-wrapper
WRAPPER_BACKEND_CONFIG=configs/backends/hybrid-opf-qwen4b.json \
WRAPPER_POLICY_CONFIG=configs/policies/hybrid-80class-v2-4b.json \
./scripts/run_server.sh
```

### Rollback Start
```bash
cd /home/admin/ZYX/redaction-wrapper
WRAPPER_BACKEND_CONFIG=configs/backends/hybrid-opf-qwen.json \
WRAPPER_POLICY_CONFIG=configs/policies/hybrid-80class-v1.json \
./scripts/run_server.sh
```

### Current Server Status (as verified)
- Host: `aitopatom-5c4b` (NVIDIA DGX Spark)
- Port: 8090
- Running: `hybrid-opf-qwen4b` + `hybrid-80class-v2-4b`
- Conda env: `opf`
- Loader mode: `causal_lm` ✓
- Head: `last_linear/head.pt` ✓
- Temperature: 1.046 ✓

---

## Artifact Preservation Note

All artifacts preserved intact:
- Model checkpoints: unchanged
- OPF artifacts: unchanged
- Legacy 9B config: unchanged
- qwen4b-tokencls config: retained as experimental
- No `spans[].value` leaks introduced

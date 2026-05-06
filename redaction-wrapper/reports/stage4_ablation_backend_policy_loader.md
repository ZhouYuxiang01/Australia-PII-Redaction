# Stage4 Ablation: Backend, Policy, and Loader Mode

**Date**: 2026-05-05
**Test set**: `opf_test_opf_format.jsonl` — 9659 records, full OPF test set
**Purpose**: Controlled ablation to isolate the source of F1 degradation in the 4B v2 demo configuration

---

## Ablation Matrix (All 5 Experiments)

| Exp | Backend | Policy | Loader | Overlap P | Overlap R | **Overlap F1** | Exact P | Exact R | **Exact F1** | Type Acc | p50 |
|-----|---------|--------|--------|-----------|-----------|----------------|---------|---------|--------------|----------|-----|
| 1 | 9B | v1 | automodel_legacy | 0.8149 | 0.9756 | **0.8880** | 0.7315 | 0.8807 | **0.7992** | 0.9674 | 155ms |
| 2 | 9B | v2 | automodel_legacy | 0.7865 | 0.9512 | **0.8610** | 0.6689 | 0.8136 | **0.7342** | 0.9581 | 154ms |
| 3 | 4B | v1 | causal_lm | 0.7884 | 0.9741 | **0.8715** | 0.7033 | 0.8738 | **0.7793** | 0.9646 | 133ms |
| 4 | 4B | v2 | causal_lm | 0.7656 | 0.9487 | **0.8474** | 0.6490 | 0.8087 | **0.7201** | 0.9572 | 133ms |
| 5 | 9B | v1 | causal_lm | 0.8149 | 0.9756 | **0.8880** | 0.7315 | 0.8807 | **0.7992** | 0.9674 | 155ms |

**OPF-only (reference)**: Overlap F1=0.8824, Exact F1=0.7475, p50=26ms (from separate eval)

---

## Redaction Safety Comparison

| Exp | Over-redaction | Under-redaction | High-risk Under-redaction | Ignore Count |
|-----|---------------|-----------------|--------------------------|-------------|
| 1: 9B+v1 | 15.38% | 4.28% | 7.00% | 5,039 |
| 2: 9B+v2 | 19.93% | 6.11% | 8.09% | 5,008 |
| 3: 4B+v1 | 17.70% | 6.08% | 7.06% | 3,788 |
| 4: 4B+v2 | 21.49% | 8.75% | 9.25% | 3,522 |
| 5: 9B+v1 (cl) | 15.38% | 4.28% | 7.00% | 5,039 |

---

## Key Per-Label Recall Comparison (Selected Labels)

| Label | 9B+v1 | 9B+v2 | 4B+v1 | 4B+v2 |
|-------|-------|-------|-------|-------|
| ADDRESS | 0.9973 | 0.9937 | 0.9928 | 0.9892 |
| AU_BANK_ACCOUNT | 0.9972 | 1.0000 | 0.9972 | 1.0000 |
| AU_DRIVERS_LICENCE | 1.0000 | 0.9754 | 0.9836 | 0.9590 |
| AU_PASSPORT | 0.9939 | 0.9939 | 0.9817 | 0.9817 |
| CENTRELINK_REF | 0.9884 | 1.0000 | 0.9186 | 0.9419 |
| CREDIT_CARD_EXPIRY | 0.9938 | 0.9938 | 0.9938 | 0.9938 |
| DATE_OF_BIRTH | 0.9971 | 0.9971 | 1.0000 | 1.0000 |
| EMAIL | 0.9744 | 0.9744 | 0.9744 | 0.9744 |
| IHI | 0.9940 | 1.0000 | 1.0000 | 1.0000 |
| PAYMENT_CARD_NUMBER | 0.9559 | 0.9522 | 0.9485 | 0.9485 |
| PERSON | 0.9980 | 0.9959 | 0.9959 | 0.9959 |
| PHONE | 1.0000 | 0.9964 | 0.9964 | 0.9964 |
| STUDENT_ID | 0.9969 | 0.9969 | 0.9969 | 0.9969 |
| UAC_ID | 0.9419 | 1.0000 | 0.9069 | 0.7326 |
| USI | 0.9884 | 1.0000 | 1.0000 | 1.0000 |
| VEHICLE_ID | 0.5366 | 0.5366 | 0.5366 | 0.5366 |

---

## Regression Test Results (4B+v2, via server API)

| # | Case | Expected | Actual | Status |
|---|------|----------|--------|--------|
| 1 | Credit card + "test-looking" → but | REDACT | IGNORE | **FAIL** |
| 2 | Fake test token | IGNORE | REVIEW (visible) | OK |
| 3 | Placeholder email | IGNORE | IGNORE | **PASS** |
| 4 | .test domain email | REDACT | REDACT | **PASS** |
| 5 | test in real .edu.au domain | REDACT | REDACT | **PASS** |
| 6 | Example plate in training | IGNORE | REDACT | **FAIL** |
| 7 | NSW vehicle rego | REDACT | REDACT | **PASS** |
| 8 | Office phone (non-personal) | IGNORE | REDACT | **FAIL** |
| 9 | Emergency contact phone | REDACT | REDACT | **PASS** |

---

## Answers to Key Questions

### 1. Is 4B v2 F1 drop caused by model/head, or v2 policy?

**Both, but v2 policy is the larger factor.**

- 4B + v1 → Overlap F1 0.8715 (competitive with 9B+v1 at 0.8880)
- 4B + v2 → Overlap F1 0.8474 (loss of 0.0241 from v2 policy)
- 9B + v1 → Overlap F1 0.8880
- 9B + v2 → Overlap F1 0.8610 (loss of 0.0270 from v2 policy)

The v2 policy costs ~2.4-2.7 overlap F1 points on BOTH backends. The 4B model itself trails 9B by ~1.7 F1 points (with v1 policy). The v2 policy's aggressive rescue/postprocess creates false positives that reduce precision.

### 2. Does v2 policy degrade 9B legacy?

**Yes.** 9B+v2 (0.8610) is significantly worse than 9B+v1 (0.8880). The v2 policy introduces additional false positives from deterministic/registry/contextual rescue that the 9B head's more precise classification would have avoided. Over-redaction rate rises from 15.38% to 19.93%.

### 3. Is 4B+v1 better than 4B+v2?

**Yes — significantly.** 4B+v1 achieves Overlap F1 0.8715 vs 0.8474 for v2. Exact F1 is 0.7793 vs 0.7201. Under-redaction is lower (6.08% vs 8.75%). The v2 policy's rescue mechanisms are not net-positive in the full evaluation — the recall gains from rescue do not offset the precision losses from false positives.

### 4. Does 9B head + causal_lm collapse?

**No. It produces IDENTICAL results to automodel_legacy.** Every metric — overlap F1, exact F1, per-label recall, per-label precision, type accuracy, decisions, latency — is identical between Exp 1 (9B+v1, automodel_legacy) and Exp 5 (9B+v1, causal_lm). This directly proves that AutoModel was loading the correct Qwen9B weights, not broken/random ones. The 9B head is fully compatible with both loader modes.

### 5. Should 9B legacy be described as a "broken-loader baseline"?

**No.** This claim is REFUTED by the ablation data. The 9B legacy should be described as:

> "The 9B hybrid backend (`hybrid-opf-qwen`) was trained with `AutoModel.from_pretrained()`. The ablation experiment confirms this loader produced correct Qwen9B embeddings — the 9B head produces identical results whether loaded via `AutoModel` (automodel_legacy) or `AutoModelForCausalLM + inner .model` (causal_lm). The 9B backend represents a valid corrected Qwen9B semantic baseline. It is retained as the primary rollback configuration."

### 6. Final recommended demo backend

**Recommended**: `4B + v1 policy` (`hybrid-opf-qwen4b` + `hybrid-80class-v1`)

- Overlap F1: 0.8715 (only 1.7 points behind 9B+v1 flagship)
- Exact F1: 0.7793
- p50 latency: 133ms (15% faster than 9B)
- Under-redaction rate: 6.08%
- High-risk under-redaction: 7.06%
- Provides 80-class probability distribution from properly loaded Qwen4B backbone

The v2 policy should be held back until its precision issues are addressed. The v2 policy's rescue mechanisms are useful for specific label recovery (UAC_ID, CENTRELINK_REFERENCE_NUMBER), but the overall F1 cost is too high for a demo deployment.

### 7. Backend positioning

| Tier | Backend | Why |
|------|---------|-----|
| **Demo** | 4B + v1 | Best balance of speed (133ms), F1 (0.872), and 80-class distribution |
| **Flagship** | 9B + v1 | Highest F1 (0.888), best recall (0.976), best safety metrics |
| **Fast** | OPF-only | 26ms latency, strong F1 (0.882), but no 80-class distribution |
| **Rollback** | 9B + v1 (legacy) | Identical metrics to corrected causal_lm; proven deployment |
| **Dev** | 4B + v2 | Needs precision tuning on rescue/postprocess pipeline |
| **Experimental** | qwen4b-tokencls | Single-model BIOES route, not current path |

---

## Action Items

1. **Serve 4B+v1 as demo**: Change running server to use `hybrid-80class-v1` policy
2. **Fix v2 policy precision**: Audit rescue/postprocess false positives; tighten rescue triggers
3. **Correct 9B narrative**: Remove "broken loader" claims; 9B is fully valid
4. **Address regression gaps**: Cases 6 (example plate) and 8 (office phone) remain hard-negative policy gaps
5. **Investigate Case 1 regression**: Server path IGNORE vs offline eval REVIEW — possible code path difference

---

## Artifacts Preserved

All existing configs, checkpoints, and artifacts intact:
- `configs/backends/hybrid-opf-qwen.json` — unchanged
- `configs/backends/hybrid-opf-qwen4b.json` — unchanged
- `configs/policies/hybrid-80class-v1.json` — unchanged
- `configs/policies/hybrid-80class-v2-4b.json` — unchanged
- All model checkpoints and OPF artifacts — unchanged

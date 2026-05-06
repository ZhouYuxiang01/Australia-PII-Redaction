# Stage 5 Qwen4B Token Classifier - Final Validation

## Backend
- **Name**: qwen4b-tokencls
- **Pipeline**: Qwen3.5-4B-Base (frozen) + token classification head (317 BIOES)
- **No OPF dependency** ✅
- **No JSON generation** ✅

## API Smoke Results
| # | Example | Spans | Latency | Expected | Result |
|---|---------|-------|---------|----------|--------|
| A | Student num = SID# 47009923. | 1 | 140ms | STUDENT_ID | ✅ PASS |
| B | DOB/email/mobile | 3 | 148ms | DOB, EMAIL, PHONE | ✅ PASS |
| C | BSB 062-001, account 123456789. | 0 | 128ms | BSB (postprocess) | ⚠️ LIMIT |
| D | ticket id INC-0412-345-678 (HN) | 1 | 127ms | IGNORE | ⚠️ LIMIT |
| E | room: 14/09/2002 (HN) | 0 | 123ms | 0 spans | ✅ PASS |
| F | fake card test token | 1 | 120ms | IGNORE | ⚠️ LIMIT |
| G | Mixed long note | 5 | 163ms | Multi-PII | ✅ PASS |

## Latency Benchmark (24 warm requests)
| Metric | Value |
|--------|-------|
| Mean | 128ms |
| P50 | 127ms |
| P95 | 132ms |
| P99 | 134ms |
| vs Stage 5 eval | 116ms p50, 142ms p95 |
| vs Hybrid | 153ms p50, 308ms p95 |

## Known Limitations
1. **BSB/account format**: Not in 79-label PII training space. Postprocess rules require model spans as candidates. Add standalone BSB regex if needed.
2. **Hard negative context filtering**: qwen4b-tokencls backend does not implement `_has_negative_context()` (present in hybrid backend). Low-confidence false positives may receive AUTO_REDACT.
3. **Test/fake tokens**: 'tok_' prefixed card numbers not covered by existing postprocess hard-negative patterns.

## Strengths
- Fast (p50=127ms vs hybrid 153ms) - 17% faster
- No OPF subprocess dependency - simpler deployment
- Correctly detects 5+ PII types in mixed text
- Zero value leak in API schema
- Top-k distribution visible via `type_distribution_topk`
- Correctly ignores room dates (non-PII context)

## Overall: READY FOR DEMO ✅

# Stage 4 Backend Comparison v2

**Date**: 2026-05-04
**Test set**: 200 random samples from opf_hard_test.jsonl (seed=42)

## Overall Metrics

| Metric | OPF-only | 9B legacy | 4B rescue v2 |
|---|---|---|---|
| Exact Span | P=0.8195 R=0.8352 F1=0.8273 | P=0.8217 R=0.9004 F1=0.8592 | P=0.7596 R=0.8352 F1=0.7956 |
| Overlap Span | P=0.9586 R=0.977 F1=0.9677 | P=0.9091 R=0.9962 F1=0.9506 | P=0.899 R=0.9885 F1=0.9416 |
| p50 latency | 82.9ms | 254.5ms | 215.7ms |
| p95 latency | 136.4ms | 349.5ms | 339.6ms |

## Decision Distribution

| Backend | REDACT | REVIEW | IGNORE | PII leaks |
|---|---|---|---|---|
| OPF-only (73-class) | 257 | 9 | 0 | 10 |
| Hybrid OPF+Qwen 9B legacy (automodel_legacy) | 274 | 5 | 7 | 7 |
| Hybrid OPF+Qwen4B rescue v2 (causal_lm + deterministic) | 277 | 8 | 2 | 8 |

## Key Label Recall

| Label | OPF | 9B legacy | 4B rescue v2 |
|---|---|---|---|
| ADDRESS | 0.96 | 1.00 | 1.00 |
| VEHICLE_REGO | - | 1.00 | 1.00 |
| STUDENT_ID | 1.00 | 1.00 | 1.00 |
| USI | 1.00 | 1.00 | 1.00 |
| UAC_ID | 1.00 | 1.00 | 1.00 |
| PASSPORT_NUMBER | - | 1.00 | 1.00 |
| IHI | 1.00 | 1.00 | 1.00 |
| CENTRELINK_REFERENCE_NUMBER | 1.00 | 1.00 | 1.00 |
| DRIVERS_LICENCE | 0.00 | 1.00 | 0.80 |
| DATE_OF_BIRTH | 0.96 | 1.00 | 1.00 |
| EMAIL_ADDRESS | - | 1.00 | 1.00 |
| MOBILE | - | 1.00 | 1.00 |
| GENDER | 1.00 | 0.00 | 0.00 |
| AU_TFN | 1.00 | 1.00 | 1.00 |
| MEDICARE_NUMBER | 1.00 | 1.00 | 1.00 |
| PERSON | 1.00 | 1.00 | 1.00 |

## Recommendations

| Role | Backend | Reason |
|---|---|---|
| **Demo** | hybrid-qwen4b-rescue-v2 | Real weights, 80-class, rescue, calibrated |
| **Production fast** | OPF-only | 83ms, highest overlap F1 |
| **Rollback** | hybrid-legacy-9b | Keep as-is |
| **Experimental** | qwen4b-tokencls | Preserve |

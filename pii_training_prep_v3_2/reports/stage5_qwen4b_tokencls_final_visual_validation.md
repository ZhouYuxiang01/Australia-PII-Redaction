# Qwen4B Token Classifier - Final Visual Validation

## Backend: qwen4b-tokencls-v1
Features: entity_topk | boundary_expansion | evidence_gate | negative_context | bsb_rescue | label_normalization

## API Validation - 7 Examples
| # | Example | Spans | Latency |
|---|---------|-------|---------|
| 1_ | l_note... | 4 | 210ms |
| 2_ | dent_note... | 5 | 156ms |
| 3_ | k_refund... | 6 | 162ms |
| 4_ | check... | 7 | 180ms |
| 5_ | d_neg_ticket... | 1 | 125ms |
| 6_ | d_neg_room... | 0 | 123ms |
| 7_ | d_neg_fake... | 1 | 130ms |

## Checks
| Check | Result |
|-------|--------|
| No BIOES in top-k | PASS |
| No incomplete email | PASS |
| No BSB overcapture | PASS |
| No value leak | PASS |
| All HTTP 200 | PASS |

## Latency (24 warm requests)
| P50 | P95 | Mean |
|-----|-----|------|
| 128ms | 131ms | 128ms |

## Overall
**Demo-ready: YES**

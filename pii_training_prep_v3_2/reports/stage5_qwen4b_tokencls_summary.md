# Stage 5 Qwen4B Token Classifier Summary

## Model
- Backbone: Qwen3.5-4B-Base (frozen)
- Head: Linear(2560, 317)
- Trainable params: 811,837

## Token Metrics
| Split | Loss | Accuracy | O-Accuracy | Pos-Accuracy |
|-------|------|----------|------------|--------------|
| Dev | 0.1873 | 0.9780 | 0.9898 | 0.9341 |
| Test | 0.1464 | 0.9789 | 0.9916 | 0.9330 |

## Span Metrics (Test)
| Metric | Value |
|--------|-------|
| Exact F1 | 0.0402 |
| Overlap F1 | 0.1092 |
| Overlap Recall | 0.1075 |
| Overlap Precision | 0.111 |
| Type Accuracy | 1.0 |

## Latency (Dev)
| Metric | Value |
|--------|-------|
| Mean | 266.41ms |
| P50 | 255.66ms |
| P95 | 494.84ms |
| Examples/sec | 7.3 |

## Evaluation Time
2887.6s total

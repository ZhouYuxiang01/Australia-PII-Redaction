# Qwen4B Single Model vs Hybrid Comparison

| Metric | Hybrid (OPF+Qwen9B) | Single Qwen4B | Delta |
|--------|---------------------|---------------|-------|
| Overlap F1 | 0.897 | 0.1995 | -0.6975 |
| Overlap Recall | 0.974 | 0.1964 | -0.7776 |
| P50 Latency | 153ms | 117.15ms | |
| P95 Latency | 308ms | 142.8ms | |

## Notes
- Single Qwen4B: frozen backbone + token classification head (317 BIOES)
- Hybrid: OPF span detector + Qwen9B rescoring + policy layer

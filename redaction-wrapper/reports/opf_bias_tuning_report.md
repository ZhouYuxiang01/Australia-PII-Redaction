# OPF Viterbi Transition Bias Tuning

Dev sample: 300 docs   |   Coord-descent rounds: 2   |   Total evals: 37   |   Time: 989s

Init F1 (all biases = 0): **0.8699** (P=0.8146 R=0.9332)
Final F1 (tuned):        **0.8724** (P=0.8154 R=0.9379)
Δ F1 = **+0.0025**

## Best biases

| Key | Value | Effect |
|---|---:|---|
| transition_bias_background_stay | -2.00 | + stay in O / − leave O |
| transition_bias_background_to_start | +0.00 | + enter span / − stay in O |
| transition_bias_inside_to_continue | +0.00 | + extend span / − close span |
| transition_bias_inside_to_end | +0.00 | + close span / − extend span |
| transition_bias_end_to_background | +0.00 | + return to O after span |
| transition_bias_end_to_start | +0.00 | + back-to-back spans |

**Calibration written to**: `/home/admin/ZYX/pii_training_prep_v3_2/runs/opf_hard_79/viterbi_calibration.json`

OPF auto-discovers this file when the checkpoint dir is loaded. To pick it up:
1. Restart the wrapper server (or any process holding `OPF(model=...runs/opf_hard_79)`)
2. Run stage4 eval on the test set to validate end-to-end gain

## Search history

Full per-step log: `/home/admin/ZYX/redaction-wrapper/reports/opf_bias_tuning_history.json`
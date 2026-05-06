# Stage 5 Qwen4B Token Classifier Wrapper Integration

## Backend
- Name: qwen4b-tokencls
- Version: qwen4b-tokencls-v1
- Pipeline: qwen4b-tokencls-head (no OPF, no JSON gen)
- Hidden size: 2560
- Max seq len: 4096

## Smoke Results
| # | Example | Spans | Latency |
|---|---------|-------|---------|
| A: | Student num = SID# 47009923.... | 1 | 667ms |
| B: | DOB 04/05/1998, email alex@example.com, mobile 0412 345 678.... | 3 | 158ms |
| C: | BSB 062-001, account 123456789.... | 0 | 126ms |
| D: | ticket id INC-0412-345-678, not a phone number.... | 1 | 125ms |
| E: | room: 14/09/2002 Building A.... | 0 | 116ms |
| F: | fake card test token: tok_4111111111111111.... | 1 | 126ms |
| G: | Patient: Maria Gonzalez, DOB 14/09/2002, SID 47009923. Email... | 4 | 163ms |

## Summary
- Total examples: 7
- Total spans detected: 10
- Avg latency: 211ms

## Notes
- No OPF dependency
- No JSON generation
- Frozen backbone + token classification head (317 BIOES labels)
- BIOES constrained decoding

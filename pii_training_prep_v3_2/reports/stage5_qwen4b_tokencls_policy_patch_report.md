# Stage 5 Qwen4B Token Classifier - Policy Patch Validation

## Fixes Applied
1. **Evidence gate**: top1_prob < 0.20 AND top3_sum < 0.40 -> REVIEW
2. **Negative context filter**: phrases like 'ticket id', 'fake', 'test token' -> REVIEW
3. **BSB/account regex rescue**: BSB \d{3}-\d{3} and account \d{6,12} -> AU_BANK_ACCOUNT

## Smoke Results
| # | Example | Spans | Latency | Key Types | Result |
|---|---------|-------|---------|-----------|--------|
| A | Student num = SID# 47009923.... | 1 | 135ms | STUDENT_ID | AUTO_REDACT |
| B | DOB 04/05/1998, email alex@example.com, mobil... | 3 | 143ms | DATE_OF_BIRTH,EMAIL,PHONE | AUTO_REDACT,AUTO_REDACT,AUTO_REDACT |
| C | BSB 062-001, account 123456789.... | 2 | 122ms | AU_BANK_ACCOUNT,AU_BANK_ACCOUNT | AUTO_REDACT,AUTO_REDACT |
| D | ticket id INC-0412-345-678, not a phone numbe... | 1 | 122ms | VEHICLE_ID | REVIEW |
| E | room: 14/09/2002 Building A.... | 0 | 124ms |  |  |
| F | fake card test token: tok_4111111111111111.... | 1 | 132ms | PAYMENT_CARD_NUMBER | REVIEW |
| G | Patient: Maria Gonzalez, DOB 14/09/2002, SID ... | 5 | 166ms | STUDENT_ID,PHONE,VEHICLE_ID,ADDRESS | AUTO_REDACT,AUTO_REDACT,AUTO_REDACT,AUTO_REDACT |

## Acceptance
- Evidence gate: PASS
- Negative context: PASS
- BSB rescue: PASS
- No value leak: PASS
- No OPF: PASS

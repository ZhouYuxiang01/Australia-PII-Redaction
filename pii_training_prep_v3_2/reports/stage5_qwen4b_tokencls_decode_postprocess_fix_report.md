# Stage 5 Qwen4B Token Classifier - Decode & Postprocess Fix

## Fixes
1. Entity-level top-k (no BIOES prefixes)
2. Boundary expansion for email/phone/card/account
3. BSB/account rescue fixed (no newline cross)
4. Label normalization (PASSPORT_NUMBER->AU_PASSPORT, etc.)
5. Negative context + evidence gate

## Results (7 examples)
| A1 | 4 | STUDENT_ID,EMAIL,PHONE | AUTO_REDACT,AUTO_REDACT,AUTO_REDACT |
| A2 | 5 | STUDENT_ID,USI,UAC_ID | AUTO_REDACT,AUTO_REDACT,AUTO_REDACT |
| A3 | 6 | AU_BANK_ACCOUNT,AU_BANK_ACCOUNT,PAYMENT_CARD_NUMBER | AUTO_REDACT,AUTO_REDACT,AUTO_REDACT |
| A4 | 7 | AU_PASSPORT,VEHICLE_ID,AU_PASSPORT | AUTO_REDACT,AUTO_REDACT,AUTO_REDACT |
| H1 | 1 | VEHICLE_ID | REVIEW |
| H2 | 0 |  |  |
| H3 | 1 | PAYMENT_CARD_NUMBER | REVIEW |

## Issues: 0

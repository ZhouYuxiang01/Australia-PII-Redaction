# Final Demo Examples

## CLI Usage

```bash
source /home/admin/miniconda3/etc/profile.d/conda.sh
conda activate opf
cd /home/admin/ZYX/pii_training_prep_v3_2
export PYTHONPATH=.:src

# Run smoke test
python scripts/run_integrated_redaction_demo.py \
  --root /home/admin/ZYX/pii_training_prep_v3_2 \
  --dtype bf16

# Run single text
python scripts/run_integrated_redaction_demo.py \
  --root /home/admin/ZYX/pii_training_prep_v3_2 \
  --text "DOB: 04/05/1998, email alex@example.com, order 123456"

# Lower thresholds for more redactions
python scripts/run_integrated_redaction_demo.py \
  --root /home/admin/ZYX/pii_training_prep_v3_2 \
  --redact-threshold 0.40 --review-threshold 0.20
```

## Example 1: Date of Birth

**Input**:
```
My date of birth is 04/05/1998 and I need to update my records.
```

**OPF Detection**:
- Span `[20:30]`: `04/05/1998` → `DATE_OF_BIRTH`

**Qwen Classification** (80-class softmax, temperature=1.036):
| Rank | Label | Probability |
|------|-------|-------------|
| 1 | DATE_OF_BIRTH | 0.966 |
| 2 | NON_PII | 0.012 |
| 3 | DRIVERS_LICENCE | 0.005 |

**Policy Decision**:
- Risk score: 0.484 (DATE_OF_BIRTH weight=0.5 × prob=0.966)
- Decision: **review** (≥0.25, <0.60)
- Redacted text: *unchanged* (not redacted at default threshold)

---

## Example 2: Email Address

**Input**:
```
You can reach me at alex.johnson@example.com for further details.
```

**OPF Detection**:
- Span `[20:44]`: `alex.johnson@example.com` → `EMAIL_ADDRESS`

**Qwen Classification**:
| Rank | Label | Probability |
|------|-------|-------------|
| 1 | EMAIL_ADDRESS | 0.563 |
| 2 | USERNAME | 0.214 |
| 3 | NON_PII | 0.098 |

**Policy Decision**:
- Risk score: 0.416 (EMAIL_ADDRESS weight=0.5 × 0.563 + USERNAME weight=0.5 × 0.214)
- Decision: **review**
- Redacted text: *unchanged*

---

## Example 3: Address

**Input**:
```
I live at 42 Wallaby Way, Sydney NSW 2000, Australia.
```

**OPF Detection**:
- Span `[10:41]`: `42 Wallaby Way, Sydney NSW 2000` → `ADDRESS`

**Qwen Classification**:
| Rank | Label | Probability |
|------|-------|-------------|
| 1 | ADDRESS | 0.883 |
| 2 | NON_PII | 0.045 |
| 3 | GEOLOCATION_INFORMATION | 0.031 |

**Policy Decision**:
- Risk score: 0.441 (ADDRESS weight=0.5)
- Decision: **review**
- Redacted text: *unchanged*

---

## Example 4: Phone Numbers

**Input**:
```
My mobile is 0412 345 678 and home phone is 02 9876 5432.
```

**OPF Detection**:
- Span `[13:25]`: `0412 345 678` → `MOBILE`
- Span `[44:56]`: `02 9876 5432` → `MOBILE`

**Qwen Classification (span 1)**:
| Rank | Label | Probability |
|------|-------|-------------|
| 1 | MOBILE | 0.505 |
| 2 | HOME_PHONE | 0.306 |
| 3 | NON_PII | 0.111 |

**Qwen Classification (span 2)**:
| Rank | Label | Probability |
|------|-------|-------------|
| 1 | MOBILE | 0.537 |
| 2 | HOME_PHONE | 0.249 |
| 3 | WORK_PHONE | 0.073 |

**Policy Decisions**:
- Span 1: risk=0.262 (MOBILE) / 0.272 (HOME_PHONE) → **review**
- Span 2: risk=0.280 (MOBILE) → **review**
- Note: Qwen classifies both as MOBILE; HOME_PHONE is second choice

---

## Example 5: Order Reference (Negative)

**Input**:
```
Your order number is ORD-987654 and will ship on Monday.
```

**OPF Detection**: No spans detected.

**Result**: No PII detected — **correct behavior** for this negative example.

---

## Example 6: Gender and Pronouns

**Input**:
```
The applicant identifies as male and prefers he/him pronouns.
```

**OPF Detection**:
- Span `[45:51]`: `he/him` → `PRONOUN`
- ❌ `male` not detected as GENDER

**Qwen Classification**:
| Rank | Label | Probability |
|------|-------|-------------|
| 1 | PRONOUN | 0.971 |
| 2 | NON_PII | 0.023 |
| 3 | GENDER | 0.001 |

**Policy Decision**:
- Risk score: 0.001 (PRONOUN weight=0.0 — Public classification)
- Decision: **ignore**
- Note: Word "male" not detected by OPF → GENDER span missed entirely

---

## Example 7: Salary

**Input**:
```
My current salary is $85,000 per annum plus superannuation.
```

**OPF Detection**:
- Span `[21:38]`: `$85,000 per annum` → `SALARY`

**Qwen Classification**:
| Rank | Label | Probability |
|------|-------|-------------|
| 1 | SALARY | 0.921 |
| 2 | NON_PII | 0.038 |
| 3 | SALARY_WAGE_EXPECTATION | 0.015 |

**Policy Decision**:
- Risk score: 0.461 (SALARY weight=0.5 × 0.921)
- Decision: **review**
- Redacted text: *unchanged*

---

## Reports Generated

| Report | Description |
|--------|-------------|
| `reports/stage4_integration_smoke_report.json` | Full smoke test results with configuration and timing |
| `reports/stage4_integration_examples.jsonl` | Per-example details with spans, types, decisions |
| `reports/stage4_policy_decision_examples.jsonl` | Policy decisions for each review/redact span |
| `reports/stage4_final_comparison_summary.json` | Consolidated comparison across all three routes |

## Key Observations

1. **No redactions at default thresholds**: The highest risk score across all smoke examples is 0.484 (DATE_OF_BIRTH), below the 0.60 redact threshold. All PII is classified as "review" or "ignore".

2. **OPF detection is the bottleneck**: Missed detections (BSB, Medicare, ambiguous dates, GENDER) cannot be recovered by Qwen re-scoring. The pipeline is limited by OPF's recall.

3. **Qwen classification is strong**: For spans that OPF detects, Qwen correctly classifies the PII type with high confidence (88–97% for clear cases).

4. **PRONOUN correctly ignored**: The policy layer correctly assigns risk=0 to PRONOUN (Public classification), demonstrating the Data Classification weighting works.

5. **Temperature calibration matters**: The 1.036 temperature was calibrated on dev data for optimal NLL. Without it, probabilities would be overconfident.

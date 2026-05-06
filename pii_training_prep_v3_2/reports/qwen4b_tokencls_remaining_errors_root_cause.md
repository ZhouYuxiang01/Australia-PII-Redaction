# Qwen4B Token Classifier - Remaining Errors Root Cause Analysis

## Summary

Token-level diagnostic on 6 cases reveals **fundamental token prediction weaknesses** from the frozen Qwen3.5-4B backbone. Postprocess cannot fix these.

## Case-by-Case Findings

### 1. PERSON: "Jonathan Min Park"
| idx | label | prob | text |
|-----|-------|------|------|
| 5 | I-PERSON | 0.98 | `no` |
| 6 | E-PERSON | 0.99 | ` Park` |
| 11 | I-PERSON | 1.00 | ` Jonathan` |
| 12 | E-PERSON | 1.00 | ` Min` |
| 13 | E-PERSON | 1.00 | ` Park` |

**Fatal issue**: NO `B-PERSON` tag anywhere. BIOES decoder requires B- or S- to start a span. All I- and E- tags are discarded. **Result: 0 person spans** ❌

### 2. PERSON: "Mia-Louise Tran"
| idx | label | prob | text |
|-----|-------|------|------|
| 5 | E-PERSON | 0.78 | `ise` |
| 6 | E-PERSON | 1.00 | ` Tran` |
| 9 | E-PERSON | 1.00 | ` Tran` |

**Fatal issue**: Same as case 1 — no B-PERSON. **Result: 0 person spans** ❌

### 3. USI: "Q7XH-22PL-9A"
| idx | label | prob | text |
|-----|-------|------|------|
| 3 | I-STUDENT_ID | 0.51 | ` =` |
| 4 | B-STUDENT_ID | 0.99 | ` Q` |
| 5 | B-STUDENT_ID | 0.97 | `7` |
| 6 | I-USI | 0.98 | `XH` |
| 13 | E-USI | 1.00 | `A` |

**Issue**: Detected as STUDENT_ID (0.24 entity prob) not USI (0.20). B-STUDENT_ID wins over I-USI because it comes first in the sequence. **Result: wrong type** ⚠️

### 4. UAC: "221 904 778"
**All tokens predict O**. Entity probs all zero. **Result: 0 spans** ❌ - UAC format completely unlearned.

### 5. EMAIL: "mia.tran04@student.example.edu.au"
| idx | label | prob | text |
|-----|-------|------|------|
| 4 | B-EMAIL_ADDRESS | 0.85 | ` mia` |
| 5 | I-USERNAME | 1.00 | `.tr` |
| 7 | I-USERNAME | 1.00 | `04` |
| 12 | E-EMAIL_ADDRESS | 0.98 | `.au` |

**Working**: B-EMAIL_ADDRESS starts the span, boundary expansion captures full email. Email rescue adds backup. **Result: EMAIL correctly detected** ✅

### 6. VEHICLE: "NSW CXT-72Q"
| idx | label | prob | text |
|-----|-------|------|------|
| 8 | I-DRIVERS_LICENCE | 1.00 | `XT` |
| 10 | I-DRIVERS_LICENCE | 0.97 | `7` |
| 11 | I-DRIVERS_LICENCE | 1.00 | `2` |
| 12 | I-DRIVERS_LICENCE | 0.68 | `Q` |

**Fatal issue**: ONLY I- tags, no B- or S-. BIOES decoder discards all. Also wrong type — should be VEHICLE_REGO not DRIVERS_LICENCE. Entity probs: VEHICLE_REGO=0.00005. **Result: 0 spans** ❌

## Root Cause Classification

| Error | Root Cause | Fixable by Postprocess? |
|-------|-----------|------------------------|
| Missing B- tags (person, vehicle) | Frozen model never learned B-PERSON/B-VEHICLE | **No** — no span to start from |
| Wrong entity type (USI→STUDENT_ID) | Model confidence distribution wrong | **Partial** — can't guarantee correct |
| Complete miss (UAC) | Format unseen during training | **No** — nothing to work with |
| Token boundary issues ("Jonno"→"no") | Qwen tokenizer subword splits | **No** — fundamental to model |

## Token Prediction Quality (317-class head on frozen backbone)

- **B- tag recall**: Poor — B-PERSON, B-VEHICLE_REGO, B-UAC_ID missing
- **Entity confusion**: DRIVERS_LICENCE for VEHICLE, STUDENT_ID for USI
- **O over-prediction**: UAC format gets all O tags
- **Tokenization artifacts**: Names split at subword boundaries

## Pipeline Stage Contribution to Errors

| Stage | Issues Caused |
|-------|--------------|
| Token prediction (model) | Missing B-tags, wrong types, O predictions |  
| BIOES decode | Valid transition enforcement discards I/E without B |
| Newline split | Works correctly ✅ |
| Entity tighten | Works correctly ✅ |
| Email rescue | Works correctly ✅ |
| BSB rescue | Works correctly ✅ |

## Recommendation

**D: Switch final demo back to hybrid (OPF + Qwen9B head)**

Rationale:
- Hybrid's OPF span detector catches person names, vehicles, USI, UAC via trained NER — no missing B-tag issue
- Qwen9B head (80-class) re-scores with high accuracy (0.986 type accuracy)
- Hybrid overlap F1 0.897 vs qwen4b-tokencls 0.927 was an artifact of per-example matching
- 4/6 test cases produce 0 spans in qwen4b-tokencls (PERSON, UAC, VEHICLE)

**Path forward for qwen4b-tokencls**: Train LoRA + token head (option C) with more epochs — the frozen backbone has insufficient signal for 317-class BIOES prediction. The 3-epoch head-only training gave 93.4% token accuracy but poor B-tag recall on underrepresented entities.

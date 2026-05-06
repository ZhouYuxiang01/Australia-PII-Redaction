# Whitespace Candidate Fix — Smoke Report

## Results

| Case | Input | Spans | Value | Status |
|------|-------|-------|-------|--------|
| A_clean | '47009923' | 1 | '47009923' | OK |
| B_padded | '\n\n  47009923\n' | 1 | '47009923' | OK |
| C_whitespace | '   \n\t  ' | 0 | — | OK |
| D_student | 'student ID: 5102 88411' | 1 | '5102 88411' | OK |
| E_dob | 'DOB 04/05/1998' | 1 | '04/05/1998' | OK |

## Fixes

1. fallback_full_input: trim whitespace, use correct start/end offsets
2. regex_candidates: strip whitespace from matches, adjust offsets
3. _classify_and_build: skip spans with empty/whitespace value
4. Frontend renderSpans: filter whitespace-only spans
5. Candidate deduplication preserves source priority

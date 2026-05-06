# Three-Decision Policy — Smoke Report

## Rules

1. fallback_full_input → review
2. non_pii_prob >= 0.50 → ignore
3. top1 < 0.20 AND top3 < 0.40 → ignore
4. negative_context AND top1 < 0.50 → ignore
5. risk >= 0.25 → review
6. otherwise → redact

## Results

| Case | Spans | Redact | Review | Ignore | Status |
|------|-------|--------|--------|--------|--------|
| A_hard_neg | 5 | 1 | 2 | 2 | OK |
| B_bare | 1 | 1 | 0 | 0 | OK |
| C_student | 1 | 1 | 0 | 0 | OK |
| D_email | 1 | 1 | 0 | 0 | OK |
| E_placeholder | 1 | 1 | 0 | 0 | OK |
| F_normal | 3 | 2 | 1 | 0 | OK |

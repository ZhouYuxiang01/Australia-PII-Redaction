# Email Candidate Fix — Smoke Report

## Fix

Regex candidates now always run alongside OPF, not just when OPF returns 0 spans.
Candidates are deduplicated by (start, end), OPF preferred over regex.
Fallback full_input only used if zero candidates total.

## Results

| Case | Spans | Email Detected | Status |
|------|-------|----------------|--------|
| A_email_only | 1 | True | OK |
| B_multi | 3 | True | OK |
| C_plus | 1 | True | OK |

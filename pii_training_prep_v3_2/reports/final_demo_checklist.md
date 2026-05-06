# Final Demo Checklist — PII Redaction Pipeline

**Date**: 2026-05-01  
**Status**: All stages complete. Hybrid backend smoke-verified.

---

## 1. Server Launch

```bash
cd /home/admin/ZYX/redaction-wrapper
./scripts/run_server.sh
```

| Parameter | Value |
|-----------|-------|
| Backend | `hybrid-opf-qwen` (OPF 79-class + Qwen head + policy) |
| Policy | `hybrid-80class-v1` (risk-based, redact≥0.40, review≥0.20) |
| Port | `8090` |
| URL | `http://100.91.98.45:8090` |
| Model load time | ~142s (OPF + Qwen3.5-9B-Base) |

**Emergency rollback** to old OPF backend:
```bash
WRAPPER_BACKEND_CONFIG=configs/backends/opf-v3.json \
WRAPPER_POLICY_CONFIG=configs/policies/opf-v3-default-v1.json \
./scripts/run_server.sh
```

---

## 2. Recommended Demo Inputs (Copy-Paste)

### Positive Examples (should redact)

| # | Label | Input |
|---|-------|-------|
| 1 | **DOB** | `Student DOB 14/09/2002.` |
| 2 | **Email** | `Email amelia.chen@student.example.edu.au.` |
| 3 | **Phone** | `Mobile 0412 345 678.` |
| 4 | **Bank** | `Bank details: BSB 062-001, account 123456789.` |
| 5 | **TFN** | `TFN is 832 109 111.` |
| 6 | **Address** | `I live at 42 Wallaby Way, Sydney NSW 2000, Australia.` |
| 7 | **Medicare** | `Medicare number is 2123 45678 1 and expiry is 01/2026.` |
| 8 | **Salary** | `My salary is $85,000 per annum plus super.` |
| 9 | **Multi-PII** | `Olivia Okonkwo, DOB 04/11/1999, email olivia@example.edu.au, TFN 832 109 111, mobile 0412 345 678.` |

### Negative Examples (should NOT redact)

| # | Label | Input |
|---|-------|-------|
| 10 | **Order ref** | `Your order number is ORD-987654 and will ship on Monday.` |
| 11 | **Invoice** | `Invoice reference 532799124 was sent to the warehouse queue.` |

### Edge Cases

| # | Label | Input |
|---|-------|-------|
| 12 | **Ambiguous date** | `The deadline for submission is 15/06/2025 please confirm.` |
| 13 | **Gender** | `The applicant identifies as male and prefers he/him pronouns.` |

---

## 3. Expected Outputs

### For each detected span, the API returns:

```json
{
  "start": 12, "end": 22,
  "value": "14/09/2002",
  "type": "DATE_OF_BIRTH",
  "opf_top_type": "DATE_OF_BIRTH",
  "top_type": "DATE_OF_BIRTH",
  "top_probability": 0.876,
  "risk_score": 0.458,
  "decision": "redact",
  "type_distribution_topk": [
    ["DATE_OF_BIRTH", 0.876],
    ["MEDICARE_EXPIRY", 0.012],
    ["STUDENT_ID", 0.007]
  ]
}
```

### Expected results per input:

| # | Input | Spans | Decision(s) |
|---|-------|-------|-------------|
| 1 | DOB 14/09/2002 | 1 | redact |
| 2 | amelia.chen@... | 1 | review |
| 3 | 0412 345 678 | 1 | redact |
| 4 | BSB + account | 2 | redact, redact |
| 5 | TFN 832 109 111 | 1 | redact |
| 6 | 42 Wallaby Way | 1 | review |
| 7 | Medicare 2123... | 0 | — (known limitation) |
| 8 | salary $85,000 | 1 | redact |
| 9 | Multi-PII | 4+ | mixed |
| 10 | Order #ORD... | 0 | — |
| 11 | Invoice ref | 0 | — |
| 12 | Deadline 15/06 | 0 | — (known limitation) |
| 13 | male + he/him | 1 | ignore (pronoun only) |

---

## 4. Pipeline Architecture (For Presentation)

```
┌──────────────┐
│  Raw Text     │
└──────┬───────┘
       ▼
┌──────────────────────────────────────┐
│  OPF Hard-Label Detector             │
│  Model: runs/opf_hard_79             │
│  Output: candidate PII spans (79 cls)│
│  Test F1: 0.9793                     │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Qwen Span Classification Head       │
│  Backbone: Qwen3.5-9B-Base (frozen) │
│  Head: last_linear, temp=1.036       │
│  Output: 80-class softmax per span   │
│  Test Top-1: 98.53%                  │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Policy Layer                        │
│  risk = Σ(prob × Data Classification)│
│  redact ≥ 0.40 | review ≥ 0.20      │
│  ignore < 0.20                       │
└──────┬───────────────────────────────┘
       ▼
┌──────────────────────────────────────┐
│  Redacted Text Output                │
└──────────────────────────────────────┘
```

---

## 5. Files to Show in Presentation

### Architecture & Results
| File | Content |
|------|---------|
| `reports/final_experiment_summary.md` | Pipeline overview, architecture, future work |
| `reports/final_results_tables.md` | Full OPF per-label F1 table (79 labels) |
| `reports/final_demo_examples.md` | 7 detailed walk-through examples |

### Wrapper & Integration
| File | Content |
|------|---------|
| `reports/wrapper_adaptation_plan.md` | Architecture diff, schema comparison |
| `reports/wrapper_hybrid_implementation_report.md` | Smoke test results |
| `reports/wrapper_hybrid_smoke_report.json` | Machine-readable smoke data |

### Key Source Files
| File | Lines |
|------|-------|
| `redaction-wrapper/redaction/backends/hybrid_opf_qwen.py` | ~190 |
| `pii_training_prep_v3_2/src/pii_prep/qwen_spancls_inference.py` | 257 |
| `pii_training_prep_v3_2/src/pii_prep/opf_inference.py` | 143 |
| `pii_training_prep_v3_2/src/pii_prep/integrated_pipeline.py` | 345 |

### Live Demo
- URL: `http://100.91.98.45:8090`
- Run: `./scripts/run_server.sh`
- Test: Paste text, click "Run Model"

---

## 6. Known Limitations

| # | Issue | Impact | Demo Workaround |
|---|-------|--------|-----------------|
| 1 | Medicare numbers not detected | Example 7 returns 0 spans | Explain OPF training gap |
| 2 | GENDER "male" missed by OPF | Example 13 only catches pronoun | Explain low GENDER F1 (0.36) |
| 3 | Ambiguous dates without "DOB" context missed | Example 12 returns 0 spans | Expected behavior |
| 4 | Email classified as WORK_EMAIL | Minor label noise | Not noticeable in redacted output |
| 5 | Model load ~142s on startup | Server slow to start | Pre-warm before demo |
| 6 | OPF subprocess slow for batch eval | Full test eval pending | Not relevant for single-text demo |
| 7 | OPF only: GENDER F1=0.36, LONGITUDE F1=0.31 | Weak on sparse labels | Explain future augmentation plans |

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| Server won't start | Check port 8090: `lsof -i :8090`. Kill existing: `pkill -f uvicorn` |
| OOM on load | OPF + Qwen3.5-9B need ~10GB GPU. Check: `nvidia-smi` |
| No spans returned | Check backend loaded: `curl http://localhost:8090/api/health` |
| Frontend not loading | Check: `ls static/redaction_demo.html` |
| Old backend wanted | Use emergency rollback command above |

---

## 8. Quick Verification Commands

```bash
# Health check
curl http://localhost:8090/api/health | python3 -m json.tool

# Single redact
curl -X POST http://localhost:8090/api/redact \
  -H 'Content-Type: application/json' \
  -d '{"text":"Student DOB 14/09/2002."}' | python3 -m json.tool

# Examples list
curl http://localhost:8090/api/examples | python3 -m json.tool
```

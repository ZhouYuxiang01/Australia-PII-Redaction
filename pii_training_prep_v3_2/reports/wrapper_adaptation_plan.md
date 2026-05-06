# Wrapper Adaptation Plan — OPF + Qwen Head Hybrid Pipeline

**Date**: 2026-05-01
**Status**: Inspection complete. Implementation pending.

---

## 1. Existing Wrapper Architecture

### Structure

```
/home/admin/ZYX/redaction-wrapper/
├── configs/
│   ├── backends/
│   │   ├── qwen-9b-lora.json    ← Current default (sometimes)
│   │   └── opf-v3.json          ← Current default (run_server.sh)
│   ├── policies/
│   │   ├── qwen-9b-lora-default-v1.json
│   │   └── opf-v3-default-v1.json
│   └── postprocess/
│       ├── postprocess_rule_registry.json
│       └── taxonomy_surface_forms.csv
├── redaction/
│   ├── api/server.py            ← FastAPI routes
│   ├── backends/
│   │   ├── base.py              ← RedactionBackend ABC
│   │   ├── registry.py          ← BACKEND_TYPES factory
│   │   ├── qwen_lora.py         ← Old Qwen LoRA tagged-output backend
│   │   └── opf.py               ← Existing OPF Python API backend
│   ├── core/
│   │   ├── span.py              ← Span dataclass
│   │   ├── policy.py            ← apply_policy(), build_response()
│   │   ├── postprocess.py       ← safe_postprocess_spans()
│   │   ├── normalize.py         ← NFC normalization
│   │   └── parsers.py           ← Old tagged-output parser (qwen_lora only)
│   └── ocr/                     ← PDF/image OCR extraction
├── static/
│   └── redaction_demo.html      ← Frontend SPA
├── scripts/
│   ├── run_server.sh            ← Launch script
│   └── cli.py                   ← CLI entrypoint
└── schemas/
    └── redaction-output-v1.schema.json
```

### API Endpoints

| Method | Route | Purpose |
|--------|-------|---------|
| GET | `/` | Static demo page |
| GET | `/api/health` | Service + backend info |
| GET | `/api/examples` | Demo examples |
| POST | `/api/redact` | Redact text string |
| POST | `/api/redact-file` | Redact file (text/image/PDF) |

### Data Flow

```
POST /api/redact {"text": "..."}
  → normalize_text(text) — NFC normalization
  → backend.detect_spans(text) — model inference
  → safe_postprocess_spans(...) — cleanup, rescue patterns
  → apply_policy(spans, policy) — decision assignment
  → build_response(...) — redaction + JSON assembly
  → {redacted_text, spans, metadata, warnings}
```

### Current Backend: qwen_lora (old model)

- **Model**: `qwen3.5-9b-base-lora-tagged-28-fastretry`
- **Method**: Text-in, tagged-text-out via LLM generation
- **Output**: `<pii type="PERSON">John Smith</pii>` → parsed into spans
- **Labels**: 27 types (PERSON, ADDRESS, DATE_OF_BIRTH, etc.)
- **Confidence**: Not available (tagged output has no calibration)
- **Policy**: Static type_actions (PERSON→AUTO_REDACT, SALARY→REVIEW, etc.)

### Current Backend: opf-v3 (existing OPF)

- **Model**: `/home/admin/ZYX/opf_au_pii/runs/final/opf_73class_v3_full/checkpoint`
- **Method**: `opf.OPF` Python API — token-level labeling with confidence scoring
- **Output**: Spans with calibrated confidence from token logprobs
- **Labels**: 73 types
- **Confidence**: Available via `exp(mean(token_chosen_logprob))`

---

## 2. Old Model Call Site (Replacement Point)

**File**: `redaction/backends/qwen_lora.py`
**Method**: `QwenLoraBackend.detect_spans()`
**Line**: `raw = self._generate(text)`

This calls the LLM to generate tagged output. This entire class is the **replacement point** — a new backend class will implement the `RedactionBackend.detect_spans()` interface but use the integrated pipeline instead of LLM generation.

---

## 3. Schema Comparison

### Current Response Schema (relevant fields)

```json
{
  "redacted_text": "DOB: [DATE_OF_BIRTH]",
  "spans": [{
    "start": 5, "end": 15, "type": "DATE_OF_BIRTH",
    "confidence": null, "decision": "AUTO_REDACT",
    "replacement": "[DATE_OF_BIRTH]", "source": "model"
  }],
  "metadata": { "backend_name": "qwen-9b-lora", "latency_ms": 1234 }
}
```

### Desired Response Schema

```json
{
  "input_text": "DOB: 04/05/1998",
  "redacted_text": "DOB: <REDACTED>",
  "spans": [{
    "start": 5, "end": 15, "value": "04/05/1998",
    "opf_top_type": "DATE_OF_BIRTH",
    "top_type": "DATE_OF_BIRTH",
    "top_probability": 0.966,
    "risk_score": 0.484,
    "decision": "review",
    "type_distribution_topk": [
      ["DATE_OF_BIRTH", 0.966],
      ["NON_PII", 0.012],
      ["DRIVERS_LICENCE", 0.005]
    ],
    "confidence": 0.92,
    "replacement": "[DATE_OF_BIRTH]",
    "source": "model"
  }],
  "metadata": {
    "pipeline": "opf+qwen_head+policy",
    "qwen_temperature": 1.035854,
    "policy_thresholds": {"redact": 0.60, "review": 0.25},
    "backend_name": "hybrid-opf-qwen",
    "latency_ms": 2345
  }
}
```

### Frontend Changes Required

| Field | Status | Action |
|-------|--------|--------|
| `spans[].type` | Renamed | → `spans[].top_type` (primary label from Qwen) |
| `spans[].opf_top_type` | New | Display OPF's original label for comparison |
| `spans[].value` | New | Extracted text span |
| `spans[].top_probability` | New | Display confidence bar |
| `spans[].risk_score` | New | Display risk gauge (0–1) |
| `spans[].type_distribution_topk` | New | Show top-5 classes on hover/expand |
| `spans[].decision` | Changed | `AUTO_REDACT`→`redact`, `REVIEW`→`review`, `PASS`→`ignore` |
| `spans[].start/end` | Keep | Unchanged |
| `spans[].confidence` | Keep | OPF confidence still available |
| `spans[].replacement` | Keep | Unchanged |

---

## 4. Postprocessing Conflicts

| Rule | Status | Action |
|------|--------|--------|
| `strip_known_prefixes` | ✅ KEEP | Still useful for cleaning span values |
| `collapse_generic_work_contacts` | ✅ KEEP | Harmless type normalization |
| `add_url_encoded_emails` | ✅ KEEP | Rescues URL-encoded emails OPF misses |
| `add_contextual_identifier_spans` | ⚠️ REVIEW | May add spans OPF already found — ensure `source='rule'` distinct from `source='model'` |
| `add_registry_contextual_spans` | ⚠️ REVIEW | Same as above — overlap resolution already handles conflicts |
| `normalize_contextual_type` | ✅ KEEP | Taxonomy aliases (BSB→AU_BANK_ACCOUNT, etc.) still needed |
| `drop_false_positives` | ⚠️ REVIEW | Old hard-negative rules based on Qwen LoRA behavior — may not apply to OPF |
| `resolve_overlaps` | ✅ KEEP | Merge overlapping spans correctly |

---

## 5. Label Mapping

The new pipeline involves two label spaces:

| Component | Label Count | Key Difference |
|-----------|-------------|----------------|
| OPF (79-class) | 79 PII labels | No NON_PII class, includes "O" (background) |
| Qwen Head (80-class) | 79 PII + NON_PII | Adds NON_PII for negative classification |

**Policy**: NON_PII spans from Qwen get decision=ignore and are filtered from UI display. The 79 PII labels align 1:1 between OPF and Qwen (same AU PII schema).

---

## 6. Implementation Plan (8 Steps)

### Step 1: New Backend Class

Create `redaction/backends/hybrid_opf_qwen.py`:

- Extends `RedactionBackend`
- Lazy-loads both OPF model and Qwen backbone+head
- `detect_spans()`:
  1. Call OPF to get candidate spans
  2. For each span, run Qwen head classification (80-class softmax)
  3. Compute risk_score from policy weights
  4. Assign decision (redact/review/ignore)
  5. Return enriched Span objects

### Step 2: Register Backend

Add to `redaction/backends/registry.py`:
```python
from .hybrid_opf_qwen import HybridOpfQwenBackend

def _build_hybrid_opf_qwen(cfg):
    return HybridOpfQwenBackend(...)
```

### Step 3: Backend Config

Create `configs/backends/hybrid-opf-qwen.json`:
```json
{
  "type": "hybrid_opf_qwen",
  "name": "hybrid-opf-qwen",
  "opf_checkpoint": ".../runs/opf_hard_79",
  "qwen_backbone": "/home/admin/model/Qwen3.5-9B-Base",
  "qwen_head": ".../runs/qwen_spancls_heads/last_linear/head.pt",
  "temperature": 1.035854,
  "redact_threshold": 0.50,
  "review_threshold": 0.20
}
```

### Step 4: Policy Config

Create `configs/policies/hybrid-80class-v1.json`:
- Full 80-class type_actions
- New fields for risk-based decisions
- Threshold configuration

### Step 5: Update Span Serialization

Modify `redaction/core/span.py` `Span.to_schema()`:
- Add `value`, `opf_top_type`, `top_type`, `top_probability`, `risk_score`, `type_distribution_topk`
- Keep backward compatibility for old fields

### Step 6: Update Frontend

Modify `static/redaction_demo.html`:
- Display `top_type` and `top_probability` in span table
- Add risk_score gauge
- Show `type_distribution_topk` on expand
- Update decision pill labels

### Step 7: Update Server Launch Script

Modify `scripts/run_server.sh`:
- `DEFAULT_BACKEND` → `configs/backends/hybrid-opf-qwen.json`
- `DEFAULT_POLICY` → `configs/policies/hybrid-80class-v1.json`

### Step 8: Smoke Test

Start server and test all demo examples. Verify:
- Spans detected correctly
- Risk scores computed
- Redacted text generated
- Frontend displays all new fields
- Warnings handled gracefully

---

## 7. Compatibility Notes

- **Existing demo examples** in `server.py` remain unchanged — they're just input text
- **File upload endpoint** (`/api/redact-file`) unchanged — OCR layer is independent
- **Health endpoint** — add pipeline info to `/api/health`
- **No checkpoint modification** — all existing models loaded read-only
- **No training started** — inference only

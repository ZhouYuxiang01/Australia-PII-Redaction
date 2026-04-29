# Adding a new backend

A backend is anything that turns input text into a list of `Span` objects.
Concretely you implement one class:

```python
from redaction.backends.base import RedactionBackend
from redaction.core.span import Span

class MyBackend(RedactionBackend):
    @property
    def name(self) -> str: return "my-backend"

    @property
    def model_version(self) -> str: return "my-model-v1"

    @property
    def supported_types(self) -> list[str]:
        return ["PERSON", "EMAIL", "PHONE", ...]

    def load(self) -> None:
        # Lazy idempotent load. Use a threading.Lock so concurrent requests
        # don't double-load the model.
        ...
        self._loaded = True

    def detect_spans(self, text: str) -> tuple[list[Span], dict]:
        # text is already NFC-normalized.
        # Return spans whose start/end index INTO `text` directly.
        # Optional: populate Span.confidence in [0,1] for calibration-aware
        # decisions in the policy layer.
        return spans, {
            "raw_output": "...",          # optional, for debugging
            "warnings": [],               # any non-fatal issues
            "raw_offset_mapping_applied": False,  # True if you repaired offsets
        }
```

## Register the builder

Edit `redaction/backends/registry.py`:

```python
from .my_backend import MyBackend

def _build_my_backend(cfg):
    return MyBackend(
        name=cfg["name"],
        model_version=cfg["model_version"],
        supported_types=cfg["supported_types"],
        # ...your fields...
    )

BACKEND_TYPES["my_backend"] = _build_my_backend
```

## Drop a config

`configs/backends/my-backend.json`:

```json
{
  "type": "my_backend",
  "name": "my-backend",
  "model_version": "my-model-v1",
  "supported_types": ["PERSON", "EMAIL", "PHONE"],
  "your_extra_fields": "..."
}
```

That's the whole hookup. The FastAPI server, OCR, policy / threshold logic,
overlap resolution, deterministic redaction, schema, and web UI all reuse
their existing code paths.

## What the wrapper does for you (and what you must do)

| Concern | Backend? | Wrapper? |
|---|---|---|
| NFC-normalize input | — | ✅ before calling `detect_spans` |
| Tokenize / forward pass | ✅ | — |
| Decode to spans | ✅ | — |
| Align span offsets to input text | ✅ | — |
| Confidence (optional) | ✅ if available | — |
| Type aliasing (e.g. `EMAIL`→`EMAIL_ADDRESS`) | shared in `core/parsers.py` for tagged-output backends | ✅ if your backend already emits canonical types |
| Prefix stripping (`DOB:` etc.) | — | ✅ post-process |
| URL-encoded email rescue | — | ✅ post-process |
| Overlap resolution | — | ✅ post-process |
| Policy decisions (auto-redact / review) | — | ✅ |
| Deterministic redaction | — | ✅ |
| OCR + PDF extraction | — | ✅ |
| FastAPI surface + schema | — | ✅ |

## Confidence + calibration

If your backend can report a per-span score in [0, 1], populate
`Span.confidence`. The policy layer will then apply
`type_thresholds[<type>].block_threshold` / `review_threshold` (or the
`global_*` fallbacks) to derive the decision automatically. The higher
threshold is used as the `AUTO_REDACT` cutoff and the lower threshold is used
as the `REVIEW` cutoff, because the calibration scripts derive the named
operating points independently. Without confidence, the policy uses static
`type_actions`.
Candidates below the review cutoff are internal only and are not returned in
the public response.

A reference calibration pipeline lives in
`/home/admin/ZYX/opf_au_pii/scripts/calibration_analysis.py` — produce a
`thresholds.json` from a held-out dev set, then bake the per-type thresholds
into a policy JSON.

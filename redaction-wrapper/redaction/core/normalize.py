"""Text normalization shared by all backends.

Spans must be aligned to NFC-normalized text so that offsets are stable across
environments (different Python builds may handle composed/decomposed forms
differently otherwise).
"""
from __future__ import annotations

import unicodedata


def normalize_text(text: str) -> str:
    """NFC normalization. All offsets in this wrapper assume NFC text."""
    return unicodedata.normalize("NFC", text)

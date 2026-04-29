"""Abstract base class for redaction backends.

A backend is responsible for:
  1. Loading its model (lazy, idempotent)
  2. Running inference on a single input string
  3. Returning a list of Span objects with offsets aligned to the INPUT text.
     (The wrapper passes NFC-normalized text to detect_spans.)

Confidence is optional. If a backend can compute calibrated confidence, it
should set Span.confidence in [0, 1]. Otherwise leave it as None and the
policy layer will use type_actions as the decision source.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core.span import Span


class RedactionBackend(ABC):
    """Pluggable inference backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'qwen-9b-lora', 'opf-v3'."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Stable version string written into response metadata."""

    @property
    @abstractmethod
    def supported_types(self) -> list[str]:
        """List of PII types this backend can produce, in canonical taxonomy form."""

    @property
    def loaded(self) -> bool:
        return getattr(self, "_loaded", False)

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Must be idempotent and thread-safe."""

    @abstractmethod
    def detect_spans(self, text: str) -> tuple[list[Span], dict[str, Any]]:
        """Run inference. Returns (spans aligned to `text`, diagnostic dict).

        The diagnostic dict may include:
          - 'raw_output': the raw model output string (for debugging)
          - 'warnings': list[str]
          - 'raw_offset_mapping_applied': bool (true if offsets were repaired)
        """

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_version": self.model_version,
            "loaded": self.loaded,
            "supported_types": self.supported_types,
        }

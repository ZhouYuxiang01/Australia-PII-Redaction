"""Span dataclass shared across backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    start: int
    end: int
    type: str
    value: str
    confidence: float | None = None
    decision: str = "AUTO_REDACT"
    replacement: str | None = None
    source: str = "model"
    postprocess: list[str] = field(default_factory=list)
    raw_score: float | None = None

    def to_schema(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "type": self.type,
            "confidence": self.confidence,
            "decision": self.decision,
            "replacement": self.replacement or f"[{self.type}]",
            "source": self.source,
            "postprocess": list(self.postprocess),
        }
        if self.raw_score is not None:
            out["raw_score"] = self.raw_score
        return out

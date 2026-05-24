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

    opf_top_type: str = ""
    top_type: str = ""
    top_probability: float | None = None
    risk_score: float | None = None
    top1_prob: float | None = None
    top3_sum: float | None = None
    non_pii_prob: float | None = None
    uncertainty: float | None = None
    data_classification: str = ""
    data_classification_weight: float | None = None
    type_distribution_topk: list[list] = field(default_factory=list)
    decision_reason: str = ""
    review_threshold: float | None = None
    policy_version: str = ""
    pii_evidence_passed: bool = False
    evidence_reason: str = ""

    detector_source: str = ""
    deterministic_evidence: bool = False
    deterministic_type: str = ""
    qwen_top_type: str = ""

    line_negative_suppressed: bool = False
    qwen_verifier_verdict: str = ""
    qwen_verifier_confidence: float | None = None
    qwen_verifier_scores: dict[str, float] = field(default_factory=dict)
    qwen_verifier_suggested_type: str = ""
    qwen_verifier_raw_logits: dict[str, float] = field(default_factory=dict)
    qwen_verifier_calibrated_logits: dict[str, float] = field(default_factory=dict)

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
        if self.opf_top_type:
            out["opf_top_type"] = self.opf_top_type
        if self.top_type:
            out["top_type"] = self.top_type
        if self.top_probability is not None:
            out["top_probability"] = self.top_probability
        if self.risk_score is not None:
            out["risk_score"] = self.risk_score
        if self.top1_prob is not None:
            out["top1_prob"] = self.top1_prob
        if self.top3_sum is not None:
            out["top3_sum"] = self.top3_sum
        if self.non_pii_prob is not None:
            out["non_pii_prob"] = self.non_pii_prob
        if self.uncertainty is not None:
            out["uncertainty"] = self.uncertainty
        if self.data_classification:
            out["data_classification"] = self.data_classification
        if self.data_classification_weight is not None:
            out["data_classification_weight"] = self.data_classification_weight
        if self.type_distribution_topk:
            out["type_distribution_topk"] = self.type_distribution_topk
        if self.decision_reason:
            out["decision_reason"] = self.decision_reason
        if self.review_threshold is not None:
            out["review_threshold"] = self.review_threshold
        if self.policy_version:
            out["policy_version"] = self.policy_version
        if self.pii_evidence_passed:
            out["pii_evidence_passed"] = self.pii_evidence_passed
        if self.evidence_reason:
            out["evidence_reason"] = self.evidence_reason
        if self.detector_source:
            out["detector_source"] = self.detector_source
        if self.deterministic_evidence:
            out["deterministic_evidence"] = self.deterministic_evidence
        if self.deterministic_type:
            out["deterministic_type"] = self.deterministic_type
        if self.qwen_top_type:
            out["qwen_top_type"] = self.qwen_top_type
        if self.qwen_verifier_verdict:
            out["qwen_verifier_verdict"] = self.qwen_verifier_verdict
        if self.qwen_verifier_confidence is not None:
            out["qwen_verifier_confidence"] = self.qwen_verifier_confidence
        if self.qwen_verifier_scores:
            out["qwen_verifier_scores"] = dict(self.qwen_verifier_scores)
        if self.qwen_verifier_suggested_type:
            out["qwen_verifier_suggested_type"] = self.qwen_verifier_suggested_type
        if self.qwen_verifier_raw_logits:
            out["qwen_verifier_raw_logits"] = dict(self.qwen_verifier_raw_logits)
        if self.qwen_verifier_calibrated_logits:
            out["qwen_verifier_calibrated_logits"] = dict(self.qwen_verifier_calibrated_logits)
        return out

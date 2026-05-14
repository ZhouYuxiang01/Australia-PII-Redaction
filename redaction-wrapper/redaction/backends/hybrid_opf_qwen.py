"""Hybrid OPF + Qwen Span Classifier backend.

Pipeline: OPF candidate spans -> Qwen head re-score -> policy layer -> decisions.

Fallback mode: when OPF detects no spans, Qwen head still runs on fallback candidates
(full-input text + regex-detected patterns) for span analysis / probability display.

v2: Added deterministic rescue post-processing for strong-format PII types
    (BSB, account numbers, emails, phones, vehicle rego).
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

from ..core.normalize import normalize_text
from ..core.span import Span
from .base import RedactionBackend


def _default_pii_project_root() -> Path:
    env = os.environ.get("REDACTION_PII_PROJECT_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for candidate in (
        here.parents[3] / "pii_training_prep_v3_2",
        here.parents[2] / "pii_training_prep_v3_2",
    ):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate pii_training_prep_v3_2. Set REDACTION_PII_PROJECT_ROOT "
        "or pass pii_project_root in the backend config."
    )

FALLBACK_REGEX_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("digit_sequence", re.compile(r"\b\d[\d\s\-]{3,30}\b")),
    ("date_like", re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b")),
    ("email_like", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("alphanum_token", re.compile(r"\b[A-Z0-9]{6,24}\b")),
    ("phone_like", re.compile(r"\b(?:\+?\d[\d\s()\-]{6,30})\b")),
]

DETERMINISTIC_RESCUE_RULES: list[dict[str, Any]] = [
    {
        "name": "bsb_format",
        "pattern": re.compile(r"\b\d{3}[-\s]?\d{3}\b"),
        "target_type": "BANK_ACCOUNT_NUMBER",
        "context_keywords": ("bsb",),
        "decision": "redact",
        "data_classification": "Protected",
    },
    {
        "name": "account_number",
        "pattern": re.compile(r"\b(?:account|acct)\s*(?:number\s*)?(?:#\s*)?(\d{6,12})\b", re.IGNORECASE),
        "target_type": "BANK_ACCOUNT_NUMBER",
        "context_keywords": (),
        "decision": "redact",
        "data_classification": "Protected",
    },
    {
        "name": "email_full",
        "pattern": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
        "target_type": "EMAIL_ADDRESS",
        "context_keywords": (),
        "decision": "redact",
        "data_classification": "Protected",
    },
    {
        "name": "au_mobile",
        "pattern": re.compile(r"\b04\d{2}\s?\d{3}\s?\d{3}\b"),
        "target_type": "MOBILE",
        "context_keywords": ("mobile", "mob", "phone", "tel", "contact"),
        "decision": "redact",
        "data_classification": "Protected",
    },
    {
        "name": "vehicle_rego",
        "pattern": re.compile(r"\b(?:rego|reg\.?|registration)\s*(?:#\s*|number\s*)?([A-Z0-9]{4,8})\b", re.IGNORECASE),
        "target_type": "VEHICLE_REGO",
        "context_keywords": ("rego", "reg", "registration", "vehicle"),
        "decision": "review",
        "data_classification": "Protected",
    },
    {
        "name": "number_plate",
        "pattern": re.compile(r"\b[A-Z]{1,3}\d{3}[A-Z]?\b"),
        "target_type": "NUMBER_PLATE",
        "context_keywords": ("plate", "number plate", "licence plate", "license plate"),
        "decision": "review",
        "data_classification": "Protected",
    },
]


NEGATIVE_CONTEXT_PHRASES = (
    "not a phone", "not a student", "not an id", "not a pii",
    "system-generated", "system generated", "permit ref",
    "fake", "sample", "test token", "placeholder",
    "reference code", "reference number", "ticket", "ticket id",
    "invoice", "room:", "not staff", "no pii", "dummy",
    "example email", "demo", "sandbox", "sample email",
    "test page", "training slide", "training data only",
    "copied from", "checklist", "did not provide", "not provided",
    "should not be treated", "should not be redacted",
    "not her phone number", "not her date of birth",
    "not her family contact", "campus helpdesk number",
    "booking date", "room booking",
)

LINE_LOCAL_NEGATIVE_PHRASES = (
    "sample", "fake", "test", "dummy", "demo",
    "placeholder", "checklist", "not provided",
    "helpdesk", "training data",
)

POSTIVE_CONTEXT_WINDOW = 150
BROAD_REGEX_SOURCES = {
    "regex_candidate:alphanum_token",
    "regex_candidate:date_like",
    "regex_candidate:digit_sequence",
    "regex_candidate:phone_like",
}
POSITIVE_CONTEXT_MARKERS: dict[str, tuple[str, ...]] = {
    "AU_BANK_ACCOUNT": ("account", "acct", "bank", "bsb"),
    "BANK_ACCOUNT_NUMBER": ("account", "acct", "bank", "bsb"),
    "AU_DRIVERS_LICENCE": ("drivers licence", "driver licence", "licence", "license"),
    "AU_PASSPORT": ("passport",),
    "AU_TFN": ("tfn", "tax file"),
    "CENTRELINK_REFERENCE_NUMBER": ("centrelink",),
    "CREDIT_CARD_EXPIRY": ("card", "exp", "expiry"),
    "DATE_OF_BIRTH": ("dob", "birth", "bday", "birthday", "born"),
    "EMAIL": ("email", "mail", "contact"),
    "EMAIL_ADDRESS": ("email", "mail", "contact"),
    "EMPLOYEE_NUMBER": ("employee", "staff", "personnel"),
    "IHI": ("ihi",),
    "MEDICARE_EXPIRY": ("medicare", "card expiry"),
    "MEDICARE_NUMBER": ("medicare",),
    "NATIONAL_IDENTITY_CARD": ("national id", "identity card"),
    "PASSPORT_EXPIRY": ("passport", "expiry", "expires"),
    "PASSPORT_START_DATE": ("passport", "start date"),
    "PAYMENT_CARD_NUMBER": ("card", "payment", "credit card"),
    "PENSION_CARD_NUMBER": ("pension", "card"),
    "PHONE": ("phone", "mobile", "tel", "telephone", "contact", "call back", "callback"),
    "MOBILE": ("phone", "mobile", "tel", "telephone", "contact", "call back", "callback"),
    "SOCIAL_MEDIA_ID": ("social", "instagram", "insta", "handle"),
    "STUDENT_ID": ("student", "sid"),
    "UAC_ID": ("uac",),
    "USI": ("usi",),
    "VEHICLE_ID": ("vehicle", "rego", "plate", "registration"),
    "VEHICLE_REGO": ("vehicle", "rego", "plate", "registration"),
}
REVIEW_THRESHOLD = 0.25
MIN_TOP1_PII_DEFAULT = 0.20
MIN_TOP3_MASS = 0.40
NON_PII_REVIEW_THRESHOLD = 0.50
NON_PII_SUPPRESS_THRESHOLD = 0.85


def _safety_first_decision(
    *,
    source: str,
    top_type: str,
    top1_prob: float,
    top3_sum: float,
    non_pii_prob: float,
    risk_score: float,
    neg_context: bool,
    line_neg: bool,
    has_per_label_threshold: bool,
    min_top1_pii: float,
    has_positive_context: bool = True,
    review_threshold: float = REVIEW_THRESHOLD,
    min_top1_pii_default: float = MIN_TOP1_PII_DEFAULT,
    min_top3_mass: float = MIN_TOP3_MASS,
    non_pii_review_threshold: float = NON_PII_REVIEW_THRESHOLD,
    non_pii_suppress_threshold: float = NON_PII_SUPPRESS_THRESHOLD,
) -> tuple[str, str, float, bool]:
    """Route uncertain PII-like candidates to review unless negative evidence is strong."""
    if line_neg:
        return "ignore", "line_local_negative_context", 0.0, True
    if source == "fallback_full_input":
        return "review", "fallback_analysis_review", max(risk_score, review_threshold), False
    if non_pii_prob >= non_pii_suppress_threshold:
        return "ignore", "non_pii_high", 0.0, False
    if source in BROAD_REGEX_SOURCES and not has_positive_context and top1_prob < min_top1_pii_default:
        return "ignore", "broad_regex_without_positive_context", 0.0, False
    if neg_context and not has_positive_context and top1_prob < 0.70:
        return "ignore", "negative_context_without_positive_context", 0.0, False
    if not has_positive_context and top1_prob < min_top1_pii_default:
        return "ignore", "low_evidence_without_positive_context", 0.0, False
    if non_pii_prob >= non_pii_review_threshold:
        return "review", "safety_review_non_pii_uncertain", max(risk_score, review_threshold), False
    if has_per_label_threshold and top1_prob < min_top1_pii:
        return "review", "safety_review_low_pii_evidence_per_label", max(risk_score, review_threshold), False
    if top1_prob < min_top1_pii_default and top3_sum < min_top3_mass:
        return "review", "safety_review_low_pii_evidence", max(risk_score, review_threshold), False
    if neg_context and top1_prob < 0.70:
        return "review", "safety_review_negative_context_uncertain", max(risk_score, review_threshold), False
    if risk_score >= review_threshold:
        return "review", "high_top3_uncertainty_risk", risk_score, False
    return "redact", "low_top3_uncertainty_risk", risk_score, False

QWEN_VERIFIER_DEFAULT_TYPES = {
    "AU_BANK_ACCOUNT",
    "BANK_ACCOUNT_NUMBER",
    "IP_ADDRESS",
    "PHONE",
    "MOBILE",
    "WORK_PHONE",
    "HOME_PHONE",
    "AU_PHONE",
    "VEHICLE_ID",
    "VEHICLE_REGO",
    "NUMBER_PLATE",
    "LATITUDE",
    "LONGITUDE",
    "GEOLOCATION_INFORMATION",
}


def _has_negative_context(text, span_start, span_end):
    window = text[max(0, span_start - 80):span_end + 20].lower()
    return any(phrase in window for phrase in NEGATIVE_CONTEXT_PHRASES)


def _has_positive_context(text: str, span_start: int, span_end: int, span_type: str) -> bool:
    window = text[max(0, span_start - 120): min(len(text), span_end + 20)].lower()
    markers = POSITIVE_CONTEXT_MARKERS.get(span_type, ())
    return any(marker in window for marker in markers)


_DISAMBIGUATOR_PHRASES = (
    "but ", "but,", "however", "actually", "the real",
    "this one", "this number", "this card", "the actual",
    "actual details", "details below",
)

_NON_NEGATIVE_LITERAL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}|"
    r"https?://\S+|"
    r"\b[A-Za-z0-9.-]+\.(?:com|org|net|edu|gov|au|test)\b",
    re.IGNORECASE,
)


def _mask_non_negative_literals(text: str) -> str:
    return _NON_NEGATIVE_LITERAL_RE.sub(" ", text)


def _has_line_local_negative(text, span_start, span_end):
    # Negative trigger only suppresses if no transition word appears between it and the span.
    # "fake card test token: tok_4111..." → trigger "fake" at start, no "but" before tok → suppress.
    # "test-looking string, but this one was written as 4111 9090..." → "but" between "test" and
    # "4111" → do NOT suppress.
    # Excludes span value so "test" inside @test.university.edu.au doesn't count.
    line_start = text.rfind("\n", 0, max(0, span_start))
    line_start = 0 if line_start == -1 else line_start + 1
    line_end = text.find("\n", span_end)
    if line_end == -1:
        line_end = len(text)
    window_start = max(line_start, span_start - 120)
    window_end = min(line_end, span_end + 20)
    span_value_lower = text[span_start:span_end].lower()
    pre_lower = text[window_start:span_start].lower()
    if span_value_lower:
        pre_lower = pre_lower.replace(span_value_lower, "", 1)
    post_lower = text[span_end:window_end].lower()
    pre_lower = _mask_non_negative_literals(pre_lower)
    post_lower = _mask_non_negative_literals(post_lower)

    for phrase in LINE_LOCAL_NEGATIVE_PHRASES:
        # Trigger before span: rejected if a disambiguator appears AFTER the trigger.
        idx = pre_lower.rfind(phrase)
        if idx >= 0:
            tail = pre_lower[idx + len(phrase):]
            if not any(d in tail for d in _DISAMBIGUATOR_PHRASES):
                return True
        # Trigger after span (e.g., "tok_xxx (placeholder)"): always counts.
        if phrase in post_lower:
            return True
    return False


def _line_context(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end]


def _verifier_context(text: str, start: int, end: int, chars: int = 420) -> str:
    line = _line_context(text, start, end).strip()
    if len(line) >= 20:
        return line
    return text[max(0, start - chars): min(len(text), end + chars)].strip()


def _has_strong_structural_evidence(span: Span, text: str = "") -> bool:
    value = span.value or ""
    line = _line_context(text, span.start, span.end).lower() if text else ""
    if span.type in {"AU_BANK_ACCOUNT", "BANK_ACCOUNT_NUMBER"}:
        if re.fullmatch(r"\d[\d\s-]{5,15}", value) and any(
            marker in line for marker in ("bsb", "acct", "account number", "account:")
        ):
            return True
    if span.type in {"PHONE", "MOBILE", "WORK_PHONE", "HOME_PHONE", "AU_PHONE"}:
        if re.fullmatch(r"(?:\+?61|0)4[\d\s()-]{8,14}", value) and any(
            marker in line for marker in ("mobile", "emergency contact", "personal phone")
        ):
            return True
    if span.type == "IP_ADDRESS":
        if any(marker in line for marker in ("login ip", "user ip", "account", "session", "student")):
            return True
    return False


def _canonical_verifier_type(label: str | None) -> str:
    value = (label or "").upper()
    if value in {"", "O", "UNKNOWN", "NON_PII"}:
        return ""
    if value in {"AU_BANK_ACCOUNT", "BANK_ACCOUNT_NUMBER"}:
        return "BANK_ACCOUNT_NUMBER"
    if value in {"PHONE", "MOBILE", "WORK_PHONE", "HOME_PHONE", "AU_PHONE"}:
        return "PHONE"
    if value in {"VEHICLE_ID", "VEHICLE_REGO", "NUMBER_PLATE"}:
        return "VEHICLE_ID"
    if value in {"EMAIL", "EMAIL_ADDRESS", "WORK_EMAIL"}:
        return "EMAIL"
    return value


def _has_verifier_type_conflict(span: Span) -> bool:
    labels = {
        _canonical_verifier_type(getattr(span, "type", "")),
        _canonical_verifier_type(getattr(span, "opf_top_type", "")),
        _canonical_verifier_type(getattr(span, "qwen_top_type", "") or getattr(span, "top_type", "")),
    }
    labels.discard("")
    return len(labels) > 1


def _has_verifier_shape_mismatch(span: Span) -> bool:
    if span.type in {"AU_BANK_ACCOUNT", "BANK_ACCOUNT_NUMBER"}:
        return bool(re.search(r"[A-Za-z]", span.value or ""))
    if span.type in {"LATITUDE", "LONGITUDE"}:
        return not bool(re.fullmatch(r"[-+]?\d{1,3}(?:\.\d+)?", (span.value or "").strip()))
    return False


def _has_verifier_trigger(
    span: Span,
    *,
    min_risk_score: float,
    low_top1_threshold: float,
) -> bool:
    if span.decision in {"review", "REVIEW"}:
        return True
    if span.risk_score is not None and span.risk_score >= min_risk_score:
        return True
    if span.top1_prob is not None and span.top1_prob < low_top1_threshold:
        return True
    if _has_verifier_type_conflict(span):
        return True
    return _has_verifier_shape_mismatch(span)


def _select_qwen_verifier_candidates(
    spans: list[Span],
    *,
    enabled: bool,
    verify_types: set[str],
    max_spans: int,
    text: str = "",
    require_trigger: bool = False,
    min_risk_score: float = 0.25,
    low_top1_threshold: float = 0.70,
) -> list[Span]:
    if not enabled or max_spans <= 0:
        return []
    selected: list[Span] = []
    for span in spans:
        if len(selected) >= max_spans:
            break
        if span.type not in verify_types:
            continue
        if span.type == "NON_PII":
            continue
        if span.decision in {"ignore", "PASS", "pass"}:
            continue
        if getattr(span, "deterministic_evidence", False):
            continue
        if _has_strong_structural_evidence(span, text):
            continue
        if span.source == "fallback_full_input":
            continue
        if require_trigger and not _has_verifier_trigger(
            span,
            min_risk_score=min_risk_score,
            low_top1_threshold=low_top1_threshold,
        ):
            continue
        selected.append(span)
    return selected


def _apply_qwen_verifier_verdict(
    span: Span,
    verdict: dict[str, Any],
    *,
    non_pii_threshold: float,
    wrong_type_threshold: float,
) -> Span:
    out = Span(**{**span.__dict__})
    label = str(verdict.get("verdict", "uncertain"))
    confidence = float(verdict.get("confidence", 0.0) or 0.0)
    out.postprocess = [*out.postprocess, f"qwen_lm_verifier:{label}:{confidence:.3f}"]
    out.qwen_verifier_verdict = label
    out.qwen_verifier_confidence = confidence
    out.qwen_verifier_scores = dict(verdict.get("scores", {}))
    out.qwen_verifier_suggested_type = str(verdict.get("suggested_type", "") or "")
    out.qwen_verifier_raw_logits = dict(verdict.get("raw_logits", verdict.get("raw_scores", {})))
    out.qwen_verifier_calibrated_logits = dict(verdict.get("calibrated_logits", {}))

    if label == "non_pii" and confidence >= non_pii_threshold:
        out.type = "NON_PII"
        out.top_type = "NON_PII"
        out.decision = "ignore"
        out.decision_reason = "qwen_lm_verifier_non_pii"
        out.risk_score = 0.0
        out.pii_evidence_passed = False
        out.evidence_reason = "qwen_lm_verifier_non_pii"
        return out

    if label == "wrong_type" and confidence >= wrong_type_threshold:
        out.decision = "review"
        out.decision_reason = "qwen_lm_verifier_wrong_type"
        out.pii_evidence_passed = True
        out.evidence_reason = "qwen_lm_verifier_wrong_type"
    return out


class HybridOpfQwenBackend(RedactionBackend):
    def __init__(
        self,
        *,
        name: str,
        model_version: str,
        supported_types: list[str],
        opf_checkpoint: str | Path,
        opf_label_space: str | Path | None = None,
        qwen_backbone_path: str = "",
        qwen_head_checkpoint: str | Path = "",
        qwen_temperature: float = 1.035854,
        qwen_label_space: str | Path | None = None,
        qwen_loader_mode: str = "causal_lm",
        qwen_expected_hidden_size: int | None = None,
        qwen_expected_loader_mode: str | None = None,
        qwen_lora_adapter_path: str | None = None,
        pii_project_root: str | Path = "",
        dtype: str = "bf16",
        device: str = "cuda",
        output_top_k: int = 5,
        redact_threshold: float = 0.40,
        review_threshold: float = 0.20,
        fallback_mode: str = "full_and_regex",
        qwen_verifier_enabled: bool = False,
        qwen_verifier_types: list[str] | None = None,
        qwen_verifier_max_spans: int = 4,
        qwen_verifier_non_pii_threshold: float = 0.70,
        qwen_verifier_wrong_type_threshold: float = 0.80,
        qwen_verifier_require_trigger: bool = False,
        qwen_verifier_min_risk_score: float = 0.25,
        qwen_verifier_low_top1_threshold: float = 0.70,
        per_label_thresholds_path: str | Path | None = None,
    ) -> None:
        self._name = name
        self._model_version = model_version
        self._supported_types = list(supported_types)
        self._opf_checkpoint = Path(opf_checkpoint)
        self._opf_label_space = Path(opf_label_space) if opf_label_space else None
        self._pii_project_root = (
            Path(pii_project_root) if pii_project_root else _default_pii_project_root()
        )
        self._qwen_backbone_path = (
            qwen_backbone_path
            or os.environ.get("REDACTION_QWEN_BACKBONE", "")
            or str(self._pii_project_root.parent / "model" / "Qwen3.5-9B-Base")
        )
        self._qwen_head_checkpoint = (
            Path(qwen_head_checkpoint)
            if qwen_head_checkpoint
            else self._pii_project_root / "runs" / "qwen_spancls_heads" / "last_linear" / "head.pt"
        )
        self._qwen_temperature = float(qwen_temperature)
        self._qwen_label_space = Path(qwen_label_space) if qwen_label_space else None
        self._qwen_loader_mode = str(qwen_loader_mode)
        self._qwen_expected_hidden_size = (
            int(qwen_expected_hidden_size) if qwen_expected_hidden_size is not None else None
        )
        self._qwen_expected_loader_mode = (
            str(qwen_expected_loader_mode) if qwen_expected_loader_mode is not None else None
        )
        self._qwen_lora_adapter_path = qwen_lora_adapter_path
        self._dtype = dtype
        self._device = device
        self._output_top_k = int(output_top_k)
        self._redact_threshold = float(redact_threshold)
        self._review_threshold = float(review_threshold)
        self._fallback_mode = fallback_mode
        self._qwen_verifier_enabled = bool(qwen_verifier_enabled)
        self._qwen_verifier_types = set(qwen_verifier_types or QWEN_VERIFIER_DEFAULT_TYPES)
        self._qwen_verifier_max_spans = int(qwen_verifier_max_spans)
        self._qwen_verifier_non_pii_threshold = float(qwen_verifier_non_pii_threshold)
        self._qwen_verifier_wrong_type_threshold = float(qwen_verifier_wrong_type_threshold)
        self._qwen_verifier_require_trigger = bool(qwen_verifier_require_trigger)
        self._qwen_verifier_min_risk_score = float(qwen_verifier_min_risk_score)
        self._qwen_verifier_low_top1_threshold = float(qwen_verifier_low_top1_threshold)

        self._per_label_top1_min: dict[str, float] = {}
        self._per_label_thresholds_path: Path | None = None
        if per_label_thresholds_path:
            path = Path(per_label_thresholds_path)
            if path.is_file():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    self._per_label_top1_min = {
                        str(k): float(v)
                        for k, v in (payload.get("top1_prob_min") or {}).items()
                    }
                    self._per_label_thresholds_path = path
                except Exception:
                    self._per_label_top1_min = {}

        self._lock = threading.Lock()
        self._loaded = False
        self._opf: Any = None
        self._qwen_cls: Any = None
        self._policy: Any = None
        self._qwen_labels: list[str] = []
        self._risk_weights: dict[str, float] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def supported_types(self) -> list[str]:
        return self._supported_types

    def load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            from opf import OPF
            self._opf = OPF(
                model=str(self._opf_checkpoint),
                device=self._device,
                output_mode="typed",
                decode_mode="viterbi",
                trim_whitespace=True,
            )
            try:
                self._opf.redact("warmup")
            except Exception:
                pass

            sys.path.insert(0, str(self._pii_project_root / "src" / "pii_prep"))
            from qwen_spancls_inference import QwenSpanClassifier
            if (
                self._qwen_expected_loader_mode is not None
                and self._qwen_expected_loader_mode != self._qwen_loader_mode
            ):
                raise ValueError(
                    f"qwen_loader_mode={self._qwen_loader_mode!r} does not match expected "
                    f"{self._qwen_expected_loader_mode!r}; refusing to load."
                )
            self._qwen_cls = QwenSpanClassifier(
                model_path=self._qwen_backbone_path,
                head_checkpoint_path=str(self._qwen_head_checkpoint),
                device=self._device,
                dtype=self._dtype,
                loader_mode=self._qwen_loader_mode,
                lora_adapter_path=self._qwen_lora_adapter_path,
            )
            if self._qwen_expected_hidden_size is not None:
                actual = int(getattr(self._qwen_cls, "hidden_size", -1))
                if actual != self._qwen_expected_hidden_size:
                    raise ValueError(
                        f"backbone hidden_size={actual} does not match expected "
                        f"{self._qwen_expected_hidden_size} for backbone {self._qwen_backbone_path!r}; "
                        "check REDACTION_QWEN4B_BACKBONE / REDACTION_QWEN_BACKBONE env vars."
                    )
            self._qwen_labels = list(self._qwen_cls.labels)

            from integrated_pipeline import PolicyLayer
            csv_path = str(self._pii_project_root / "docs" / "Data Sensitivity.csv")
            self._policy = PolicyLayer(
                csv_path=csv_path,
                redact_threshold=self._redact_threshold,
                review_threshold=self._review_threshold,
            )
            self._risk_weights = dict(self._policy.risk_weights)

            self._loaded = True

    def _build_fallback_candidates(self, text: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()

        if self._fallback_mode in ("full_and_regex", "full_only"):
            stripped = text.strip()
            if stripped:
                start = text.index(stripped[0]) if stripped[0] in text else 0
                end = start + len(stripped)
                key = (start, end)
                if key not in seen:
                    candidates.append({
                        "start": start, "end": end, "value": stripped,
                        "opf_top_type": None, "source": "fallback_full_input",
                    })
                    seen.add(key)

        if self._fallback_mode in ("full_and_regex", "regex_only"):
            for pattern_name, pattern in FALLBACK_REGEX_PATTERNS:
                for match in pattern.finditer(text):
                    match_val = match.group(0)
                    stripped_val = match_val.strip()
                    if not stripped_val:
                        continue
                    start_adj = match.start() + (len(match_val) - len(match_val.lstrip()))
                    end_adj = match.end() - (len(match_val) - len(match_val.rstrip()))
                    if start_adj >= end_adj:
                        continue
                    key = (start_adj, end_adj)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({
                        "start": start_adj, "end": end_adj, "value": stripped_val,
                        "opf_top_type": None,
                        "source": "regex_candidate:" + pattern_name,
                    })

        return candidates

    def _classify_and_build(
        self, text: str, candidates: list[dict[str, Any]]
    ) -> list[Span]:
        if not candidates:
            return []

        qwen_result = self._qwen_cls.classify_spans(
            text, candidates,
            output_full_distribution=False,
            top_k=self._output_top_k,
            include_non_pii=False,
        )

        spans: list[Span] = []
        for i, sr in enumerate(qwen_result.get("spans", [])):
            cs = candidates[i] if i < len(candidates) else {}
            source = cs.get("source", "model")

            qwen_top_type = sr.get("top_type", "NON_PII")
            opf_type = cs.get("opf_top_type") or ""

            type_dist = sr.get("type_distribution", {})
            sorted_dist = sorted(type_dist.items(), key=lambda x: x[1], reverse=True)
            topk = sorted_dist[: self._output_top_k]
            top1_prob = sorted_dist[0][1] if sorted_dist else 0.0
            top_prob = top1_prob
            top_type = topk[0][0] if topk else qwen_top_type

            top3_sum = sum(prob for _, prob in sorted_dist[:3])
            uncertainty = max(0.0, 1.0 - top3_sum)

            non_pii_prob = type_dist.get("NON_PII", 0.0)

            dc_weight = self._policy.risk_weights.get(top_type, 0.5)
            dc_label = "Highly Protected" if dc_weight >= 0.9 else "Protected" if dc_weight >= 0.4 else "Public"

            risk_score = uncertainty * dc_weight

            min_top1_pii = self._per_label_top1_min.get(top_type, MIN_TOP1_PII_DEFAULT)

            neg_context = _has_negative_context(text, int(cs.get("start", 0)), int(cs.get("end", 0)))
            line_neg = _has_line_local_negative(text, int(cs.get("start", 0)), int(cs.get("end", 0)))
            positive_context = _has_positive_context(text, int(cs.get("start", 0)), int(cs.get("end", 0)), top_type)

            decision, reason, risk_score, line_negative_suppressed = _safety_first_decision(
                source=source,
                top_type=top_type,
                top1_prob=top1_prob,
                top3_sum=top3_sum,
                non_pii_prob=non_pii_prob,
                risk_score=risk_score,
                neg_context=neg_context,
                line_neg=line_neg,
                has_per_label_threshold=top_type in self._per_label_top1_min,
                min_top1_pii=min_top1_pii,
                has_positive_context=positive_context,
            )
            pii_evidence_passed = decision != "ignore"
            evidence_reason = reason

            span_value = sr.get("value", cs.get("value", ""))
            if not span_value or not span_value.strip():
                continue

            span = Span(
                start=int(sr.get("start", cs.get("start", 0))),
                end=int(sr.get("end", cs.get("end", 0))),
                type=top_type,
                value=span_value,
                confidence=top1_prob,
                decision=decision,
                replacement=f"[{top_type}]",
                source=source,
                postprocess=[],
            )
            span.opf_top_type = opf_type
            span.top_type = top_type
            span.top_probability = top_prob
            span.top1_prob = top1_prob
            span.top3_sum = top3_sum
            span.non_pii_prob = non_pii_prob
            span.risk_score = risk_score
            span.uncertainty = uncertainty
            span.data_classification = dc_label
            span.data_classification_weight = dc_weight
            span.type_distribution_topk = topk
            span.decision_reason = reason
            span.line_negative_suppressed = line_negative_suppressed
            span.review_threshold = REVIEW_THRESHOLD
            span.policy_version = "top3_risk_v1"
            span.pii_evidence_passed = pii_evidence_passed
            span.evidence_reason = evidence_reason
            span.qwen_top_type = qwen_top_type
            span.detector_source = "qwen"
            span.deterministic_evidence = False
            spans.append(span)

        return spans

    def _apply_deterministic_rescue(self, text: str, spans: list[Span]) -> list[Span]:
        if not spans:
            return spans

        for span in spans:
            span_val = span.value
            span_start = span.start
            span_end = span.end

            text_window = text[max(0, span_start - POSTIVE_CONTEXT_WINDOW):span_end + 20]
            neg_context = _has_negative_context(text, span_start, span_end)
            line_neg_context = _has_line_local_negative(text, span_start, span_end)

            for rule in DETERMINISTIC_RESCUE_RULES:
                target_type = rule["target_type"]
                pattern = rule["pattern"]
                rule_name = rule["name"]
                target_decision = rule["decision"]
                context_keywords = rule.get("context_keywords", ())

                match = pattern.search(span_val)
                if not match:
                    match = pattern.search(text[span_start:span_end])

                if not match:
                    matched_full = pattern.search(text_window)
                    if matched_full:
                        m_start = matched_full.start()
                        matched_val = matched_full.group(0)
                        if rule_name == "account_number":
                            matched_val = matched_full.group(1)
                            m_start = matched_full.start(1)
                        if (span_start <= m_start < span_end or
                                max(span_start, m_start) < min(span_end, m_start + len(matched_val))):
                            match = matched_full

                if not match:
                    continue

                context_hit = any(kw in text_window.lower() for kw in context_keywords) if context_keywords else True

                if not context_hit and context_keywords and rule_name == "bsb_format":
                    continue

                span.deterministic_evidence = True
                span.deterministic_type = target_type
                span.detector_source = f"rescue:{rule_name}"

                span.type = target_type
                span.top_type = target_type
                span.replacement = f"[{target_type}]"

                dc_weight = self._policy.risk_weights.get(target_type, 0.6)
                dc_label = (
                    "Highly Protected" if dc_weight >= 0.9
                    else "Protected" if dc_weight >= 0.4
                    else "Public"
                )
                span.data_classification = dc_label
                span.data_classification_weight = dc_weight

                if line_neg_context:
                    span.decision = "ignore"
                    span.decision_reason = f"rescue_{rule_name}_line_negative"
                    span.risk_score = 0.0
                    span.policy_version = "top3_risk_v3_neg_priority"
                    span.pii_evidence_passed = False
                    span.evidence_reason = f"line_negative_override_{rule_name}"
                elif neg_context:
                    span.decision = "review"
                    span.decision_reason = f"rescue_{rule_name}_negative_context"
                    span.risk_score = 0.20
                    span.policy_version = "top3_risk_v3_neg_priority"
                    span.pii_evidence_passed = True
                    span.evidence_reason = f"window_negative_review_{rule_name}"
                elif target_decision == "redact":
                    if span.decision in ("ignore", "review"):
                        span.decision = "redact"
                        span.decision_reason = f"rescue_{rule_name}"
                        span.risk_score = max(span.risk_score or 0.0, 0.85)
                        span.confidence = max(span.confidence or 0.0, 0.80)
                        span.pii_evidence_passed = True
                        span.evidence_reason = f"deterministic_{rule_name}"
                        span.policy_version = "top3_risk_v2_rescue"
                    else:
                        span.decision_reason = f"rescue_{rule_name}_confirmed"
                        span.evidence_reason = f"deterministic_{rule_name}"
                elif target_decision == "review":
                    if span.decision == "ignore":
                        span.decision = "review"
                        span.decision_reason = f"rescue_{rule_name}_review"
                        span.risk_score = max(span.risk_score or 0.0, 0.25)
                        span.pii_evidence_passed = True
                        span.evidence_reason = f"deterministic_{rule_name}"
                        span.policy_version = "top3_risk_v2_rescue"

                span.postprocess.append(f"rescue_{rule_name}")
                break

        return spans

    def _apply_selective_qwen_verifier(self, text: str, spans: list[Span]) -> list[Span]:
        selected = _select_qwen_verifier_candidates(
            spans,
            enabled=self._qwen_verifier_enabled,
            verify_types=self._qwen_verifier_types,
            max_spans=self._qwen_verifier_max_spans,
            text=text,
            require_trigger=self._qwen_verifier_require_trigger,
            min_risk_score=self._qwen_verifier_min_risk_score,
            low_top1_threshold=self._qwen_verifier_low_top1_threshold,
        )
        if not selected:
            return spans

        verifier = getattr(self._qwen_cls, "verify_span_lm", None)
        if verifier is None:
            for span in selected:
                span.postprocess.append("qwen_lm_verifier_unavailable")
            return spans

        selected_ids = {id(span) for span in selected}
        out: list[Span] = []
        for span in spans:
            if id(span) not in selected_ids:
                out.append(span)
                continue
            try:
                verdict = verifier(
                    text=text,
                    context=_verifier_context(text, span.start, span.end),
                    candidate=span.value,
                    proposed_type=span.type,
                    opf_type=span.opf_top_type,
                    qwen_type=span.qwen_top_type or span.top_type,
                )
            except Exception as exc:
                span.postprocess.append("qwen_lm_verifier_error:" + exc.__class__.__name__)
                out.append(span)
                continue
            out.append(_apply_qwen_verifier_verdict(
                span,
                verdict,
                non_pii_threshold=self._qwen_verifier_non_pii_threshold,
                wrong_type_threshold=self._qwen_verifier_wrong_type_threshold,
            ))
        return out

    def detect_spans(self, text: str) -> tuple[list[Span], dict[str, Any]]:
        self.load()
        text = normalize_text(text)
        warnings: list[str] = []

        opf_result = self._opf.redact(text)
        opf_detected = getattr(opf_result, "detected_spans", ())

        seen: set[tuple[int, int]] = set()
        all_candidates: list[dict[str, Any]] = []

        for ds in opf_detected:
            start = int(getattr(ds, "start", -1))
            end = int(getattr(ds, "end", -1))
            value = getattr(ds, "text", None)
            if value is None and 0 <= start < end <= len(text):
                value = text[start:end]
            if not (0 <= start < end <= len(text)):
                warnings.append(f"opf_span_dropped_invalid_offsets:{start}:{end}")
                continue
            key = (start, end)
            seen.add(key)
            all_candidates.append({
                "start": start, "end": end, "value": value or "",
                "opf_top_type": str(getattr(ds, "label", "UNKNOWN")),
                "source": "model",
            })

        for pattern_name, pattern in FALLBACK_REGEX_PATTERNS:
            for match in pattern.finditer(text):
                match_val = match.group(0)
                stripped_val = match_val.strip()
                if not stripped_val:
                    continue
                start_adj = match.start() + (len(match_val) - len(match_val.lstrip()))
                end_adj = match.end() - (len(match_val) - len(match_val.rstrip()))
                if start_adj >= end_adj:
                    continue
                key = (start_adj, end_adj)
                if key in seen:
                    continue
                seen.add(key)
                all_candidates.append({
                    "start": start_adj, "end": end_adj, "value": stripped_val,
                    "opf_top_type": None,
                    "source": "regex_candidate:" + pattern_name,
                })

        has_candidates = bool(all_candidates)
        if not has_candidates and text.strip():
            stripped = text.strip()
            start = text.index(stripped[0]) if stripped[0] in text else 0
            end = start + len(stripped)
            all_candidates.append({
                "start": start, "end": end, "value": stripped,
                "opf_top_type": None, "source": "fallback_full_input",
            })

        spans = self._classify_and_build(text, all_candidates)

        spans = self._apply_deterministic_rescue(text, spans)

        spans = self._apply_selective_qwen_verifier(text, spans)

        spans = [s for s in spans if s.type != "NON_PII"]

        if not spans and opf_detected:
            for cs in all_candidates:
                if cs.get("source") != "model":
                    continue
                opf_type = cs.get("opf_top_type", "UNKNOWN")
                if opf_type == "O":
                    continue
                span_value = cs.get("value", "")
                if not span_value or not span_value.strip():
                    continue
                span = Span(
                    start=cs["start"], end=cs["end"], type=opf_type,
                    value=span_value, confidence=0.5,
                    decision="review",
                    replacement=f"[{opf_type}]", source="model",
                )
                span.opf_top_type = opf_type
                span.top_type = opf_type
                span.top_probability = 1.0
                span.risk_score = 0.0
                span.uncertainty = 0.0
                span.type_distribution_topk = [[opf_type, 1.0]]
                span.decision_reason = "low_confidence_fallback"
                spans.append(span)

        diag: dict[str, Any] = {"warnings": warnings, "raw_offset_mapping_applied": False}
        diag["opf_detected"] = len(opf_detected)
        diag["regex_candidates"] = len(all_candidates) - len(opf_detected)
        diag["total_candidates"] = len(all_candidates)
        diag["deterministic_rescues"] = sum(
            1 for s in spans if s.deterministic_evidence
        )
        diag["qwen_lm_verifier_enabled"] = self._qwen_verifier_enabled
        return spans, diag

    def info(self) -> dict[str, Any]:
        base = super().info()
        base["pipeline"] = "opf+qwen_head+policy+rescue+neg_priority"
        base["qwen_temperature"] = self._qwen_temperature
        base["qwen_labels"] = len(self._qwen_labels)
        base["qwen_loader_mode"] = self._qwen_loader_mode
        base["redact_threshold"] = self._redact_threshold
        base["review_threshold"] = self._review_threshold
        base["fallback_mode"] = self._fallback_mode
        base["deterministic_rescue"] = True
        base["rescue_rules"] = [r["name"] for r in DETERMINISTIC_RESCUE_RULES]
        base["qwen_lm_verifier_enabled"] = self._qwen_verifier_enabled
        base["qwen_lm_verifier_types"] = sorted(self._qwen_verifier_types)
        base["qwen_lm_verifier_max_spans"] = self._qwen_verifier_max_spans
        base["qwen_lm_verifier_require_trigger"] = self._qwen_verifier_require_trigger
        base["qwen_lm_verifier_min_risk_score"] = self._qwen_verifier_min_risk_score
        base["qwen_lm_verifier_low_top1_threshold"] = self._qwen_verifier_low_top1_threshold
        return base

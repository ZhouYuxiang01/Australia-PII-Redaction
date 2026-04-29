"""OPF token-tagger backend.

Uses OpenPrivacyFilter's Python API (opf.OPF) plus a shallow re-implementation
of `opf._core.runtime.predict_text` that also captures per-token chosen-label
logprobs. Each detected span gets a calibrated-style confidence:

    confidence(span) = exp( mean(token_chosen_logprob[i] for i overlapping span) )

This is the same aggregation used by the offline calibration pipeline that
produced `thresholds_dev.json`, so the per-type thresholds in the policy
file apply directly here.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

from ..core.normalize import normalize_text
from ..core.span import Span
from .base import RedactionBackend


# OPF emits its own label set (PERSON, AU_TFN, etc.) which already lines up
# with the canonical taxonomy used by our policy / schema. No alias remap needed
# here — the 73-class label_space is the source of truth on the OPF side.


class OpfBackend(RedactionBackend):
    def __init__(
        self,
        *,
        name: str,
        model_version: str,
        supported_types: list[str],
        checkpoint_path: str | Path,
        device: str = "cuda",
        decode_mode: str = "viterbi",
        trim_whitespace: bool = True,
        emit_confidence: bool = True,
    ) -> None:
        self._name = name
        self._model_version = model_version
        self._supported_types = list(supported_types)
        self._checkpoint_path = Path(checkpoint_path)
        self._device = device
        self._decode_mode = decode_mode
        self._trim_whitespace = trim_whitespace
        self._emit_confidence = bool(emit_confidence)
        self._lock = threading.Lock()
        self._loaded = False
        self._opf: Any = None

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
                model=str(self._checkpoint_path),
                device=self._device,
                output_mode="typed",
                decode_mode=self._decode_mode,
                trim_whitespace=self._trim_whitespace,
            )
            # Force a one-shot warmup so the first request isn't slow.
            try:
                self._opf.redact("warmup")
            except Exception:
                pass
            self._loaded = True

    def detect_spans(self, text: str) -> tuple[list[Span], dict[str, Any]]:
        self.load()
        text = normalize_text(text)
        if self._emit_confidence:
            return self._detect_with_confidence(text)
        return self._detect_simple(text)

    # ------------------------------------------------------------------
    # No-confidence path: just call OPF.redact() and map spans through.
    # ------------------------------------------------------------------
    def _detect_simple(self, text: str) -> tuple[list[Span], dict[str, Any]]:
        result = self._opf.redact(text)
        spans: list[Span] = []
        warnings: list[str] = []
        for ds in getattr(result, "detected_spans", ()):
            label = getattr(ds, "label", None) or "UNKNOWN"
            start = int(getattr(ds, "start", -1))
            end = int(getattr(ds, "end", -1))
            value = getattr(ds, "text", None)
            if value is None and 0 <= start < end <= len(text):
                value = text[start:end]
            if not (0 <= start < end <= len(text)):
                warnings.append(f"opf_span_dropped_invalid_offsets:{label}:{(value or '')[:40]}")
                continue
            spans.append(Span(
                start=start, end=end, type=label, value=value or "",
                confidence=None, source="model", postprocess=[],
            ))
        if getattr(result, "warning", None):
            warnings.append(f"opf_runtime_warning:{result.warning}")
        return spans, {
            "raw_output": None, "warnings": warnings,
            "raw_offset_mapping_applied": False,
        }

    # ------------------------------------------------------------------
    # Confidence path: shallow re-implementation of opf._core.runtime.predict_text
    # that also returns per-token chosen-label logprobs, which we aggregate to
    # span-level confidence.
    # ------------------------------------------------------------------
    def _detect_with_confidence(self, text: str) -> tuple[list[Span], dict[str, Any]]:
        import torch
        import torch.nn.functional as F
        from opf._core.sequence_labeling import (
            ExampleAggregation, TokenizedExample, example_to_windows,
        )
        from opf._core.spans import (
            decode_text_with_offsets, discard_overlapping_spans_by_label,
            labels_to_spans, token_spans_to_char_spans,
            trim_char_spans_whitespace,
        )

        runtime, decoder = self._opf.get_prediction_components()
        warnings: list[str] = []

        token_ids = tuple(int(t) for t in runtime.encoding.encode(text, allowed_special="all"))
        if not token_ids:
            return [], {"raw_output": None, "warnings": warnings,
                        "raw_offset_mapping_applied": False}

        background = int(runtime.label_info.background_token_label)
        example = TokenizedExample(
            tokens=token_ids,
            labels=tuple(background for _ in token_ids),
            example_id="wrapper-confidence",
            text=text,
        )
        agg = ExampleAggregation(logprob_logsumexp=[], counts=[], labels=[], token_ids=[])

        # Forward windows -> per-token avg log_softmax over labels.
        for window in example_to_windows(example, runtime.n_ctx):
            if not window.tokens:
                continue
            window_tokens = torch.tensor(
                [list(window.tokens)], device=runtime.device, dtype=torch.int32,
            )
            attention_mask = torch.ones_like(window_tokens, dtype=torch.bool)
            with torch.no_grad():
                logits = runtime.model(window_tokens, attention_mask=attention_mask)
            log_probs = F.log_softmax(logits.float(), dim=-1)[0].detach().cpu()
            for tok_pos, is_valid in enumerate(window.mask):
                if not bool(is_valid):
                    continue
                tok_idx = int(window.offsets[tok_pos])
                if tok_idx < 0:
                    continue
                agg.ensure_capacity(tok_idx)
                score_vec = log_probs[tok_pos]
                existing = agg.logprob_logsumexp[tok_idx]
                if existing is None:
                    agg.logprob_logsumexp[tok_idx] = score_vec.clone()
                else:
                    agg.logprob_logsumexp[tok_idx] = torch.logaddexp(existing, score_vec)
                agg.counts[tok_idx] += 1
                agg.record_token_id(tok_idx, int(window.tokens[tok_pos]), example.example_id)
                agg.length = max(agg.length, tok_idx + 1)

        # Average over windows where tokens overlap.
        token_positions: list[int] = []
        token_score_vectors: list[torch.Tensor] = []
        for tok_idx in range(agg.length):
            if tok_idx >= len(agg.logprob_logsumexp):
                continue
            score_sum = agg.logprob_logsumexp[tok_idx]
            count = agg.counts[tok_idx]
            if score_sum is None or count <= 0:
                continue
            avg_logprob = score_sum - math.log(float(count))
            token_positions.append(tok_idx)
            token_score_vectors.append(avg_logprob)

        if not token_score_vectors:
            return [], {"raw_output": None, "warnings": warnings,
                        "raw_offset_mapping_applied": False}

        stacked = torch.stack(token_score_vectors, dim=0)
        if decoder is not None:
            decoded_labels = decoder.decode(stacked)
            if len(decoded_labels) != len(token_positions):
                decoded_labels = stacked.argmax(dim=1).tolist()
        else:
            decoded_labels = stacked.argmax(dim=1).tolist()

        # Per-token chosen-label logprob, indexed by tok_idx.
        chosen_logprob_by_tok_idx: dict[int, float] = {}
        for vec_idx, tok_idx in enumerate(token_positions):
            chosen_logprob_by_tok_idx[tok_idx] = float(stacked[vec_idx, int(decoded_labels[vec_idx])])

        predicted_labels_by_index = {
            tok_idx: int(label)
            for tok_idx, label in zip(token_positions, decoded_labels)
        }
        predicted_token_spans = labels_to_spans(predicted_labels_by_index, runtime.label_info)

        decoded_text, char_starts, char_ends = decode_text_with_offsets(
            token_ids, runtime.encoding,
        )
        decoded_mismatch = decoded_text != text
        if decoded_mismatch:
            warnings.append("opf_decoded_mismatch_offsets_use_decoded_text")
        source_text = decoded_text if decoded_mismatch else text

        predicted_char_spans = token_spans_to_char_spans(
            predicted_token_spans, char_starts, char_ends,
        )
        if runtime.trim_span_whitespace:
            predicted_char_spans = trim_char_spans_whitespace(
                predicted_char_spans, source_text,
            )
        if runtime.discard_overlapping_predicted_spans:
            predicted_char_spans = discard_overlapping_spans_by_label(predicted_char_spans)

        spans: list[Span] = []
        for label_idx, start, end in predicted_char_spans:
            if not (0 <= start < end <= len(source_text)):
                continue
            label = (
                str(runtime.label_info.span_class_names[label_idx])
                if 0 <= int(label_idx) < len(runtime.label_info.span_class_names)
                else f"label_{label_idx}"
            )
            value = source_text[start:end]
            confidence, raw_logprob, n_tok = self._span_confidence(
                start, end, char_starts, char_ends, chosen_logprob_by_tok_idx,
            )
            spans.append(Span(
                start=int(start), end=int(end), type=label, value=value,
                confidence=confidence, raw_score=raw_logprob, source="model",
                postprocess=[f"confidence_n_tokens={n_tok}"] if n_tok > 0 else [],
            ))

        return spans, {
            "raw_output": None, "warnings": warnings,
            "raw_offset_mapping_applied": bool(decoded_mismatch),
        }

    @staticmethod
    def _span_confidence(start: int, end: int, char_starts, char_ends,
                          chosen_logprob_by_tok_idx: dict[int, float]
                          ) -> tuple[float | None, float | None, int]:
        """Mean of chosen-label logprobs for tokens that overlap [start, end)."""
        lps: list[float] = []
        n = min(len(char_starts), len(char_ends))
        for tok_idx in range(n):
            cs = int(char_starts[tok_idx]); ce = int(char_ends[tok_idx])
            if cs >= end:
                break
            if ce <= start:
                continue
            lp = chosen_logprob_by_tok_idx.get(tok_idx)
            if lp is None:
                continue
            lps.append(lp)
        if not lps:
            return None, None, 0
        mean_lp = sum(lps) / len(lps)
        try:
            conf = math.exp(mean_lp)
        except OverflowError:
            conf = 1.0
        return float(conf), float(mean_lp), len(lps)

from __future__ import annotations

import json
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

VERIFIER_LABEL_CHOICES = {
    "valid_pii": " A",
    "non_pii": " B",
    "wrong_type": " C",
    "uncertain": " D",
}

TYPE_SUGGESTION_CANDIDATES = (
    "BANK_ACCOUNT_NUMBER",
    "PHONE",
    "IP_ADDRESS",
    "VEHICLE_ID",
    "LATITUDE",
    "LONGITUDE",
    "GEOLOCATION_INFORMATION",
    "EMAIL",
    "PERSON",
    "ADDRESS",
    "DATE_OF_BIRTH",
    "PAYMENT_CARD_NUMBER",
)


def _import_transformers():
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    return AutoModel, AutoModelForCausalLM, AutoTokenizer


def _resolve_attn_implementation(requested: str | None) -> str | None:
    """Pick the fastest attention backend available in this runtime."""
    requested = (requested or os.environ.get("REDACTION_QWEN_ATTN_IMPLEMENTATION") or "").strip()
    if not requested:
        return None
    if requested in {"fla", "flash_linear_attention", "flash-linear-attention"}:
        if find_spec("fla") is None:
            print(
                "[qwen] flash-linear-attention requested but fla is not installed; "
                "using Transformers default attention paths.",
                flush=True,
            )
        else:
            print(
                "[qwen] flash-linear-attention available; Qwen3.5 linear_attention "
                "layers will use the FLA kernels exposed by Transformers.",
                flush=True,
            )
        # FLA is used internally by Qwen3.5 linear_attention layers; it is not
        # a valid AutoModel attn_implementation value for full attention layers.
        return None
    if requested == "flash_attention_2" and find_spec("flash_attn") is None:
        print(
            "[qwen] flash_attention_2 requested but flash_attn is not installed; "
            "using Transformers default attention paths.",
            flush=True,
        )
        return None
    return requested


class QwenSpanClassifier:
    def __init__(
        self,
        model_path: str,
        head_checkpoint_path: str,
        *,
        device: str = "cuda",
        dtype: str = "bf16",
        loader_mode: str = "causal_lm",
        lora_adapter_path: str | None = None,
        attn_implementation: str | None = None,
    ):
        model_path = str(model_path)
        head_checkpoint_path = str(head_checkpoint_path)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        use_bf16 = bool(
            self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
            and dtype == "bf16"
        )
        torch_dtype = (
            torch.bfloat16
            if use_bf16
            else torch.float16 if self.device.type == "cuda" else torch.float32
        )

        AutoModel, AutoModelForCausalLM, AutoTokenizer = _import_transformers()
        self.causal_lm = None
        self._verifier_baseline_cache: dict[tuple[tuple[str, str], str], dict[str, float]] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "torch_dtype": torch_dtype,
        }
        resolved_attn = _resolve_attn_implementation(attn_implementation)
        if resolved_attn:
            model_kwargs["attn_implementation"] = resolved_attn

        if loader_mode == "automodel_legacy":
            # Preserves the historical broken-AutoModel path (random-init backbone) for
            # heads trained against that same broken cache. Only set for legacy 9B head.
            self.backbone = AutoModel.from_pretrained(model_path, **model_kwargs)
            config = self.backbone.config
        else:
            causal = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            self.causal_lm = causal
            inner = None
            for attr in ("model", "language_model"):
                cand = getattr(causal, attr, None)
                if cand is not None and hasattr(cand, "embed_tokens"):
                    inner = cand
                    break
            self.backbone = inner if inner is not None else causal
            config = causal.config

        self.backbone.to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        if lora_adapter_path is not None:
            from peft import PeftModel
            self.backbone = PeftModel.from_pretrained(self.backbone, lora_adapter_path)
            self.backbone = self.backbone.to(self.device)
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.causal_lm = None

        if self.causal_lm is not None:
            self.causal_lm.to(self.device)
            self.causal_lm.eval()
            for param in self.causal_lm.parameters():
                param.requires_grad = False

        if hasattr(config, "hidden_size"):
            hidden_size = int(config.hidden_size)
        elif hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            hidden_size = int(config.text_config.hidden_size)
        else:
            raise ValueError("could not determine hidden_size from Qwen config")

        ckpt = torch.load(head_checkpoint_path, map_location="cpu", weights_only=True)
        self.labels: list[str] = ckpt["labels"]
        self.num_labels = len(self.labels)
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.experiment = ckpt.get("experiment", "last_linear")

        head_state = ckpt["head_state_dict"]
        stored_input_dim = int(head_state["weight"].shape[1])
        if stored_input_dim != hidden_size:
            raise ValueError(
                f"head/backbone dim mismatch: head expects {stored_input_dim}, backbone produces {hidden_size}. "
                f"Wrong model_path ({model_path}) or wrong head checkpoint ({head_checkpoint_path})."
            )
        self.hidden_size = hidden_size

        self.head = nn.Linear(hidden_size, self.num_labels)
        self.head.load_state_dict(head_state)
        self.head.to(self.device)
        self.head.requires_grad_(False)
        self.head.eval()

    @classmethod
    def from_project_root(
        cls,
        root: str | Path = ".",
        *,
        model_path: str = "/home/admin/model/Qwen3.5-9B-Base",
        experiment: str = "last_linear",
        device: str = "cuda",
        dtype: str = "bf16",
    ) -> QwenSpanClassifier:
        root = Path(root)
        head_path = root / "runs" / "qwen_spancls_heads" / experiment / "head.pt"
        return cls(model_path, str(head_path), device=device, dtype=dtype)

    def _char_span_to_token_range(self, offsets, start: int, end: int):
        token_indices = []
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_end <= tok_start:
                continue
            if tok_end <= start:
                continue
            if tok_start >= end:
                break
            if tok_start < end and tok_end > start:
                token_indices.append(idx)
        if not token_indices:
            return None
        return token_indices[0], token_indices[-1] + 1

    def _pool_last_token(self, hidden, span_token_range):
        start_tok, end_tok = span_token_range
        span_hidden = hidden[start_tok:end_tok]
        return span_hidden[-1]

    def classify_spans(
        self,
        text: str,
        candidate_spans: list[dict[str, Any]],
        *,
        output_full_distribution: bool = True,
        top_k: int = 5,
        include_non_pii: bool = True,
    ) -> dict[str, Any]:
        if not candidate_spans:
            return {
                "text": text,
                "spans": [],
                "summary": {"span_count": 0, "detected_pii": 0},
            }

        for span in candidate_spans:
            span_start = int(span["start"])
            span_end = int(span["end"])
            if span_end > len(text):
                span_end = len(text)
            if "value" in span and span.get("value") != text[span_start:span_end]:
                pass

        encoded = self.tokenizer(
            [text],
            padding=False,
            truncation=True,
            max_length=1536,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        offsets = encoded["offset_mapping"][0].tolist()

        token_ranges = []
        valid_indices = []
        for i, span in enumerate(candidate_spans):
            mapped = self._char_span_to_token_range(offsets, int(span["start"]), int(span["end"]))
            if mapped is not None:
                token_ranges.append(mapped)
                valid_indices.append(i)

        if not valid_indices:
            return {
                "text": text,
                "spans": [],
                "summary": {"span_count": len(candidate_spans), "mapped_spans": 0},
            }

        with torch.no_grad():
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_hidden_states": False,
                "return_dict": True,
            }
            try:
                outputs = self.backbone(**kwargs, use_cache=False)
            except TypeError:
                outputs = self.backbone(**kwargs)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                kwargs["output_hidden_states"] = True
                try:
                    outputs = self.backbone(**kwargs, use_cache=False)
                except TypeError:
                    outputs = self.backbone(**kwargs)
                hidden = outputs.hidden_states[-1]

        pooled = []
        for tr in token_ranges:
            pooled.append(self._pool_last_token(hidden[0], tr))

        embeddings = torch.stack(pooled, dim=0).float()
        logits = self.head(embeddings)
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)

        non_pii_idx = self.labels.index("NON_PII") if "NON_PII" in self.labels else -1

        result_spans = []
        for j, idx in enumerate(valid_indices):
            span = candidate_spans[idx]
            prob_vec = probs[j]
            top_values, top_indices = prob_vec.topk(k=min(len(self.labels), max(top_k, len(self.labels))))

            type_distribution = {}
            display_values, display_indices = prob_vec.topk(k=len(self.labels))
            for k in range(len(self.labels)):
                label = self.labels[int(display_indices[k])]
                if not include_non_pii and label == "NON_PII":
                    continue
                prob_val = float(display_values[k].detach().item())
                if prob_val > 0.0 or output_full_distribution:
                    type_distribution[label] = prob_val

            top_label = self.labels[int(top_indices[0])]
            if top_label == "NON_PII" and len(top_indices) > 1:
                top_label = self.labels[int(top_indices[1])]

            pii_probs = prob_vec.clone()
            if non_pii_idx >= 0:
                pii_probs[non_pii_idx] = 0.0
                pii_sum = pii_probs.sum()
                if pii_sum > 0:
                    pii_probs = pii_probs / pii_sum

            top_pii_idx = int(pii_probs.argmax())
            result_spans.append({
                "start": int(span["start"]),
                "end": int(span["end"]),
                "value": text[int(span["start"]):int(span["end"])],
                "type_distribution": type_distribution,
                "top_type": top_label,
                "top_probability": float(top_values[0]),
                "top_pii_type": self.labels[top_pii_idx] if non_pii_idx >= 0 else top_label,
                "top_pii_probability": float(pii_probs[top_pii_idx]) if non_pii_idx >= 0 else float(top_values[0]),
            })

        detected_pii = sum(1 for s in result_spans if s["top_type"] != "NON_PII")
        return {
            "text": text,
            "spans": result_spans,
            "summary": {
                "span_count": len(result_spans),
                "mapped_spans": len(valid_indices),
                "detected_pii": detected_pii,
            },
        }

    def _continuation_score(self, prompt: str, continuation: str, max_length: int = 1024) -> float:
        """Average log-probability of a constrained verifier continuation."""
        if self.causal_lm is None:
            raise RuntimeError("qwen causal lm is not available for verifier scoring")
        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )["input_ids"][0]
        full = self.tokenizer(
            prompt + continuation,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = full["input_ids"].to(self.device)
        prompt_len = int(prompt_ids.shape[0])
        if input_ids.shape[1] <= prompt_len:
            return -1e9
        with torch.no_grad():
            try:
                outputs = self.causal_lm(input_ids=input_ids, use_cache=False, return_dict=True)
            except TypeError:
                outputs = self.causal_lm(input_ids=input_ids, return_dict=True)
            logits = outputs.logits[:, :-1, :].float()
            labels = input_ids[:, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            start = max(0, prompt_len - 1)
            target = labels[:, start:]
            selected = log_probs[:, start:, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)
            if selected.numel() == 0:
                return -1e9
            return float(selected.mean().detach().item())

    def _single_token_id(self, token_text: str) -> int:
        ids = self.tokenizer(token_text, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"verifier verbalizer is not one token: {token_text!r} -> {ids}")
        return int(ids[0])

    def _next_token_logits(self, prompt: str, max_length: int = 1024) -> torch.Tensor:
        if self.causal_lm is None:
            raise RuntimeError("qwen causal lm is not available for verifier scoring")
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        with torch.no_grad():
            try:
                outputs = self.causal_lm(input_ids=input_ids, use_cache=False, return_dict=True)
            except TypeError:
                outputs = self.causal_lm(input_ids=input_ids, return_dict=True)
        return outputs.logits[0, -1, :].float()

    def _score_single_token_choices(
        self,
        *,
        prompt: str,
        baseline_prompt: str,
        choices: dict[str, str],
        max_length: int = 1024,
    ) -> dict[str, Any]:
        logits = self._next_token_logits(prompt, max_length=max_length)
        raw_logits = {
            label: float(logits[self._single_token_id(verbalizer)].detach().item())
            for label, verbalizer in choices.items()
        }

        baseline_key = (tuple(choices.items()), baseline_prompt)
        baseline_logits = self._verifier_baseline_cache.get(baseline_key)
        if baseline_logits is None:
            base = self._next_token_logits(baseline_prompt, max_length=max_length)
            baseline_logits = {
                label: float(base[self._single_token_id(verbalizer)].detach().item())
                for label, verbalizer in choices.items()
            }
            self._verifier_baseline_cache[baseline_key] = baseline_logits

        calibrated_logits = {
            label: raw_logits[label] - baseline_logits[label]
            for label in choices
        }
        score_tensor = torch.tensor(
            [calibrated_logits[label] for label in choices],
            dtype=torch.float32,
        )
        probs = F.softmax(score_tensor, dim=0)
        scores = {
            label: float(probs[i].detach().item())
            for i, label in enumerate(choices)
        }
        winner = max(scores.items(), key=lambda item: item[1])[0]
        return {
            "winner": winner,
            "confidence": scores[winner],
            "scores": scores,
            "raw_logits": raw_logits,
            "baseline_logits": baseline_logits,
            "calibrated_logits": calibrated_logits,
        }

    @staticmethod
    def _canonical_suggested_type(label: str | None) -> str:
        value = (label or "").upper()
        if value in {"", "O", "UNKNOWN", "NON_PII"}:
            return ""
        if value in {"AU_BANK_ACCOUNT", "BANK_ACCOUNT_NUMBER"}:
            return "BANK_ACCOUNT_NUMBER"
        if value in {"PHONE", "MOBILE", "WORK_PHONE", "HOME_PHONE", "AU_PHONE"}:
            return "PHONE"
        if value in {"VEHICLE_ID", "VEHICLE_REGO", "NUMBER_PLATE"}:
            return "VEHICLE_ID"
        if value in {"EMAIL_ADDRESS", "WORK_EMAIL"}:
            return "EMAIL"
        return value

    def _suggested_type_candidates(self, proposed_type: str, opf_type: str, qwen_type: str) -> list[str]:
        proposed = self._canonical_suggested_type(proposed_type)
        out: list[str] = []
        for label in (qwen_type, opf_type, *TYPE_SUGGESTION_CANDIDATES):
            candidate = self._canonical_suggested_type(label)
            if not candidate or candidate == proposed or candidate in out:
                continue
            out.append(candidate)
        return out[:10]

    def _suggest_type_lm(
        self,
        *,
        guidance: str,
        context: str,
        candidate: str,
        proposed_type: str,
        opf_type: str,
        qwen_type: str,
    ) -> dict[str, Any]:
        candidates = self._suggested_type_candidates(proposed_type, opf_type, qwen_type)
        if not candidates:
            return {}
        letters = "ABCDEFGHIJ"
        choices = {
            label: " " + letters[i]
            for i, label in enumerate(candidates)
        }
        option_text = "\n".join(f"{letters[i]}: {label}" for i, label in enumerate(candidates))
        type_guidance = (
            "PII type correction task. Choose exactly one option letter.\n"
            "Bank account values are numeric identifiers such as BSB/account numbers, not bank names. "
            "Phone numbers are personal when tied to a person. IP addresses are personal when tied to "
            "a person/account/session. Vehicle identifiers are personal when tied to a person or case. "
            "Geolocation values are personal when they identify a person's location.\n"
        )
        prompt = (
            type_guidance
            + "\nType correction task. Choose the option letter for the candidate's best PII type.\n"
            + "Options:\n"
            + option_text
            + "\n\nContext:\n"
            + context.strip()[:900]
            + "\n\nCandidate: "
            + json.dumps(candidate, ensure_ascii=False)
            + "\nOriginal proposed type: "
            + proposed_type
            + "\nOPF type: "
            + (opf_type or "unknown")
            + "\nQwen head type: "
            + (qwen_type or "unknown")
            + "\n\nAnswer:"
        )
        baseline_prompt = (
            type_guidance
            + "\nType correction task. Choose the option letter for the candidate's best PII type.\n"
            + "Options:\n"
            + option_text
            + "\n\nContext:\nN/A\n\nCandidate: \"\"\nOriginal proposed type: unknown\n"
            + "OPF type: unknown\nQwen head type: unknown\n\nAnswer:"
        )
        return self._score_single_token_choices(
            prompt=prompt,
            baseline_prompt=baseline_prompt,
            choices=choices,
        )

    def verify_span_lm(
        self,
        *,
        text: str,
        context: str,
        candidate: str,
        proposed_type: str,
        opf_type: str = "",
        qwen_type: str = "",
    ) -> dict[str, Any]:
        """Use Qwen's LM head as a constrained semantic verifier."""
        guidance = (
            "PII verification task. Choose exactly one option letter.\n"
            "A: valid_pii - the candidate is real personal or sensitive data of the proposed type.\n"
            "B: non_pii - the candidate is an organization name, system value, public/shared contact, "
            "asset code, demo/example value, or otherwise not personal/sensitive data.\n"
            "C: wrong_type - the candidate is personal/sensitive data, but the proposed type is wrong.\n"
            "D: uncertain - the context is insufficient.\n"
            "Type hints: bank account values are numeric identifiers such as BSB/account numbers, "
            "not bank institution names. IP addresses are PII only when tied to a person/account/session. "
            "Phone numbers are PII only when tied to a person, not reception/helpdesk/main lines. "
            "Vehicle identifiers are PII when tied to a person or case, not asset/demo/system codes. "
            "Latitude/longitude/geolocation is PII when it identifies a person's location.\n"
            "Examples:\n"
            "Context: NAT gateway 10.88.12.7 routed traffic. Candidate: \"10.88.12.7\"; "
            "Proposed type: IP_ADDRESS -> B\n"
            "Context: Student login IP 203.45.67.89. Candidate: \"203.45.67.89\"; "
            "Proposed type: IP_ADDRESS -> A\n"
            "Context: bank name written as Southern Mutual. Candidate: \"Southern Mutual\"; "
            "Proposed type: BANK_ACCOUNT_NUMBER -> B\n"
            "Context: contact number 0412 345 678. Candidate: \"0412 345 678\"; "
            "Proposed type: BANK_ACCOUNT_NUMBER -> C\n"
        )
        prompt = (
            guidance
            + "\nContext:\n"
            + context.strip()[:900]
            + "\n\nCandidate: "
            + json.dumps(candidate, ensure_ascii=False)
            + "\nProposed type: "
            + proposed_type
            + "\nOPF type: "
            + (opf_type or "unknown")
            + "\nQwen head type: "
            + (qwen_type or "unknown")
            + "\n\nAnswer:"
        )
        baseline_prompt = (
            guidance
            + "\nContext:\nN/A\n\nCandidate: \"\"\nProposed type: unknown\n"
            + "OPF type: unknown\nQwen head type: unknown\n\nAnswer:"
        )
        scored = self._score_single_token_choices(
            prompt=prompt,
            baseline_prompt=baseline_prompt,
            choices=VERIFIER_LABEL_CHOICES,
        )
        verdict = scored["winner"]
        result = {
            "verdict": verdict,
            "confidence": scored["confidence"],
            "scores": scored["scores"],
            "raw_logits": scored["raw_logits"],
            "baseline_logits": scored["baseline_logits"],
            "calibrated_logits": scored["calibrated_logits"],
        }
        if verdict == "wrong_type":
            type_scored = self._suggest_type_lm(
                guidance=guidance,
                context=context,
                candidate=candidate,
                proposed_type=proposed_type,
                opf_type=opf_type,
                qwen_type=qwen_type,
            )
            if type_scored:
                result["suggested_type"] = type_scored["winner"]
                result["suggested_type_confidence"] = type_scored["confidence"]
                result["suggested_type_scores"] = type_scored["scores"]
        return result

"""Qwen tagged-output backend.

Works for any Qwen-family model that has been fine-tuned to produce
'<pii type="X">value</pii>' tagged output. Supports two loading modes:
  - 'lora': base model + PEFT/LoRA adapter directory
  - 'full': single full-finetune model directory (no adapter)
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..core.normalize import normalize_text
from ..core.parsers import parse_annotated_output, repair_offsets_to_input
from ..core.span import Span
from .base import RedactionBackend


DEFAULT_SYSTEM_PROMPT = (
    "You are an Australian PII redaction system. Return the input text with "
    "each supported PII span wrapped as <pii type=\"TYPE\">exact text</pii>. "
    "Preserve every character. Do not explain.\n"
    "Supported types:\n- "
)


class QwenLoraBackend(RedactionBackend):
    def __init__(
        self,
        *,
        name: str,
        model_version: str,
        supported_types: list[str],
        mode: str = "lora",
        base_model_path: str | Path | None = None,
        adapter_path: str | Path | None = None,
        full_model_path: str | Path | None = None,
        max_new_tokens: int = 512,
        max_concurrent_generate: int = 4,
        system_prompt: str | None = None,
    ) -> None:
        if mode not in {"lora", "full"}:
            raise ValueError(f"mode must be 'lora' or 'full', got {mode!r}")
        if mode == "lora" and (base_model_path is None or adapter_path is None):
            raise ValueError("lora mode requires both base_model_path and adapter_path")
        if mode == "full" and full_model_path is None:
            raise ValueError("full mode requires full_model_path")
        self._name = name
        self._model_version = model_version
        self._supported_types = list(supported_types)
        self._mode = mode
        self._base_model_path = Path(base_model_path) if base_model_path else None
        self._adapter_path = Path(adapter_path) if adapter_path else None
        self._full_model_path = Path(full_model_path) if full_model_path else None
        self._max_new_tokens = int(max_new_tokens)
        self._semaphore = threading.BoundedSemaphore(max(1, int(max_concurrent_generate)))
        self._system_prompt = system_prompt
        self._lock = threading.Lock()
        self._loaded = False
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None

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
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self._mode == "lora":
                from peft import PeftModel
                tok_src = self._adapter_path
                base = AutoModelForCausalLM.from_pretrained(
                    str(self._base_model_path),
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                model = PeftModel.from_pretrained(base, str(self._adapter_path))
            else:
                tok_src = self._full_model_path
                model = AutoModelForCausalLM.from_pretrained(
                    str(self._full_model_path),
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            tokenizer = AutoTokenizer.from_pretrained(str(tok_src), trust_remote_code=True, use_fast=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            model.eval()
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._loaded = True

    def _build_system(self) -> str:
        if self._system_prompt:
            return self._system_prompt
        return DEFAULT_SYSTEM_PROMPT + "\n- ".join(self._supported_types)

    def _generate(self, text: str) -> str:
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        messages = [
            {"role": "system", "content": self._build_system()},
            {"role": "user", "content": text},
        ]
        encoded = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        if isinstance(encoded, self._torch.Tensor):
            encoded = {"input_ids": encoded}
        device = getattr(self._model, "device", None) or next(self._model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        prompt_len = encoded["input_ids"].shape[-1]
        eos_ids = [self._tokenizer.eos_token_id]
        try:
            im_end = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end is not None and im_end >= 0:
                eos_ids.append(im_end)
        except Exception:
            pass
        with self._semaphore, self._torch.no_grad():
            output_ids = self._model.generate(
                **encoded,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=eos_ids,
            )
        gen = output_ids[0][prompt_len:]
        return self._tokenizer.decode(gen, skip_special_tokens=True).strip()

    def detect_spans(self, text: str) -> tuple[list[Span], dict[str, Any]]:
        self.load()
        text = normalize_text(text)
        raw = self._generate(text)
        parsed_text, spans = parse_annotated_output(raw)
        parsed_text = normalize_text(parsed_text)
        spans, warnings, repaired = repair_offsets_to_input(text, parsed_text, spans)
        return spans, {
            "raw_output": raw,
            "warnings": warnings,
            "raw_offset_mapping_applied": repaired,
        }

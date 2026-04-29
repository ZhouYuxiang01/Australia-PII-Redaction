"""Backend registry / factory.

Build a backend from a JSON config of the form:

    {
      "type": "qwen_lora" | "opf",
      "name": "...",
      "model_version": "...",
      "supported_types": [...],
      ...type-specific fields...
    }

The factory keeps `redaction.backends` model-agnostic; new backends register
themselves in BACKEND_TYPES below.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .base import RedactionBackend
from .opf import OpfBackend
from .qwen_lora import QwenLoraBackend


def _build_qwen_lora(cfg: dict[str, Any]) -> RedactionBackend:
    return QwenLoraBackend(
        name=cfg["name"],
        model_version=cfg["model_version"],
        supported_types=cfg["supported_types"],
        mode=cfg.get("mode", "lora"),
        base_model_path=cfg.get("base_model_path"),
        adapter_path=cfg.get("adapter_path"),
        full_model_path=cfg.get("full_model_path"),
        max_new_tokens=int(cfg.get("max_new_tokens", 512)),
        max_concurrent_generate=int(cfg.get("max_concurrent_generate", 4)),
        system_prompt=cfg.get("system_prompt"),
    )


def _build_opf(cfg: dict[str, Any]) -> RedactionBackend:
    return OpfBackend(
        name=cfg["name"],
        model_version=cfg["model_version"],
        supported_types=cfg["supported_types"],
        checkpoint_path=cfg["checkpoint_path"],
        device=cfg.get("device", "cuda"),
        decode_mode=cfg.get("decode_mode", "viterbi"),
        trim_whitespace=bool(cfg.get("trim_whitespace", True)),
        emit_confidence=bool(cfg.get("emit_confidence", True)),
    )


BACKEND_TYPES: dict[str, Callable[[dict[str, Any]], RedactionBackend]] = {
    "qwen_lora": _build_qwen_lora,
    "opf": _build_opf,
}


def load_backend_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_backend(cfg: dict[str, Any]) -> RedactionBackend:
    btype = cfg.get("type")
    if btype not in BACKEND_TYPES:
        raise ValueError(f"Unknown backend type: {btype!r}. Known: {sorted(BACKEND_TYPES)}")
    return BACKEND_TYPES[btype](cfg)


def build_backend_from_path(path: str | Path) -> RedactionBackend:
    return build_backend(load_backend_config(path))

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

from ..core.paths import expand_env_placeholders
from .base import RedactionBackend
from .opf import OpfBackend
from .hybrid_opf_qwen import HybridOpfQwenBackend
from .qwen_lora import QwenLoraBackend
from .qwen4b_tokencls import Qwen4BTokenClsBackend


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


def _build_hybrid_opf_qwen(cfg: dict[str, Any]) -> RedactionBackend:
    return HybridOpfQwenBackend(
        name=cfg["name"],
        model_version=cfg["model_version"],
        supported_types=cfg["supported_types"],
        opf_checkpoint=cfg["opf_checkpoint"],
        opf_label_space=cfg.get("opf_label_space"),
        qwen_backbone_path=cfg.get("qwen_backbone_path", ""),
        qwen_head_checkpoint=cfg.get("qwen_head_checkpoint", ""),
        qwen_temperature=float(cfg.get("qwen_temperature", 1.035854)),
        qwen_label_space=cfg.get("qwen_label_space"),
        qwen_loader_mode=cfg.get("qwen_loader_mode", "causal_lm"),
        qwen_expected_hidden_size=cfg.get("qwen_expected_hidden_size"),
        qwen_expected_loader_mode=cfg.get("qwen_expected_loader_mode"),
        qwen_lora_adapter_path=cfg.get("qwen_lora_adapter_path"),
        pii_project_root=cfg.get("pii_project_root", ""),
        dtype=cfg.get("dtype", "bf16"),
        device=cfg.get("device", "cuda"),
        output_top_k=int(cfg.get("output_top_k", 5)),
        redact_threshold=float(cfg.get("redact_threshold", 0.40)),
        review_threshold=float(cfg.get("review_threshold", 0.20)),
        qwen_verifier_enabled=bool(cfg.get("qwen_verifier_enabled", False)),
        qwen_verifier_types=cfg.get("qwen_verifier_types"),
        qwen_verifier_max_spans=int(cfg.get("qwen_verifier_max_spans", 4)),
        qwen_verifier_non_pii_threshold=float(cfg.get("qwen_verifier_non_pii_threshold", 0.70)),
        qwen_verifier_wrong_type_threshold=float(cfg.get("qwen_verifier_wrong_type_threshold", 0.80)),
        qwen_verifier_require_trigger=bool(cfg.get("qwen_verifier_require_trigger", False)),
        qwen_verifier_min_risk_score=float(cfg.get("qwen_verifier_min_risk_score", 0.25)),
        qwen_verifier_low_top1_threshold=float(cfg.get("qwen_verifier_low_top1_threshold", 0.70)),
        per_label_thresholds_path=cfg.get("per_label_thresholds_path"),
    )



def _build_qwen4b_tokencls(cfg: dict[str, Any]) -> RedactionBackend:
    return Qwen4BTokenClsBackend(
        name=cfg["name"],
        model_version=cfg["model_version"],
        supported_types=cfg["supported_types"],
        model_path=cfg.get("model_path", "/home/admin/model/Qwen3.5-4B-Base"),
        checkpoint_path=cfg.get("checkpoint_path", ""),
        token_label_to_id_path=cfg.get("token_label_to_id_path", ""),
        id_to_token_label_path=cfg.get("id_to_token_label_path", ""),
        max_seq_len=int(cfg.get("max_seq_len", 4096)),
        dtype=cfg.get("dtype", "bf16"),
        device=cfg.get("device", "cuda"),
        output_top_k=int(cfg.get("output_top_k", 5)),
        pii_project_root=cfg.get("pii_project_root", ""),
    )


BACKEND_TYPES: dict[str, Callable[[dict[str, Any]], RedactionBackend]] = {
    "qwen_lora": _build_qwen_lora,
    "opf": _build_opf,
    "hybrid_opf_qwen": _build_hybrid_opf_qwen,
    "qwen4b_tokencls": _build_qwen4b_tokencls,
}


def load_backend_config(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return expand_env_placeholders(raw)


def build_backend(cfg: dict[str, Any]) -> RedactionBackend:
    btype = cfg.get("type")
    if btype not in BACKEND_TYPES:
        raise ValueError(f"Unknown backend type: {btype!r}. Known: {sorted(BACKEND_TYPES)}")
    return BACKEND_TYPES[btype](cfg)


def build_backend_from_path(path: str | Path) -> RedactionBackend:
    return build_backend(load_backend_config(path))

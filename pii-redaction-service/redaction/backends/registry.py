"""Backend registry / factory for the deployable hybrid service."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..core.paths import expand_env_placeholders
from .base import RedactionBackend
from .hybrid_opf_qwen import HybridOpfQwenBackend


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

BACKEND_TYPES: dict[str, Callable[[dict[str, Any]], RedactionBackend]] = {
    "hybrid_opf_qwen": _build_hybrid_opf_qwen,
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

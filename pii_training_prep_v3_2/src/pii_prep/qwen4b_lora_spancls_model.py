"""Qwen4B LoRA Span Classification Model.

Loads Qwen3.5-4B-Base with causal_lm loader, applies LoRA adapters,
and adds an 80-class span classification head.

Backbone loading follows the corrected causal_lm pattern:
    AutoModelForCausalLM.from_pretrained(...) → .model (inner text model)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class Qwen4BLoRASpanCls(nn.Module):
    """Qwen4B backbone + LoRA + 80-class SpanCls head."""

    def __init__(
        self,
        backbone_path: str,
        head_checkpoint: Optional[str] = None,
        num_labels: int = 80,
        hidden_size: int = 2560,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list[str] | None = None,
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        # Load backbone as AutoModelForCausalLM, then take inner .model
        causal = AutoModelForCausalLM.from_pretrained(
            backbone_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        self.backbone = causal.model  # inner text model (corrected loader)
        self.config = causal.config
        del causal  # free the LM head
        torch.cuda.empty_cache()

        # Apply LoRA
        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        # gradient checkpointing disabled for speed — enough VRAM on DGX

        # Classification head: Linear(hidden_size → num_labels)
        self.head = nn.Linear(hidden_size, num_labels, bias=True)

        # Cast head to match backbone dtype
        self.head = self.head.to(dtype=dtype)

        # Initialize head from baseline checkpoint if provided
        if head_checkpoint is not None:
            self._init_head_from_checkpoint(head_checkpoint)
            self.head = self.head.to(dtype=dtype)  # re-cast after loading

    def _init_head_from_checkpoint(self, path: str):
        """Load head weights from frozen baseline checkpoint."""
        sd = torch.load(path, map_location="cpu")
        if "head_state_dict" in sd:
            hsd = sd["head_state_dict"]
            self.head.load_state_dict({"weight": hsd["weight"], "bias": hsd["bias"]})
        elif "weight" in sd:
            self.head.load_state_dict(sd)
        else:
            raise KeyError(f"Cannot find head weights in checkpoint at {path}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_starts: torch.Tensor,
        span_ends: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning [batch, num_labels] logits.

        Pooling: last token of each span — must match the wrapper's
        `_pool_last_token` in qwen_spancls_inference.py, otherwise the head
        learns a different input distribution than what it sees at deploy.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            span_starts: [batch] integer positions
            span_ends: [batch] integer positions (exclusive end)

        Returns:
            logits: [batch, num_labels]
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        batch_size = hidden.size(0)
        span_embeddings = torch.zeros(batch_size, self.hidden_size, device=hidden.device, dtype=hidden.dtype)

        for i in range(batch_size):
            start = span_starts[i].item()
            end = span_ends[i].item()
            end = min(end, hidden.size(1))
            if end > start:
                span_embeddings[i] = hidden[i, end - 1]

        return self.head(span_embeddings)

    def forward_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_starts: torch.Tensor,
        span_ends: torch.Tensor,
    ) -> torch.Tensor:
        """Optimized batch forward: compute hidden states once, pool many spans.

        Pooling: last token of span (matches inference wrapper).

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            span_starts: [num_spans]
            span_ends: [num_spans]

        Returns:
            logits: [num_spans, num_labels]
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = outputs.last_hidden_state  # [1, seq_len, hidden_size]

        logits_list = []
        for i in range(len(span_starts)):
            start = span_starts[i].item()
            end = span_ends[i].item()
            end = min(end, hidden.size(1))
            if end > start:
                span_emb = hidden[0, end - 1]
            else:
                span_emb = torch.zeros(self.hidden_size, device=hidden.device, dtype=hidden.dtype)
            logits_list.append(self.head(span_emb))

        return torch.stack(logits_list, dim=0)

    def get_lora_params(self):
        """Return parameters that should receive LoRA learning rates."""
        lora_params = []
        for name, param in self.named_parameters():
            if "lora_" in name:
                lora_params.append(param)
        return lora_params

    def get_head_params(self):
        """Return head parameters (should use higher LR)."""
        return list(self.head.parameters())

    def save_lora(self, path: str):
        """Save LoRA adapter weights only."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(path)

    def save_head(self, path: str):
        """Save classification head."""
        torch.save(self.head.state_dict(), path)

    def save_full(self, lora_path: str, head_path: str):
        """Save both LoRA adapter and head."""
        self.save_lora(lora_path)
        self.save_head(head_path)

    @classmethod
    def from_pretrained(
        cls,
        backbone_path: str,
        lora_path: str,
        head_path: str,
        num_labels: int = 80,
        hidden_size: int = 2560,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Load pre-trained LoRA model for inference."""
        import json
        # Load adapter config
        adapter_config = json.load(open(Path(lora_path) / "adapter_config.json"))
        r = adapter_config.get("r", 16)
        alpha = adapter_config.get("lora_alpha", 32)
        dropout = adapter_config.get("lora_dropout", 0.05)
        target_modules = adapter_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])

        model = cls(
            backbone_path=backbone_path,
            num_labels=num_labels,
            hidden_size=hidden_size,
            lora_r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            lora_target_modules=target_modules,
            dtype=dtype,
        )

        # Load LoRA weights
        from peft import PeftModel
        model.backbone = PeftModel.from_pretrained(model.backbone, lora_path)

        # Load head
        model.head.load_state_dict(torch.load(head_path, map_location="cpu"))

        return model

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.train(mode)
        return self

    def eval(self):
        return self.train(False)

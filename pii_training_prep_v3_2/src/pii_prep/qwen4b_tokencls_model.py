from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class Qwen4BTokenClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int = 2560,
        num_labels: int = 317,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        with torch.set_grad_enabled(self.training):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[-1]

        logits = self.classifier(hidden_states.float())

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )

        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states,
        }

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask, labels=None)
        return result['logits']

    def trainable_parameters(self) -> dict[str, int]:
        return {
            'backbone_trainable': sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
            'head_trainable': sum(p.numel() for p in self.classifier.parameters() if p.requires_grad),
            'total_trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def load_model(
    model_path: str,
    num_labels: int = 317,
    freeze_backbone: bool = True,
    device: torch.device | None = None,
    use_bf16: bool = False,
) -> tuple[Qwen4BTokenClassifier, Any, int]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    backbone = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
    )

    config = backbone.config
    if hasattr(config, 'hidden_size') and config.hidden_size is not None:
        hidden_size = config.hidden_size
    elif hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
        hidden_size = config.text_config.hidden_size
    else:
        hidden_size = 2560

    model = Qwen4BTokenClassifier(
        backbone=backbone,
        hidden_size=hidden_size,
        num_labels=num_labels,
        freeze_backbone=freeze_backbone,
    )

    if device is not None:
        model.to(device)

    return model, tokenizer, hidden_size

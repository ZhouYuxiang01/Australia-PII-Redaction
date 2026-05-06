from __future__ import annotations

import json
import torch
import torch.nn.functional as F
from collections import Counter
from typing import Any
from pathlib import Path


def load_tokencls_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if limit is not None and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


class TokenClsCollator:
    def __init__(self, tokenizer: Any, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_offset_mapping = []
        batch_record_ids = []
        batch_texts = []

        for row in rows:
            input_ids = row['input_ids']
            labels = row['labels']
            attn_mask = row.get('attention_mask', [1] * len(input_ids))
            offsets = row.get('offset_mapping', [[0, 0]] * len(input_ids))

            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]
                attn_mask = attn_mask[:self.max_length]
                offsets = offsets[:self.max_length]

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_attention_mask.append(torch.tensor(attn_mask, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))
            batch_offset_mapping.append(offsets)
            batch_record_ids.append(row.get('record_id', ''))
            batch_texts.append(row.get('text', ''))

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            batch_attention_mask, batch_first=True, padding_value=0
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels, batch_first=True, padding_value=-100
        )

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'labels': padded_labels,
            'offset_mapping': batch_offset_mapping,
            'record_ids': batch_record_ids,
            'texts': batch_texts,
        }


def compute_token_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    o_label_id: int = 0,
) -> dict[str, float]:
    preds = logits.argmax(dim=-1)
    mask = labels != -100

    total = mask.sum().item()
    if total == 0:
        return {
            'token_accuracy': 0.0,
            'o_token_accuracy': 0.0,
            'positive_token_accuracy': 0.0,
            'total_tokens': 0,
            'positive_tokens': 0,
            'o_tokens': 0,
        }

    correct = (preds == labels) & mask
    accuracy = correct.sum().item() / total

    o_mask = (labels == o_label_id) & mask
    o_total = o_mask.sum().item()
    o_correct = (correct & o_mask).sum().item()
    o_accuracy = o_correct / max(1, o_total)

    pos_mask = (labels != o_label_id) & mask
    pos_total = pos_mask.sum().item()
    pos_correct = (correct & pos_mask).sum().item()
    pos_accuracy = pos_correct / max(1, pos_total)

    return {
        'token_accuracy': round(accuracy, 6),
        'o_token_accuracy': round(o_accuracy, 6),
        'positive_token_accuracy': round(pos_accuracy, 6),
        'total_tokens': total,
        'positive_tokens': pos_total,
        'o_tokens': o_total,
    }


def compute_per_label_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int = 317,
) -> dict[str, Any]:
    preds = logits.argmax(dim=-1)
    mask = labels != -100

    per_label = {}
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    for lid in range(1, num_labels):
        tp = ((preds == lid) & (labels == lid) & mask).sum().item()
        fp = ((preds == lid) & (labels != lid) & mask).sum().item()
        fn = ((preds != lid) & (labels == lid) & mask).sum().item()

        if tp + fp + fn > 0:
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-9, precision + recall)
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        if tp + fp + fn > 0:
            per_label[lid] = {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'support': tp + fn,
            }

    micro_precision = micro_tp / max(1, micro_tp + micro_fp)
    micro_recall = micro_tp / max(1, micro_tp + micro_fn)
    micro_f1 = 2 * micro_precision * micro_recall / max(1e-9, micro_precision + micro_recall)

    return {
        'micro_precision': round(micro_precision, 4),
        'micro_recall': round(micro_recall, 4),
        'micro_f1': round(micro_f1, 4),
        'per_label': per_label,
    }

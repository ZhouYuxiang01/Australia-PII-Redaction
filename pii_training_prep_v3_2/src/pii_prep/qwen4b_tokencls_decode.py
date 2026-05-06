from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any


BIOES_TAGS = ['B', 'I', 'E', 'S', 'O']

VALID_PREV_FOR_CURRENT = {
    'B': {'E', 'S', 'O', 'B'},
    'I': {'B', 'I'},
    'E': {'B', 'I'},
    'S': {'E', 'S', 'O', 'B'},
    'O': {'E', 'S', 'O', 'B'},
}

VALID_NEXT_FOR_CURRENT = {
    'B': {'I', 'E'},
    'I': {'I', 'E'},
    'E': {'B', 'S', 'O'},
    'S': {'B', 'S', 'O'},
    'O': {'B', 'S', 'O'},
}


def token_label_to_tag(label_id: int, id_to_label: dict[str, str]) -> str:
    if label_id == 0:
        return 'O'
    label_name = id_to_label.get(str(label_id), 'O')
    if label_name.startswith('B-'):
        return 'B'
    elif label_name.startswith('I-'):
        return 'I'
    elif label_name.startswith('E-'):
        return 'E'
    elif label_name.startswith('S-'):
        return 'S'
    return 'O'


def token_label_to_type(label_id: int, id_to_label: dict[str, str]) -> str:
    if label_id <= 0:
        return 'O'
    label_name = id_to_label.get(str(label_id), 'O')
    if '-' in label_name:
        return label_name.split('-', 1)[1]
    return label_name


def greedy_decode(
    logits: torch.Tensor,
    id_to_label: dict[str, str],
) -> list[int]:
    preds = logits.argmax(dim=-1).tolist()
    return preds


def constrained_decode(
    logits: torch.Tensor,
    id_to_label: dict[str, str],
) -> list[int]:
    probs = F.softmax(logits, dim=-1)
    n = logits.shape[0]

    tags = [token_label_to_tag(probs[0].argmax().item(), id_to_label)]

    for i in range(1, n):
        prev_tag = tags[-1]
        valid_next = VALID_NEXT_FOR_CURRENT.get(prev_tag, {'B', 'S', 'O'})

        valid_ids = []
        for lid in range(logits.shape[1]):
            tag = token_label_to_tag(lid, id_to_label)
            if tag in valid_next:
                valid_ids.append(lid)

        if not valid_ids:
            valid_ids = [0]

        valid_probs = probs[i, valid_ids]
        best_valid_idx = valid_probs.argmax().item()
        best_label_id = valid_ids[best_valid_idx]
        tags.append(token_label_to_tag(best_label_id, id_to_label))

    return []
    # Reconstruct from tags
    # We need actual label_ids, not just tags
    # Let's redo this properly


def constrained_decode_v2(
    logits: torch.Tensor,
    id_to_label: dict[str, str],
    label_to_id: dict[str, int],
) -> list[int]:
    probs = F.softmax(logits, dim=-1)
    n = logits.shape[0]

    preds = [0] * n

    for i in range(n):
        if i == 0:
            valid_tags = {'B', 'S', 'O'}
        else:
            prev_tag = token_label_to_tag(preds[i - 1], id_to_label)
            valid_tags = VALID_NEXT_FOR_CURRENT.get(prev_tag, {'B', 'S', 'O'})

        valid_ids = []
        for lid in range(logits.shape[1]):
            tag = token_label_to_tag(lid, id_to_label)
            if tag in valid_tags:
                valid_ids.append(lid)

        if not valid_ids:
            valid_ids = [0]

        best_score = -float('inf')
        best_id = valid_ids[0]
        for lid in valid_ids:
            score = probs[i, lid].item()
            if score > best_score:
                best_score = score
                best_id = lid

        preds[i] = best_id

    return preds


def logits_to_spans(
    logits: torch.Tensor,
    offset_mapping: list[tuple[int, int]],
    id_to_label: dict[str, str],
    label_to_id: dict[str, int],
    constrained: bool = True,
) -> list[dict[str, Any]]:
    if constrained:
        preds = constrained_decode_v2(logits, id_to_label, label_to_id)
    else:
        preds = greedy_decode(logits, id_to_label)

    probs = F.softmax(logits, dim=-1)
    confidence = probs[torch.arange(len(preds)), preds].tolist()

    spans = []
    i = 0
    n = len(preds)

    while i < n:
        lid = preds[i]
        tag = token_label_to_tag(lid, id_to_label)

        if tag == 'O':
            i += 1
            continue

        if tag == 'S':
            pii_type = token_label_to_type(lid, id_to_label)
            if pii_type != 'O' and i < len(offset_mapping):
                start, end = offset_mapping[i]
                value = ''
                if start < end:
                    spans.append({
                        'start': start, 'end': end,
                        'type': pii_type,
                        'confidence': round(confidence[i], 6),
                        'num_tokens': 1,
                    })
            i += 1
            continue

        if tag == 'B':
            pii_type = token_label_to_type(lid, id_to_label)
            span_start_idx = i
            span_end_idx = i + 1
            span_confidence = [confidence[i]]

            j = i + 1
            while j < n:
                next_lid = preds[j]
                next_tag = token_label_to_tag(next_lid, id_to_label)
                if next_tag in ('I', 'E'):
                    span_end_idx = j + 1
                    span_confidence.append(confidence[j])
                    if next_tag == 'E':
                        j += 1
                        break
                    j += 1
                else:
                    break

            if span_start_idx < len(offset_mapping) and span_end_idx <= len(offset_mapping):
                start = offset_mapping[span_start_idx][0]
                end = offset_mapping[span_end_idx - 1][1]
                if start < end and pii_type != 'O':
                    spans.append({
                        'start': start, 'end': end,
                        'type': pii_type,
                        'confidence': round(sum(span_confidence) / len(span_confidence), 6),
                        'num_tokens': span_end_idx - span_start_idx,
                    })

            i = max(span_end_idx, i + 1)
            continue

        i += 1

    # Filter whitespace-only spans
    spans = [s for s in spans
             if s['start'] < s['end'] and s['confidence'] > 0]

    # Resolve overlapping spans (keep higher confidence, longer)
    spans.sort(key=lambda s: (-s['confidence'], -(s['end'] - s['start'])))
    resolved = []
    for span in spans:
        overlaps = False
        for rs in resolved:
            if span['start'] < rs['end'] and span['end'] > rs['start']:
                overlaps = True
                break
        if not overlaps:
            resolved.append(span)

    resolved.sort(key=lambda s: s['start'])
    return resolved

'''
Stage 5 Task 1: Build Qwen4B token classification datasets.

Converts char-level span annotations from data/splits/*.jsonl into
token-level BIOES label sequences for Qwen3.5-4B token classification.

Output:
  data/train/qwen4b_tokencls_train.jsonl
  data/train/qwen4b_tokencls_dev.jsonl
  data/train/qwen4b_tokencls_test.jsonl

Reports:
  reports/stage5_qwen4b_tokencls_dataset_report.json
  reports/stage5_qwen4b_tokencls_alignment_errors.json
  reports/stage5_qwen4b_tokencls_overlap_conflicts.json
  reports/stage5_qwen4b_tokencls_label_distribution.json
'''
from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path('/home/admin/ZYX/pii_training_prep_v3_2')
MODEL_PATH = '/home/admin/model/Qwen3.5-4B-Base'
MAX_SEQ_LEN = 4096
STRIDE = 512

BIOES_TAGS = ['B', 'I', 'E', 'S']

# Data Classification weight rank for overlap resolution
WEIGHT_RANK = {'Highly Protected': 3, 'Protected': 2, 'Public': 1, None: 0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def build_classification_weights(csv_path: Path, pii_labels: list[str]) -> dict[str, str]:
    '''
    Map canonical PII labels to Data Classification from CSV.
    Uses the _canonical_label overrides from integrated_pipeline.py
    plus fuzzy matching.
    '''
    # Build CSV name -> classification map
    csv_map: dict[str, str] = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = (row.get('Name') or '').strip()
            classification = (row.get('Data Classification') or '').strip()
            if not name or not classification:
                continue
            csv_map[name] = classification

    # Canonicalization overrides (from integrated_pipeline.py)
    overrides: dict[str, str] = {}
    for csv_name, classification in csv_map.items():
        code = re.sub(r'[^0-9A-Za-z]+', '_', csv_name.split(' - ')[0].split(' / ')[0].strip())
        code = re.sub(r'_+', '_', code).strip('_').upper()
        overrides[code] = classification

    result: dict[str, str] = {}
    for label in pii_labels:
        if label in overrides:
            result[label] = overrides[label]
        else:
            result[label] = 'Protected'  # default
    return result


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------
def resolve_overlaps(
    spans: list[dict[str, Any]],
    classification_weights: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    '''
    Resolve overlapping spans deterministically.

    Priority:
      1. Higher Data Classification (Highly Protected > Protected > Public)
      2. Higher source confidence (teacher_confidence or training_weight)
      3. Longer span
      4. Earlier start
      5. Lexicographic label name
    '''
    if len(spans) <= 1:
        return list(spans), []

    sorted_spans = sorted(spans, key=lambda s: (s['start'], -(s['end'] - s['start'])))
    resolved: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for span in sorted_spans:
        overlapping = [
            (i, rs) for i, rs in enumerate(resolved)
            if span['start'] < rs['end'] and span['end'] > rs['start']
        ]

        if not overlapping:
            resolved.append(span)
            continue

        # Check if current span wins all conflicts
        keep_current = True
        for oi, other in overlapping:
            curr_weight = WEIGHT_RANK.get(
                classification_weights.get(span.get('top_type', ''), None), 0
            )
            other_weight = WEIGHT_RANK.get(
                classification_weights.get(other.get('top_type', ''), None), 0
            )

            if curr_weight > other_weight:
                continue  # current wins
            elif curr_weight < other_weight:
                keep_current = False
                break
            else:
                # Same weight tier - check confidence
                curr_conf = span.get('teacher_confidence', span.get('training_weight', 0.5))
                other_conf = other.get('teacher_confidence', other.get('training_weight', 0.5))
                if curr_conf > other_conf:
                    continue
                elif curr_conf < other_conf:
                    keep_current = False
                    break
                else:
                    # Same confidence - check span length
                    curr_len = span['end'] - span['start']
                    other_len = other['end'] - other['start']
                    if curr_len > other_len:
                        continue
                    elif curr_len < other_len:
                        keep_current = False
                        break
                    else:
                        # Same length - earlier start wins
                        if span['start'] < other['start']:
                            continue
                        elif span['start'] > other['start']:
                            keep_current = False
                            break
                        else:
                            # Tie-break by label name
                            if span.get('top_type', '') < other.get('top_type', ''):
                                continue
                            else:
                                keep_current = False
                                break

        if keep_current:
            # Remove all overlapping spans, add current
            for oi, other in overlapping:
                conflicts.append({
                    'winner': {
                        'start': span['start'], 'end': span['end'],
                        'value': span.get('value', ''),
                        'top_type': span.get('top_type', ''),
                    },
                    'loser': {
                        'start': other['start'], 'end': other['end'],
                        'value': other.get('value', ''),
                        'top_type': other.get('top_type', ''),
                    },
                    'resolution_reason': 'priority_tiebreak',
                })
            resolved = [rs for i, rs in enumerate(resolved)
                        if i not in [oi for oi, _ in overlapping]]
            resolved.append(span)
        else:
            # Current span loses
            for oi, other in overlapping:
                conflicts.append({
                    'winner': {
                        'start': other['start'], 'end': other['end'],
                        'value': other.get('value', ''),
                        'top_type': other.get('top_type', ''),
                    },
                    'loser': {
                        'start': span['start'], 'end': span['end'],
                        'value': span.get('value', ''),
                        'top_type': span.get('top_type', ''),
                    },
                    'resolution_reason': 'priority_tiebreak',
                })

    return resolved, conflicts


# ---------------------------------------------------------------------------
# BIOES label assignment
# ---------------------------------------------------------------------------
def assign_bioes_labels(
    token_offsets: list[tuple[int, int]],
    spans: list[dict[str, Any]],
    label_to_id: dict[str, int],
    o_id: int,
    ignore_id: int,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    '''
    Assign BIOES token labels based on char-level spans.

    Returns (labels, included_spans, alignment_errors).
    '''
    n_tokens = len(token_offsets)
    labels = [o_id] * n_tokens
    included_spans: list[dict[str, Any]] = []
    alignment_errors: list[dict[str, Any]] = []

    for span in spans:
        start = int(span['start'])
        end = int(span['end'])
        top_type = str(span.get('top_type', ''))
        value = str(span.get('value', ''))

        if not top_type:
            alignment_errors.append({
                'span': {'start': start, 'end': end, 'value': value},
                'reason': 'missing_top_type',
            })
            continue

        if not value.strip():
            alignment_errors.append({
                'span': {'start': start, 'end': end, 'value': value, 'top_type': top_type},
                'reason': 'whitespace_only_value',
            })
            continue

        # Find tokens that overlap with the char span
        token_indices = []
        for idx, (tok_start, tok_end) in enumerate(token_offsets):
            if tok_end <= tok_start:
                continue  # skip special tokens
            if tok_end <= start:
                continue
            if tok_start >= end:
                break
            if tok_start < end and tok_end > start:
                token_indices.append(idx)

        if not token_indices:
            alignment_errors.append({
                'span': {'start': start, 'end': end, 'value': value, 'top_type': top_type},
                'reason': 'no_tokens_overlap_span',
            })
            continue

        # Check alignment quality: does token span roughly match char span?
        first_token_start = token_offsets[token_indices[0]][0]
        last_token_end = token_offsets[token_indices[-1]][1]

        if first_token_start > end or last_token_end < start:
            alignment_errors.append({
                'span': {'start': start, 'end': end, 'value': value, 'top_type': top_type},
                'token_range': [first_token_start, last_token_end],
                'reason': 'token_and_char_bounds_mismatch',
            })
            continue

        # Assign BIOES labels
        if len(token_indices) == 1:
            label_id = label_to_id.get(f'S-{top_type}', o_id)
            labels[token_indices[0]] = label_id
        else:
            for i, ti in enumerate(token_indices):
                if i == 0:
                    labels[ti] = label_to_id.get(f'B-{top_type}', o_id)
                elif i == len(token_indices) - 1:
                    labels[ti] = label_to_id.get(f'E-{top_type}', o_id)
                else:
                    labels[ti] = label_to_id.get(f'I-{top_type}', o_id)

        included_spans.append({
            'start': start,
            'end': end,
            'value': value,
            'top_type': top_type,
            'token_start': token_indices[0],
            'token_end': token_indices[-1] + 1,
            'num_tokens': len(token_indices),
        })

    return labels, included_spans, alignment_errors


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def window_sequence(
    input_ids: list[int],
    attention_mask: list[int],
    offset_mapping: list[tuple[int, int]],
    labels: list[int],
    char_start: int,
    char_end: int,
    max_seq_len: int = MAX_SEQ_LEN,
    stride: int = STRIDE,
) -> list[dict[str, Any]]:
    '''
    Split a long token sequence into overlapping windows.

    Returns list of window dicts.
    '''
    total_len = len(input_ids)
    if total_len <= max_seq_len:
        return [{
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'offset_mapping': offset_mapping,
            'labels': labels,
            'original_char_start': char_start,
            'original_char_end': char_end,
        }]

    windows = []
    window_start = 0
    while window_start < total_len:
        window_end = min(window_start + max_seq_len, total_len)
        win_input_ids = input_ids[window_start:window_end]
        win_attention_mask = attention_mask[window_start:window_end]
        win_offsets = offset_mapping[window_start:window_end]
        win_labels = labels[window_start:window_end]

        # Determine char range for this window
        win_char_start = char_start
        if win_offsets:
            win_char_start = win_offsets[0][0] if win_offsets[0][0] != win_offsets[0][1] else char_start
            valid_offsets = [(a, b) for a, b in win_offsets if a < b]
            win_char_end = valid_offsets[-1][1] if valid_offsets else char_end
        else:
            win_char_end = char_end

        # Mark partial BIOES spans at boundaries as O
        # If window starts in the middle of a span (I-tag or E-tag),
        # convert remaining tags to O
        for i, lid in enumerate(win_labels):
            if lid == -100:
                continue
            label_name = ''  # We'll handle via pattern
            if i == 0 and window_start > 0:
                # Check if we're in the middle of a span by looking at original labels
                prev_lid = labels[window_start - 1] if window_start > 0 else -100
                curr_lid = win_labels[0]
                # If previous token was part of a span and current is too,
                # but current starts with I/E, reset to O
                if prev_lid > 0 and curr_lid > 0:
                    win_labels[i] = 0  # O

        if window_end >= total_len:
            # Last window
            windows.append({
                'input_ids': win_input_ids,
                'attention_mask': win_attention_mask,
                'offset_mapping': win_offsets,
                'labels': win_labels,
                'original_char_start': win_char_start,
                'original_char_end': win_char_end,
            })
            break

        windows.append({
            'input_ids': win_input_ids,
            'attention_mask': win_attention_mask,
            'offset_mapping': win_offsets,
            'labels': win_labels,
            'original_char_start': win_char_start,
            'original_char_end': win_char_end,
        })
        window_start += max_seq_len - stride

    return windows


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------
def build_dataset(split: str, records: list[dict[str, Any]],
                  tokenizer: Any, label_to_id: dict[str, int],
                  o_id: int, ignore_id: int,
                  classification_weights: dict[str, str],
                  max_seq_len: int, stride: int,
                  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    '''
    Build token classification dataset for one split.

    Returns (output_rows, stats).
    '''
    output_rows: list[dict[str, Any]] = []
    all_alignment_errors: list[dict[str, Any]] = []
    all_overlap_conflicts: list[dict[str, Any]] = []
    label_counter: Counter[int] = Counter()
    skipped_partial: int = 0
    windowed_count: int = 0
    alignment_success: int = 0
    alignment_failure: int = 0

    for rec_idx, rec in enumerate(records):
        if (rec_idx + 1) % 10000 == 0:
            print(f'    {split}: processed {rec_idx + 1}/{len(records)} records', flush=True)

        record_id = rec.get('id', f'unknown-{rec_idx}')
        text = rec.get('text', '')
        raw_spans = rec.get('spans', [])

        # Tokenize
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_len,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        offset_mapping = encoded['offset_mapping']

        # Convert offset_mapping to list of tuples
        token_offsets = [(int(a), int(b)) for a, b in offset_mapping]

        # Determine char range
        all_offsets = [(a, b) for a, b in token_offsets if a < b]
        char_start = all_offsets[0][0] if all_offsets else 0
        char_end = all_offsets[-1][1] if all_offsets else len(text)

        # Resolve overlapping spans
        if raw_spans:
            resolved_spans, conflicts = resolve_overlaps(raw_spans, classification_weights)
        else:
            resolved_spans, conflicts = [], []

        for conflict in conflicts:
            conflict['record_id'] = record_id
            conflict['split'] = split
        all_overlap_conflicts.extend(conflicts)

        # Assign BIOES labels
        labels, included_spans, errors = assign_bioes_labels(
            token_offsets, resolved_spans, label_to_id, o_id, ignore_id
        )

        for err in errors:
            err['record_id'] = record_id
            err['split'] = split
        all_alignment_errors.extend(errors)
        alignment_failure += len(errors)
        alignment_success += len(included_spans)

        # Mark padding/special tokens as ignore_index
        n = len(input_ids)
        while len(labels) < n:
            labels.append(o_id)
        labels = labels[:n]

        for i in range(n):
            if attention_mask[i] == 0:
                labels[i] = ignore_id
            # Special tokens (token_offsets where start == end)
            ts, te = token_offsets[i]
            if ts == te and ts == 0 and te == 0:
                labels[i] = ignore_id

        # Count labels
        for lid in labels:
            if lid != ignore_id:
                label_counter[lid] += 1

        # Check if windowing is needed
        total_len = len(input_ids)

        if total_len > max_seq_len:
            windows = window_sequence(
                input_ids, attention_mask, token_offsets,
                labels, char_start, char_end,
                max_seq_len=max_seq_len, stride=stride,
            )
            windowed_count += 1
        else:
            windows = [{
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'offset_mapping': token_offsets,
                'labels': labels,
                'original_char_start': char_start,
                'original_char_end': char_end,
            }]

        # Build output rows
        for win_idx, win in enumerate(windows):
            # Skip windows with no positive labels (still include for eval though)
            win_included = [
                span for span in included_spans
                if span['token_start'] >= 0 and span['token_end'] <= len(win['labels'])
            ]

            # Check for partial spans at window boundaries
            win_skipped = []
            for span in included_spans:
                if span['token_start'] < 0 or span['token_end'] > len(win['labels']):
                    win_skipped.append({
                        'start': span['start'], 'end': span['end'],
                        'value': span['value'], 'top_type': span['top_type'],
                        'reason': 'span_cut_by_window',
                    })
                    skipped_partial += 1

            output_rows.append({
                'record_id': record_id,
                'split': split,
                'window_index': win_idx,
                'text': text,
                'input_ids': win['input_ids'],
                'attention_mask': win['attention_mask'],
                'offset_mapping': win['offset_mapping'],
                'labels': win['labels'],
                'original_char_start': win['original_char_start'],
                'original_char_end': win['original_char_end'],
                'included_spans': win_included,
                'skipped_spans': win_skipped,
                'source': rec.get('metadata', {}).get('source_type', '') if isinstance(rec.get('metadata'), dict) else '',
            })

    stats = {
        'split': split,
        'input_records': len(records),
        'output_windows': len(output_rows),
        'windowed_record_count': windowed_count,
        'alignment_success': alignment_success,
        'alignment_failure': alignment_failure,
        'overlap_conflict_count': len(all_overlap_conflicts),
        'skipped_partial_span_count': skipped_partial,
        'label_counts': dict(label_counter.most_common()),
        'o_token_count': label_counter.get(o_id, 0),
        'positive_token_count': sum(v for k, v in label_counter.items() if k > 0),
        'total_labeled_tokens': sum(label_counter.values()),
    }

    return output_rows, {
        'rows': output_rows,
        'stats': stats,
        'alignment_errors': all_alignment_errors,
        'overlap_conflicts': all_overlap_conflicts,
    }


def main() -> int:
    t0 = time.time()

    # Load label space
    label_space_path = ROOT / 'pii_schema' / 'token_label_space_317.json'
    if not label_space_path.exists():
        print(f'ERROR: label space not found at {label_space_path}', file=sys.stderr)
        return 1

    label_space = json.loads(label_space_path.read_text(encoding='utf-8'))
    pii_labels = [l for l in label_space['token_labels'] if l != 'O']
    # Extract base PII labels from BIOES labels
    base_pii_labels = sorted(set(
        l[2:] for l in label_space['token_labels'] if l.startswith(('B-', 'I-', 'E-', 'S-'))
    ))
    assert len(base_pii_labels) == 79, f'Expected 79 base PII labels, got {len(base_pii_labels)}'

    # Load label-to-id mapping
    l2i = json.loads((ROOT / 'pii_schema' / 'token_label_to_id_317.json').read_text(encoding='utf-8'))
    label_to_id = l2i['label_to_id']
    o_id = label_to_id['O']
    ignore_id = -100

    # Load classification weights from CSV
    csv_path = ROOT / 'docs' / 'Data Sensitivity.csv'
    classification_weights = build_classification_weights(csv_path, base_pii_labels)

    # Load tokenizer
    from transformers import AutoTokenizer
    print('Loading tokenizer...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f'Tokenizer loaded: pad_token_id={tokenizer.pad_token_id}', flush=True)

    # Process splits
    splits = ['train', 'dev', 'test']
    all_data: dict[str, dict] = {}
    report_stats: dict[str, Any] = {
        'stage': '5',
        'task': 'token_cls_dataset_builder',
        'label_space': {
            'num_token_labels': len(label_space['token_labels']),
            'num_pii_labels': 79,
            'bioes_tags': BIOES_TAGS,
            'ignore_index': ignore_id,
            'o_label_id': o_id,
        },
        'config': {
            'model_path': MODEL_PATH,
            'max_seq_len': MAX_SEQ_LEN,
            'stride': STRIDE,
        },
        'splits': {},
    }
    all_alignment_errors: list[dict[str, Any]] = []
    all_overlap_conflicts: list[dict[str, Any]] = []
    all_label_counts: Counter[int] = Counter()

    for split in splits:
        print(f'\nProcessing {split} split...', flush=True)
        input_path = ROOT / 'data' / 'splits' / f'{split}.jsonl'
        records = load_jsonl(input_path)
        print(f'  Loaded {len(records)} records', flush=True)

        output_rows, data = build_dataset(
            split, records, tokenizer, label_to_id, o_id, ignore_id,
            classification_weights, MAX_SEQ_LEN, STRIDE,
        )

        # Write output
        output_path = ROOT / 'data' / 'train' / f'qwen4b_tokencls_{split}.jsonl'
        write_jsonl(output_path, output_rows)
        print(f'  Wrote {len(output_rows)} windows to {output_path}', flush=True)

        all_data[split] = data
        report_stats['splits'][split] = data['stats']
        all_alignment_errors.extend(data['alignment_errors'])
        all_overlap_conflicts.extend(data['overlap_conflicts'])
        for k, v in data['stats']['label_counts'].items():
            all_label_counts[k] += v

    # Compute aggregate stats
    for split in splits:
        stats = report_stats['splits'][split]
        total_tokens = stats['total_labeled_tokens']
        o_ratio = stats['o_token_count'] / max(1, total_tokens)
        pos_ratio = stats['positive_token_count'] / max(1, total_tokens)
        stats['o_token_ratio'] = round(o_ratio, 6)
        stats['positive_token_ratio'] = round(pos_ratio, 6)

    # Generate reports
    reports_dir = ROOT / 'reports'

    # 1. Dataset report
    report_stats['total_alignment_errors'] = len(all_alignment_errors)
    report_stats['total_overlap_conflicts'] = len(all_overlap_conflicts)
    report_stats['wall_time_seconds'] = round(time.time() - t0, 3)
    write_json(reports_dir / 'stage5_qwen4b_tokencls_dataset_report.json', report_stats)

    # 2. Alignment errors
    write_json(
        reports_dir / 'stage5_qwen4b_tokencls_alignment_errors.json',
        {
            'total_errors': len(all_alignment_errors),
            'by_reason': dict(Counter(e.get('reason', 'unknown') for e in all_alignment_errors)),
            'examples': all_alignment_errors[:100],
        },
    )

    # 3. Overlap conflicts
    write_json(
        reports_dir / 'stage5_qwen4b_tokencls_overlap_conflicts.json',
        {
            'total_conflicts': len(all_overlap_conflicts),
            'examples': all_overlap_conflicts[:50],
        },
    )

    # 4. Label distribution
    id_to_label = json.loads(
        (ROOT / 'pii_schema' / 'id_to_token_label_317.json').read_text(encoding='utf-8')
    )['id_to_label']
    per_label = {}
    for lid, count in sorted(all_label_counts.items()):
        label_name = id_to_label.get(str(lid), f'UNKNOWN_{lid}')
        per_label[label_name] = count

    # Per-PII-type span counts from output files
    per_pii_spans: Counter[str] = Counter()
    for split in splits:
        for row in all_data[split]['rows']:
            for span in row.get('included_spans', []):
                per_pii_spans[span.get('top_type', 'UNKNOWN')] += 1

    write_json(
        reports_dir / 'stage5_qwen4b_tokencls_label_distribution.json',
        {
            'per_token_label_counts': per_label,
            'per_pii_type_span_counts': dict(per_pii_spans.most_common()),
            'total_token_labels': sum(all_label_counts.values()),
        },
    )

    # Summary
    elapsed = time.time() - t0
    print(f'\n{"=" * 60}')
    print(f'Dataset building complete in {elapsed:.1f}s')
    print(f'{"=" * 60}')
    for split in splits:
        s = report_stats['splits'][split]
        print(f'  {split}: {s["input_records"]} records -> {s["output_windows"]} windows')
        print(f'    O ratio: {s[o_token_ratio]:.4f}, pos ratio: {s[positive_token_ratio]:.4f}')
        print(f'    alignment errors: {s["alignment_failure"]}')
        print(f'    overlap conflicts: {s["overlap_conflict_count"]}')
        print(f'    skipped partial: {s["skipped_partial_span_count"]}')
    print(f'\n  Total alignment errors: {len(all_alignment_errors)}')
    print(f'  Total overlap conflicts: {len(all_overlap_conflicts)}')
    print(f'\nReports written to: {reports_dir}')
    for name in sorted([
        'stage5_qwen4b_tokencls_dataset_report.json',
        'stage5_qwen4b_tokencls_alignment_errors.json',
        'stage5_qwen4b_tokencls_overlap_conflicts.json',
        'stage5_qwen4b_tokencls_label_distribution.json',
    ]):
        p = reports_dir / name
        if p.exists():
            print(f'  {name}: {p.stat().st_size} bytes')

    # Acceptance checks
    print(f'\n{"=" * 60}')
    print('ACCEPTANCE CHECKS')
    print(f'{"=" * 60}')
    all_pass = True

    # Check label count
    label_count = len(label_space['token_labels'])
    print(f'  Label count: {label_count} (expected 317) -> {PASS if label_count == 317 else FAIL}')
    if label_count != 317:
        all_pass = False

    # Check no BIOES NON_PII
    no_nonpii = all('NON_PII' not in tl for tl in label_space['token_labels'])
    print(f'  No NON_PII BIOES: {no_nonpii} -> {PASS if no_nonpii else FAIL}')
    if not no_nonpii:
        all_pass = False

    # Check O exists
    print(f'  O label exists at id={o_id}: {PASS if o_id == 0 else FAIL}')
    if o_id != 0:
        all_pass = False

    # Check lengths match for first few rows
    len_ok = True
    for split in splits:
        rows = all_data[split]['rows']
        for row in rows[:5]:
            if len(row['input_ids']) != len(row['labels']):
                print(f'  Length mismatch in {split}: input_ids={len(row[input_ids])}, labels={len(row[labels])} -> FAIL')
                len_ok = False
                all_pass = False
                break
    if len_ok:
        print(f'  Length check (first 5 rows per split): PASS')

    # Check no invalid label IDs
    max_id = max(label_to_id.values())
    invalid_found = False
    for split in splits:
        for row in all_data[split]['rows'][:100]:
            for lid in row['labels']:
                if lid != ignore_id and (lid < 0 or lid > max_id):
                    print(f'  Invalid label ID {lid} in {split} -> FAIL')
                    invalid_found = True
                    all_pass = False
                    break
            if invalid_found:
                break
    if not invalid_found:
        print(f'  No invalid label IDs (first 100 rows per split): PASS')

    # Check ignore_index for special tokens
    print(f'  ignore_index={ignore_id}: PASS' if ignore_id == -100 else f'  ignore_index={ignore_id}: FAIL')

    print(f'\n  OVERALL: {"ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"}')

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())

"""Stage 5 Task 4: Dev/test evaluation and hybrid comparison."""
from __future__ import annotations

import json, sys, time, math
from pathlib import Path
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path('/home/admin/ZYX/pii_training_prep_v3_2')
MODEL_PATH = '/home/admin/model/Qwen3.5-4B-Base'
NUM_LABELS = 317
O_LABEL_ID = 0
IGNORE_INDEX = -100
BATCH_SIZE = 4

sys.path.insert(0, str(ROOT / 'src' / 'pii_prep'))


class TokenClsDataset(Dataset):
    def __init__(self, path):
        self.rows = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            'input_ids': torch.tensor(row['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(row.get('attention_mask', [1]*len(row['input_ids'])), dtype=torch.long),
            'labels': torch.tensor(row['labels'], dtype=torch.long),
            'offset_mapping': row.get('offset_mapping', []),
            'record_id': row.get('record_id', ''),
            'text': row.get('text', ''),
        }


def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b['input_ids'] for b in batch], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b['attention_mask'] for b in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b['labels'] for b in batch], batch_first=True, padding_value=IGNORE_INDEX
    )
    return {
        'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels,
        'offset_mapping': [b['offset_mapping'] for b in batch],
        'record_ids': [b['record_id'] for b in batch],
        'texts': [b['text'] for b in batch],
    }


def load_model_and_data():
    from qwen4b_tokencls_model import load_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    model, tokenizer, hidden_size = load_model(
        MODEL_PATH, num_labels=NUM_LABELS, freeze_backbone=True,
        device=device, use_bf16=use_bf16,
    )
    ckpt_path = ROOT / 'runs' / 'qwen4b_tokencls_head_only' / 'best_head.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.classifier.load_state_dict(ckpt['head_state_dict'])
        print(f'Loaded checkpoint from {ckpt_path}', flush=True)
    else:
        print(f'Warning: no checkpoint at {ckpt_path}, using random head', flush=True)
    model.eval()
    return model, tokenizer, hidden_size, device


def logits_to_spans(logits, offset_mapping, id_to_label, label_to_id, constrained=True):
    from qwen4b_tokencls_decode import logits_to_spans as lts
    return lts(logits, offset_mapping, id_to_label, label_to_id, constrained=constrained)


def labels_to_spans(labels, offset_mapping, id_to_label):
    spans = []
    i = 0
    n = len(labels)
    while i < n:
        lid = labels[i]
        if lid <= 0:
            i += 1
            continue
        label_name = id_to_label.get(str(lid), 'O')
        if label_name.startswith('B-'):
            pii_type = label_name.split('-', 1)[1]
            span_start = i
            span_end = i + 1
            j = i + 1
            while j < n:
                nlid = labels[j]
                nl_name = id_to_label.get(str(nlid), 'O')
                if nl_name.startswith('I-') or nl_name.startswith('E-'):
                    span_end = j + 1
                    if nl_name.startswith('E-'):
                        j += 1
                        break
                    j += 1
                else:
                    break
            if span_start < len(offset_mapping) and span_end <= len(offset_mapping):
                start = offset_mapping[span_start][0]
                end = offset_mapping[span_end - 1][1]
                if start < end:
                    spans.append({'start': start, 'end': end, 'type': pii_type})
            i = max(span_end, i + 1)
        elif label_name.startswith('S-'):
            pii_type = label_name.split('-', 1)[1]
            if i < len(offset_mapping):
                start, end = offset_mapping[i]
                if start < end:
                    spans.append({'start': start, 'end': end, 'type': pii_type})
            i += 1
        else:
            i += 1
    return spans


def span_overlap(a, b):
    return max(0, min(a['end'], b['end']) - max(a['start'], b['start']))


def compute_span_metrics(gold_spans, pred_spans):
    tp_exact = 0
    tp_overlap = 0
    matched_pairs = []

    gold_unmatched = list(range(len(gold_spans)))
    pred_unmatched = list(range(len(pred_spans)))

    for gi, gs in enumerate(gold_spans):
        best_overlap = 0
        best_pi = -1
        for pi, ps in enumerate(pred_spans):
            overlap = span_overlap(gs, ps)
            if overlap > best_overlap:
                best_overlap = overlap
                best_pi = pi
        if best_overlap > 0 and gs['type'] == pred_spans[best_pi]['type']:
            g_len = gs['end'] - gs['start']
            p_len = pred_spans[best_pi]['end'] - pred_spans[best_pi]['start']
            if gs['start'] == pred_spans[best_pi]['start'] and gs['end'] == pred_spans[best_pi]['end']:
                tp_exact += 1
            tp_overlap += 1
            matched_pairs.append((gs, pred_spans[best_pi], best_overlap))
            if best_pi in pred_unmatched:
                pred_unmatched.remove(best_pi)
            if gi in gold_unmatched:
                gold_unmatched.remove(gi)

    n_gold = len(gold_spans)
    n_pred = len(pred_spans)

    exact_precision = tp_exact / max(1, n_pred)
    exact_recall = tp_exact / max(1, n_gold)
    exact_f1 = 2 * exact_precision * exact_recall / max(1e-9, exact_precision + exact_recall)

    overlap_precision = tp_overlap / max(1, n_pred)
    overlap_recall = tp_overlap / max(1, n_gold)
    overlap_f1 = 2 * overlap_precision * overlap_recall / max(1e-9, overlap_precision + overlap_recall)
    overlap_f2 = 5 * overlap_precision * overlap_recall / max(1e-9, 4 * overlap_precision + overlap_recall)

    type_correct = sum(1 for g, p, _ in matched_pairs if g['type'] == p['type'])
    type_accuracy = type_correct / max(1, len(matched_pairs))

    per_label = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    for gs in gold_spans:
        per_label[gs['type']]['fn'] += 1
    for ps in pred_spans:
        per_label[ps['type']]['fp'] += 1
    for gs, ps, _ in matched_pairs:
        per_label[gs['type']]['tp'] += 1
        per_label[gs['type']]['fn'] -= 1
        per_label[ps['type']]['fp'] -= 1

    per_label_metrics = {}
    for label, counts in sorted(per_label.items()):
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        per_label_metrics[label] = {
            'precision': round(precision, 4), 'recall': round(recall, 4),
            'f1': round(f1, 4), 'tp': tp, 'fp': fp, 'fn': fn,
        }

    return {
        'n_gold': n_gold, 'n_pred': n_pred,
        'exact': {'precision': round(exact_precision, 4), 'recall': round(exact_recall, 4), 'f1': round(exact_f1, 4)},
        'overlap': {'precision': round(overlap_precision, 4), 'recall': round(overlap_recall, 4), 'f1': round(overlap_f1, 4), 'f2': round(overlap_f2, 4)},
        'type_accuracy': round(type_accuracy, 4),
        'per_label': per_label_metrics,
    }


def evaluate_split(model, ds, device, split_name, id_to_label, label_to_id, output_path, error_path):
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    o_correct = 0
    o_total = 0
    pos_correct = 0
    pos_total = 0
    all_gold_spans = []
    all_pred_spans = []
    error_examples = []
    latencies = []
    total_examples = 0

    t_start = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            t_batch = time.time()
            result = model(ids, mask, labels)
            batch_time = time.time() - t_batch
            latencies.append(batch_time)

            total_loss += result['loss'].item()
            logits = result['logits']

            preds = logits.argmax(dim=-1)
            valid = labels != IGNORE_INDEX
            total_correct += (preds == labels)[valid].sum().item()
            total_tokens += valid.sum().item()
            o_valid = (labels == O_LABEL_ID) & valid
            o_correct += (preds == labels)[o_valid].sum().item()
            o_total += o_valid.sum().item()
            pos_valid = (labels != O_LABEL_ID) & valid
            pos_correct += (preds == labels)[pos_valid].sum().item()
            pos_total += pos_valid.sum().item()

            for i in range(len(batch['record_ids'])):
                total_examples += 1
                gold_labels = batch['labels'][i][batch['labels'][i] != IGNORE_INDEX].tolist()
                full_labels = batch['labels'][i].tolist()
                offsets_i = batch['offset_mapping'][i]
                logits_i = logits[i]

                valid_len = batch['attention_mask'][i].sum().item()
                full_labels_clipped = full_labels[:int(valid_len)]
                offsets_i_clipped = offsets_i[:int(valid_len)]
                logits_i_clipped = logits_i[:int(valid_len)]
                gold_spans = labels_to_spans(full_labels_clipped, offsets_i_clipped, id_to_label)
                pred_spans = logits_to_spans(logits_i_clipped, offsets_i_clipped, id_to_label, label_to_id, constrained=True)

                gs_copy = [{'start': s['start'], 'end': s['end'], 'type': s['type']} for s in gold_spans]
                ps_copy = [{'start': s['start'], 'end': s['end'], 'type': s['type']} for s in pred_spans]

                all_gold_spans.extend(gs_copy)
                all_pred_spans.extend(ps_copy)

                if len(gs_copy) != len(ps_copy) and len(error_examples) < 200:
                    error_examples.append({
                        'record_id': batch['record_ids'][i],
                        'text_snippet': batch['texts'][i][:200],
                        'gold_spans': gs_copy,
                        'predicted_spans': ps_copy,
                    })

            if (batch_idx + 1) % 500 == 0:
                print(f'  {split_name}: {batch_idx+1}/{len(loader)} batches', flush=True)

    eval_time = time.time() - t_start
    n_batches = len(loader)
    latencies.sort()

    span_metrics = compute_span_metrics(all_gold_spans, all_pred_spans)

    token_metrics = {
        'token_accuracy': round(total_correct / max(1, total_tokens), 6),
        'o_token_accuracy': round(o_correct / max(1, o_total), 6),
        'positive_token_accuracy': round(pos_correct / max(1, pos_total), 6),
        'total_tokens': total_tokens,
        'positive_tokens': pos_total,
    }

    report = {
        'split': split_name,
        'loss': round(total_loss / max(1, n_batches), 6),
        'token_metrics': token_metrics,
        'span_metrics': span_metrics,
        'latency': {
            'mean_ms': round(1000 * sum(latencies) / max(1, len(latencies)), 2),
            'p50_ms': round(1000 * latencies[len(latencies)//2], 2) if latencies else 0,
            'p95_ms': round(1000 * latencies[int(len(latencies)*0.95)], 2) if latencies else 0,
            'p99_ms': round(1000 * latencies[int(len(latencies)*0.99)], 2) if latencies else 0,
            'examples_per_sec': round(total_examples / max(1, eval_time), 2),
            'total_time_seconds': round(eval_time, 1),
        },
        'total_examples': total_examples,
        'total_gold_spans': len(all_gold_spans),
        'total_pred_spans': len(all_pred_spans),
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    with open(error_path, 'w') as f:
        for ex in error_examples:
            f.write(json.dumps(ex, ensure_ascii=False, separators=(',', ':')) + '\n')

    return report


def generate_comparison(single4b_dev, single4b_test):
    hybrid_ref = {
        'overlap_precision': 0.831,
        'overlap_recall': 0.974,
        'overlap_f1': 0.897,
        'exact_f1': 0.831,
        'type_accuracy': 0.986,
        'p50_latency_ms': 153,
        'p95_latency_ms': 308,
    }

    single4b = single4b_test['span_metrics']

    comparison = {
        'models': {
            'hybrid_opf_qwen9b': hybrid_ref,
            'single_qwen4b_tokencls': {
                'overlap_precision': single4b['overlap']['precision'],
                'overlap_recall': single4b['overlap']['recall'],
                'overlap_f1': single4b['overlap']['f1'],
                'exact_f1': single4b['exact']['f1'],
                'type_accuracy': single4b['type_accuracy'],
                'p50_latency_ms': single4b_dev.get('latency', {}).get('p50_ms', 0),
                'p95_latency_ms': single4b_dev.get('latency', {}).get('p95_ms', 0),
            },
        },
        'deltas': {
            'overlap_f1': round(single4b['overlap']['f1'] - hybrid_ref['overlap_f1'], 4),
            'overlap_recall': round(single4b['overlap']['recall'] - hybrid_ref['overlap_recall'], 4),
            'type_accuracy': round(single4b['type_accuracy'] - hybrid_ref['type_accuracy'], 4),
        },
    }

    with open(ROOT / 'reports' / 'stage5_single4b_vs_hybrid_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    md = f"""# Qwen4B Single Model vs Hybrid Comparison

## Models
| Metric | Hybrid (OPF+Qwen9B) | Single Qwen4B TokenCLS | Delta |
|--------|---------------------|----------------------|-------|
| Overlap F1 | {hybrid_ref['overlap_f1']} | {single4b['overlap']['f1']} | {comparison['deltas']['overlap_f1']:+.4f} |
| Overlap Recall | {hybrid_ref['overlap_recall']} | {single4b['overlap']['recall']} | {comparison['deltas']['overlap_recall']:+.4f} |
| Overlap Precision | {hybrid_ref['overlap_precision']} | {single4b['overlap']['precision']} | |
| Exact F1 | {hybrid_ref['exact_f1']} | {single4b['exact']['f1']} | |
| Type Accuracy | {hybrid_ref['type_accuracy']} | {single4b['type_accuracy']} | {comparison['deltas']['type_accuracy']:+.4f} |
| P50 Latency | {hybrid_ref['p50_latency_ms']}ms | {single4b_dev.get('latency', {}).get('p50_ms', 0)}ms | |
| P95 Latency | {hybrid_ref['p95_latency_ms']}ms | {single4b_dev.get('latency', {}).get('p95_ms', 0)}ms | |

## Notes
- Single Qwen4B model uses frozen backbone + token classification head (317 BIOES labels)
- Hybrid uses OPF span detector + Qwen9B rescoring head + policy layer
- Single model has no OPF dependency, no subprocess overhead
"""

    with open(ROOT / 'reports' / 'stage5_single4b_vs_hybrid_comparison.md', 'w') as f:
        f.write(md)


def main():
    id_to_label = json.loads((ROOT / 'pii_schema' / 'id_to_token_label_317.json').read_text())['id_to_label']
    label_to_id = json.loads((ROOT / 'pii_schema' / 'token_label_to_id_317.json').read_text())['label_to_id']

    t0 = time.time()
    model, tokenizer, hidden_size, device = load_model_and_data()

    dev_ds = TokenClsDataset(ROOT / 'data' / 'train' / 'qwen4b_tokencls_dev.jsonl')
    test_ds = TokenClsDataset(ROOT / 'data' / 'train' / 'qwen4b_tokencls_test.jsonl')
    print(f'Dev: {len(dev_ds)} rows, Test: {len(test_ds)} rows', flush=True)

    print('\nEvaluating dev...', flush=True)
    dev_report = evaluate_split(
        model, dev_ds, device, 'dev', id_to_label, label_to_id,
        ROOT / 'reports' / 'stage5_qwen4b_tokencls_dev_eval.json',
        ROOT / 'reports' / 'stage5_qwen4b_tokencls_error_examples.jsonl',
    )

    print('\nEvaluating test...', flush=True)
    test_report = evaluate_split(
        model, test_ds, device, 'test', id_to_label, label_to_id,
        ROOT / 'reports' / 'stage5_qwen4b_tokencls_test_eval.json',
        ROOT / 'reports' / 'stage5_qwen4b_tokencls_test_error_examples.jsonl',
    )

    latency_data = {
        'dev': dev_report['latency'],
        'test': test_report['latency'],
        'batch_size': BATCH_SIZE,
        'device': str(device),
    }
    with open(ROOT / 'reports' / 'stage5_qwen4b_tokencls_latency.json', 'w') as f:
        json.dump(latency_data, f, indent=2)

    generate_comparison(dev_report, test_report)

    summary_md = f"""# Stage 5 Qwen4B Token Classifier Summary

## Model
- Backbone: Qwen3.5-4B-Base (frozen)
- Head: Linear({hidden_size}, 317)
- Trainable params: {model.trainable_parameters()['head_trainable']:,}

## Token Metrics
| Split | Loss | Accuracy | O-Accuracy | Pos-Accuracy |
|-------|------|----------|------------|--------------|
| Dev | {dev_report['loss']:.4f} | {dev_report['token_metrics']['token_accuracy']:.4f} | {dev_report['token_metrics']['o_token_accuracy']:.4f} | {dev_report['token_metrics']['positive_token_accuracy']:.4f} |
| Test | {test_report['loss']:.4f} | {test_report['token_metrics']['token_accuracy']:.4f} | {test_report['token_metrics']['o_token_accuracy']:.4f} | {test_report['token_metrics']['positive_token_accuracy']:.4f} |

## Span Metrics (Test)
| Metric | Value |
|--------|-------|
| Exact F1 | {test_report['span_metrics']['exact']['f1']} |
| Overlap F1 | {test_report['span_metrics']['overlap']['f1']} |
| Overlap Recall | {test_report['span_metrics']['overlap']['recall']} |
| Overlap Precision | {test_report['span_metrics']['overlap']['precision']} |
| Type Accuracy | {test_report['span_metrics']['type_accuracy']} |

## Latency (Dev)
| Metric | Value |
|--------|-------|
| Mean | {dev_report['latency']['mean_ms']}ms |
| P50 | {dev_report['latency']['p50_ms']}ms |
| P95 | {dev_report['latency']['p95_ms']}ms |
| Examples/sec | {dev_report['latency']['examples_per_sec']} |

## Evaluation Time
{time.time() - t0:.1f}s total
"""

    with open(ROOT / 'reports' / 'stage5_qwen4b_tokencls_summary.md', 'w') as f:
        f.write(summary_md)

    print(f'\n{"="*60}')
    print('EVALUATION SUMMARY')
    print(f'{"="*60}')
    print(f'Dev:  acc={dev_report["token_metrics"]["token_accuracy"]:.4f} pos_acc={dev_report["token_metrics"]["positive_token_accuracy"]:.4f} overlap_f1={dev_report["span_metrics"]["overlap"]["f1"]}')
    print(f'Test: acc={test_report["token_metrics"]["token_accuracy"]:.4f} pos_acc={test_report["token_metrics"]["positive_token_accuracy"]:.4f} overlap_f1={test_report["span_metrics"]["overlap"]["f1"]}')
    print(f'Latency (dev): p50={dev_report["latency"]["p50_ms"]}ms p95={dev_report["latency"]["p95_ms"]}ms')
    print(f'Reports in: {ROOT / "reports"}')
    print(f'Time: {time.time() - t0:.1f}s')

    return 0


if __name__ == '__main__':
    sys.exit(main())

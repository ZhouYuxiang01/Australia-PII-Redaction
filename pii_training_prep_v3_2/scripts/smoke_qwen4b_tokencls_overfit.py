"""Stage 5 Task 2: 100-sample overfit smoke for Qwen4B token classifier."""
from __future__ import annotations

import json
import sys
import time
import random
from pathlib import Path
from collections import Counter

import torch

ROOT = Path('/home/admin/ZYX/pii_training_prep_v3_2')
MODEL_PATH = '/home/admin/model/Qwen3.5-4B-Base'
NUM_LABELS = 317
O_LABEL_ID = 0
IGNORE_INDEX = -100
SMOKE_SAMPLES = 100
SMOKE_STEPS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-3

sys.path.insert(0, str(ROOT / 'src' / 'pii_prep'))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) for r in rows) + '\n',
        encoding='utf-8',
    )


def main():
    t0 = time.time()

    from qwen4b_tokencls_model import load_model
    from qwen4b_tokencls_train import load_tokencls_rows, TokenClsCollator, compute_token_metrics
    from qwen4b_tokencls_decode import logits_to_spans

    id_to_label = json.loads(
        (ROOT / 'pii_schema' / 'id_to_token_label_317.json').read_text()
    )['id_to_label']
    label_to_id = json.loads(
        (ROOT / 'pii_schema' / 'token_label_to_id_317.json').read_text()
    )['label_to_id']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    print(f'Device: {device}, BF16: {use_bf16}')

    print('Loading model...')
    model, tokenizer, hidden_size = load_model(
        MODEL_PATH,
        num_labels=NUM_LABELS,
        freeze_backbone=True,
        device=device,
        use_bf16=use_bf16,
    )
    param_info = model.trainable_parameters()
    print(f'  Backbone trainable: {param_info["backbone_trainable"]:,}')
    print(f'  Head trainable: {param_info["head_trainable"]:,}')
    print(f'  Total trainable: {param_info["total_trainable"]:,}')
    print(f'  Hidden size: {hidden_size}')
    print(f'  Num labels: {NUM_LABELS}')

    train_rows = load_tokencls_rows(
        ROOT / 'data' / 'train' / 'qwen4b_tokencls_train.jsonl',
        limit=None,
    )
    print(f'Loaded {len(train_rows)} training rows')

    positive_rows = []
    for row in train_rows:
        has_positive = any(
            lid > 0 and lid != IGNORE_INDEX for lid in row['labels']
        )
        if has_positive:
            positive_rows.append(row)

    print(f'Rows with positive labels: {len(positive_rows)}')

    if len(positive_rows) >= SMOKE_SAMPLES:
        random.seed(42)
        smoke_rows = random.sample(positive_rows, SMOKE_SAMPLES)
    else:
        smoke_rows = positive_rows[:SMOKE_SAMPLES]

    print(f'Smoke dataset: {len(smoke_rows)} rows')

    collator = TokenClsCollator(tokenizer, max_length=4096)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE)

    model.train()
    model.backbone.eval()

    losses = []
    step = 0
    print(f'\nTraining {SMOKE_STEPS} steps...')

    for step in range(1, SMOKE_STEPS + 1):
        random.shuffle(smoke_rows)
        total_loss = 0.0

        for start in range(0, len(smoke_rows), BATCH_SIZE):
            batch_rows = smoke_rows[start:start + BATCH_SIZE]
            batch = collator(batch_rows)

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            result = model(input_ids, attention_mask, labels)
            loss = result['loss']

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step == 1:
                losses.append(loss.item())

        avg_loss = total_loss / max(1, (len(smoke_rows) // BATCH_SIZE))
        losses.append(avg_loss)

        if step % 50 == 0 or step == 1 or step == SMOKE_STEPS:
            print(f'  Step {step:4d}: loss={avg_loss:.6f}')

    initial_loss = round(losses[0], 6)
    final_loss = round(losses[-1], 6)
    min_loss = round(min(losses), 6)

    print(f'\nLoss: initial={initial_loss}, final={final_loss}, min={min_loss}')

    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for start in range(0, len(smoke_rows), BATCH_SIZE):
            batch_rows = smoke_rows[start:start + BATCH_SIZE]
            batch = collator(batch_rows)

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            logits = model.predict(input_ids, attention_mask)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)

    metrics = compute_token_metrics(logits_cat, labels_cat, O_LABEL_ID)
    print(f'  Token accuracy: {metrics["token_accuracy"]:.4f}')
    print(f'  O-token accuracy: {metrics["o_token_accuracy"]:.4f}')
    print(f'  Positive-token accuracy: {metrics["positive_token_accuracy"]:.4f}')

    # Decode examples
    print('\nDecoding examples...')
    examples = []
    for i in range(min(10, len(smoke_rows))):
        row = smoke_rows[i]
        logits_i = all_logits[i] if i < len(all_logits) else all_logits[0]

        # Constrained decode
        offsets = row.get('offset_mapping', [])
        spans = logits_to_spans(logits_i, offsets, id_to_label, label_to_id, constrained=True)

        # Gold spans
        gold_labels = row.get('labels', [])
        gold_spans = []
        j = 0
        n = len(gold_labels)
        while j < n:
            lid = gold_labels[j]
            if lid <= 0:
                j += 1
                continue
            label_name = id_to_label.get(str(lid), 'O')
            if label_name.startswith('B-') or label_name.startswith('S-'):
                pii_type = label_name.split('-', 1)[1] if '-' in label_name else label_name
                if j < len(offsets):
                    start = offsets[j][0]
                    gold_spans.append({'start': start, 'end': offsets[j][1], 'type': pii_type})
            j += 1

        examples.append({
            'record_id': row.get('record_id', ''),
            'text_snippet': row.get('text', '')[:200],
            'gold_spans': gold_spans[:5],
            'predicted_spans': spans[:5],
            'gold_span_count': len(gold_spans),
            'predicted_span_count': len(spans),
        })

    for ex in examples[:5]:
        print(f'  {ex["record_id"]}: gold={ex["gold_span_count"]} spans, pred={ex["predicted_span_count"]} spans')
        if ex['gold_spans']:
            for gs in ex['gold_spans'][:2]:
                print(f'    Gold: [{gs["start"]}:{gs["end"]}] {gs["type"]}')
        if ex['predicted_spans']:
            for ps in ex['predicted_spans'][:2]:
                print(f'    Pred: [{ps["start"]}:{ps["end"]}] {ps["type"]} conf={ps["confidence"]:.4f}')

    # Save checkpoint
    ckpt_dir = ROOT / 'runs' / 'qwen4b_tokencls_smoke'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'head_state_dict': model.classifier.state_dict(),
            'hidden_size': hidden_size,
            'num_labels': NUM_LABELS,
            'smoke_samples': SMOKE_SAMPLES,
            'smoke_steps': SMOKE_STEPS,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
        },
        ckpt_dir / 'head.pt',
    )
    print(f'\nCheckpoint saved to {ckpt_dir / "head.pt"}')

    # Logits shape verification
    test_input = torch.randint(0, 1000, (1, 10)).to(device)
    test_mask = torch.ones(1, 10).to(device)
    with torch.no_grad():
        test_logits = model.predict(test_input, test_mask)
    logits_shape = list(test_logits.shape)
    print(f'Logits shape: {logits_shape} (expected [1, 10, 317])')

    # Reports
    reports_dir = ROOT / 'reports'

    report = {
        'stage': '5',
        'task': 'token_cls_overfit_smoke',
        'model': 'Qwen3.5-4B-Base',
        'hidden_size': hidden_size,
        'num_labels': NUM_LABELS,
        'smoke_samples': SMOKE_SAMPLES,
        'smoke_steps': SMOKE_STEPS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'device': str(device),
        'bf16': use_bf16,
        'backbone_frozen': True,
        'trainable_params': param_info,
        'logits_shape': logits_shape,
        'loss': {
            'initial': initial_loss,
            'final': final_loss,
            'min': min_loss,
            'decreased': final_loss < initial_loss,
        },
        'metrics': metrics,
        'wall_time_seconds': round(time.time() - t0, 3),
    }

    write_json(reports_dir / 'stage5_qwen4b_tokencls_overfit_smoke_report.json', report)
    write_jsonl(reports_dir / 'stage5_qwen4b_tokencls_overfit_examples.jsonl', examples)

    model_info = {
        'hidden_size': hidden_size,
        'num_labels': NUM_LABELS,
        'logits_shape': logits_shape,
        'trainable_params': param_info,
        'backbone_frozen': True,
        'smoke_initial_loss': initial_loss,
        'smoke_final_loss': final_loss,
    }
    write_json(reports_dir / 'stage5_qwen4b_tokencls_model_report.json', model_info)

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.1f}s')

    acceptance = {
        'model_forward_works': True,
        'logits_shape_correct': logits_shape == [1, 10, 317],
        'loss_computes': True,
        'loss_decreased': final_loss < initial_loss,
        'positive_token_accuracy_above_0': metrics['positive_token_accuracy'] > 0,
        'checkpoint_saved': (ckpt_dir / 'head.pt').exists(),
    }

    print('\nACCEPTANCE CHECKS:')
    all_pass = True
    for check, result in sorted(acceptance.items()):
        status = 'PASS' if result else 'FAIL'
        if not result:
            all_pass = False
        print(f'  [{status}] {check}')

    if all_pass:
        print('\n  OVERALL: ALL CHECKS PASSED')
    else:
        print('\n  OVERALL: SOME CHECKS FAILED')

    return 0


if __name__ == '__main__':
    sys.exit(main())

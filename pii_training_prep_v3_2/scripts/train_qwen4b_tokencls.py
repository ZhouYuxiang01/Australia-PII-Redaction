"""Stage 5 Task 3: Full Qwen4B token classifier head-only training."""
from __future__ import annotations

import json, sys, time, random, math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path('/home/admin/ZYX/pii_training_prep_v3_2')
MODEL_PATH = '/home/admin/model/Qwen3.5-4B-Base'
NUM_LABELS = 317
O_LABEL_ID = 0
IGNORE_INDEX = -100

BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
EPOCHS = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 4096
LOG_EVERY = 200
EVAL_EVERY = 2000

sys.path.insert(0, str(ROOT / 'src' / 'pii_prep'))


class TokenClsDataset(Dataset):
    def __init__(self, path: Path):
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
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'offset_mapping': [b['offset_mapping'] for b in batch],
        'record_ids': [b['record_id'] for b in batch],
        'texts': [b['text'] for b in batch],
    }


def compute_batch_metrics(logits, labels):
    preds = logits.argmax(dim=-1)
    mask = labels != IGNORE_INDEX
    total = mask.sum().item()
    if total == 0:
        return 0, 0, 0, 0, 0, 0

    correct = (preds == labels) & mask
    acc = correct.sum().item() / total

    o_mask = (labels == O_LABEL_ID) & mask
    o_total = o_mask.sum().item()
    o_acc = (correct & o_mask).sum().item() / max(1, o_total)

    pos_mask = (labels != O_LABEL_ID) & mask
    pos_total = pos_mask.sum().item()
    pos_acc = (correct & pos_mask).sum().item() / max(1, pos_total)

    return acc, o_acc, pos_acc, total, o_total, pos_total


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    o_correct = 0
    o_total = 0
    pos_correct = 0
    pos_total = 0

    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            result = model(ids, mask, labels)
            loss = result['loss']
            total_loss += loss.item()

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

    return {
        'loss': total_loss / max(1, len(loader)),
        'token_accuracy': round(total_correct / max(1, total_tokens), 6),
        'o_token_accuracy': round(o_correct / max(1, o_total), 6),
        'positive_token_accuracy': round(pos_correct / max(1, pos_total), 6),
        'total_tokens': total_tokens,
        'positive_tokens': pos_total,
    }


def main():
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_bf16 = device.type == 'cuda' and torch.cuda.is_bf16_supported()

    from qwen4b_tokencls_model import load_model

    print(f'Device: {device}, BF16: {use_bf16}', flush=True)
    print('Loading model...', flush=True)
    model, tokenizer, hidden_size = load_model(
        MODEL_PATH, num_labels=NUM_LABELS, freeze_backbone=True,
        device=device, use_bf16=use_bf16,
    )
    pi = model.trainable_parameters()
    print(f'Trainable: {pi}', flush=True)

    print('Loading datasets...', flush=True)
    train_ds = TokenClsDataset(ROOT / 'data' / 'train' / 'qwen4b_tokencls_train.jsonl')
    dev_ds = TokenClsDataset(ROOT / 'data' / 'train' / 'qwen4b_tokencls_dev.jsonl')
    print(f'Train: {len(train_ds)} rows, Dev: {len(dev_ds)} rows', flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    opt = torch.optim.AdamW(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=len(train_loader) * EPOCHS)

    run_dir = ROOT / 'runs' / 'qwen4b_tokencls_head_only'
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / 'training_log.jsonl'
    with open(log_file, 'w') as lf:
        lf.write('')

    best_dev_acc = 0.0
    best_checkpoint = None
    global_step = 0
    history = []

    effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION
    print(f'Training: {EPOCHS} epochs, batch={BATCH_SIZE}, grad_acc={GRADIENT_ACCUMULATION}, eff_batch={effective_batch}', flush=True)
    print(f'LR={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}', flush=True)
    print(f'Log every {LOG_EVERY}, eval every {EVAL_EVERY} steps', flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.backbone.eval()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_correct = 0
        epoch_tokens = 0
        epoch_pos_correct = 0
        epoch_pos_total = 0
        t_epoch = time.time()

        for batch_idx, batch in enumerate(train_loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            result = model(ids, mask, labels)
            loss = result['loss'] / GRADIENT_ACCUMULATION
            loss.backward()

            batch_acc, _, batch_pos_acc, batch_tokens, _, batch_pos_tokens = compute_batch_metrics(
                result['logits'].detach(), labels
            )
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION
            epoch_steps += 1
            epoch_correct += int(batch_acc * batch_tokens)
            epoch_tokens += batch_tokens
            epoch_pos_correct += int(batch_pos_acc * batch_pos_tokens)
            epoch_pos_total += batch_pos_tokens

            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0 or (batch_idx + 1) == len(train_loader):
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % LOG_EVERY == 0 and global_step > 0:
                    current_loss = epoch_loss / max(1, epoch_steps)
                    current_acc = epoch_correct / max(1, epoch_tokens)
                    current_pos_acc = epoch_pos_correct / max(1, epoch_pos_total)
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    print(f'  [E{epoch} S{global_step}] loss={current_loss:.4f} acc={current_acc:.4f} pos_acc={current_pos_acc:.4f} lr={lr:.2e} time={elapsed:.0f}s', flush=True)
                    epoch_loss = 0.0
                    epoch_steps = 0
                    epoch_correct = 0
                    epoch_tokens = 0
                    epoch_pos_correct = 0
                    epoch_pos_total = 0

                    history.append({
                        'epoch': epoch,
                        'global_step': global_step,
                        'loss': round(current_loss, 6),
                        'token_accuracy': round(current_acc, 6),
                        'positive_token_accuracy': round(current_pos_acc, 6),
                        'lr': lr,
                        'wall_time': round(elapsed, 1),
                    })
                    with open(log_file, 'a') as lf:
                        lf.write(json.dumps(history[-1]) + '\n')

                if global_step % EVAL_EVERY == 0 and global_step > 0:
                    print(f'  [E{epoch} S{global_step}] Running dev eval...', flush=True)
                    dev_metrics = evaluate(model, dev_loader, device)
                    print(f'    dev_loss={dev_metrics["loss"]:.4f} dev_acc={dev_metrics["token_accuracy"]:.4f} dev_pos_acc={dev_metrics["positive_token_accuracy"]:.4f}', flush=True)

                    if dev_metrics['token_accuracy'] > best_dev_acc:
                        best_dev_acc = dev_metrics['token_accuracy']
                        best_checkpoint = {
                            'head_state_dict': {k: v.cpu().clone() for k, v in model.classifier.state_dict().items()},
                            'hidden_size': hidden_size,
                            'num_labels': NUM_LABELS,
                            'global_step': global_step,
                            'epoch': epoch,
                            'dev_metrics': dev_metrics,
                        }
                        torch.save(best_checkpoint, run_dir / 'best_head.pt')
                        print(f'    Saved best checkpoint (dev_acc={best_dev_acc:.4f})', flush=True)

        epoch_time = time.time() - t_epoch
        print(f'  Epoch {epoch} done in {epoch_time:.0f}s', flush=True)

    final_dev_metrics = evaluate(model, dev_loader, device)
    print(f'\nFinal dev: loss={final_dev_metrics["loss"]:.4f} acc={final_dev_metrics["token_accuracy"]:.4f} pos_acc={final_dev_metrics["positive_token_accuracy"]:.4f}', flush=True)

    if best_checkpoint is None:
        best_checkpoint = {
            'head_state_dict': {k: v.cpu().clone() for k, v in model.classifier.state_dict().items()},
            'hidden_size': hidden_size,
            'num_labels': NUM_LABELS,
            'global_step': global_step,
            'epoch': EPOCHS,
            'dev_metrics': final_dev_metrics,
        }
        torch.save(best_checkpoint, run_dir / 'best_head.pt')

    torch.save({
        'head_state_dict': model.classifier.state_dict(),
        'hidden_size': hidden_size,
        'num_labels': NUM_LABELS,
    }, run_dir / 'token_head.pt')

    config = {
        'model_path': MODEL_PATH,
        'hidden_size': hidden_size,
        'num_labels': NUM_LABELS,
        'freeze_backbone': True,
        'batch_size': BATCH_SIZE,
        'gradient_accumulation': GRADIENT_ACCUMULATION,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'train_rows': len(train_ds),
        'dev_rows': len(dev_ds),
        'best_dev_accuracy': best_dev_acc,
        'final_dev_metrics': final_dev_metrics,
        'trainable_params': pi,
        'wall_time_seconds': round(time.time() - t0, 1),
    }
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    with open(run_dir / 'best_dev_metrics.json', 'w') as f:
        json.dump({'best_dev_accuracy': best_dev_acc, 'final_dev_metrics': final_dev_metrics, 'history': history}, f, indent=2)

    with open(ROOT / 'reports' / 'stage5_qwen4b_tokencls_full_train_report.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f'\nTraining complete. Best dev acc: {best_dev_acc:.4f}', flush=True)
    print(f'Checkpoints: {run_dir}', flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())

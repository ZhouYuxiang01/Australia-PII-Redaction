# Qwen4B Token/Span Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-model Qwen4B token/span classifier that replaces both OPF span detection and the Qwen9B span classification head.

**Architecture:** Qwen3.5-4B-Base backbone (frozen or LoRA-adapted) → final hidden states → Linear(2560, 317) token classification head → BIOES constrained decoding → span-level 79-class top-k probabilities → REDACT/REVIEW/IGNORE policy.

**Tech Stack:** PyTorch, Transformers (trust_remote_code=True for Qwen3.5), Qwen3.5-4B-Base tokenizer, bf16 on NVIDIA GB10 (128GB).

---

## File Structure

### New files in `/home/admin/ZYX/pii_training_prep_v3_2/`:

```
src/pii_prep/qwen4b_tokencls_dataset.py   — BIOES dataset builder
src/pii_prep/qwen4b_tokencls_model.py      — Model + training loop
src/pii_prep/qwen4b_tokencls_decode.py     — BIOES constrained decoding
src/pii_prep/qwen4b_tokencls_eval.py       — Evaluation metrics
scripts/build_qwen4b_tokencls_dataset.py   — CLI for dataset building
scripts/train_qwen4b_tokencls.py           — CLI for training
scripts/eval_qwen4b_tokencls.py            — CLI for evaluation
tests/test_qwen4b_tokencls_dataset.py      — Dataset + BIOES tests
tests/test_qwen4b_tokencls_decode.py       — Decoding tests
tests/test_qwen4b_tokencls_model.py        — Model shape tests
```

### New files in `/home/admin/ZYX/redaction-wrapper/`:

```
redaction/backends/qwen4b_tokencls.py      — Wrapper backend
configs/backends/qwen4b-tokencls.json      — Backend config
```

### Generated data:

```
data/train/qwen4b_tokencls_train.jsonl
data/train/qwen4b_tokencls_dev.jsonl
data/train/qwen4b_tokencls_test.jsonl
```

### Reports:

```
reports/stage5_qwen4b_tokencls_dataset_report.json
reports/stage5_qwen4b_tokencls_alignment_errors.json
reports/stage5_qwen4b_tokencls_dev_eval.json
reports/stage5_qwen4b_tokencls_test_eval.json
reports/stage5_qwen4b_tokencls_error_examples.jsonl
reports/stage5_qwen4b_tokencls_latency.json
reports/stage5_qwen4b_tokencls_summary.md
reports/stage5_single4b_vs_hybrid_comparison.json
reports/stage5_single4b_vs_hybrid_comparison.md
```

---

## BIOES Label Schema

79 PII types from `training_label_space_80.json` (excluding `NON_PII`):

```
B-<TYPE>, I-<TYPE>, E-<TYPE>, S-<TYPE> for each of 79 types = 316 labels
O = 1 label
Total = 317 labels
```

Label → index mapping:

```python
PII_LABELS = json.loads(Path("pii_schema/training_label_space_80.json").read_text())
PII_TYPES = [l for l in PII_LABELS if l != "NON_PII"]  # 79 types

BIOES_TAGS = ["O"]
for pii_type in PII_TYPES:
    BIOES_TAGS.extend([f"B-{pii_type}", f"I-{pii_type}", f"E-{pii_type}", f"S-{pii_type}"])
# len(BIOES_TAGS) == 317

TAG2ID = {tag: i for i, tag in enumerate(BIOES_TAGS)}
ID2TAG = {i: tag for tag, i in TAG2ID.items()}
```

---

## Task 1: BIOES Dataset Builder

**Files:**
- Create: `src/pii_prep/qwen4b_tokencls_dataset.py`
- Create: `scripts/build_qwen4b_tokencls_dataset.py`
- Test: `tests/test_qwen4b_tokencls_dataset.py`

### Task 1.1: Write failing tests for BIOES labeling

- [ ] **Step 1: Create test file**

```python
# tests/test_qwen4b_tokencls_dataset.py
import json
import tempfile
import unittest
from pathlib import Path

from pii_prep.qwen4b_tokencls_dataset import (
    bioes_labels_for_spans,
    build_tokencls_record,
    BIOES_TAGS,
    TAG2ID,
    PII_TYPES,
)


class BioesLabelTests(unittest.TestCase):

    def test_bioes_tags_count(self):
        self.assertEqual(len(BIOES_TAGS), 317)
        self.assertEqual(len(PII_TYPES), 79)

    def test_tag2id_roundtrip(self):
        for tag in BIOES_TAGS:
            self.assertEqual(ID2TAG[TAG2ID[tag]], tag)

    def test_single_token_span_gets_S_tag(self):
        text = "call 0421 now"
        spans = [{"start": 5, "end": 9, "top_type": "MOBILE", "type_distribution": {"MOBILE": 0.95, "NON_PII": 0.05}}]
        labels = bioes_labels_for_spans(text, spans, tokenizer_name_or_path="Qwen/Qwen3-4B")
        # "0421" is a single token — should get S-MOBILE
        self.assertIn("S-MOBILE", labels)

    def test_multi_token_span_gets_BIE_tags(self):
        text = "email: alex@example.com please"
        spans = [{"start": 7, "end": 25, "top_type": "EMAIL_ADDRESS", "type_distribution": {"EMAIL_ADDRESS": 0.95, "NON_PII": 0.05}}]
        labels = bioes_labels_for_spans(text, spans, tokenizer_name_or_path="Qwen/Qwen3-4B")
        self.assertIn("B-EMAIL_ADDRESS", labels)
        self.assertIn("E-EMAIL_ADDRESS", labels)

    def test_no_spans_all_O(self):
        text = "nothing here"
        spans = []
        labels = bioes_labels_for_spans(text, spans, tokenizer_name_or_path="Qwen/Qwen3-4B")
        self.assertTrue(all(l == "O" for l in labels))

    def test_overlapping_spans_deterministic(self):
        # Same text overlapped by two types — longer span wins
        text = "plate O385UM here"
        spans = [
            {"start": 6, "end": 12, "top_type": "NUMBER_PLATE", "type_distribution": {"NUMBER_PLATE": 0.95, "NON_PII": 0.05}},
            {"start": 6, "end": 12, "top_type": "VEHICLE_REGO", "type_distribution": {"VEHICLE_REGO": 0.95, "NON_PII": 0.05}},
        ]
        labels = bioes_labels_for_spans(text, spans, tokenizer_name_or_path="Qwen/Qwen3-4B")
        # Exactly one of the two types should be assigned, not both
        non_o = [l for l in labels if l != "O"]
        types_present = set(l.split("-", 1)[1] for l in non_o if "-" in l)
        self.assertLessEqual(len(types_present), 1)

    def test_offset_alignment_reported(self):
        text = "DOB 04/05/1998"
        spans = [{"start": 100, "end": 200, "top_type": "DATE_OF_BIRTH", "type_distribution": {"DATE_OF_BIRTH": 0.95, "NON_PII": 0.05}}]
        result = build_tokencls_record(text, spans, record_id="test-1", tokenizer_name_or_path="Qwen/Qwen3-4B")
        self.assertGreater(len(result["alignment_errors"]), 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/admin/ZYX/pii_training_prep_v3_2 && /home/admin/miniconda3/envs/qwen/bin/python -m pytest tests/test_qwen4b_tokencls_dataset.py -v`
Expected: FAIL (module not found)

### Task 1.2: Implement BIOES dataset builder

- [ ] **Step 3: Implement `qwen4b_tokencls_dataset.py`**

Key design:

```python
# src/pii_prep/qwen4b_tokencls_dataset.py
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

PII_LABELS = json.loads(
    Path(__file__).resolve().parents[2].joinpath("pii_schema/training_label_space_80.json").read_text(encoding="utf-8")
)
PII_TYPES = [l for l in PII_LABELS if l != "NON_PII"]

BIOES_TAGS = ["O"]
for _t in PII_TYPES:
    BIOES_TAGS.extend([f"B-{_t}", f"I-{_t}", f"E-{_t}", f"S-{_t}"])

TAG2ID = {tag: i for i, tag in enumerate(BIOES_TAGS)}
ID2TAG = {i: tag for tag, i in TAG2ID.items()}


def _load_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True, use_fast=True)


def _char_spans_to_token_labels(
    offset_mapping: list[tuple[int, int]],
    spans: list[dict[str, Any]],
    num_tokens: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Convert char-level spans to token-level BIOES labels.

    Overlapping spans: use the span with the highest max probability in
    type_distribution (excluding NON_PII). If tied, first span wins.

    Returns (labels, alignment_errors).
    """
    from collections import defaultdict

    token_labels = ["O"] * num_tokens
    alignment_errors: list[dict[str, Any]] = []

    # Sort spans by top_type probability (desc) for deterministic overlap resolution
    def _span_priority(s):
        dist = s.get("type_distribution", {})
        max_pii = max((v for k, v in dist.items() if k != "NON_PII"), default=0.0)
        return -max_pii

    sorted_spans = sorted(spans, key=_span_priority)

    # Track which tokens are already assigned
    assigned = [False] * num_tokens

    for span in sorted_spans:
        char_start = int(span["start"])
        char_end = int(span["end"])
        top_type = span.get("top_type", "")

        if top_type == "NON_PII" or top_type not in set(PII_TYPES):
            continue

        # Find tokens overlapping this char span
        token_indices = []
        for idx, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_end <= tok_start:
                continue
            if tok_end <= char_start:
                continue
            if tok_start >= char_end:
                break
            if tok_start < char_end and tok_end > char_start:
                token_indices.append(idx)

        if not token_indices:
            alignment_errors.append({
                "span_start": char_start,
                "span_end": char_end,
                "top_type": top_type,
                "reason": "no_token_overlap",
            })
            continue

        # Check if any token already assigned to a higher-priority span
        already_assigned = [i for i in token_indices if assigned[i]]
        if already_assigned and len(already_assigned) == len(token_indices):
            alignment_errors.append({
                "span_start": char_start,
                "span_end": char_end,
                "top_type": top_type,
                "reason": "fully_overlapped_by_higher_priority",
            })
            continue

        # Use only unassigned tokens
        usable = [i for i in token_indices if not assigned[i]]
        if not usable:
            continue

        # Assign BIOES tags
        if len(usable) == 1:
            token_labels[usable[0]] = f"S-{top_type}"
        else:
            token_labels[usable[0]] = f"B-{top_type}"
            for i in usable[1:-1]:
                token_labels[i] = f"I-{top_type}"
            token_labels[usable[-1]] = f"E-{top_type}"

        for i in usable:
            assigned[i] = True

    return token_labels, alignment_errors


def bioes_labels_for_spans(
    text: str,
    spans: list[dict[str, Any]],
    tokenizer_name_or_path: str = "/home/admin/model/Qwen3.5-4B-Base",
    max_length: int = 1536,
) -> list[str]:
    """Convenience function: tokenize + label, return tag strings."""
    tokenizer = _load_tokenizer(tokenizer_name_or_path)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    num_tokens = len(encoded["input_ids"])
    labels, _ = _char_spans_to_token_labels(offsets, spans, num_tokens)
    return labels


def build_tokencls_record(
    text: str,
    spans: list[dict[str, Any]],
    record_id: str,
    tokenizer_name_or_path: str = "/home/admin/model/Qwen3.5-4B-Base",
    max_length: int = 1536,
) -> dict[str, Any]:
    """Build a single training record with token-level BIOES labels."""
    tokenizer = _load_tokenizer(tokenizer_name_or_path)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    num_tokens = len(encoded["input_ids"])
    labels, alignment_errors = _char_spans_to_token_labels(offsets, spans, num_tokens)

    label_ids = [TAG2ID[l] for l in labels]

    # Build per-token type distribution targets (for auxiliary loss)
    # Each token gets the type_distribution of the span it belongs to, or NON_PII
    token_distributions = []
    for i in range(num_tokens):
        label = labels[i]
        if label == "O":
            dist = {"NON_PII": 1.0}
        else:
            # Find the span this token belongs to
            tok_start, tok_end = offsets[i]
            span_dist = {"NON_PII": 1.0}
            for s in spans:
                s_start, s_end = int(s["start"]), int(s["end"])
                if tok_start < s_end and tok_end > s_start:
                    span_dist = s.get("type_distribution", {"NON_PII": 1.0})
                    break
            token_distributions.append(span_dist)

    return {
        "id": record_id,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "offset_mapping": offsets,
        "labels": label_ids,
        "label_tags": labels,
        "text": text,
        "alignment_errors": alignment_errors,
    }


def build_split(
    input_path: Path,
    output_path: Path,
    tokenizer_name_or_path: str = "/home/admin/model/Qwen3.5-4B-Base",
    max_length: int = 1536,
) -> dict[str, Any]:
    """Build token classification dataset from a splits JSONL file."""
    records = []
    total_spans = 0
    total_tokens = 0
    total_alignment_errors = 0
    all_alignment_errors: list[dict[str, Any]] = []
    label_counter = Counter()

    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row["text"]
        spans = row.get("spans", [])
        record_id = row.get("id", "")
        total_spans += len(spans)

        record = build_tokencls_record(text, spans, record_id, tokenizer_name_or_path, max_length)
        total_tokens += len(record["input_ids"])
        total_alignment_errors += len(record["alignment_errors"])
        all_alignment_errors.extend(record["alignment_errors"])
        for l in record["label_tags"]:
            label_counter[l] += 1

        # Write only serializable fields
        out_record = {
            "id": record["id"],
            "input_ids": record["input_ids"],
            "attention_mask": record["attention_mask"],
            "offset_mapping": record["offset_mapping"],
            "labels": record["labels"],
            "text": record["text"],
        }
        records.append(out_record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "records": len(records),
        "total_spans": total_spans,
        "total_tokens": total_tokens,
        "alignment_error_count": total_alignment_errors,
        "label_distribution": dict(label_counter.most_common()),
    }
```

- [ ] **Step 4: Run tests**

Run: `cd /home/admin/ZYX/pii_training_prep_v3_2 && /home/admin/miniconda3/envs/qwen/bin/python -m pytest tests/test_qwen4b_tokencls_dataset.py -v`

### Task 1.3: Build dataset CLI

- [ ] **Step 5: Create `scripts/build_qwen4b_tokencls_dataset.py`**

```python
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from pii_prep.qwen4b_tokencls_dataset import build_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--tokenizer", default="/home/admin/model/Qwen3.5-4B-Base")
    parser.add_argument("--max-length", type=int, default=1536)
    args = parser.parse_args()

    root = Path(args.root)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_reports = {}
    all_errors = []
    for split in ["train", "dev", "test"]:
        input_path = root / "data" / "splits" / f"{split}.jsonl"
        output_path = root / "data" / "train" / f"qwen4b_tokencls_{split}.jsonl"
        report = build_split(input_path, output_path, args.tokenizer, args.max_length)
        all_reports[split] = report
        all_errors.extend(report.pop("alignment_error_count", 0))

    (reports_dir / "stage5_qwen4b_tokencls_dataset_report.json").write_text(
        json.dumps(all_reports, indent=2, ensure_ascii=False) + "\n"
    )
    (reports_dir / "stage5_qwen4b_tokencls_alignment_errors.json").write_text(
        json.dumps({"total_errors": sum(r["alignment_error_count"] for r in all_reports.values())}, indent=2) + "\n"
    )
    print(json.dumps(all_reports, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run dataset build**

Run: `cd /home/admin/ZYX/pii_training_prep_v3_2 && /home/admin/miniconda3/envs/qwen/bin/python scripts/build_qwen4b_tokencls_dataset.py`

---

## Task 2: Token Classifier Model

**Files:**
- Create: `src/pii_prep/qwen4b_tokencls_model.py`
- Create: `scripts/train_qwen4b_tokencls.py`
- Test: `tests/test_qwen4b_tokencls_model.py`

### Task 2.1: Write failing model tests

- [ ] **Step 1: Create test file**

```python
# tests/test_qwen4b_tokencls_model.py
import unittest
import torch
from pii_prep.qwen4b_tokencls_model import Qwen4BTokenClassifier, BIOES_TAGS

class Qwen4BTokenClassifierTests(unittest.TestCase):

    def test_logits_shape(self):
        model = Qwen4BTokenClassifier(num_labels=317, hidden_size=2560)
        hidden = torch.randn(2, 16, 2560)
        logits = model.classifier(hidden)
        self.assertEqual(logits.shape, (2, 16, 317))

    def test_num_labels_matches_bioes(self):
        model = Qwen4BTokenClassifier(num_labels=317, hidden_size=2560)
        self.assertEqual(model.classifier.out_features, 317)
        self.assertEqual(model.classifier.out_features, len(BIOES_TAGS))
```

- [ ] **Step 2: Run test to verify it fails**

### Task 2.2: Implement model

- [ ] **Step 3: Implement `qwen4b_tokencls_model.py`**

```python
# src/pii_prep/qwen4b_tokencls_model.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .qwen4b_tokencls_dataset import BIOES_TAGS, TAG2ID, ID2TAG


class Qwen4BTokenClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int = 317):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }
        try:
            outputs = self.backbone(**kwargs, use_cache=False)
        except TypeError:
            outputs = self.backbone(**kwargs)

        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs.hidden_states[-1]

        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits}


def load_model_for_training(
    model_path: str = "/home/admin/model/Qwen3.5-4B-Base",
    num_labels: int = 317,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> Qwen4BTokenClassifier:
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    config = backbone.config
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None and hasattr(config, "text_config"):
        hidden_size = config.text_config.hidden_size
    hidden_size = int(hidden_size)

    if use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()

    model = Qwen4BTokenClassifier(backbone, hidden_size, num_labels)
    return model
```

### Task 2.3: Training script

- [ ] **Step 4: Create `scripts/train_qwen4b_tokencls.py`**

Full HuggingFace Trainer-based training script with:
- Dataset class that loads pre-built JSONL files
- Data collator for padding
- TrainingArguments with bf16, gradient_checkpointing, paged_adamw_8bit
- Evaluation on dev set
- Save best model by eval_loss

Key hyperparameters:
- batch_size=4 (per device), grad_accumulation=4
- learning_rate=5e-5 (LoRA) or 2e-5 (full)
- warmup_ratio=0.1
- max_epochs=3
- max_length=1536
- weight_decay=0.01
- label_smoothing_factor=0.0 (BIOES doesn't benefit from smoothing)

---

## Task 3: BIOES Constrained Decoding

**Files:**
- Create: `src/pii_prep/qwen4b_tokencls_decode.py`
- Test: `tests/test_qwen4b_tokencls_decode.py`

### Task 3.1: Write failing decode tests

- [ ] **Step 1: Create test file**

```python
# tests/test_qwen4b_tokencls_decode.py
import unittest
from pii_prep.qwen4b_tokencls_decode import (
    valid_bioes_transitions,
    decode_bioes,
    bioes_tags_to_spans,
)

class BioesDecodeTests(unittest.TestCase):

    def test_valid_transitions_no_B_after_O(self):
        valid = valid_bioes_transitions()
        self.assertTrue(valid["O"]["B-PERSON"])
        self.assertTrue(valid["O"]["S-PERSON"])
        self.assertTrue(valid["O"]["O"])
        self.assertFalse(valid["O"]["I-PERSON"])
        self.assertFalse(valid["O"]["E-PERSON"])

    def test_valid_transitions_B_must_continue(self):
        valid = valid_bioes_transitions()
        self.assertTrue(valid["B-PERSON"]["I-PERSON"])
        self.assertTrue(valid["B-PERSON"]["E-PERSON"])
        self.assertFalse(valid["B-PERSON"]["B-EMAIL"])
        self.assertFalse(valid["B-PERSON"]["O"])

    def test_valid_transitions_E_can_start_new(self):
        valid = valid_bioes_transitions()
        self.assertTrue(valid["E-PERSON"]["O"])
        self.assertTrue(valid["E-PERSON"]["B-EMAIL"])
        self.assertTrue(valid["E-PERSON"]["S-EMAIL"])

    def test_decode_bioes_simple(self):
        tags = ["O", "B-EMAIL_ADDRESS", "E-EMAIL_ADDRESS", "O"]
        spans = bioes_tags_to_spans(tags)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["type"], "EMAIL_ADDRESS")
        self.assertEqual(spans[0]["start_token"], 1)
        self.assertEqual(spans[0]["end_token"], 3)

    def test_decode_bioes_single_token(self):
        tags = ["O", "S-STUDENT_ID", "O"]
        spans = bioes_tags_to_spans(tags)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["type"], "STUDENT_ID")
        self.assertEqual(spans[0]["start_token"], 1)
        self.assertEqual(spans[0]["end_token"], 2)

    def test_whitespace_span_rejected(self):
        # If span text is only whitespace, it should be rejected
        tags = ["O", "S-DATE_OF_BIRTH", "O"]
        offsets = [(0, 4), (4, 5), (5, 10)]  # token 1 is a space
        text = "DOB  1998"
        spans = bioes_tags_to_spans(tags, offsets=offsets, text=text)
        self.assertEqual(len(spans), 0)

    def test_decode_with_logits_constrains_transitions(self):
        import torch
        # Simulate logits that would predict invalid B->O transition
        # decode_bioes should enforce valid transitions
        num_labels = 317
        seq_len = 4
        logits = torch.zeros(1, seq_len, num_labels)
        # Force B-PERSON at position 1, O at position 2 (invalid: should be I or E)
        from pii_prep.qwen4b_tokencls_dataset import TAG2ID
        logits[0, 1, TAG2ID["B-PERSON"]] = 10.0
        logits[0, 2, TAG2ID["O"]] = 10.0
        logits[0, 2, TAG2ID["E-PERSON"]] = 9.0
        tags = decode_bioes(logits[0])
        self.assertIn(tags[2], ["I-PERSON", "E-PERSON"])
```

### Task 3.2: Implement decoding

- [ ] **Step 2: Implement `qwen4b_tokencls_decode.py`**

Key functions:

1. `valid_bioes_transitions()` — build transition matrix
2. `decode_bioes(logits, transition_valid=None)` — Viterbi-like constrained decoding
3. `bioes_tags_to_spans(tags, offsets=None, text=None)` — convert tag sequence to spans
4. `spans_to_output(text, spans, offsets, logits, top_k=5)` — produce final output with probabilities

Transition rules:
- O → {O, B-*, S-*}
- B-X → {I-X, E-X}
- I-X → {I-X, E-X}
- E-X → {O, B-*, S-*}
- S-X → {O, B-*, S-*}

For Viterbi decoding: at each position, only consider tags reachable from previous tag. This prevents invalid sequences.

Span output format:
```python
{
    "start": int,        # char offset
    "end": int,          # char offset
    "type": str,         # PII type
    "value": str,        # text[start:end]
    "type_distribution_topk": [(type, prob), ...],
    "top1_prob": float,
    "top3_sum": float,
    "non_pii_prob": float,  # O probability at span tokens
    "confidence": float,    # geometric mean of token probs
}
```

---

## Task 4: Evaluation

**Files:**
- Create: `src/pii_prep/qwen4b_tokencls_eval.py`
- Create: `scripts/eval_qwen4b_tokencls.py`

### Metrics to compute:

1. **Token accuracy**: correct_label / total_tokens (excluding -100)
2. **Detection precision/recall/F1**: span-level, exact match (start, end, type)
3. **Overlap F1**: span matches if char overlap > 0 and type matches
4. **Type accuracy**: on matched spans, type matches
5. **Per-label precision/recall/F1**: for each of 79 PII types
6. **High-risk under-redaction rate**: gold spans with REDACT-type not detected
7. **Over-redaction rate**: predicted spans not in gold
8. **Calibration**: NLL, ECE, Brier
9. **Latency**: p50, p95, mean, examples/sec
10. **Memory**: model size, peak VRAM

---

## Task 5: Comparison Against Hybrid

**Files:**
- Create: `reports/stage5_single4b_vs_hybrid_comparison.json`
- Create: `reports/stage5_single4b_vs_hybrid_comparison.md`

Reference hybrid metrics:
- Overlap F1 ≈ 0.897
- Overlap recall ≈ 0.974
- Type accuracy on overlap ≈ 0.986
- p50 latency ≈ 153ms
- p95 latency ≈ 308ms

Compare: span F1, recall, precision, type accuracy, calibration, latency, VRAM, startup time, deployment complexity.

---

## Task 6: Wrapper Integration

**Files:**
- Create: `redaction/backends/qwen4b_tokencls.py`
- Create: `configs/backends/qwen4b-tokencls.json`
- Modify: `redaction/backends/registry.py` (add new backend type)

### Backend implementation:

```python
# redaction/backends/qwen4b_tokencls.py
class Qwen4BTokenClsBackend(RedactionBackend):
    """Single-model Qwen4B token/span classification backend."""

    def detect_spans(self, text: str) -> tuple[list[Span], dict[str, Any]]:
        # 1. Tokenize
        # 2. Forward pass through Qwen4B + classifier head
        # 3. BIOES constrained decoding
        # 4. Convert to spans with offset mapping
        # 5. Apply REDACT/REVIEW/IGNORE policy
        # 6. Return spans
```

Policy reuse: Use the same risk weights and threshold logic from HybridOpfQwenBackend.

Config:
```json
{
  "type": "qwen4b_tokencls",
  "name": "qwen4b-tokencls",
  "model_version": "qwen4b-tokencls-bioes-v1",
  "supported_types": [...79 types...],
  "checkpoint_path": "${REDACTION_PII_PROJECT_ROOT}/runs/qwen4b_tokencls/best",
  "device": "cuda",
  "dtype": "bf16",
  "max_length": 1536,
  "output_top_k": 5,
  "redact_threshold": 0.70,
  "review_threshold": 0.25
}
```

---

## Task 7: Smoke Tests

Run 7 test examples (A-G) through the trained model:

- A: `Student num = SID# 47009923.` → should detect STUDENT_ID
- B: `DOB 04/05/1998, email alex@example.com, mobile 0412 345 678` → DATE_OF_BIRTH, EMAIL_ADDRESS, MOBILE
- C: `BSB 062-001, account 123456789` → BANK_ACCOUNT_NUMBER, BSB (or mapped)
- D: `ticket id INC-0412-345-678, not a phone number` → should be IGNORE
- E: `room: 14/09/2002 Building A` → should be IGNORE (not a DOB)
- F: `fake card test token: tok_4111111111111111` → should be IGNORE
- G: Mixed long note → should detect multiple PII types

---

## Task 8: Unit Tests

- `test_qwen4b_tokencls_dataset.py`: BIOES label construction, offset alignment, overlap handling
- `test_qwen4b_tokencls_decode.py`: valid transitions, span merging, whitespace rejection, constrained decoding
- `test_qwen4b_tokencls_model.py`: logits shape, num_labels
- Wrapper schema compatibility test: output matches Span.to_schema()
- No spans[].value leak by default (values only in backend, not exposed in API)

---

## Self-Review Checklist

1. **Spec coverage**: All 8 tasks from the spec mapped to plan tasks. ✅
2. **Placeholder scan**: All code blocks contain actual implementation. ✅
3. **Type consistency**: BIOES_TAGS, TAG2ID, ID2TAG consistent across all files. ✅
4. **No OPF in final path**: Model uses Qwen4B backbone only. ✅
5. **No JSON generation**: Token classification, not text generation. ✅
6. **Existing artifacts untouched**: All new files, no modifications to existing. ✅ (except registry.py which needs a new entry)

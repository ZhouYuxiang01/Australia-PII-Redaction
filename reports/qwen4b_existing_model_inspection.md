# Qwen4B Existing Model Inspection

**Date**: 2026-05-02
**Inspector**: Sisyphus
**Purpose**: Assess the existing full fine-tuned Qwen3.5-4B JSON-output model for potential reuse as a token/span classifier backbone.

---

## 1. Model Location

| Item | Path |
|------|------|
| **Fine-tuned model** | `/home/admin/ZYX/Qwen3.5_4b_base_Full_73class/outputs/qwen3_5_4b_base_full_73class` |
| **Base model** | `/home/admin/model/Qwen3.5-4B-Base` |
| **Project root** | `/home/admin/ZYX/Qwen3.5_4b_base_Full_73class/` |
| **Weight file** | `model.safetensors` (8.4 GB) |
| **Best checkpoint** | `checkpoint-1900` |

## 2. Architecture

| Property | Value |
|----------|-------|
| **Architecture** | `Qwen3_5ForCausalLM` |
| **Model type** | `qwen3_5_text` |
| **hidden_size** | **2560** |
| **num_hidden_layers** | 32 |
| **num_attention_heads** | 16 |
| **num_key_value_heads** | 4 |
| **head_dim** | 256 |
| **vocab_size** | 248,320 |
| **max_position_embeddings** | 262,144 |
| **dtype** | bfloat16 |
| **Layer pattern** | 3× linear_attention + 1× full_attention (8 blocks) |
| **intermediate_size** | 9216 |
| **tie_word_embeddings** | true |
| **tokenizer** | Qwen3.5 tokenizer with chat template |

## 3. Training History

- **Paradigm**: Full-parameter SFT (not LoRA)
- **Base model**: Qwen3.5-4B-Base (text-only extraction from multimodal checkpoint)
- **Framework**: TRL SFTTrainer with `assistant_only_loss=True`
- **Training recipe**: `safe_full` profile
  - batch_size=1, grad_accumulation=16
  - learning_rate=2e-5, warmup_steps=150
  - optim=paged_adamw_8bit, gradient_checkpointing=True
  - max_length=1280
- **Data**: 30,400 training rows from 19,000 AU PII records
- **Runtime**: Checkpoints at steps 1500 and 1900
- **GPU**: Training done on single GPU (evidenced by safetensors being single file)

## 4. Label Space

The model was trained on the **OPF canonical 73-class taxonomy** (`custom_label_space_73.v1.1.1.json`).

**74 labels total** (73 PII types + `O`):

Key merges applied:
- `EMAIL_ADDRESS` + `WORK_EMAIL` → `EMAIL`
- `AU_PHONE` + `WORK_PHONE` → `PHONE`
- `PAYMENT_CARD_NUMBER` + `HASHED_PAYMENT_CARD_NUMBER` → `PAYMENT_CARD_NUMBER`
- `NUMBER_PLATE` + `VEHICLE_REGO` → `VEHICLE_ID`

Classes present in 73-class but NOT in the 80-class training space:
- `BSB` (absorbed into `BANK_ACCOUNT_INFORMATION` in 80-class)
- `CREDIT_CARD_CVV` (not present in 80-class)
- `AU_DRIVERS_LICENCE` (renamed to `DRIVERS_LICENCE` in 80-class)
- `AU_PASSPORT` (renamed to `PASSPORT_NUMBER` in 80-class)
- `AU_BANK_ACCOUNT` (split into `BANK_ACCOUNT_NUMBER` + `BANK_ACCOUNT_INFORMATION` in 80-class)

## 5. Model Output Format

The model generates JSON spans via the SFT training format:

```json
{"spans":[{"start":36,"end":43,"type":"BSB","value":"062-001"}]}
```

**Evaluation results** (from `RESULT_SUMMARY.md`):
- **JSON validity**: 99.5% (995/1000 parseable)
- **Raw offsets**: Very poor - 867/1000 rows with invalid spans
  - 5,660 offset-value mismatches
  - 619 out-of-range offsets
- **After repair** (value-to-offset resolver):
  - Typed exact F1: 0.8795
  - Untyped exact F1: 0.9126
- **Hard negatives**: 0% false positive (4,011 rows)
- **Trap examples**: 0% false positive (100 rows)

## 6. Smoke Inference Results

7 test examples run, key observations:

- **GPU**: NVIDIA GB10, 128.5GB (used ~8.4GB for model)
- **Load time**: 46.1s
- **Inference latency**: 1.4-7.7s per example (1-114 tokens)
- **JSON valid rate**: 7/7 (100%)
- **Output schema inconsistent**: Model uses `pii`, `au_pii`, `spans` interchangeably
- **Missing detections**: Simple examples (DOB, EMAIL alone) returned empty arrays
- **Hard negative**: Correctly passed (ticket ID not flagged)
- **Offset accuracy**: Poor (Example 5 BSB offset [36:43] wrong vs actual position)

## 7. Hidden State Extraction

**Verified**: `output_hidden_states=True` works correctly.

| Property | Value |
|----------|-------|
| Hidden state layers | 33 (1 embedding + 32 transformer) |
| Shape per layer | `(batch, seq_len, 2560)` |
| Forward pass latency | 0.119s for 31 tokens |
| Span pooling | mean/first/last extraction confirmed |
| hidden_size | **2560** (vs 4096 for 9B) |

The pooling formula used by the 9B classifier is `(mean + first + last) / 3.0`, which is compatible with 4B hidden states.

## 8. Key Differences from 9B Span Classifier

| Feature | 4B Model | 9B Span Classifier |
|---------|----------|-------------------|
| hidden_size | 2560 | 4096 |
| Head type | LM head (causal) | Linear(4096, 80) |
| Label space | 73 PII + O | 79 PII + NON_PII (80) |
| Training | Full SFT JSON gen | Frozen backbone + head |
| Model size | 8.4GB | ~18GB |
| Memory required | ~8.5GB | ~18GB+ |
| Output | JSON spans | Probability distribution |
| Offset quality | Poor raw | N/A (token-level) |

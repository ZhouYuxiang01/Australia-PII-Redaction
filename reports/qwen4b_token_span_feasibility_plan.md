# Qwen4B Token/Span Classification Feasibility Plan

**Date**: 2026-05-02
**Author**: Sisyphus
**Status**: Recommendation (Option A)

---

## Executive Summary

The existing Qwen3.5-4B full fine-tuned model (`Qwen3.5_4b_base_Full_73class`) can serve as a **token/span classifier backbone**. The JSON-output head is irrelevant — we discard it and use only the transformer backbone. The 4B backbone's hidden_size=2560 is smaller than the 9B's 4096, but the architecture (32 layers, hybrid linear+full attention) is identical. Label mapping from 73→80 classes is required.

**Recommendation: Option A — Use Qwen4B Base as token/span classifier backbone.**

---

## 1. Approach Options

### Option A: Use Qwen4B Base as backbone ⭐ RECOMMENDED

Start from the fresh Qwen3.5-4B-Base model and train a classification head.

**Advantages:**
- Clean architecture: no chat template artifacts, no generation mode leaks
- Direct compatibility with existing `FrozenQwenSpanClassifier` class
- Only need to change: `hidden_size=2560`, model path
- No label mapping from 73-class needed (train on 80-class directly)
- Proven pipeline: `qwen_spancls_cache.py` → `qwen_spancls_heads.py` → evaluate
- 8.4GB GPU memory fits easily on 128GB GPU

**Disadvantages:**
- No PII-specific knowledge in backbone (generic Qwen3.5 text model)
- Must classify all 80 labels from scratch

### Option B: Use the full fine-tuned 4B JSON model as backbone

Start from the existing SFT checkpoint and train a classification head.

**Advantages:**
- Backbone has seen 19,000 AU PII records
- May have better PII-specific representations
- Potentially better OOD detection

**Disadvantages:**
- Chat template artifact (model expects chat-formatted input with `<think>` tags)
- Loads with `Qwen3_5ForCausalLM` — need to strip LM head
- Need to handle text-only extraction from the LM (already done: model is `qwen3_5_text` type)
- 73-class knowledge may bias toward merged labels
- Additional complexity for label mapping 73→80

### Option C: Compare both by training small heads

Train small classification heads on both the Base and the fine-tuned backbone.

**Advantages:**
- Empirical data on which backbone is better
- Clear winner selection

**Disadvantages:**
- 2x compute cost
- Complexity of setting up two pipelines
- The fine-tuned model's SFT doesn't guarantee better span embeddings (it was trained for JSON generation, not span classification)

### Option D: Existing 4B model is unsuitable

This would be incorrect. The model is very much suitable as a backbone.

---

## 2. Technical Feasibility Analysis

### 2.1 Backbone Compatibility

| Component | 9B Pipeline | 4B Adaptation | Status |
|-----------|-------------|---------------|--------|
| Model class | AutoModel/AutoModelForCausalLM | Same | ✅ Compatible |
| hidden_size | 4096 | 2560 | Need new head dim |
| num_hidden_layers | 32 | 32 | ✅ Identical |
| Layer types | linear + full attention hybrid | Same | ✅ Identical |
| tokenizer | Qwen3.5 | Same vocab | ✅ Compatible |
| pad/eos tokens | 248044/248044 | Same | ✅ Compatible |

### 2.2 Head Modification

Current 9B head: `nn.Linear(4096, 80)`

4B head: `nn.Linear(2560, 80)` — **only one line changes** in `qwen_spancls_smoke.py` (and `qwen_spancls_heads.py`), changing the parameter `input_dim` from hidden_size derived from config.

```python
# In qwen_spancls_smoke.py: load_backbone_and_tokenizer already reads hidden_size from config
# No code change needed — hidden_size=2560 is auto-detected

# FrozenQwenSpanClassifier uses hidden_size parameter:
class FrozenQwenSpanClassifier(nn.Module):
    def __init__(self, backbone, hidden_size, num_labels):
        # hidden_size=2560 comes from config, num_labels=80
        self.classifier = nn.Linear(hidden_size, num_labels)  # Linear(2560, 80)
```

### 2.3 Embedding Caching

The `qwen_spancls_cache.py` pipeline requires:
1. Load model → tokenizer → extract hidden states
2. Pool span embeddings: `(mean + first + last) / 3.0`
3. Save to disk

Operation for 4B: identical code, just different model path. Fast (0.119s per 31 tokens forward pass).

### 2.4 Label Mapping (73 → 80)

If using Option B, a label mapping is needed. Key differences:

```
73-class → 80-class mapping:
  EMAIL → {EMAIL_ADDRESS, WORK_EMAIL}             (split, ambiguous)
  PHONE → {MOBILE, HOME_PHONE, WORK_PHONE}         (split, ambiguous)
  VEHICLE_ID → {NUMBER_PLATE, VEHICLE_REGO}         (split, ambiguous)
  PERSON → {PERSON, FIRST_NAME, LAST_NAME}          (split, ambiguous)
  PAYMENT_CARD_NUMBER → {PAYMENT_CARD_NUMBER, HASHED_PAYMENT_CARD_NUMBER}  (split)
  AU_DRIVERS_LICENCE → DRIVERS_LICENCE              (rename)
  AU_PASSPORT → PASSPORT_NUMBER                     (rename)
  AU_BANK_ACCOUNT → BANK_ACCOUNT_NUMBER             (partial map)
  BSB → {BANK_ACCOUNT_INFORMATION}                  (novel mapping needed)
  CREDIT_CARD_CVV → (no 80-class equivalent)        (drop)
```

For the ambiguous splits (EMAIL→{EMAIL_ADDRESS, WORK_EMAIL}), the mapping cannot be deterministically disambiguated without context. This makes **Option B more complex**.

### 2.5 Chat Template Issues (Option B only)

The fine-tuned model uses a Qwen3 chat template with `<think>` tags. The smoke test showed the model generates thinking tags in output. When used as a backbone for span classification, we only need hidden states. However:

1. The model may expect chat-formatted input for best representations
2. The `chat_template` attribute must be disabled or bypassed for raw tokenization
3. Tokenizer may add system/user/assistant wrapper tokens

For Option A (Base model), no chat template interference.

---

## 3. Implementation Plan (Option A)

### Phase 1: Cache Span Embeddings (1-2 hours)

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2
# Modify model_path to 4B Base
python3 -m pii_prep.qwen_spancls_cache \
  --model-path /home/admin/model/Qwen3.5-4B-Base \
  --batch-size 1 \
  --cache-dtype float16
```

**Expected outputs:**
- `data/cache/qwen_spancls_embeddings_train.pt`
- `data/cache/qwen_spancls_embeddings_dev.pt`
- `data/cache/qwen_spancls_embeddings_test.pt`

Each file: `{records, mean_embeddings, first_embeddings, last_embeddings}` with embedding_dim=2560.

### Phase 2: Train Classification Heads (1-3 hours)

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2
python3 -m pii_prep.qwen_spancls_heads \
  --batch-size 1024 \
  --max-epochs 30 \
  --patience 5
```

**Four experiments** will run:
1. `mean_linear`: Linear(2560, 80) on mean-pooled embeddings
2. `first_linear`: Linear(2560, 80) on first-token embeddings
3. `last_linear`: Linear(2560, 80) on last-token embeddings
4. `concat_mlp`: Sequential(Linear(7680, 1024), GELU, Dropout, Linear(1024, 80)) on concatenated [mean, first, last]

### Phase 3: Evaluate and Select

```bash
cd /home/admin/ZYX/pii_training_prep_v3_2
python3 -m pii_prep.qwen_spancls_smoke \
  --model-path /home/admin/model/Qwen3.5-4B-Base \
  --train-limit 1000 --dev-limit 200
```

**Success criteria:**
- Test top-1 accuracy > 0.85 (comparable to 9B results)
- Non-PII accuracy > 0.90
- Calibrated ECE < 0.05

---

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 4B underperforms 9B significantly | Medium | Low | 4B is faster and cheaper; accept performance gap or try Option B |
| Chat template interference (Opt B) | High | Medium | Use Base model instead (Option A) |
| 73→80 label mapping ambiguity | High | Medium | Skip Option B; train on 80-class directly with Base |
| GPU OOM during caching | Low | High | 128GB GPU is more than sufficient for 8.4GB model |
| flash-linear-attention not installed | High | Low | Falls back to torch implementation (slower but correct) |

---

## 5. Resource Estimates

| Resource | 4B Model | 9B Model (comparison) |
|----------|----------|----------------------|
| GPU memory (model only) | 8.4 GB | ~18 GB |
| GPU memory (inference) | ~10 GB | ~20 GB |
| Embedding cache size (3 splits) | ~3 GB (fp16) | ~6 GB (fp16) |
| Cache build time | ~1 hour | ~2 hours |
| Head training time (4 expts) | ~1 hour | ~2 hours |
| Inference latency (31 tokens) | 0.119s | ~0.2s |

---

## 6. Decision Matrix

| Criterion | Option A (Base) | Option B (SFT) | Option C (Compare) |
|-----------|-----------------|-----------------|---------------------|
| Code changes required | 0 (auto-detect hidden_size) | 3-5 files | 2x Option A |
| Label mapping needed | No (80-class native) | Yes (73→80 complex) | No |
| Chat template issues | No | Yes | Only for B |
| PII knowledge in backbone | No | Yes | Both tested |
| Time to results | ~3 hours | ~3 hours + mapping | ~6 hours |
| Simplicity | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Risk | Low | Medium | Low |

---

## 7. Final Recommendation

**Option A: Use Qwen4B Base as token/span classifier backbone.**

Rationale:
1. The existing code automatically adapts to any hidden_size via config
2. No label mapping complexity (train directly on 80-class space)
3. No chat template interference
4. Fastest path to results (~3 hours)
5. If 4B backbone underperforms, Option B can be tested as fallback using the same evaluation pipeline

**Next step**: Cache 4B Base embeddings for all 3 splits → train heads → compare vs 9B benchmark.

# PII Training Preparation Spec v3.2

> **修订历史**：
> - v2.1：ChatGPT 起草，过度设计、跳过数据生成
> - v3：Claude 重写，复用 19000 条已有数据，简化 schema
> - v3.1：采纳 ChatGPT 4 条修正，模型升级到 Qwen3.5-9B
> - **v3.2（当前版本）**：采纳团队第二轮 6 条修订，纠正 v3.1 关于 Qwen3.5 架构的事实错误
>
> **v3.2 相对 v3.1 的变更**：
> 1. OPF 表述改软（"工程复杂度上升，本项目不优先采用"而非"不能处理"）
> 2. 纯 SFT 必须配概率校准评估门槛 + 升级条款（ECE / Brier / NLL）
> 3. Qwen3.5 架构事实纠正：MoE 不是 Gated Attention，thinking 默认开启不是禁用，vision 是 early fusion 不是独立 encoder
> 4. 视觉路径降级为探索性扩展，不进主验收
> 5. format_candidates 明确为候选上界，非过滤器
> 6. Hard negatives 拆为 document-level + candidate-level 两类
> 7. Vision encoder 不能独立 strip（影响显存预算）
> 8. Thinking mode 配置正确写法（推理时主动 disable）

---

## 1. 项目目标重述

把已有的 27B teacher PII 检测能力蒸馏到 Qwen3.5-9B student，让 student 在推理时输出**每个 span 在 79 个 PII 标签上的概率分布**。下游 policy layer 基于这个分布 + CSV 的 Data Classification 决定 redact / review / ignore。

视觉输入（图片/PDF）作为**探索性扩展（exploratory extension）**，不进入主交付与主验收。Capstone 主线仍以 text pipeline 为准；视觉能力仅在主线达标后用剩余时间做演示性实现。

**核心设计原则：**

1. 训练数据格式服务于"输出 79 维分布"这一目标
2. 概率分布反映"格式 + 上下文"两类证据，**不引入"频率先验"**
3. Policy 与模型解耦：模型只输出分布，policy 在推理时由独立 layer 处理
4. 充分复用已有 19000 条标注，避免重新调用 teacher
5. **主线不依赖视觉能力**；视觉路径效果不达标不影响 capstone 验收

---

## 2. 概率分布的定义（核心概念）

对每个 span，输出一个 80 维的概率分布 `type_distribution`（79 个 PII 标签 + `NON_PII`）：

```
type_distribution[T] = 综合"格式匹配"和"上下文信号"两类证据后,
                       该 span 是 type T 的可信度
```

**两类证据：**

- **格式匹配**：长度、字符集、校验规则。硬约束，规则可写。
- **上下文信号**：关键词、字段名、语义场景。软证据，由 teacher 推理。

**不包含**：领域频率先验（"BSB 在大学场景占 X%"这种）。

**两层概率构造方法：**

| 层 | 处理 | 实现 |
|---|---|---|
| 层 1 | 格式硬过滤 | 写规则代码，返回该 span 的 format-valid candidate 集合 |
| 层 2 | 候选集上分配概率 | 由 teacher 基于上下文在 candidate 上分布概率 |

**示例**："123456" 这个 span：

| 上下文 | 层 1 候选 | 层 2 输出（teacher 给） |
|---|---|---|
| 无上下文 | {BSB, STUDENT_ID, EMPLOYEE_NUMBER, UAC_ID, PERSONNEL_NUMBER, NON_PII} | 接近均匀分布 |
| `BSB 123456 transferred` | 同上 | BSB 高度集中（~0.92） |
| `Order #123456 shipped` | 同上 | NON_PII 主导（~0.85） |

---

## 3. 79 标签分类与格式过滤策略

> **v3.1 修订**：v3 要求 79 标签都写 format_filter 不现实。语义型标签没有稳定字符格式，强行写规则会得到无意义的"始终 pass"过滤器。改为按规则强度分三类。

### 3.1 三类标签

**A 类：Pattern-based（有 regex / checksum / 长度规则）**

可以写严格 format_filter。共约 30 个标签：

```
AU_TFN, EMAIL_ADDRESS, WORK_EMAIL, IP_ADDRESS, PAYMENT_CARD_NUMBER,
MEDICARE_NUMBER, MEDICARE_EXPIRY, AU_PHONE, MOBILE, WORK_PHONE, HOME_PHONE,
DATE_OF_BIRTH, PASSPORT_NUMBER, PASSPORT_EXPIRY, PASSPORT_START_DATE,
DRIVERS_LICENCE, BSB, BANK_ACCOUNT_NUMBER, CREDIT_CARD_CVV, CREDIT_CARD_EXPIRY,
NUMBER_PLATE, VEHICLE_REGO, USI, UAC_ID, IHI, NATIONAL_IDENTITY_CARD,
PENSION_CARD_NUMBER, CENTRELINK_REFERENCE_NUMBER, EMPLOYEE_NUMBER,
PERSONNEL_NUMBER, STUDENT_ID, LATITUDE, LONGITUDE, GEOLOCATION_INFORMATION
```

**B 类：Context-based（无固定格式，靠语义）**

format_filter 始终返回 True（任何 span 都"格式上可能"是这些标签）。判断完全交给层 2 teacher。共约 25 个标签：

```
PERSON, FIRST_NAME, LAST_NAME, ADDRESS, ABORIGINALITY, GENDER, PRONOUN,
RELIGION_BELIEF, RACIAL_ETHNIC_ORIGIN, SEXUAL_ORIENTATION, NATIONALITY,
CITIZENSHIP_STATUS, MARITAL_STATUS, MILITARY_VETERAN_STATUS,
CARING_RESPONSIBILITIES, DISABILITY_OR_SPECIFIC_CONDITION,
MEDICAL_INFORMATION, COUNSELLING_RECORDS, MEDICAL_CERTIFICATE,
SPECIAL_CONSIDERATION, CRIMINAL_RECORDS, EMPLOYMENT_INFORMATION,
CONTRACT_TYPE, SOCIO_ECONOMIC_STATUS, NEXT_OF_KIN
```

**C 类：Document/Media-type（描述性，需要特殊文本描述样本）**

格式过滤特殊处理。共约 12 个标签：

```
AUDIO_INFORMATION, CAMERA_FOOTAGE_AUDIO, FACIAL_RECOGNITION, FINGERPRINT,
SIGNATURE, VOICE_RECOGNITION, COOKIE_INFORMATION, DEVICE_ID, USERNAME,
SOCIAL_MEDIA_ACCOUNT, SOCIAL_MEDIA_ID, SOCIAL_MEDIA_HISTORY,
WEBSITE_HISTORY, SCHOLARSHIP, SUBJECT_RESULTS, WAM_SCORE, SANCTIONS,
PERSONAL_DEBT, SALARY, SALARY_WAGE_EXPECTATION, WORKERS_COMPENSATION_CLAIM,
HASHED_PAYMENT_CARD_NUMBER
```

C 类可以有部分格式提示（`USERNAME` 是字母数字、`HASHED_PAYMENT_CARD_NUMBER` 是 hex 字符串），但严格度低于 A 类。

### 3.2 格式过滤实现

```python
class FormatFilter:
    """A 类标签：严格 regex/长度/校验"""
    
    @staticmethod
    def is_au_tfn(value):
        cleaned = value.replace(" ", "").replace("-", "")
        return cleaned.isdigit() and len(cleaned) == 9
    
    @staticmethod
    def is_bsb(value):
        cleaned = value.replace("-", "").replace(" ", "")
        return cleaned.isdigit() and len(cleaned) == 6
    
    @staticmethod
    def is_email(value):
        return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value))
    
    # ... 其余 A 类标签
    
    @staticmethod
    def context_based(value):
        """B 类标签：始终返回 True"""
        return True
    
    @staticmethod
    def get_candidates(span_value):
        candidates = set()
        
        # A 类格式检查
        if FormatFilter.is_au_tfn(span_value):
            candidates.add("AU_TFN")
        if FormatFilter.is_bsb(span_value):
            candidates.add("BSB")
        # ... 全部 A 类
        
        # B 类无条件加入
        candidates.update(B_CLASS_LABELS)
        
        # C 类按弱规则
        if FormatFilter.is_username_format(span_value):
            candidates.add("USERNAME")
        # ... 全部 C 类
        
        # NON_PII 始终是候选
        candidates.add("NON_PII")
        
        return candidates
```

### 3.3 校验级规则（决定 `rule_verified`）

仅 A 类标签的子集有强校验规则：

| 标签 | 校验规则 |
|---|---|
| `AU_TFN` | TFN 加权校验位算法 |
| `PAYMENT_CARD_NUMBER` | Luhn 算法 |
| `MEDICARE_NUMBER` | Medicare 校验位算法 |
| `BSB` | 前 3 位匹配澳洲银行 BSB 注册表 |
| `IP_ADDRESS` | 各段 0-255 范围检查 |

`rule_verified=True` 的样本，`training_weight` 提升到 0.9-1.0。

---

## 4. 数据架构

### 4.1 数据来源分类

| 类别 | 来源 | 数量 | 处理方式 |
|---|---|---|---|
| 类别 A：现有标注（高置信） | 19000 条里 confidence ≥ 0.85 | ~13000 条 | 转换为 one-hot + label smoothing |
| 类别 B：现有标注（中低置信） | 19000 条里 confidence < 0.85 | ~6000 条 | 同上但降低 training_weight |
| 类别 C：裸 span 训练样本 | 新生成 | ~2000 条 | 5 档 ranking 让 teacher 标注 |
| 类别 D：hard negatives | 已存在于现有数据中 | ~30000 条 | 转换为 `{NON_PII: 1.0}` 空 spans |

### 4.2 类别 A 转换：confidence → distribution（v3.1 修订）

> **v3 错误**：直接把 0.883 当 IP_ADDRESS 概率、剩余 0.117 平摊到 candidates。这是把**未校准的自评 confidence** 当**校准的概率分布**用，方法不严谨。
>
> **v3.1 修正**：confidence 主要用作 training_weight；distribution 设为 one-hot + label smoothing。

输入（现有格式）：

```json
{"start": 412, "end": 426, "type": "IP_ADDRESS",
 "value": "242.30.143.150", "confidence": 0.883}
```

转换算法：

```python
LABEL_SMOOTHING = 0.05  # 留 5% 给 NON_PII 防 over-confidence

def convert_high_conf_label(span):
    candidates = format_filter(span["value"])
    teacher_type = span["type"]
    teacher_conf = span["confidence"]
    
    if teacher_type not in candidates:
        return mark_for_review(span)  # 异常：teacher 标的 type 不通过格式
    
    # One-hot + label smoothing（不再用 confidence 当概率）
    distribution = {teacher_type: 1.0 - LABEL_SMOOTHING,
                    "NON_PII": LABEL_SMOOTHING}
    
    # confidence 用作 training_weight
    if teacher_conf >= 0.85:
        weight = 0.8  # 高置信 sonnet 标注
    else:
        weight = 0.4 + 0.4 * (teacher_conf / 0.85)  # 0.4 - 0.8 线性
    
    # 校验通过的样本权重升至 1.0
    if rule_verify(span):
        weight = min(1.0, weight + 0.2)
    
    return {
        "start": span["start"],
        "end": span["end"],
        "value": span["value"],
        "type_distribution": distribution,
        "top_type": teacher_type,
        "training_weight": weight,
        "format_candidates": list(candidates),
        "rule_verified": rule_verify(span),
        "source": "sonnet_high_conf"
    }
```

输出（v3.2 格式）：

```json
{
  "start": 412, "end": 426, "value": "242.30.143.150",
  "type_distribution": {"IP_ADDRESS": 0.95, "NON_PII": 0.05},
  "top_type": "IP_ADDRESS",
  "training_weight": 1.0,
  "format_candidates": ["IP_ADDRESS", "DEVICE_ID", "NON_PII"],
  "rule_verified": true,
  "source": "sonnet_high_conf"
}
```

**说明**：原有 confidence 0.883 的信息**没有丢**，它编码进了 training_weight，反映"sonnet 对这个标注的把握程度"。但 distribution 本身是 one-hot，不假装 0.117 是 DEVICE_ID 的概率。

### 4.3 类别 B 转换：低置信样本

confidence < 0.85 表示 sonnet 不确定。处理方式与类别 A 相同，但：

- `training_weight` 范围 0.4 - 0.7（vs 类别 A 的 0.7 - 1.0）
- 标记 `source = "sonnet_low_conf"`，可在训练时单独评估

可选：对 confidence < 0.6 的样本，二次调用 teacher 用 5 档 ranking 重标。这增加约 6000 次 teacher 调用，但可显著提升这部分样本的可靠性。

### 4.4 类别 C 生成：裸 span 训练样本

**目标**：教 student 在缺乏上下文时输出合适均匀的分布，以及在不同上下文强度下调整分布的尖锐程度。

**生成流程**：

```
1. 从 79 标签的格式规则反向生成 span 实例（每个 span 跨多个候选 type）
   例：6 位数字 → 同时是 BSB / STUDENT_ID / EMPLOYEE_NUMBER 候选
   例：4 位数字 → STUDENT_ID / EMPLOYEE_NUMBER / NON_PII 候选
   例：纯人名 → PERSON / FIRST_NAME / LAST_NAME 候选

2. 为每个 span 生成 4 种上下文变体：
   - 裸 span（仅该 span，无上下文）
   - 极弱上下文（"Reference: <span>"）
   - 强正向上下文（"BSB <span>" / "DOB <span>"）
   - 反向上下文（"Order #<span>" / "Invoice <span>"）

3. 用 5 档 ranking 让 teacher 标注每个变体（self-consistency: 跑 3 次取多数）
```

**5 档 ranking prompt 模板**：

```
You will assess the likelihood that a given text span belongs to each candidate
PII type, given the context.

Span: "{span_value}"
Context: "{context}"  (or "NO_CONTEXT" if bare span)

This span passes format checks for these candidate types:
{candidate_list}

For each candidate, assign EXACTLY ONE of these 5 verdicts:
- "strong_for"      : Context strongly suggests this type
- "weak_for"        : Context mildly suggests this type
- "neutral"         : No evidence either way
- "weak_against"    : Context mildly contradicts this type
- "strong_against"  : Context strongly contradicts this type

Also assess NON_PII (the span being not a PII at all).

Output JSON only:
{
  "verdicts": {
    "BSB": "strong_for",
    "STUDENT_ID": "weak_against",
    ...
    "NON_PII": "weak_against"
  }
}
```

**Verdict → 概率转换**（仅作为 soft label 构造规则，不当真实概率）：

```python
def verdicts_to_distribution(verdicts, candidates):
    multipliers = {
        "strong_for": 3.0, "weak_for": 1.5, "neutral": 1.0,
        "weak_against": 0.5, "strong_against": 0.1
    }
    
    raw = {c: multipliers[verdicts.get(c, "neutral")] for c in candidates}
    total = sum(raw.values())
    return {c: w/total for c, w in raw.items()}

def aggregate_self_consistency(runs):
    """3 次 teacher 调用的结果聚合"""
    # 对每个 candidate，取 3 次 verdict 的多数
    # 如果 3 次都不同，标记为 high_uncertainty
    ...
```

**Self-consistency 检查**：同一 span 调用 3 次，分布稳定的样本 weight=0.7，分布漂移大的样本 weight=0.3 或丢弃。

### 4.5 类别 C 数据规模

| 上下文类型 | 每种 span 类型样本数 | span 类型数 | 总计 |
|---|---|---|---|
| 裸 span | 50 | 10 个歧义组 | 500 |
| 极弱上下文 | 50 | 10 | 500 |
| 强正向上下文 | 50 | 10 | 500 |
| 反向上下文 | 50 | 10 | 500 |

**总计：~2000 条**，乘 self-consistency × 3 = 6000 次 teacher 调用。

### 4.6 10 个歧义组

按 CSV Category Type 分组：

1. 数字 ID 组（6-9 位）：BSB, STUDENT_ID, EMPLOYEE_NUMBER, UAC_ID, PERSONNEL_NUMBER, AU_TFN
2. 电话号码组：AU_PHONE, MOBILE, WORK_PHONE, HOME_PHONE
3. 日期组：DATE_OF_BIRTH, PASSPORT_EXPIRY, PASSPORT_START_DATE, MEDICARE_EXPIRY, CREDIT_CARD_EXPIRY
4. Email 组：EMAIL_ADDRESS, WORK_EMAIL
5. 名字组：PERSON, FIRST_NAME, LAST_NAME
6. 地址组：ADDRESS, GEOLOCATION_INFORMATION
7. 坐标组：LATITUDE, LONGITUDE
8. 车辆组：NUMBER_PLATE, VEHICLE_REGO
9. 卡号组：PAYMENT_CARD_NUMBER, MEDICARE_NUMBER, IHI
10. 社交媒体组：SOCIAL_MEDIA_ACCOUNT, SOCIAL_MEDIA_ID, USERNAME

---

## 5. Schema 定义

### 5.1 训练样本顶层结构

```json
{
  "id": "AU-PII-V3-00001",
  "text": "完整的训练文本",
  "metadata": {
    "source_type": "email | slack | voicemail_transcript | ...",
    "data_category": "A | B | C | D",
    "language": "en-AU"
  },
  "spans": [
    { /* span 对象 */ }
  ]
}
```

### 5.2 Span 对象结构

```json
{
  "start": 412,
  "end": 426,
  "value": "242.30.143.150",
  "type_distribution": {
    "IP_ADDRESS": 0.95,
    "NON_PII": 0.05
  },
  "top_type": "IP_ADDRESS",
  "source": "sonnet_high_conf | sonnet_low_conf | qwen_5way_ranking | rule_verified",
  "training_weight": 0.8,
  "format_candidates": ["IP_ADDRESS", "DEVICE_ID", "NON_PII"],
  "rule_verified": false
}
```

### 5.3 NON_PII 的处理

`NON_PII` 是 80 维分布中的一维（79 PII 标签 + NON_PII），但语义上不同：

- **训练时**：作为分类目标的一个 class，loss 计算正常进行
- **推理时**：student 输出 80 维分布。下游 policy layer 使用完整 80 维做风险评估（不只看 top_type）

### 5.3.1 Hard negatives 的两个层次（v3.2 新增）

> **v3.2 修订**：v3.1 把 hard negatives 笼统当一类。实际上有两个不同训练目标，应分开。

**类型 1：Document-level hard negative**

整段文本里没有任何 PII span。训练样本格式：

```json
{"text": "Order #123456 shipped today.", "spans": []}
```

**训练目标**：教 student "不要乱抽 span"。这控制 precision，避免 over-redaction。

**类型 2：Candidate-level ambiguous negative**

文本里识别出某个 span，但 distribution 中 NON_PII 概率主导：

```json
{
  "text": "The report references 123456 in section 3.",
  "spans": [{
    "value": "123456",
    "type_distribution": {"NON_PII": 0.75, "STUDENT_ID": 0.10, "BSB": 0.10, "EMPLOYEE_NUMBER": 0.05}
  }]
}
```

**训练目标**：教 student "对疑似 span 输出 NON_PII 主导的分布"。这控制 recall calibration，让模型在识别歧义 span 时仍能输出有用的概率信号给下游 policy。

**两类的区别**：

| 维度 | Document-level | Candidate-level |
|---|---|---|
| 输出 spans 列表 | 空 `[]` | 非空，但 NON_PII 主导 |
| 训练的能力 | 不抽 span | 抽 span 但标对分布 |
| 控制的指标 | Precision | Recall + calibration |
| 数据来源 | 类别 D 中无歧义干扰项 | 类别 C 反向上下文 + 类别 D 中可能产生歧义的样本 |

**数据生成时分开统计**：类别 D 在生成阶段就要标记 `subtype: "doc_level" | "candidate_level"`，验证器分别统计两类比例。建议比例：document-level 占 hard negatives 的 60-70%，candidate-level 占 30-40%。

### 5.4 哪些 v2.1 字段被砍掉

| v2.1 字段 | v3.2 处理 | 理由 |
|---|---|---|
| `pii_probability` | **砍掉** | 与 `1 - type_distribution["NON_PII"]` 等价 |
| `policy_target` | **砍掉**，移到推理 policy layer | Policy 不应硬编码进训练数据 |
| `hard_label` | **砍掉**，由 `source` 字段隐式表达 | source 已包含可信度信息 |
| `ambiguity` | **砍掉** | 由 distribution 的熵自然表达 |
| `label_quality.weight` | 重命名为 `training_weight` | 简化嵌套结构 |

---

## 6. Validator 设计

```python
def validate_record(record, full_text):
    errors = []
    
    for span in record["spans"]:
        # 1. Offset 一致性
        if full_text[span["start"]:span["end"]] != span["value"]:
            errors.append(f"Offset mismatch at span {span}")
        
        # 2. type_distribution 合法性
        dist = span["type_distribution"]
        if abs(sum(dist.values()) - 1.0) > 0.02:
            errors.append(f"Distribution doesn't sum to 1: {sum(dist.values())}")
        if any(v < 0 or v > 1 for v in dist.values()):
            errors.append(f"Invalid probability value")
        
        # 3. top_type 一致性
        if span["top_type"] != max(dist, key=dist.get):
            errors.append(f"top_type doesn't match argmax")
        
        # 4. type_distribution 标签合法性
        valid_labels = set(TAXONOMY_79) | {"NON_PII"}
        for label in dist.keys():
            if label not in valid_labels:
                errors.append(f"Invalid label: {label}")
        
        # 5. 分布与 format_candidates 一致
        non_zero_types = {t for t, p in dist.items() if p > 1e-4}
        if not non_zero_types.issubset(set(span["format_candidates"])):
            errors.append(f"Distribution has types outside format_candidates")
        
        # 6. training_weight 范围
        if not 0 <= span["training_weight"] <= 1:
            errors.append(f"Invalid training_weight")
    
    return errors
```

数据集级别检查：

- 每个 type 的样本数 ≥ 30
- 类别 D（hard negatives）占比 15-25%
- 类别 C 在每个歧义组上至少 50 条
- 平均文本长度 / span 密度统计

### 6.1 关于 format_candidates 的语义说明

> **重要：format_candidates 是候选上界，不是最终过滤器。**

`format_candidates` 表示该 span 在格式上"理论可能"是哪些标签的集合，由 3.2 节的 `FormatFilter.get_candidates()` 生成。理解这个字段时注意：

- **A 类标签**：format_candidates 是真过滤——该字段实际起到限制作用（"123456" 不会出现 EMAIL_ADDRESS 在 candidates 里）
- **B 类标签**：format_candidates 始终包含全部 25 个 B 类标签（context_based() 永远返回 True），因此对 B 类几乎无过滤作用
- **C 类标签**：部分有弱过滤（USERNAME 字符集），但仍宽松

因此 validator 第 5 项检查（分布 ⊆ format_candidates）只对 A 类标签有约束力。**B/C 类标签的判断主要靠 teacher context 推理 + training_weight 控制，不靠 format_candidates 验证。**

不要误以为 validator 通过 = 所有标签都被合理验证。

---

## 7. Student 模型训练（Qwen3.5-9B）

### 7.1 模型选择理由

| 维度 | Qwen3.5-9B |
|---|---|
| 参数量 | 约 9.6B（FP16 权重 ~19.3GB），dense 而非 MoE 9B 子集 |
| 架构 | **Gated Delta Networks + sparse Mixture-of-Experts（MoE）的 hybrid 架构**（注：v3.1 写"Gated Attention"是错的，已纠正）|
| 上下文 | 原生 262K，本任务用 4096 截断 |
| 多模态 | **Unified vision-language foundation, early fusion training on multimodal tokens**（视觉与文本在 token 层早期融合，**不是独立 vision encoder**）|
| Thinking 模式 | **默认开启**，PII 任务推理时需主动 disable（`enable_thinking=false` 或对应 chat template 参数）|
| License | Apache 2.0 |
| HF Repo | `Qwen/Qwen3.5-9B`（post-trained）/ `Qwen/Qwen3.5-9B-Base`（pre-trained only）|
| 兼容生态 | Transformers / vLLM / SGLang / KTransformers |
| 工具链 TBD | LoRA target_modules 具体名称（DeltaNet 不是标准 attention 模块）；Unsloth 训练稳定性需首次 sanity check 验证 |

**选 Qwen3.5-9B 而非 Qwen3-8B 理由**：
- 多模态能力可作为 capstone 视觉路径**探索性扩展**基础（注：仅探索性，不进主验收）
- 与 27B teacher 同家族，tokenizer 一致便于 distillation
- 262K 上下文对长文档场景留有余量

**Early fusion 架构对部署的影响**：
- 视觉部分**无法独立 strip**——视觉 token 与文本 token 共享底层权重
- 部署时即使只处理文本，也加载完整模型权重
- 这影响 7.7 节的显存预算估算

### 7.2 输入输出格式

**输入**：原始文本（训练用 max 4096 token 截断）

**输出**：JSON 格式的 spans 列表

```json
{
  "spans": [
    {
      "start": 412,
      "end": 426,
      "value": "242.30.143.150",
      "type_distribution": {"IP_ADDRESS": 0.92, "DEVICE_ID": 0.05, "NON_PII": 0.03},
      "top_type": "IP_ADDRESS"
    }
  ]
}
```

### 7.3 Loss 设计：纯 SFT（做法 1）

> **v3.1 决定**：使用纯 SFT next-token prediction loss，不引入 auxiliary classification head。这是最简单可行的方案，作为 baseline 先跑通。
>
> **已知 limitation**：纯 SFT 学到的概率精度有限——模型可能输出 "0.9" 比 "0.92" 频繁，因为 "0.9" 在预训练语料中出现更多。如果第一轮训练后概率精度不够，再升级到做法 2（auxiliary head）。

**Loss 计算**：

```python
loss = sum over output_tokens of -log P(token | input, prev_tokens)
```

完全标准的 SFT。不引入：
- KL divergence on type_distribution（因为没有显式 classifier head）
- offset alignment loss（移到后处理校验，见 7.5）

**样本权重应用**：

```python
weighted_loss = loss * training_weight
```

`training_weight` 来自数据 schema 的字段，反映样本可信度。

### 7.4 Loss 增强：confidence 通过权重影响训练

虽然没有显式 KL loss，但通过两种方式让模型学到分布：

1. **生成时输出完整 distribution**：模型在生成 JSON 时必须输出所有非零概率的 type 及数值。模型学到的是 **JSON 文本中的概率字符串**。
2. **样本权重稀疏化**：高 training_weight 样本主导学习信号，低权重样本作为 regularization。

### 7.4.1 概率校准评估门槛（v3.2 新增）

> **关键约束**：纯 SFT 学到的是 JSON 文本中的概率字符串，不是真正的 classifier logits。模型可能输出"语法上合理"但**校准性差**的概率（比如倾向输出整数化的 0.9 而不是 0.92，或在难样本上 over-confident）。
>
> 因此 v3.2 强制规定：第一轮训练结束后，必须用以下指标评估概率校准。**任何一项不达标，必须升级到做法 2（auxiliary classification head）**。

**评估指标**：

| 指标 | 含义 | 阈值（建议起始值） |
|---|---|---|
| **ECE** (Expected Calibration Error) | 预测概率与实际频率的偏差 | ≤ 0.10 |
| **Brier score** | 概率分布的均方误差 | ≤ 0.20 |
| **NLL** (Negative Log-Likelihood) | 概率预测的对数似然 | < baseline NLL（teacher 在 dev set 上的值）|
| **Top-3 accuracy** | 正确 type 在前 3 名的比例 | ≥ 0.90 |
| **Risk-score calibration** | Policy layer 用 risk_score 决策的实际 precision/recall | redact 模式 precision ≥ 0.95；review 模式 recall ≥ 0.85 |

**评估流程**：

```
1. 从训练集独立的 dev set 中抽 500 条样本（含各类别均匀分布）
2. Student 推理输出 type_distribution
3. 与 teacher distribution 比较（teacher 视为 ground truth 概率）
4. 计算 5 项指标
5. 任意一项超过阈值 → 触发升级到做法 2
```

**升级条款**：

如果纯 SFT 不达标，做法 2 改为：
- 保留 SFT 损失主体
- 在 type_distribution 输出位置加 auxiliary classification head（80 维 softmax）
- 用 KL divergence 训练这个 head（teacher distribution → student head）
- 推理时优先用 head 输出，JSON 中的概率字符串作为 fallback

预估升级工作量：1-2 周（主要是 head 接入和重新训练）。

### 7.5 Offset 后处理（替代 v3 的 offset_alignment_loss）

> **v3.1 修正**：Generative model 输出 JSON 中的 `start`/`end` 数字不可微，无法加 loss term。改为推理后校验。

```python
def fix_offsets(text, generated_spans):
    fixed_spans = []
    for span in generated_spans:
        value = span["value"]
        # 优先用 model 给的 offset
        if text[span["start"]:span["end"]] == value:
            fixed_spans.append(span)
            continue
        
        # 在原文中查找 value
        matches = [i for i in range(len(text)) if text[i:i+len(value)] == value]
        
        if len(matches) == 1:
            # 唯一匹配：修复 offset
            span["start"] = matches[0]
            span["end"] = matches[0] + len(value)
            span["offset_fixed"] = True
            fixed_spans.append(span)
        elif len(matches) > 1:
            # 多处匹配：标记 ambiguous，需要人工或上下文消歧
            span["offset_ambiguous"] = True
            span["candidate_offsets"] = matches
            fixed_spans.append(span)
        else:
            # 找不到：丢弃
            log_warning(f"Span value not found in text: {value}")
    
    return fixed_spans
```

### 7.6 训练超参

| 超参 | 值 | 备注 |
|---|---|---|
| 模型 | Qwen3.5-9B-Base | 用 base 而非 instruct，避免 chat template 干扰 JSON 输出 |
| Fine-tune 方法 | LoRA | DeltaNet 模块的 target_modules 待 PEFT 文档确认 |
| LoRA r | 64 | |
| LoRA alpha | 128 | |
| LoRA target_modules | TBD：参考 Unsloth Qwen3.5 配置 | DeltaNet 不是标准 attention 模块 |
| Learning rate | 1e-4 | LoRA 标准 |
| Batch size | 4 (per device) | DGX Spark 120GB 充足 |
| Gradient accumulation | 8 | effective batch = 32 |
| Epochs | 3-5 | validation loss 早停 |
| Max sequence length | 4096 | 99% 训练样本在此范围内 |
| Thinking mode | **主动 disabled**（`enable_thinking=false`）| Qwen3.5 默认开启 thinking，PII 任务用不上，主动关闭节省 inference token |
| Training framework | Unsloth（推荐，TBD 验证）/ Transformers + PEFT（备选）| Unsloth 对 Qwen3.5 支持有 GGUF 仓库，但首次 fine-tune 需 sanity check |

### 7.7 部署量化

训练用 FP16 或 BF16，部署根据显存权衡量化级别。

> **重要**：Qwen3.5-9B 是 early fusion 多模态模型，**视觉部分无法独立卸载**。即使部署只处理文本，仍需加载完整模型权重。下表显存估算已计入此因素。

| 量化级别 | 模型权重显存 | 24GB GPU 部署可行性 | 精度风险 |
|---|---|---|---|
| FP16 | ~19.3GB | 紧张：留 KV cache + activation 仅 ~4GB，文本短可行，长文本 OOM 风险 | 无 |
| INT8 | ~9.7GB | ✅ 充足：~14GB buffer 给 KV cache + activation | 低 |
| Q4_K_M | ~5GB | ✅ 极宽松 | 中（PII 任务格式判断可能掉点，需测）|

**计划**：
- Week 7：测 INT8 vs FP16 精度差异（用 capstone benchmark harness）
- Week 9：决定最终量化级别（默认 INT8，备选 Q4_K_M）

显存预算（24GB GPU、INT8）：
- Model weights (INT8, 含融合的视觉部分): ~9.7GB
- KV cache (4096 context): ~2-3GB
- Activation: ~2GB
- **总计：~14-15GB，留 9-10GB buffer**

注：v3.1 把 vision encoder 单独列出来是错的——early fusion 架构下视觉权重已包含在 model weights 中。

---

## 8. 视觉路径（探索性扩展，不进主验收）

> **v3.2 重要变更**：v3.1 把视觉路径写成主线 deliverable 一部分。但 capstone 文档第 4 节明确将 OCR 和 image-based PII detection 列为 **out of scope**。v3.2 把视觉路径降级为**探索性扩展**：
>
> - **不进入主交付与主验收**
> - **主线评估指标不依赖视觉**
> - **Week 4-7 主训练不分配时间给视觉**
> - 仅在主线达标且时间充裕时（Week 11-12）做演示性实现
> - 即使视觉效果不佳，**capstone 验收不受影响**

### 8.1 总体策略

主交付仅包含文本路径。视觉作为可选 demo 模块，触发条件：

| 阶段 | 条件 | 决策 |
|---|---|---|
| Week 8 中期评估 | 文本路径主线指标达标（precision/recall/calibration 全部过线） | 启动视觉探索 |
| Week 8 中期评估 | 文本路径有问题，需要继续优化 | **不启动视觉**，专注主线 |
| Week 11 时间评估 | 剩余时间 ≥ 2 周 | 实现 8.2 路径 A demo |
| Week 11 时间评估 | 剩余时间 < 2 周 | 跳过视觉，写 limitations 章节即可 |

### 8.2 路径 A：Base model 视觉直接处理（探索性）

**做法**：
- 利用 Qwen3.5-9B 的 unified vision-language 能力
- 不需要专门 fine-tune（fine-tune 数据全是文本）
- Prompt engineering 让它输出与 text path 一致的 JSON schema

**Prompt 模板**：

```
You are a PII detection model. Identify all PII spans in the following image.

Output JSON only, with this schema:
{
  "spans": [
    {
      "value": "<exact text from image>",
      "type_distribution": {<79 PII labels + NON_PII>},
      "top_type": "<argmax label>",
      "bounding_box": {"x": 0, "y": 0, "width": 100, "height": 30}
    }
  ]
}

The 79 PII labels are:
[full taxonomy list]

For each detected PII:
1. Extract the exact text value
2. Estimate type_distribution based on visual context and format
3. Provide bounding box coordinates if possible
```

**评估方法（如果启动）**：
- 用 30-50 张测试图片（合成的护照、驾照、表单截图）做演示性评估
- 指标：visual span recall、type accuracy（不进入主 benchmark harness）

### 8.3 Fallback 路径 C：OCR + Text Student（更探索性）

**仅在路径 A 完成且效果差时考虑**。OCR 集成本身就是 1 周工作量，建议除非有强需求否则跳过。

```python
def process_image(image, primary_path="vision_base"):
    if primary_path == "vision_base":
        result = qwen35_vision_inference(image)
        if result["span_recall_estimate"] < threshold:
            primary_path = "ocr_fallback"
    
    if primary_path == "ocr_fallback":
        ocr_result = run_ocr(image)  # Tesseract / EasyOCR / Azure Doc Intel
        text = ocr_result["text"]
        bboxes = ocr_result["bboxes"]
        
        text_spans = text_student_inference(text)
        
        for span in text_spans:
            span["bounding_box"] = map_offset_to_bbox(
                span["start"], span["end"], bboxes
            )
        
        return text_spans
```

### 8.4 视觉路径在 capstone report 中的定位

- 写入 "Future Work" 或 "Exploratory Extensions" 章节
- 明确说明：**not part of primary deliverable**
- 强调主交付的文本 pipeline 已完成评估

---

## 9. Policy Layer 设计（推理时，与训练解耦）

### 9.1 风险加权决策（v3.1 修订）

> **v3 错误**：只看 top_type，遗漏了 top_type=NON_PII 但 PII 高位概率显著的情况。
>
> **v3.1 修正**：用 risk_score 加权全 80 维分布。

### 9.2 配置

```yaml
# policy_v1.yaml

# 不同 Data Classification 的风险权重
risk_weights:
  Highly_Protected: 1.0
  Protected: 0.6
  Public: 0.1
  NON_PII: 0.0

# 决策阈值
thresholds:
  redact: 0.6
  review: 0.25
  # < 0.25 → ignore
```

### 9.3 决策算法

```python
def decide_policy(span, csv_classification, policy_config):
    dist = span["type_distribution"]
    
    risk_score = 0.0
    for type_, prob in dist.items():
        if type_ == "NON_PII":
            continue  # NON_PII 风险权重为 0
        classification = csv_classification[type_]  # Highly_Protected / Protected / Public
        risk_weight = policy_config["risk_weights"][classification]
        risk_score += prob * risk_weight
    
    thresholds = policy_config["thresholds"]
    if risk_score >= thresholds["redact"]:
        return "redact"
    elif risk_score >= thresholds["review"]:
        return "review"
    else:
        return "ignore"
```

**ChatGPT 反馈例子验证**：

输入分布 `{NON_PII: 0.45, AU_TFN: 0.40, STUDENT_ID: 0.15}`：

```
risk_score = 0.45 * 0 + 0.40 * 1.0 (TFN, Highly_Protected) + 0.15 * 0.6 (STUDENT_ID, Protected)
           = 0 + 0.40 + 0.09
           = 0.49
```

0.49 ≥ 0.25 但 < 0.6，**触发 review**（不会被 ignore）。这正是期望行为。

### 9.4 Per-type 阈值覆盖（可选）

某些极敏感标签可单独设阈值：

```yaml
per_type_overrides:
  AU_TFN:
    standalone_redact_threshold: 0.30  # 单独看 TFN 概率，>0.30 直接 redact
  AU_PASSPORT:
    standalone_redact_threshold: 0.30
```

---

## 10. Build Order

### 10.1 第一阶段：数据准备（Week 1-3）

| 步骤 | 工作 | 工时估计 |
|---|---|---|
| 1 | 写 A 类约 30 标签的格式过滤规则 | 3 天 |
| 2 | 写核心校验规则（TFN / Luhn / Medicare / BSB / IP） | 2 天 |
| 3 | 转换现有 19000 条数据（类别 A + B + D） | 2 天 |
| 4 | 跑 validator，修复异常样本 | 2 天 |
| 5 | 设计并测试 5 档 ranking prompt（用 Qwen 27B teacher） | 2-3 天 |
| 6 | 生成 2000 条类别 C 数据（含 self-consistency × 3） | 4 天 |
| 7 | 整体数据集 sanity check | 1 天 |

**关键节点**：Week 3 末有完整的 ~21000 条 v3.2 格式训练数据。

### 10.2 第二阶段：训练（Week 4-7）

| 步骤 | 工作 | 工时估计 |
|---|---|---|
| 1 | 配置 Unsloth + Qwen3.5-9B fine-tune pipeline（含工具链 sanity check） | 2-3 天 |
| 2 | 确定 LoRA target_modules（DeltaNet + MoE 模块名） | 1 天 |
| 3 | 第一轮训练（baseline，纯 SFT） | 2-3 天 |
| 4 | **概率校准评估（7.4.1 节）+ error analysis** | 2 天 |
| 5 | 决策点：升级到做法 2 还是继续调超参 | 0.5 天 |
| 6 | 第二轮训练（按上一步决策走） | 2-3 天 |
| 7 | INT8 量化测试 | 2 天 |

### 10.3 第三阶段：评估与部署（Week 8-13，对齐 capstone 时间线）

详见 capstone 文档第 9 节。Week 8 关键决策点：
- 概率校准是否最终达标（不达标可能仍需第三轮训练）
- 量化级别选择（INT8 / Q4_K_M）
- 是否启动视觉路径探索性扩展（取决于剩余时间）

Week 11-12 可选活动：
- 视觉路径 A demo 实现（仅在主线达标且时间充裕时）
- 额外的 robustness 测试与 capstone report 撰写

---

## 11. 与 v3 / v3.1 / v2.1 的差异总结

| 维度 | v2.1 | v3 | v3.1 | v3.2 |
|---|---|---|---|---|
| 概率信号 | 双概率解耦 | 单一 80 维分布 | 同 v3 | 同 v3 |
| Confidence 用法 | 没说 | 直接当概率分布 | 改为 training_weight | 同 v3.1 |
| 79 标签格式过滤 | 没说 | 全部要写 | A/B/C 三类 | **format_candidates 明确为上界** |
| Policy 决策 | 不在数据里 | 看 top_type | 风险加权全分布 | 同 v3.1 |
| Offset alignment | 没说 | 写成 loss term | 改为后处理校验 | 同 v3.1 |
| Student 模型 | 没说 | Qwen3 8B | Qwen3.5-9B（架构描述有误） | **Qwen3.5-9B（架构纠正）** |
| Loss | 没说 | KL + span + offset | 纯 SFT（做法 1）| **纯 SFT + 校准评估门槛** |
| 视觉路径 | 没提 | 没提 | 路径 A 主走 + OCR fallback | **降级为探索性扩展** |
| 训练 framework | 没说 | Transformers + PEFT | Unsloth 主推 | **Unsloth TBD，需首次验证** |
| OPF 表述 | / | 没提 | "处理不了" | **"工程复杂度上升，本项目不优先采用"** |
| Hard negatives | 没说 | 笼统 NON_PII | 笼统 NON_PII | **拆为 doc-level + candidate-level** |
| Vision encoder | / | / | 可独立 strip（错误）| **early fusion，不可 strip** |
| Thinking mode | / | / | 默认禁用（错误）| **默认开启，需主动 disable** |

---

## 12. 已知风险与缓解

| 风险 | 缓解 |
|---|---|
| Qwen3.5-9B 工具链不成熟，LoRA target_modules 不明确 | Week 4 第 1 周专门用于 framework 调试；Unsloth 优先尝试，PEFT + Transformers 备选 |
| 纯 SFT 学不到精确概率（如 0.92 vs 0.9） | 7.4.1 节强制评估门槛；ECE / Brier / NLL 任一不达标即升级到做法 2（auxiliary head）|
| 5 档 ranking 仍可能给出不一致 verdict | Self-consistency: 同一 span 调用 3 次取多数 |
| Qwen3.5-9B JSON 输出 offset 飘 | 后处理校验 + value 重定位 |
| INT8 量化掉点严重 | Week 7 早测，必要时加 quantization-aware training |
| 低频标签（IHI / UAC_ID）训练样本不足 | 类别 C 生成时按低频标签倾斜 |
| Sonnet 自评 confidence 校准性差 | v3.1 已不直接用作概率，只作权重 |
| Early fusion 架构无法独立 strip vision，部署显存压力 | 已重新计算显存预算；INT8 部署留 9-10GB buffer 充足 |
| 视觉路径效果差 | 不影响主线，写入 limitations 即可 |
| Thinking mode 误开启增加 inference 成本 | 推理时显式传 `enable_thinking=false` |

---

## 13. 待项目方确认的事项

1. **CSV 的 Data Classification 字段是否权威**：v3.1 用它驱动 risk_weights。如果 University Privacy Office 有更新版本，需先获取。
2. **24GB 部署 GPU 的具体型号**：影响 INT8 / Q4 量化性能（A10 / L4 / RTX 3090 / A5000 等表现差异较大）。
3. **类别 C 生成是否走 Qwen 27B**：还是继续用 Sonnet？建议用 Qwen 27B 保持 teacher 一致性。
4. **训练完成后的人工 verification 计划**：从 type_distribution 高熵样本中抽样 200-300 条做人工标注作为 gold test set。
5. **概率校准评估的具体阈值**（7.4.1 节）：ECE / Brier / NLL 阈值起始值是建议值，需 Week 6-7 实测校准。
6. **LoRA target_modules 具体名称**：Qwen3.5 DeltaNet + MoE 模块结构需在 Week 4 framework 调试阶段从 PEFT/Unsloth 文档或 model config 确定。
7. **（仅探索性）视觉测试图片来源**：仅在主线达标后才需要决定。合成生成 vs 公开数据集（FUNSD / CORD）。

---

## 14. 对外部反馈的回应

> 此节记录方案演化中的判断，避免后续团队讨论时重复争论已经决定的事。

### 14.1 v3 → v3.1 采纳的修订（4 项）

1. ✅ Confidence 不直接当概率 → 改为 training_weight + label smoothing one-hot
2. ✅ 79 标签不强制都有 format_filter → 分 A/B/C 三类
3. ✅ Policy 不只看 top_type → 改为 risk_score 加权
4. ✅ offset_alignment_loss 不可微 → 改为后处理校验

### 14.2 v3.1 → v3.2 采纳的修订（6 项）

1. ✅ OPF 表述改软（见 14.3 重新表述）
2. ✅ 纯 SFT 必须配概率校准评估门槛 + 升级条款（见 7.4.1）
3. ✅ Qwen3.5 实现细节标 TBD（LoRA target_modules、Unsloth 稳定性）
4. ✅ 视觉路径降级为探索性扩展（见第 8 节）
5. ✅ format_candidates 明确为候选上界（见 6.1）
6. ✅ Hard negatives 拆为 document-level + candidate-level（见 5.3.1）

### 14.3 关于 OPF 的最终表述（v3.2 修订）

> **v3.1 错误**：v3.1 写"OPF 处理不了 multi-type span"过于绝对。
>
> **v3.2 修正**：

标准单头 BIOES OPF 不天然支持同一 span 多标签（如 NUMBER_PLATE + VEHICLE_REGO 同位置）。但通过 multi-label head、multi-task head、span-level classifier 或多次解码等工程方案，OPF 是**可以**处理这种情况的。

**本项目不优先采用 OPF 的理由**（不是"OPF 不能"）：

1. 工程复杂度上升：multi-label head 设计、训练、推理都比生成式更复杂
2. v3.1/v3.2 schema 已经为生成式设计，切换需重写训练 pipeline
3. 19000 条已有数据格式天然适合生成式模型直接学
4. Qwen3.5-9B 自带视觉能力，覆盖 capstone 探索性扩展需求
5. capstone 13 周时间紧，避免方案切换成本

**结论**：本项目走 Qwen3.5-9B 生成式路线。OPF 作为参考 baseline 在 capstone report 中提及但不实现。

### 14.4 不采纳的反馈（3 项及理由）

1. ❌ "对外恢复 pii_probability + 隐藏 NON_PII" → `pii_probability = 1 - P(NON_PII)` 是确定性派生量，无需单独存储。下游 policy layer 需要看完整 80 维（含 NON_PII）才能算 risk_score，刻意隐藏反而麻烦。
2. ❌ "Hard negatives 不让 Qwen 输出 NON_PII span" → 这是对训练目标的误解。已在 5.3.1 节用 document-level vs candidate-level 拆分清楚。
3. ❌ "纯 SFT 完全不能做 KL loss" → 半对半错。纯 SFT 能 work，只是学到的概率精度有限（已在 7.3 / 7.4.1 写明 known limitation 及升级路径）。

---

**文档版本**：v3.2  
**最后更新**：2026-04-30  
**关键负责人**：[TBD]  
**下游依赖**：student 模型训练 pipeline、policy layer 实现、benchmark harness  
**视觉路径**：探索性扩展，不进入主交付

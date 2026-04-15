# Teacher-Student LoRA Distillation Project

这个项目当前采用的是一条更明确的 teacher-student 流程：

1. 原始文本放在 `data/raw/au_pii_19000.json`
2. teacher 先做 span 标注
3. teacher 输出被清洗成 student 可训练数据
4. 再构造成 BIO token 标签
5. 最后进行 student 的 LoRA / BIO distillation 训练

## 当前推荐流程

对于你现在的本地环境，**推荐使用一键脚本**，不要直接运行 `src/train_distill.py`。

原因是：

- 你当前使用的是本地 `Qwen3.5-27B` GGUF teacher
- GGUF teacher 适合做 teacher labelling
- 但 `src/train_distill.py` 需要的是 Hugging Face 格式的 teacher 模型，不适合直接拿 GGUF 跑

因此，当前最稳妥的做法是通过 `scripts/run_train.sh` 走完整流程。

## 项目结构

- `config/` - 配置文件
- `data/raw/` - 原始数据
- `data/teacher/` - teacher 标注输出（运行时自动生成）
- `data/processed_teacher/` - teacher 清洗后的中间数据（自动生成）
- `data/processed_bio/` - BIO token 数据（自动生成）
- `outputs/` - 训练输出（自动生成）
- `src/` - 数据与训练脚本
- `scripts/` - 一键运行脚本

## 快速开始

### 1. 小样本 smoke test

建议先只跑少量样本，确认 teacher 输出质量：

```bash
TEACHER_MODEL_PATH=/home/admin/model/Qwen3.5/Qwen3.5-27B-Q4_K_M-GGUF MAX_SAMPLES=10 bash scripts/run_train.sh
```

### 2. 全量运行

确认 teacher 输出没问题后，再跑全量：

```bash
TEACHER_MODEL_PATH=/home/admin/model/Qwen3.5/Qwen3.5-27B-Q4_K_M-GGUF bash scripts/run_train.sh
```

如果仓库中存在 `data/raw/cleaned_test_set.json`，脚本会自动把它当作独立验证集使用，而不是再从 teacher 训练数据里切一部分做验证。

### 3. 如果使用远程 teacher API

```bash
TEACHER_API_URL=https://api.example.com TEACHER_API_KEY=xxx bash scripts/run_train.sh
```

## 训练过程中会自动生成的目录

下面这些目录都是**中间产物或输出**，删除后可以再次生成：

- `data/teacher/`
- `data/processed_teacher/`
- `data/processed_bio/`
- `outputs/`

## 常见问题

### 为什么不要直接运行 `src/train_distill.py`

如果 teacher 是 GGUF 文件或只包含 GGUF 的目录，`src/train_distill.py` 会报错；这属于预期行为。

### 如果脚本中途停止怎么办

如果 `scripts/run_train.sh` 提示 teacher 没有解析出有效 BIO 标签，说明 teacher 输出里仍然混入了 reasoning 或非 JSON 文本。此时应先检查：

- `data/teacher/teacher_labels.jsonl`
- teacher 是否真的只输出 JSON
- `MAX_SAMPLES` 小样本下是否已经稳定

## 当前配置位置

- 训练与 LoRA 参数：`config/lora_config.yaml`
- teacher 标注脚本：`src/teacher/teacher_labeling.py`
- BIO 数据构建：`src/build_bio_dataset.py`
- BIO distillation 训练：`src/train_bio_distill.py`
- 一键运行入口：`scripts/run_train.sh`

## 下一步建议

当前仓库已经清理完旧产物，所以下一步最合理的是：

1. 先做一次小样本 smoke test
2. 检查 teacher 输出是不是干净 JSON
3. 再决定是否跑全量训练

这样可以避免直接全量运行后才发现 teacher 输出格式不稳定。

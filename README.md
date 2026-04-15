# LoRA 微调项目

这是一个面向 LoRA 微调的项目骨架，适合你当前已有 `19000` 条 JSON 样本的数据集。

## 结构说明

- `config/` - 训练参数、模型与 LoRA 配置
- `data/raw/` - 原始数据，用于存放你的 JSON 文件
- `data/processed/` - 处理后的训练/验证数据
- `src/` - 数据准备与训练脚本
- `scripts/` - 运行脚本
- `requirements.txt` - 依赖包

## 使用方法

1. 将你的原始 JSON 数据放入 `data/raw/au_pii_19000.json`
2. 先运行 teacher labelling，生成 teacher 输出：

```bash
python src/teacher/teacher_labeling.py --input data/raw/au_pii_19000.json --output_dir data/teacher --api_url <YOUR_API_URL> --api_key <YOUR_API_KEY>
```

如果你要本地运行 `qwen-3.5-9b`，请改成：

```bash
python src/teacher/teacher_labeling.py --input data/raw/au_pii_19000.json --output_dir data/teacher --model_path /path/to/qwen-3.5-9b
```

`/path/to/qwen-3.5-9b` 可以是你当前机器上已有模型所在的任何目录，
不必一定放到项目目录里。

如果你希望把模型放在项目内，建议使用项目下的 `models/qwen-3.5-9b/` 或 `data/teacher/model/` 目录，然后同样用 `--model_path` 指向它。

> 注意：本地 `qwen-3.5-9b` 生成会非常慢。建议先测试少量记录：

```bash
python src/teacher/teacher_labeling.py --input data/raw/au_pii_19000.json --output_dir data/teacher --model_path /home/admin/model/Qwen3.5/Qwen3.5-9b --max_samples 10
```

3. 检查 `data/teacher/teacher_labels.jsonl` 中的 teacher 输出是否符合 span detection 要求
4. 把 teacher 输出转换成 student 训练目标数据：

```bash
python src/teacher/prepare_teacher_student_data.py --input data/teacher/teacher_labels.jsonl --output_dir data/processed_teacher --val_ratio 0.05
```

5. 启动真实蒸馏训练：

```bash
python src/train_distill.py --config config/lora_config.yaml
```

本地蒸馏需要一个可访问的 teacher 模型；请在 `config/lora_config.yaml` 的 `distillation.teacher_model_name` 中指定 teacher 模型名称或本地路径。

你也可以一键运行整个 pipeline：

```bash
TEACHER_MODEL_PATH=/path/to/qwen-3.5-9b bash scripts/run_train.sh
```

或使用远程 teacher API：

```bash
TEACHER_API_URL=https://api.example.com TEACHER_API_KEY=xxx bash scripts/run_train.sh
```

6. 如果你想继续使用普通 LoRA 训练，请把 `config/lora_config.yaml` 的 `train_file` / `validation_file` 改为：

```yaml
train_file: "data/processed_teacher/train.jsonl"
validation_file: "data/processed_teacher/val.jsonl"
```

7. 启动普通微调：

```bash
python src/train_lora.py --config config/lora_config.yaml
```

## 常见调整

- 如果你的 JSON 每条记录是 `{"prompt": ..., "response": ...}`，请修改 `text_key` 与 `target_key`
- 如果你需要根据 PDF 里的项目要求调整 `batch_size`、`learning_rate`、`validation_split`、`model_name`，请直接修改 `config/lora_config.yaml`

## 注意

- 19000 条样本是一个合理的规模，建议保留 5%-10% 作为验证集
- 若模型参数较大，可开启 `gradient_checkpointing` 或调整 `per_device_train_batch_size`

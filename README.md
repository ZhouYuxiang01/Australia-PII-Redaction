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

1. 将你的原始 JSON 数据放入 `data/raw/dataset.json`
2. 根据数据格式修改 `config/lora_config.yaml` 中的 `text_key` / `target_key`
3. 运行数据准备脚本：

```bash
python src/data_preparation.py --input data/raw/dataset.json --output_dir data/processed
```

4. 启动微调：

```bash
python src/train_lora.py --config config/lora_config.yaml
```

## 常见调整

- 如果你的 JSON 每条记录是 `{"prompt": ..., "response": ...}`，请修改 `text_key` 与 `target_key`
- 如果你需要根据 PDF 里的项目要求调整 `batch_size`、`learning_rate`、`validation_split`、`model_name`，请直接修改 `config/lora_config.yaml`

## 注意

- 19000 条样本是一个合理的规模，建议保留 5%-10% 作为验证集
- 若模型参数较大，可开启 `gradient_checkpointing` 或调整 `per_device_train_batch_size`

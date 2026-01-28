# 训练指南

## 概述

Medical-Agentic-RAG 的训练分为两个阶段:

1. **阶段一**: Embedding模型对比学习微调
2. **阶段二**: Agentic RAG RL训练 (VERL+GRPO)

---

## 阶段一: Embedding微调

### 训练目标

通过对比学习统一图像和文本的向量空间，提高多模态检索的召回率。

### 数据准备

#### 1. 构建训练数据

```python
# 数据格式
{
    "query_text": "描述查询",           # 查询文本
    "pos_text": "正样本文本",           # 匹配的文档文本
    "pos_image": "path/to/image.jpg",   # 可选的图像路径
}
```

#### 2. 生成训练数据

```python
from MMRAG.task.preprocess import generate_training_data

generate_training_data(
    dataset_names=["medmax", "iu_xray", "pmc_vqa"],
    output_dir="./datasets/training_data",
    vlm_model="Qwen3-VL-8B-Instruct"
)
```

### 训练配置

#### 单卡训练

```bash
cd train/finetune_embedding.py
python main.py \
    --model_path "Qwen/Qwen3-VL-Embedding-2B" \
    --data_path "./datasets/training_data" \
    --output_dir "./checkpoints/embedding" \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_epochs 3
```

#### 多卡训练 (FSDP)

```bash
cd train/finetune_embedding.py
torchrun --nproc_per_node=4 main.py \
    --model_path "Qwen/Qwen3-VL-Embedding-2B" \
    --data_path "./datasets/training_data" \
    --output_dir "./checkpoints/embedding" \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --fsdp
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature` | 0.05 | 对比学习温度系数 |
| `learning_rate` | 2e-5 | 学习率 |
| `batch_size` | 4 | 每卡batch size |
| `gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `max_seq_length` | 2048 | 最大序列长度 |
| `pooling_strategy` | "mean" | 池化策略 (mean/last_token) |

### 训练监控

#### 使用Weights & Biases

```python
import wandb

wandb.init(
    project="medical-rag",
    name="embedding-finetune-run1",
    config={"lr": 2e-5, "batch_size": 16}
)
```

#### 使用TensorBoard

```bash
tensorboard --logdir ./checkpoints/embedding/runs
```

### 评估检索性能

```bash
python scripts/eval_retrieval.py \
    --model_path "./checkpoints/embedding" \
    --test_data "./datasets/test_data.json" \
    --metrics recall@5,recall@10,mrr
```

---

## 阶段二: Agentic RAG RL训练

### 训练目标

通过强化学习训练智能代理，使其能够:
- 自主决定检索策略
- 合理使用文本和图像检索工具
- 基于检索结果生成准确回答

### 依赖安装

```bash
# 安装VERL框架
pip install verl
```

### 数据格式

```python
{
    "query": "问题文本",
    "image": "path/to/image.jpg",  # 可选
    "ground_truth": "标准答案",
    "conversation": [
        {"role": "user", "content": [...]},
        {"role": "assistant", "content": [...]}
    ]
}
```

### 训练启动

```bash
python train/rl_train.py \
    --config configs/rl_train_config.yaml \
    --base_model "Qwen3-VL-4B-Instruct" \
    --embedding_model "./checkpoints/embedding" \
    --output_dir "./checkpoints/rl_agent"
```

### GRPO配置

```yaml
# configs/rl_train_config.yaml
rl:
  algorithm: "grpo"
  group_size: 8
  epsilon: 0.2

reward:
  type: "exact_match"  # exact_match/f1/llm_judge
  weights:
    accuracy: 1.0
    retrieval_relevance: 0.5

training:
  num_iterations: 1000
  batch_size: 32
  learning_rate: 1e-6
```

### 奖励函数

#### 1. 精确匹配奖励

```python
def exact_match_reward(prediction: str, ground_truth: str) -> float:
    """选项匹配奖励"""
    pred_label = extract_answer_label(prediction)
    gt_label = extract_answer_label(ground_truth)
    return 1.0 if pred_label == gt_label else 0.0
```

#### 2. F1分数奖励

```python
def f1_reward(prediction: str, ground_truth: str) -> float:
    """F1分数奖励，适用于开放式回答"""
    pred_tokens = set(tokenize(prediction))
    gt_tokens = set(tokenize(ground_truth))

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1
```

---

## 训练技巧

### 1. 混合精度训练

```python
from torch.distributed.fsdp import MixedPrecision

mixed_precision = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)
```

### 2. 梯度检查点

```python
model.gradient_checkpointing_enable()
```

### 3. 学习率调度

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)
```

### 4. 早停策略

```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    ...,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

---

## 常见问题

### Q: 显存不足怎么办？

**A**: 尝试以下方法:
1. 减小 `batch_size`
2. 增加 `gradient_accumulation_steps`
3. 启用梯度检查点
4. 使用 DeepSpeed ZeRO-3
5. 减小 `max_seq_length`

### Q: 训练发散怎么办？

**A**:
1. 降低学习率 (尝试 1e-6 ~ 5e-6)
2. 增加 warmup 步数
3. 使用梯度裁剪 `max_grad_norm=1.0`

### Q: 检索性能提升不明显？

**A**:
1. 检查数据质量，确保正负样本区分明显
2. 调整负样本采样策略
3. 增加训练数据量
4. 尝试不同的温度系数

# 评估指南

## 评估体系

Medical-Agentic-RAG 的评估分为三个层次:

1. **检索性能评估**: 向量检索的准确性
2. **生成质量评估**: RAG系统的回答质量
3. **端到端评估**: 完整任务的性能

---

## 1. 检索性能评估

### 评估指标

| 指标 | 说明 | 公式 |
|------|------|------|
| Recall@k | Top-k中包含正确答案的比例 | $\\frac{1}{|Q|} \\sum_{q \\in Q} \\mathbb{1}(\\text{gold} \\in \\text{top-k})$ |
| MRR | 平均倒数排名 | $\\frac{1}{|Q|} \\sum_{q \\in Q} \\frac{1}{\\text{rank}_q}$ |
| NDCG@k | 归一化折损累积增益 | 考虑排名的加权指标 |

### 运行评估

```bash
python scripts/eval_retrieval.py \
    --model_path "Qwen/Qwen3-VL-Embedding-2B" \
    --test_data "./datasets/test_retrieval.json" \
    --vector_db "milvus" \
    --metrics recall@5,recall@10,mrr \
    --output "./results/retrieval_metrics.json"
```

### 测试数据格式

```python
[
    {
        "query": "胸部X光显示肺炎症状",
        "query_image": "path/to/query_img.jpg",  # 可选
        "gold_doc_ids": ["doc_001", "doc_002"],
        "gold_passages": ["The chest X-ray reveals..."]
    }
]
```

### 结果示例

```json
{
    "recall@5": 0.72,
    "recall@10": 0.85,
    "mrr": 0.58,
    "num_queries": 1000
}
```

---

## 2. 生成质量评估

### 评估方法

#### 2.1 自动评估

**基于规则的评估**:

```python
def evaluate_answer(prediction: str, ground_truth: str, metric: str) -> float:
    if metric == "exact_match":
        return float(prediction.strip() == ground_truth.strip())
    elif metric == "contains":
        return float(ground_truth in prediction)
    elif metric == "f1":
        return compute_f1(prediction, ground_truth)
```

**基于模型的评估 (GPT-4)**:

```python
EVALUATION_PROMPT = """
请评估以下回答的质量。

问题: {question}
标准答案: {ground_truth}
模型回答: {prediction}

请从以下维度评分(1-5分):
1. 准确性: 回答是否 medically accurate
2. 完整性: 是否涵盖了关键信息
3. 相关性: 是否回答了问题

请以JSON格式返回:
{"accuracy": X, "completeness": Y, "relevance": Z}
"""
```

#### 2.2 人工评估

创建评估模板:

```python
eval_template = {
    "sample_id": "",
    "question": "",
    "prediction": "",
    "ground_truth": "",
    "ratings": {
        "correctness": 0,  # 0-5
        "completeness": 0,
        "conciseness": 0,
        "clinical_relevance": 0
    },
    "comments": ""
}
```

### 运行生成评估

```bash
# 使用本地VLM评估
python eval/eval_transformers.py \
    --model_path "Qwen3-VL-4B-Instruct" \
    --dataset "medmax" \
    --output_dir "./results/generation"

# 使用OpenAI API评估
python eval/eval_openai.py \
    --model "Qwen3-VL-4B-Instruct" \
    --api_base "http://localhost:8000/v1" \
    --dataset "pmc_vqa" \
    --output_dir "./results/generation"
```

---

## 3. 端到端评估

### VQA任务评估

#### MedMax评估

```python
from MMRAG.task.medmax import MedMaxDataset
from eval.dataset import EvaluationDataset

dataset = MedMaxDataset("medmax", vlm_service)
results = dataset.evaluate(index, conversation)
```

#### PMC-VQA评估

```python
from MMRAG.task.pmc_vqa import PMCVQADataset

dataset = PMCVQADataset("pmc_vqa", vlm_service)
accuracy = dataset.evaluate_batch(predictions, ground_truths)
```

### 多模态Benchmark

| Benchmark | 任务类型 | 评估指标 |
|-----------|----------|----------|
| MedQA | 文本QA | Accuracy |
| MMLU-med | 多项选择 | Accuracy |
| MedXpertQA | VQA | Accuracy |
| VQA-RAD | 放射学VQA | Accuracy |
| MIMIC-CXR | 图像分类 | AUC-ROC |
| ChestX-ray14 | 图像分类 | AUC-ROC |

### 运行Benchmark评估

```bash
# 评估所有benchmarks
python eval/run_benchmarks.py \
    --model "Qwen3-VL-4B-Instruct" \
    --benchmarks medqa,vqa_rad,medxpertqa \
    --output_dir "./results/benchmarks"

# 评估单个benchmark
python eval/eval_transformers.py \
    --model_path "Qwen3-VL-4B-Instruct" \
    --dataset "vqa_rad" \
    --batch_size 8
```

---

## 4. 对比实验

### 基线方法

| 方法 | 说明 |
|------|------|
| Naive VLM | 直接使用VLM，不使用RAG |
| Text-only RAG | 仅使用文本检索 |
| Image-only RAG | 仅使用图像检索 |
| Multi-modal RAG | 同时使用文本和图像检索 |
| Agentic RAG | 使用RL代理动态选择检索策略 |

### 运行对比实验

```bash
python eval/compare_methods.py \
    --methods naive,text_rag,multimodal_rag,agentic_rag \
    --dataset "medmax" \
    --metrics accuracy,recall,latency \
    --output "./results/comparison.csv"
```

---

## 5. 结果可视化

### 生成报告

```python
from eval.visualize import generate_report

generate_report(
    results_dir="./results",
    output_file="./results/evaluation_report.html",
    include_charts=True
)
```

### 可视化指标

```python
# 混淆矩阵
plot_confusion_matrix(predictions, labels, save_path="confusion_matrix.png")

# PR曲线
plot_pr_curve(results, save_path="pr_curve.png")

# 检索结果分布
plot_retrieval_distribution(scores, save_path="score_dist.png")
```

---

## 6. 持续评估

### 集成到CI/CD

```yaml
# .github/workflows/eval.yml
name: Evaluation
on: [push]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run evaluation
        run: |
          python eval/run_benchmarks.py \
            --model "Qwen3-VL-4B-Instruct" \
            --benchmarks medqa
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: eval-results
          path: ./results/
```

### 性能回归检测

```python
def check_regression(current_results, baseline_results, threshold=0.05):
    """检测性能回归"""
    regressions = []
    for metric, current_val in current_results.items():
        baseline_val = baseline_results.get(metric, 0)
        if current_val < baseline_val * (1 - threshold):
            regressions.append({
                "metric": metric,
                "baseline": baseline_val,
                "current": current_val,
                "drop": (baseline_val - current_val) / baseline_val
            })
    return regressions
```

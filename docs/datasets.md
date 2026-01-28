# 数据集说明文档

## 支持的数据集

### 1. MedMax

**来源**: [mint-medmax/medmax_data](https://huggingface.co/datasets/mint-medmax/medmax_data)

**特点**:
- 多任务医疗数据集
- 包含VQA、图像生成、报告理解等任务
- 部分数据需要凭证访问

**数据结构**:
```python
{
    "text": "Instruction + Context + Response",
    "tokens": {...},           # 预分词结果
    "image_path": [...],       # 图像路径列表
    "task": "VQA/Image Generation/Report Understanding",
    "source": "数据来源",
    "credential": "yes/no"     # 是否需要凭证
}
```

**处理流程**:
```python
from MMRAG.task.medmax import MedMaxDataset
from MMRAG.model_service.vlm_service import OpenAIVLMService

vlm_service = OpenAIVLMService(model_name="Qwen3-VL-4B-Instruct")
dataset = MedMaxDataset("medmax", vlm_service)
dataset.process_dataset(dataset_size=100, rewrite=True)
```

### 2. IU-Xray

**来源**: [IU Chest X-Ray Collection](https://openi.nlm.nih.gov/)

**特点**:
- 胸部X光片及报告
- 放射学领域
- 图像-报告配对

**数据结构**:
```python
{
    "image_paths": [...],
    "report": "Findings and impression",
    "question": "Generated question",
    "answer": "Answer from report"
}
```

### 3. Harvard-FairVLMed

**来源**: Harvard FairVLMed Dataset

**特点**:
- 眼科学领域
- 眼底图像
- 公平性评估

### 4. PMC-OA

**来源**: PubMed Central Open Access

**特点**:
- 病理学图像
- 开放获取
- 大规模

### 5. PMC-VQA

**来源**: [PMC-VQA](https://github.com/xiaoman-zhang/PMC-VQA)

**特点**:
- 病理视觉问答
- 多选题格式
- 专业知识密集

## 数据预处理

### 通用处理流程

```python
# 统一数据格式
{
    "question": str,          # 问题文本
    "image_paths": List[str], # 图像路径列表
    "content": str,           # 改写后的内容
    "index": int,             # 样本索引
    "chunk_id": str,          # 唯一标识
    "dataset_id": str,        # 数据集名称
    "answer": str,            # 原始答案
    "answer_label": str,      # 提取的选项标签
    "key_words": List[str],   # 关键词
    "source": str             # 数据来源
}
```

### 图像处理

```python
# 保存图像到本地
def save_images(examples, output_dir):
    image_paths = []
    for idx, image in enumerate(examples['images']):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        path = f"{output_dir}/images/{chunk_id}_img{idx}.jpeg"
        image.save(path)
        image_paths.append(path)
    return image_paths
```

### 内容改写

使用VLM对问题进行改写，提高检索质量：

```python
REWRITE_CONTENT_PROMPT = """
请根据以下医疗问题，改写为一个可以用于检索相似病例的查询语句。
保留关键医学术语，使查询更加规范和专业。

问题: {question}

改写后的查询:
"""
```

## 数据加载

### 从Hugging Face加载

```python
from datasets import load_dataset

# MedMax
dataset = load_dataset("mint-medmax/medmax_data", split="test")

# MedVL-Thinker
dataset = load_dataset("UCSC-VLAA/MedVLThinker-Eval")
```

### 从本地加载

```python
from datasets import load_from_disk

# 加载预处理后的数据
dataset = load_from_disk("./datasets/processed_datasets/medmax")
```

## 构建训练数据

### Embedding微调数据

**正样本构建**:
```python
{
    "query_text": "胸部X光显示什么异常?",
    "pos_text": "The chest X-ray shows bilateral infiltrates...",
    "pos_image": "path/to/chest_xray.jpg"
}
```

**负样本采样**:
```python
def sample_negatives(query_vector, corpus_vectors, top_k=10, delta=0.1):
    """
    检索top-k相似样本，选取满足 s < s+ + delta 的作为负样本
    避免假阴性(FN)问题
    """
    similarities = cosine_similarity([query_vector], corpus_vectors)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    negatives = []
    pos_similarity = similarities[sorted_indices[0]]  # 正样本相似度

    for idx in sorted_indices[1:top_k+1]:
        if similarities[idx] < pos_similarity + delta:
            negatives.append(idx)

    return negatives
```

### RL训练数据

```python
{
    "messages": [
        {"role": "user", "content": [...]},
        {"role": "assistant", "content": [...]}
    ],
    "tool_calls": [...],  # 工具调用序列
    "answer": "ground truth"
}
```

## 评估数据格式

### VQA评估

```python
{
    "question": "What abnormality is seen in the chest X-ray?",
    "choices": ["A. Normal", "B. Pneumonia", "C. Fracture"],
    "answer": "B",
    "image": "path/to/image.jpg"
}
```

### 检索评估

```python
{
    "query": "胸部X光显示肺炎",
    "gold_doc_ids": ["doc_001", "doc_002"],
    "metadata": {...}
}
```

**评估指标**:
- Recall@k: Top-k中包含正确答案的比例
- MRR: 平均倒数排名
- NDCG: 归一化折损累积增益

## 数据目录结构

```
datasets/
├── processed_datasets/      # 预处理后的数据
│   ├── medmax/
│   ├── iu_xray/
│   ├── harvard_fairvlmed/
│   └── pmc_vqa/
├── raw/                     # 原始数据（下载）
│   └── ...
└── MedQA/                   # MedQA医学教材
    └── data_clean/
        └── textbooks/
            ├── en/          # 英文教材
            └── zh_paragraph/ # 中文教材
```

## 数据准备脚本

```bash
# 下载并预处理所有数据集
python scripts/prepare_data.py --datasets medmax,iu_xray,pmc_vqa

# 仅预处理特定数据集
python scripts/prepare_data.py --datasets medmax --size 1000
```

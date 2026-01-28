# Medical-Agentic-RAG

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Medical-Agentic-RAG 是一个针对医疗领域的多模态智能问答系统，集成文本、图像和结构化数据，提升医疗信息检索和患者咨询的效率与准确性。

## 项目特点

- **多模态RAG**: 支持文本、图像的联合检索与生成
- **医疗专业化**: 针对医疗领域优化，支持放射学、病理学、眼科学等多个科室
- **Agentic架构**: 基于强化学习的智能代理系统，可自主决策检索策略
- **高效Embedding**: 基于Qwen3-VL-Embedding的统一多模态向量表示
- **灵活部署**: 支持vLLM本地部署和OpenAI兼容API

## 项目架构

```
Medical-Agentic-RAG/
├── MMRAG/                      # 核心RAG框架
│   ├── MMRAG.py               # 主RAG类
│   ├── base.py                # 基础数据类
│   ├── DB/                    # 向量数据库实现
│   ├── model_service/         # 模型服务
│   ├── task/                  # 数据集处理
│   └── agent/                 # Agentic RAG组件
├── train/                     # 训练模块
├── eval/                      # 评估模块
├── configs/                   # 配置文件
└── docs/                      # 文档
```

## 快速开始

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/yourusername/Medical-Agentic-RAG.git
cd Medical-Agentic-RAG

# 创建虚拟环境
conda create -n medrag python=3.10
conda activate medrag

# 安装依赖
pip install -r requirements.txt
```

### 启动服务

```bash
# 启动Embedding服务 (vLLM)
bash MMRAG/scripts/embedding_server.sh

# 启动VLM服务
bash MMRAG/scripts/vlm_server.sh
```

### 基础使用

```python
import asyncio
from MMRAG.MMRAG import MMRAG
from MMRAG.model_service.embedding_service import OpenaiEmbeddingService
from MMRAG.model_service.vlm_service import VLMService

async def main():
    # 初始化服务
    embedding_service = OpenaiEmbeddingService()
    llm_service = VLMService()

    # 创建RAG实例
    rag = MMRAG(
        embedding_service=embedding_service,
        llm_service=llm_service,
        workspace="./workspace"
    )

    # 数据入库
    await rag.ingest_dataset("UCSC-VLAA/MedVLThinker-Eval", dataset_size=100)

    # 执行检索
    query = [{"text": "描述胸部X光片发现"}]
    results = await rag.retrieve(query, top_k=5)

    for r in results:
        print(f"Score: {r.metadata.get('score')}: {r.content[:100]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

## 支持的模型

### Embedding模型
- Qwen3-VL-Embedding-2B (推荐)
- 自定义微调模型

### 生成模型
- Qwen3-VL-4B/8B-Instruct
- MedGemma-4B

## 支持的数据集

| 数据集 | 类型 | 领域 |
|--------|------|------|
| MedMax | 多任务医疗 | 通用 |
| IU-Xray | 放射学 | 胸部X光 |
| Harvard-FairVLMed | VQA | 眼科学 |
| PMC-OA | 病理学 | 开放获取 |
| PMC-VQA | VQA | 病理 |

## 评估Benchmark

- **文本QA**: MedQA, MMLU-med
- **图像分类**: MIMIC-CXR, ChestX-ray14
- **VQA**: MedXpertQA, VQA-RAD

## 训练

### Embedding微调

```bash
# 使用FSDP进行多卡训练
cd train/finetune_embedding.py
torchrun --nproc_per_node=4 main.py
```

详见 [docs/training.md](docs/training.md)

## 文档

- [架构设计](docs/architecture.md) - 系统架构详解
- [数据集说明](docs/datasets.md) - 数据集准备与使用
- [训练指南](docs/training.md) - 模型微调教程
- [评估指南](docs/evaluation.md) - 性能评估方法
- [API文档](docs/api_reference.md) - API接口说明

## 贡献

欢迎提交Issue和Pull Request。

## 许可证

本项目采用 MIT 许可证。

## 致谢

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) - 基础模型
- [MedMax](https://huggingface.co/datasets/mint-medmax/medmax_data) - 医疗数据集
- [VERL](https://github.com/volcengine/verl) - RL训练框架

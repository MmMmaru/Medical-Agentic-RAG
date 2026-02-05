# MMRAG 测试文档

本文档描述 Medical Multi-modal RAG 系统的测试模块。

## 测试模块位置

- **测试脚本**: `MMRAG/tests/test_rag.py`
- **测试类**: `TestMMRAG`

## 功能概述

测试模块提供以下三类测试功能：

1. **检索性能测试** (`test_retrieval_performance`)
2. **端到端 RAG 测试** (`test_e2e_rag`)
3. **混合检索对比测试** (`test_hybrid_retrieval`, `test_hybrid_retrieval_batch`)

## 环境要求

### 服务依赖

测试需要以下服务正在运行：

| 服务 | 默认URL | 用途 |
|------|---------|------|
| Embedding服务 | http://localhost:8001/v1 | 多模态向量嵌入 |
| VLM服务 | http://localhost:8000/v1 | 视觉语言模型推理 |

### 环境变量

```bash
# 测试工作目录
export MMRAG_TEST_WORKSPACE="./test_workspace"

# 默认测试数据集大小
export MMRAG_TEST_DATASET_SIZE="100"

# 服务URL配置
export EMBEDDING_SERVICE_URL="http://localhost:8001/v1"
export VLM_SERVICE_URL="http://localhost:8000/v1"
```

## 使用方法

### 命令行运行

```bash
# 运行所有测试
python MMRAG/tests/test_rag.py --test all

# 仅运行检索性能测试
python MMRAG/tests/test_rag.py --test retrieval --dataset pmc-oa --dataset-size 100

# 仅运行端到端RAG测试
python MMRAG/tests/test_rag.py --test e2e --dataset pmc-oa --dataset-size 50

# 运行混合检索对比测试
python MMRAG/tests/test_rag.py --test hybrid --dataset pmc-oa --dataset-size 50

# 测试单个查询的混合检索
python MMRAG/tests/test_rag.py --test hybrid --query "What is shown in this medical figure?"

# 启用Reranker
python MMRAG/tests/test_rag.py --test e2e --use-reranker
```

### 编程使用

```python
import asyncio
from MMRAG.tests.test_rag import TestMMRAG

async def run_tests():
    # 初始化测试器
    tester = TestMMRAG(
        workspace="./test_workspace",
        embedding_url="http://localhost:8001/v1",
        vlm_url="http://localhost:8000/v1",
        use_reranker=False
    )

    try:
        # 1. 检索性能测试
        metrics = await tester.test_retrieval_performance(
            dataset_name="pmc-oa",
            dataset_size=100
        )
        print(f"Recall@5: {metrics.recall_at_5}")
        print(f"MRR: {metrics.mrr}")

        # 2. 端到端RAG测试
        results = await tester.test_e2e_rag(
            dataset_name="pmc-oa",
            dataset_size=50
        )

        # 3. 混合检索对比
        summary = await tester.test_hybrid_retrieval_batch(
            dataset_name="pmc-oa",
            dataset_size=50
        )

    finally:
        await tester.cleanup()

asyncio.run(run_tests())
```

## 测试功能详解

### 1. 检索性能测试

**方法**: `test_retrieval_performance`

**评估指标**:
- **Recall@K**: Top-K结果中包含正确答案的比例 (K=1, 5, 10)
- **MRR (Mean Reciprocal Rank)**: 正确答案倒数排名的平均值
- **平均查询时间**: 每次检索的平均耗时

**测试流程**:
1. 加载 PMCOADataset 数据集
2. 对每个样本，使用 `question` 作为查询
3. 执行向量检索 (naive_retrieve)
4. 检查 gold chunk_id 是否在检索结果中
5. 计算并输出各项指标

**输出示例**:
```
Recall@1: 0.2500 | Recall@5: 0.6500 | Recall@10: 0.8200 | MRR: 0.4523 | Avg Time: 0.2341s
```

### 2. 端到端 RAG 测试

**方法**: `test_e2e_rag`

**测试流程**:
1. 加载数据集
2. 对每个样本:
   - 执行混合检索 (hybrid_retrieve)
   - 可选: 使用 Reranker 精排序
   - 构建上下文 (build_context)
   - 调用 VLM 生成答案
   - 评估答案准确性
3. 统计整体准确率

**评估方式**:
- 精确匹配 (Exact Match)
- 包含匹配 (Containment)
- Token F1 Score

**输出结果**:
- 每个样本的查询、预测答案、真实答案
- 检索到的文档块数量
- 查询耗时
- 整体准确率

### 3. 混合检索对比测试

**方法**: `test_hybrid_retrieval`, `test_hybrid_retrieval_batch`

**对比维度**:
| 维度 | Dense检索 | Sparse检索 | Hybrid检索 |
|------|-----------|------------|------------|
| 实现方式 | 向量相似度 | BM25 | RRF融合 |
| 优势 | 语义理解 | 关键词匹配 | 综合优势 |
| 适用场景 | 语义相关 | 精确匹配 | 通用场景 |

**测试流程**:
1. 执行 Dense 检索 (向量检索)
2. 执行 Sparse 检索 (BM25)
3. 执行 Hybrid 检索 (RRF融合)
4. 对比三种方法的:
   - 检索耗时
   - Recall@K
   - Gold Chunk 排名
   - MRR

**输出示例**:
```
============================================================
Batch Hybrid Retrieval Comparison Summary:
  Dense:
    Recall@10: 0.7200
    MRR: 0.4521
    Avg Time: 0.1234s
  Sparse:
    Recall@10: 0.6800
    MRR: 0.4123
    Avg Time: 0.0456s
  Hybrid:
    Recall@10: 0.8500
    MRR: 0.5234
    Avg Time: 0.1567s
============================================================
```

## 数据结构

### RetrievalMetrics

```python
@dataclass
class RetrievalMetrics:
    recall_at_1: float      # Recall@1
    recall_at_5: float      # Recall@5
    recall_at_10: float     # Recall@10
    mrr: float              # Mean Reciprocal Rank
    total_queries: int      # 总查询数
    successful_queries: int # 成功查询数
    avg_query_time: float   # 平均查询时间
```

### E2ETestResult

```python
@dataclass
class E2ETestResult:
    query: str                    # 查询文本
    ground_truth: str             # 真实答案
    predicted_answer: str         # 预测答案
    retrieved_chunks: List[DataChunk]  # 检索到的文档块
    accuracy: float               # 准确率
    query_time: float             # 查询耗时
```

## 结果保存

测试结果自动保存到工作目录：

- `retrieval_metrics_{dataset}_{timestamp}.json` - 检索性能指标
- `e2e_results_{dataset}_{timestamp}.json` - 端到端测试结果
- `hybrid_comparison_{timestamp}.json` - 单查询混合检索对比
- `hybrid_batch_summary_{dataset}_{timestamp}.json` - 批量混合检索对比

## 注意事项

1. **服务依赖**: 确保 embedding 和 VLM 服务已启动
2. **数据集准备**: 确保数据集已预处理并保存到 `./datasets/preprocessed_datasets/`
3. **工作目录**: 测试会创建向量数据库和BM25索引，确保有足够的磁盘空间
4. **内存使用**: 大数据集测试可能消耗较多内存，建议分批测试

## 更新记录

- **2026-02-04**: 创建测试模块 `MMRAG/tests/test_rag.py`
  - 实现检索性能测试 (Recall@K, MRR)
  - 实现端到端 RAG 测试
  - 实现混合检索对比测试

# Medical-Agentic-RAG 项目文档

## 1. 项目背景

针对医疗领域的多模态准确问答，构建一个集成文本、图像和结构化数据的智能问答系统，提升医疗信息检索和患者咨询的效率与准确性。

**项目环境**
你的开发环境未配置，不需要进行测试和环境配置。
## 2. 模型 (Models)

| 模型类型 | 模型名称 | 用途 |
|---------|---------|------|
| 多模态大模型 | Qwen3-VL-4B | 主要VLM，用于视觉问答和embedding |
| 医疗领域模型 | MedGemma-4B | 医疗场景VLM，辅助生成查询 |
| Embedding模型 | Qwen3-VL-Embedding (微调) | 多模态检索 |
| Reranker | BGE-Reranker / Medical-Reranker | 检索结果精排序 |

## 3. 数据集 (Datasets)

### 3.1 基础数据集
基于 MedVL-thinker 和 Med-Max 数据集构建：

| 医学领域 | 数据集 | 数据类型 | 实现文件 |
|---------|--------|---------|---------|
| 放射学 | IU-Xray | 胸部X光图像 + 报告 | `data/IU_Xray.py` |
| 眼科学 | Harvard-FairVLMed | 眼底图像 + 临床数据 | `data/Harvard_fairVLMed.py` |
| 病理学 | PMC-OA | 开放获取医学文献 | `data/pmc_oa.py` |
| VQA | PMC-VQA | 医学视觉问答对 | `data/pmc_vqa.py` |
| 评估 | MedVLThinker-Eval | 多模态医疗VQA评估 | `data/medvlthinker.py` |

### 3.2 检索数据格式
- **Query**: 可以是图像(image)、文本(text)或图文组合
- **Document**: 图像(image) + 报告(report)
- **来源**: PMC-OA (PubMed Central Open Access)

## 4. 评估基准 (Benchmarks)

| 任务类型 | 基准测试 |
|---------|---------|
| 文本问答 | MedQA, MMLU-Med |
| 图像分类 | MIMIC-CXR, ChestX-ray14 |
| 视觉问答(VQA) | MedXpertQA, VQA-RAD |

## 5. 技术方案 (Methods)

### 5.1 Naive Method (基线方法)

**目标**: 建立性能基线，评估VLM直接回答的能力。

**方法**:
- 基于 VLM (Qwen3-VL / MedGemma) 直接进行端到端问答
- 不引入外部知识检索

**当前结果**:
- PMC-VQA 100题准确率: 0.45 (45%)

### 5.2 多模态RAG构建 (Stage 1)

**目标**: 通过检索增强提升回答准确性，构建多路召回+精排系统。

#### 5.2.1 多路检索机制

```
Query (Text/Image)
    │
    ├──→ [路径1] BM25文本检索器 → 候选文档
    │
    └──→ [路径2] Qwen3-VL-Embedding多模态检索器 → 候选文档
                      │
                      ↓
              RRF融合两路结果 (initial_top_k=100)
                      │
                      ↓
              Reranker精排序 → Top-K文档 (top_k=5/10)
                      │
                      ↓
              VLM生成答案
```

**检索流程**:
1. **BM25文本检索**: 基于稀疏向量，无需关键词提取
2. **多模态Embedding检索**: 基于Qwen3-VL-Embedding的稠密向量检索
3. **RRF融合**: 使用Reciprocal Rank Fusion融合两路检索结果
4. **Reranker重排**: 使用医疗领域reranker模型精排序

#### 5.2.2 数据准备与处理

**Query类型**:
- 图像问答: 图像 + 问题
- 文本检索: 纯文本问题

**文档格式**:
- Image + Report (来源于PMC-OA)

**领域过滤 (TODO)**:
- 利用MLLM生成领域标签(domain)
- 在子集中进行定向搜索，提升检索效率

#### 5.2.3 Embedding模型微调

**目标**: 统一图像-文本向量空间，通过对比学习提升召回率。

**正样本构建**:
- **图像问答数据**: 通过 Qwen3-VL-8B 或 MedGemma-4B 将病例图像和报告改写为(image+问题)格式，检索对应的(image+report)作为正样本
- **文本检索数据**: 改写文本问题，检索对应(image+report)作为正样本
- **策略**: 对于VQA数据集生成report；对于带report的数据集生成query

**负样本采样 (困难负样本)**:
1. 先检索Top-10候选样本
2. 选取满足条件 $s < s^+ + \delta$ 的样本作为负样本
3. 目的: 避免假阴性(FN)，提升模型判别能力

**评估指标**:
- **Recall@K**: Top-K结果中包含Gold Document的比例
  - Recall@5, Recall@10为主要观测指标
- **MRR (Mean Reciprocal Rank)**: 正确文档倒数排名的平均值
  - 反映模型整体排序质量

### 5.3 Agentic RAG (Stage 2)

**目标**: 引入Agent能力，通过工具调用和强化学习优化问答流程。

**技术栈**:
- 框架: VERL ( Efficient RL for LLM/VLM)
- 算法: GRPO (Group Relative Policy Optimization)

**奖励函数**:
- 基础版本: 简单的文本匹配 (exact match / F1 score)
- 进阶版本: 医学实体匹配 + 语义相似度

**工具设计**:
1. **文本检索工具**: 基于文本query检索相关文档
2. **图像检索工具**: 基于图像或图文query检索相关文档

**Agent决策流程**:
```
用户Query → Agent决策 → [选择工具] → 检索结果 → VLM生成 → 答案
                ↓
         [直接回答] (当不需要检索时)
```

## 6. 项目规范

- 该文档使用中文编写
- 每个模块实现时需在项目根目录维护 `docs/xx.md` 文档
- 功能更新后需同步更新对应文档，并在CLAUDE.md记录变更
- 使用subagents并行编写代码和测试案例
- 开发环境: Windows系统

## 7. 任务清单 (TODO)

### Stage 0: 基础设施
- [x] 项目基础结构搭建
- [x] 数据集加载模块 (IU-Xray, Harvard-FairVLMed, PMC-OA, PMC-VQA, MedVLThinker-Eval)
- [ ] Milvus向量数据库部署与配置
- [x] 测试文档 (docs/testing.md) - MMRAG测试模块
- [ ] 文档完善 (docs/api_reference.md, docs/architecture.md, docs/datasets.md)

### Stage 1: 多模态RAG (当前重点)
- [x] BM25文本检索器实现 (MMRAG/DB/bm25_storage.py)
- [x] Qwen3-VL-Embedding服务封装
- [x] RRF融合算法实现 (MMRAG/retrieval_fusion.py)
- [x] MMRAG主类混合检索集成 (MMRAG/MMRAG.py hybrid_retrieve)
- [ ] Reranker服务集成
- [ ] 领域标签生成模块 (MLLM-based domain分类)
- [x] Embedding微调数据生成管道
  - [x] VQA数据集 → Report生成
  - [x] Report数据集 → Query生成
- [ ] 困难负样本采样实现
- [ ] 微调训练脚本 (DDP/FSDP)
- [x] 检索性能评估 (Recall@K, MRR) - `MMRAG/tests/test_rag.py`
- [x] 端到端RAG评测 - `MMRAG/tests/test_rag.py`

### Stage 2: Agentic RAG
- [ ] VERL框架集成
- [ ] GRPO训练配置
- [ ] 文本检索工具定义
- [ ] 图像检索工具定义
- [ ] Agent决策逻辑实现
- [ ] 奖励函数实现 (文本匹配 → 语义匹配)
- [ ] RL训练管道
- [ ] Agentic RAG评测

### Stage 3: 评估与优化
- [ ] MedQA基准测试
- [ ] MMLU-Med基准测试
- [ ] MIMIC-CXR分类评估
- [ ] MedXpertQA/VQA-RAD评估
- [ ] 与Naive方法对比分析
- [ ] 错误案例分析

### Stage 4: 部署与文档
- [ ] 模型量化与优化
- [ ] API服务封装
- [ ] 前端交互界面 (可选)
- [ ] 完整技术报告
- [ ] 代码仓库整理与开源准备

---

**最后更新**: 2026-02-04

---

## 8. 更新记录

| 日期 | 更新内容 | 相关文件 |
|------|---------|---------|
| 2026-02-04 | 完成 MedVLThinker-Eval 数据集类实现 | `data/medvlthinker.py` |
| 2026-02-04 | 创建 MMRAG 测试模块 | `MMRAG/tests/test_rag.py` |
| 2026-02-04 | 更新数据集文档 | `docs/datasets.md` |

# 架构设计文档

## 系统架构概览

Medical-Agentic-RAG 采用分层架构设计，包含以下核心层次：

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │    Web UI    │ │   API Server │ │   Evaluation Tools   │ │
│  └──────────────┘ └──────────────┘ └──────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Agentic RAG Layer                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    RL Agent                         │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ │   │
│  │  │   Planner    │ │ Tool Executor│ │   Memory    │ │   │
│  │  └──────────────┘ └──────────────┘ └─────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    Core RAG Layer                           │
│  ┌──────────────────┐  ┌──────────────────────────────┐    │
│  │   MMRAG Engine   │  │      Retrieval Pipeline      │    │
│  │  - Data Ingest   │  │  - Query Understanding       │    │
│  │  - Vector Store  │  │  - Multi-modal Retrieval     │    │
│  │  - Reranking     │  │  - Result Fusion             │    │
│  └──────────────────┘  └──────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    Model Service Layer                      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐    │
│  │  Embedding   │ │    VLM       │ │   Reranker       │    │
│  │   Service    │ │  Service     │ │   Service        │    │
│  └──────────────┘ └──────────────┘ └──────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    Data Layer                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐    │
│  │   Milvus     │ │    Faiss     │ │   Dataset Cache  │    │
│  │   (Vector)   │ │   (Vector)   │ │   (Images/Text)  │    │
│  └──────────────┘ └──────────────┘ └──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件详解

### 1. MMRAG Engine (MMRAG/MMRAG.py)

多模态RAG的核心引擎，负责文档的入库、检索和生成。

**核心功能**:

| 方法 | 功能 | 异步支持 |
|------|------|---------|
| `ingest_dataset()` | 数据集入库 | ✅ |
| `retrieve()` | 向量检索 | ✅ |
| `aquery()` | 完整RAG查询 | ✅ (待实现) |
| `delete_document()` | 文档删除 | ✅ |

**数据流**:
```
Raw Dataset → DataChunk → Embedding → Vector Store
```

### 2. Vector Storage

#### MilvusVectorStorage (MMRAG/DB/milvus_vectorDB.py)

基于 Milvus Lite 的本地向量存储实现。

**特点**:
- 本地文件存储 (`.db`)
- 支持动态字段
- 自动持久化
- 支持 IP (内积) 和 L2 距离度量

**Schema设计**:
```python
{
    "chunk_id": VARCHAR (primary key),
    "vector": FLOAT_VECTOR (dim=2048),
    "doc_id": VARCHAR,
    "content": TEXT,
    "image_paths": JSON,
    "file_path": VARCHAR,
    "metadata": JSON (dynamic)
}
```

### 3. Embedding Service

#### VllmEmbeddingService

基于 vLLM 的本地 Embedding 服务部署。

**配置参数**:
- `tensor_parallel_size`: 2 (张量并行)
- `max_model_len`: 4096
- `gpu_memory_utilization`: 0.8

#### OpenaiEmbeddingService

兼容 OpenAI API 的 Embedding 服务。

**接口**:
```python
POST /v1/embeddings
{
    "messages": [...],
    "model": "Qwen3-VL-Embedding-2B",
    "encoding_format": "float"
}
```

### 4. Agentic RAG 架构

#### RL Agent

使用 GRPO (Group Relative Policy Optimization) 算法训练的智能代理。

**组件**:

| 组件 | 职责 |
|------|------|
| Planner | 规划检索策略，决定调用哪些工具 |
| Tool Executor | 执行检索工具，获取证据 |
| Memory | 维护对话历史和中间结果 |

#### 工具系统

**检索工具**:
1. `TextRetrievalTool` - 文本检索
2. `ImageRetrievalTool` - 图像检索

**工具调用格式**:
```python
{
    "tool": "text_retrieval",
    "arguments": {
        "query": "胸部X光片发现",
        "top_k": 5
    }
}
```

### 5. 数据流设计

#### 文档入库流程

```mermaid
sequenceDiagram
    participant User
    participant MMRAG
    participant Embedding
    participant VectorDB

    User->>MMRAG: ingest_dataset(dataset_name)
    MMRAG->>MMRAG: _process_dataset()
    loop Batch Processing
        MMRAG->>Embedding: async_embed_batch(batch)
        Embedding->>Embedding: encode(text+image)
        Embedding-->>MMRAG: vectors
        MMRAG->>VectorDB: upsert(chunks)
    end
    VectorDB-->>MMRAG: success
    MMRAG-->>User: done
```

#### 查询流程

```mermaid
sequenceDiagram
    participant User
    participant MMRAG
    participant Embedding
    participant VectorDB
    participant VLM

    User->>MMRAG: retrieve(query)
    MMRAG->>Embedding: async_embed_batch(query)
    Embedding-->>MMRAG: query_vector
    MMRAG->>VectorDB: search(query_vector, top_k)
    VectorDB-->>MMRAG: results
    MMRAG-->>User: ranked_chunks
```

## 模块依赖关系

```
MMRAG/
├── base.py              # 无依赖 (基础数据类)
├── utils.py             # 标准库
├── MMRAG.py             # 依赖: base, utils, DB, model_service
├── DB/
│   ├── base.py          # 依赖: base
│   ├── milvus_vectorDB.py # 依赖: base, utils
│   └── faiss_vectorDB.py  # 依赖: base, utils
├── model_service/
│   ├── embedding_service.py
│   ├── vlm_service.py
│   └── rerank_service.py
├── agent/               # (待实现)
│   ├── rl_agent.py
│   ├── tools.py
│   └── reward.py
└── task/
    ├── common.py
    └── *.py             # 数据集处理
```

## 扩展性设计

### 1. 新增向量数据库

继承 `BaseVectorStorage` 接口:
```python
class NewVectorStorage(BaseVectorStorage):
    async def upsert(self, chunks: list[DataChunk]) -> None: ...
    async def search(self, query_vector: list[float], top_k: int) -> list[DataChunk]: ...
    async def delete_by_doc_id(self, doc_id: str) -> None: ...
```

### 2. 新增Embedding服务

继承 `EmbeddingService` 接口:
```python
class NewEmbeddingService(EmbeddingService):
    async def async_embed_batch(self, inputs, batch_size) -> List[List[float]]: ...
```

### 3. 新增数据集处理

在 `MMRAG/task/` 下创建新的数据集类:
```python
class NewDataset(Dataset):
    def process_dataset(self, ...): ...
    def evaluate(self, index, conversation): ...
```

# API参考文档

## MMRAG 核心类

### MMRAG

主RAG类，提供多模态检索和生成功能。

```python
class MMRAG:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: VLMService,
        workspace: str = "./workspace",
    )
```

**参数**:
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `embedding_service` | EmbeddingService | 必填 | Embedding服务实例 |
| `llm_service` | VLMService | 必填 | VLM服务实例 |
| `workspace` | str | "./workspace" | 工作目录 |

#### ingest_dataset

```python
async def ingest_dataset(
    self,
    dataset_name: str,
    dataset_size: int = None,
    filter: str = None
) -> None
```

将数据集导入向量数据库。

**参数**:
- `dataset_name`: HuggingFace数据集名称或本地路径
- `dataset_size`: 导入的样本数量 (None表示全部)
- `filter`: 过滤条件

**示例**:
```python
await rag.ingest_dataset("UCSC-VLAA/MedVLThinker-Eval", dataset_size=100)
```

#### retrieve

```python
async def retrieve(
    self,
    query: list[dict[str, str]],
    top_k: int = 5
) -> list[DataChunk]
```

执行向量检索。

**参数**:
- `query`: 查询列表，每个元素为 `{"text": "...", "image": "..."}`
- `top_k`: 返回结果数量

**返回**: `list[DataChunk]`

**示例**:
```python
query = [{"text": "胸部X光发现"}]
results = await rag.retrieve(query, top_k=5)
for chunk in results:
    print(chunk.content)
    print(chunk.image_paths)
```

#### delete_document

```python
async def delete_document(self, doc_id: str) -> None
```

删除指定文档的所有分块。

---

## Embedding服务

### EmbeddingService (基类)

```python
class EmbeddingService(ABC):
    embedding_dim: int

    @abstractmethod
    async def async_embed_batch(
        self,
        inputs: list[dict[str, Any]],
        batch_size: int = 128
    ) -> list[list[float]]
```

### VllmEmbeddingService

基于vLLM的本地Embedding服务。

```python
class VllmEmbeddingService(EmbeddingService):
    def __init__(
        self,
        model_path: str = None,
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.8,
    )
```

**配置参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_path` | `~/.cache/modelscope/...` | 模型路径 |
| `tensor_parallel_size` | 2 | 张量并行数 |
| `gpu_memory_utilization` | 0.8 | GPU内存利用率 |
| `max_model_len` | 4096 | 最大序列长度 |
| `max_num_seqs` | 128 | 最大并发序列数 |

### OpenaiEmbeddingService

兼容OpenAI API的Embedding服务。

```python
class OpenaiEmbeddingService(EmbeddingService):
    def __init__(
        self,
        model_name: str = "Qwen3-VL-Embedding-2B",
        api_key: str = None,
        base_url: str = "http://localhost:8001/v1"
    )
```

---

## 向量存储

### BaseVectorStorage (基类)

```python
class BaseVectorStorage(ABC):
    @abstractmethod
    async def upsert(self, chunks: list[DataChunk]) -> None

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5
    ) -> list[DataChunk]

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str) -> None
```

### MilvusVectorStorage

```python
class MilvusVectorStorage(BaseVectorStorage):
    def __init__(
        self,
        workspace: str,
        embedding_dim: int,
        auto_save: bool = True,
        save_interval: int = 100
    )
```

**方法**:

#### upsert
```python
async def upsert(self, chunks: list[DataChunk]) -> None
```
插入或更新文本块。

#### search
```python
async def search(
    self,
    query_vector: list[float],
    top_k: int = 5
) -> list[DataChunk]
```
向量检索。

#### backup
```python
async def backup(self, backup_name: str = None) -> str
```
创建数据库备份。

#### restore
```python
async def restore(self, backup_name: str) -> None
```
从备份恢复。

---

## 数据类

### DataChunk

```python
@dataclass
class DataChunk:
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    content: Optional[str] = None
    image_paths: list[str] = field(default_factory=list)
    vector: list[float] = field(default_factory=list)
    dataset_id: Optional[str] = None
    file_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)
```

**方法**:
- `to_dict() -> dict`: 转换为字典

### QueryResult

```python
@dataclass
class QueryResult:
    answer: str
    query: str
    references: list[DataChunk] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

---

## 工具函数

### utils.py

#### compute_mdhash_id

```python
def compute_mdhash_id(content: str, prefix: str = "chunk_") -> str
```
计算内容的MD5哈希ID。

**参数**:
- `content`: 输入内容
- `prefix`: ID前缀

**返回**: 哈希字符串

#### encode_image_paths_to_base64

```python
def encode_image_paths_to_base64(
    image_paths: list[str],
    max_size: int = 1024
) -> list[dict]
```
将图像路径列表编码为base64格式。

**返回**: OpenAI格式的消息列表

#### extract_catogorical_answer

```python
def extract_catogorical_answer(text: str) -> Optional[str]
```
从文本中提取分类答案(A/B/C/D/E)。

---

## 数据集类

### BaseDataset

所有数据集类的基类。

```python
class BaseDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        vlm_service: VLMService,
        dataset_size: int = None
    )

    def process_dataset(
        self,
        dataset_size: int = None,
        rewrite: bool = True
    ) -> None

    def evaluate(
        self,
        index: int,
        conversation: dict
    ) -> int
```

### 具体数据集

| 类名 | 文件 | 说明 |
|------|------|------|
| MedMaxDataset | `task/medmax.py` | MedMax数据集 |
| IUXrayDataset | `task/IU_Xray.py` | IU-Xray数据集 |
| PMCVQADataset | `task/pmc_vqa.py` | PMC-VQA数据集 |

---

## 配置常量

### common.py

```python
# 内容改写提示词
REWRITE_CONTENT_PROMPT = """
请根据以下医疗问题，改写为一个可以用于检索相似病例的查询语句...
"""

# 默认配置
DEFAULT_EMBEDDING_DIM = 2048
DEFAULT_TOP_K = 5
DEFAULT_BATCH_SIZE = 128
```

---

## 错误处理

### 异常类

```python
class MMRAGError(Exception):
    """MMRAG基础异常"""
    pass

class VectorStorageError(MMRAGError):
    """向量存储异常"""
    pass

class EmbeddingServiceError(MMRAGError):
    """Embedding服务异常"""
    pass

class DatasetError(MMRAGError):
    """数据集处理异常"""
    pass
```

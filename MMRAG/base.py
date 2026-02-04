
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Dict

@dataclass
class DataChunk:
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    content: Optional[str] = None
    image_paths: List[str] = field(default_factory=list)
    vector: List[float] = field(default_factory=list)
    dataset_id: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)  # ✅ 推荐：永不为 None

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "image_paths": self.image_paths,
            "vector": self.vector,
            "file_path": self.file_path,
            "metadata": self.metadata or {}
        }

def dict_to_datachunk(data: dict) -> DataChunk:
    return DataChunk(
        doc_id=data.get("doc_id"),
        chunk_id=data.get("chunk_id"),
        content=data.get("content"),
        image_paths=data.get("image_paths", []),
        vector=data.get("vector"),
        file_path=data.get("file_path"),
        metadata={k: v for k, v in data.items() if k not in {"doc_id", "chunk_id", "content", "image_paths", "vector", "file_path"}}
    )

@dataclass
class QueryResult:
    answer: str                     # LLM 生成的回答
    query: str                      # 原始查询
    references: List[DataChunk] = field(default_factory=list)     # 引用的文本块
    metadata: Dict = field(default_factory=dict)                 # 附加信息（耗时、模型等）

@dataclass
class RetrievalResult:
    """检索结果统一格式"""
    chunk_id: str
    content: str
    score: float                      # 融合后的分数
    source: str                       # 来源：dense/sparse/hybrid
    metadata: dict = field(default_factory=dict)  # 原始分数、排名等

@dataclass
class BaseVectorStorage(ABC):
    """向量存储接口定义"""

    @abstractmethod
    async def upsert(self, chunks: List[DataChunk]) -> None:
        """插入或更新文本块的向量表示"""
        pass

    @abstractmethod
    async def search(self, query_vector: List[float], top_k: int = 5) -> List[DataChunk]:
        """基于向量的检索"""
        pass

    @abstractmethod
    async def delete_by_doc_id(self, doc_id: str) -> None:
        """删除指定文档的所有文本块"""
        pass


from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any

@dataclass
class TextChunk:
    id: str                    # 唯一标识
    content: str               # 文本内容
    vector: list[float]        # 向量表示
    doc_id: str                # 所属文档ID
    chunk_index: int           # 在文档中的顺序
    file_path: str             # 原始文件路径
    metadata: dict = None      # 额外元数据

@dataclass
class QueryResult:
    answer: str                     # LLM 生成的回答
    references: list[TextChunk]     # 引用的文本块
    query: str                      # 原始查询
    metadata: dict                  # 附加信息（耗时、模型等）
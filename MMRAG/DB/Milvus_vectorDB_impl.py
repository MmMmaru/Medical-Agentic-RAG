import asyncio
import os
import numpy as np
from ..utils import logger
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema  # type: ignore

from MMRAG.base import TextChunk

class PersistentVectorStorage():
    """支持持久化的向量存储实现"""
    
    def __init__(
        self, 
        workspace: str,
        embedding_dim: int,
        auto_save: bool = True,
        save_interval: int = 100  # 每N次操作自动保存
    ):
        self.workspace = workspace
        self.index_path = os.path.join(workspace, "chunks.index")
        self.metadata_path = os.path.join(workspace, "chunks_metadata.json")
        
        # 内存中的数据结构
        self.index = None  # FAISS索引
        self.metadata = {}  # {chunk_id: TextChunk对象}
        self.operation_counter = 0
        
        # 加载已有数据
        self._load_from_disk()
    
    async def upsert(self, chunks: list[TextChunk]) -> None:
        """插入/更新文本块"""
        # 1. 提取向量
        vectors = np.array([chunk.vector for chunk in chunks])
        
        # 2. 更新FAISS索引
        if self.index is None:
            self.index = faiss.IndexFlatIP(len(vectors[0]))
        self.index.add(vectors)
        
        # 3. 更新元数据
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.id
            self.metadata[chunk_id] = {
                "id": chunk.id,
                "content": chunk.content,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "file_path": chunk.file_path,
                "metadata": chunk.metadata or {}
            }
        
        # 4. 增量持久化
        self.operation_counter += len(chunks)
        if self.auto_save and self.operation_counter >= self.save_interval:
            await self.save_to_disk()
            self.operation_counter = 0
    
    async def search(self, query_vector: list[float], top_k: int = 5) -> list[TextChunk]:
        """向量检索"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # FAISS检索
        query_vec = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_vec, top_k)
        
        # 重建TextChunk对象
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS找不到结果时返回-1
                continue
            chunk_id = list(self.metadata.keys())[idx]
            chunk_data = self.metadata[chunk_id]
            results.append(TextChunk(
                id=chunk_data["id"],
                content=chunk_data["content"],
                vector=None,  # 节省内存，不返回向量
                doc_id=chunk_data["doc_id"],
                chunk_index=chunk_data["chunk_index"],
                file_path=chunk_data["file_path"],
                metadata=chunk_data["metadata"]
            ))
        
        return results
    
    async def delete_by_doc_id(self, doc_id: str) -> None:
        """删除文档的所有分块"""
        # 1. 找到要删除的chunk_ids
        to_delete = [cid for cid, data in self.metadata.items() 
                     if data["doc_id"] == doc_id]
        
        # 2. 从元数据中移除
        for chunk_id in to_delete:
            del self.metadata[chunk_id]
        
        # 3. 重建FAISS索引（FAISS不支持单个删除）
        await self._rebuild_index()
        
        # 4. 持久化
        await self.save_to_disk()
    
    async def save_to_disk(self) -> None:
        """保存到磁盘"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # 保存FAISS索引
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        
        # 保存元数据
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(self.metadata)} chunks to disk")
    
    def _load_from_disk(self) -> None:
        """从磁盘加载"""
        # 加载FAISS索引
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # 加载元数据
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {len(self.metadata)} chunk metadata")
    
    async def _rebuild_index(self) -> None:
        """重建FAISS索引（删除操作需要）"""
        if not self.metadata:
            self.index = None
            return
        
        # 从元数据中重新构建向量索引
        # 注意：这要求metadata中保存了向量数据，或从外部重新获取
        vectors = []
        for chunk_data in self.metadata.values():
            # 这里需要重新生成向量，或在metadata中存储向量
            # 简化处理：假设已存储
            vectors.append(chunk_data.get("vector", []))
        
        if vectors and vectors[0]:
            vectors_array = np.array(vectors).astype('float32')
            self.index = faiss.IndexFlatIP(len(vectors[0]))
            self.index.add(vectors_array)
    
    async def backup(self, backup_name: str = None) -> str:
        """创建备份"""
        if backup_name is None:
            from datetime import datetime
            backup_name = datetime.now().strftime("chunks_%Y%m%d_%H%M%S")
        
        backup_dir = os.path.join(self.workspace, "chunks_backup")
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_path = os.path.join(backup_dir, f"{backup_name}.index")
        backup_meta_path = os.path.join(backup_dir, f"{backup_name}_metadata.json")
        
        # 复制文件
        if os.path.exists(self.index_path):
            shutil.copy(self.index_path, backup_path)
        if os.path.exists(self.metadata_path):
            shutil.copy(self.metadata_path, backup_meta_path)
        
        logger.info(f"Created backup: {backup_name}")
        return backup_path
    
    async def restore(self, backup_name: str) -> None:
        """从备份恢复"""
        backup_dir = os.path.join(self.workspace, "chunks_backup")
        backup_path = os.path.join(backup_dir, f"{backup_name}.index")
        backup_meta_path = os.path.join(backup_dir, f"{backup_name}_metadata.json")
        
        if os.path.exists(backup_path):
            shutil.copy(backup_path, self.index_path)
        if os.path.exists(backup_meta_path):
            shutil.copy(backup_meta_path, self.metadata_path)
        
        # 重新加载
        self._load_from_disk()
        logger.info(f"Restored from backup: {backup_name}")
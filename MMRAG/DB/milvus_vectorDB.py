import asyncio
import os
import shutil
import json
from typing import List, Dict, Any, Optional
from utils import logger
from base import DataChunk, dict_to_datachunk, BaseVectorStorage

try:
    from pymilvus import MilvusClient, DataType
except ImportError:
    logger.error("pymilvus not installed. Please install it via `pip install pymilvus`")
    raise

class MilvusVectorStorage(BaseVectorStorage):
    """基于 Milvus Lite 的向量存储实现"""
    
    def __init__(
        self, 
        workspace: str,
        embedding_dim: int,
        auto_save: bool = True,  # 保留参数以兼容接口，Milvus自动保存
        save_interval: int = 100
    ):
        self.workspace = workspace
        
        if not os.path.exists(workspace):
            os.makedirs(workspace)
        self.db_path = os.path.join(workspace, "milvus_rag.db")
        self.collection_name = "mmrag_chunks"
        self.embedding_dim = embedding_dim
        
        # 初始化 Milvus Client (Lite版本会在本地创建文件)
        self.client = MilvusClient(uri=self.db_path)
        self.similarity_metric = "IP"  # 内积
        
        # 检查并创建集合
        self._init_collection()
            
    def _init_collection(self):
        """初始化集合"""
        if self.collection_name in self.client.list_collections():
            logger.info(f"Connected to Milvus collection: {self.collection_name}")
            return

        # 使用 Schema 显式创建，避免简化 API 的类型推断问题
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        
        # 添加主键 (VARCHAR)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=256, is_primary=True)
        # 添加向量字段
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        
        # 准备索引参数
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type=self.similarity_metric, 
            index_type="FLAT"  # 使用 FLAT 确保 Lite 版兼容性
        )
        
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            logger.info(f"Created Milvus collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    async def upsert(self, chunks: list[DataChunk]) -> None:
        """插入/更新文本块"""
        if not chunks:
            return

        data = []
        for chunk in chunks:
            # 兼容处理 id
            chunk_id = getattr(chunk, 'chunk_id', getattr(chunk, 'id', None))
            if not chunk_id:
                logger.warning("Skipping chunk without id")
                continue

            data.append(chunk.to_dict())
        
        # Milvus upsert
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                data=data
            )
            logger.info(f"Upserted {len(data)} chunks to Milvus")
        except Exception as e:
            logger.error(f"Failed to upsert to Milvus: {e}")
            raise

    async def search(self, query_vector: list[float], top_k: int = 5) -> list[DataChunk]:
        """向量检索"""
        try:
            # 执行搜索
            res = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=top_k,
                output_fields=["chunk_id", "content", "doc_id", "file_path", "image_paths", "*"], # * 获取所有动态字段
                search_params={"metric_type": "IP", "params": {}} 
            )
            
            results = []
            for hits in res:
                for hit in hits:
                    entity = hit['entity']
                    entity['score'] = hit['distance']
                    results.append(dict_to_datachunk(entity))
            
            return results
            
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return []
        
    def num_vectors(self) -> int:
        return self.client.num_entities
    
    async def delete_by_doc_id(self, doc_id: str) -> None:
        """删除文档的所有分块"""
        try:
            # Milvus 支持通过表达式删除
            self.client.delete(
                collection_name=self.collection_name,
                filter=f'doc_id == "{doc_id}"'
            )
            logger.info(f"Deleted chunks for doc_id: {doc_id}")
        except Exception as e:
            logger.error(f"Failed to delete by doc_id: {e}")
            raise
    
    async def save_to_disk(self) -> None:
        """Milvus Lite 自动持久化，此处仅做接口兼容或手动flush"""
        # Milvus Lite 基于文件，通常操作即刻生效，无需显式保存索引文件
        pass
     
    async def backup(self, backup_name: str = None) -> str:
        """创建备份 (复制 .db 文件)"""
        if backup_name is None:
            from datetime import datetime
            backup_name = datetime.now().strftime("milvus_backup_%Y%m%d_%H%M%S")
        
        backup_dir = os.path.join(self.workspace, "chunks_backup")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Milvus Lite 生成的文件可能除了 .db 还有 lock 文件，主要备份 .db
        # 注意：在生产环境中，应该确保持久化完成或暂停写入再备份
        backup_path = os.path.join(backup_dir, f"{backup_name}.db")
        
        if os.path.exists(self.db_path):
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        return ""
    
    async def restore(self, backup_name: str) -> None:
        """从备份恢复"""
        backup_dir = os.path.join(self.workspace, "chunks_backup")
        backup_path = os.path.join(backup_dir, f"{backup_name}.db")
        
        if os.path.exists(backup_path):
            # Close existing client connection if possible or just overwrite
            # Milvus Lite might lock the file. 
            self.client.close()
            
            shutil.copy2(backup_path, self.db_path)
            
            # Re-initialize client
            self.client = MilvusClient(uri=self.db_path)
            logger.info(f"Restored from backup: {backup_name}")
        else:
            logger.error(f"Backup not found: {backup_path}")


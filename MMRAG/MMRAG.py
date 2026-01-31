from typing import Any, Dict, List
import os
import json
from datasets import load_dataset
from torch.utils.data import DataLoader
import asyncio

import sys
# 确保项目根目录在 path 中
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(project_root, os.getcwd())
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

from data.medmax import MedMaxDataset
from model_service.embedding_service import EmbeddingService, OpenaiEmbeddingService
from model_service.vlm_service import VLMService
from DB.milvus_vectorDB import MilvusVectorStorage
from utils import logger, compute_mdhash_id
from base import DataChunk, dict_to_datachunk

class MMRAG:
    """
    Docstring for MMRAG
    1、支持文件持久化
    2、支持多模态数据存储与检索（多路检索，图片文本分开然后进行关联合并）
    3、支持向量化存储与检索
    """
    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: VLMService,
        workspace: str = "./workspace",
    ):
        self.workspace = workspace
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.insert_batch_size = 128
        
        # 初始化持久化组件
        self.vector_storage = MilvusVectorStorage(
            workspace=workspace,
            embedding_dim=embedding_service.embedding_dim,
            auto_save=True,
            save_interval=50
        )
    
    async def aquery(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        多模态RAG异步查询接口
        Args:
            query: 用户查询文本
            image_paths: 可选的图像路径列表
            top_k: 检索的top k数量
        Returns:
            模型生成的回答文本
        """
        pass
    async def ainsert(data: dict[str, dict[str, Any]]) -> None:
        """
        多模态RAG异步插入接口
        Args:
            data: 包含文本和图像数据的字典
        """
        pass

    async def ainit_rag(self):
        dir_list = [
            "datasets/preprocessed_dataset/medmax",
        ]
        for dir_name in dir_list:
            await self.ingest_dataset(dir_name, dataset_size=100)
    
    async def ingest_dataset(self, dataset_name: str, dataset_size: int = None, filter: str = None) -> None:
        """文档入库（带持久化）"""
        # 1. 生成文档ID
        dataset = load_dataset(dataset_name)
        if dataset_size:
            dataset = dataset['train'].select(range(dataset_size))
        # 2. 处理文档
        # dataset = self._process_dataset(dataset, dataset_size)
        doc_id = compute_mdhash_id(dataset_name, prefix="doc_")
        chunk_ids = []

        async def process_chunk(data: List[Dict]):
            # 向量化图片和文本

            vectors = await self.embedding_service.async_embed_batch(data, batch_size=self.insert_batch_size)
            
            # 4. 处理为chunks
            chunks = []
            for ex, vector in zip(data, vectors):
                chunk = dict_to_datachunk(ex)
                chunk.vector = vector
                chunk.doc_id = doc_id
                chunk.file_path = f"{dataset_name}_{ex['index']}"
                chunk_ids.append(chunk.chunk_id)
                chunks.append(chunk)
            
            # 5. 存储到向量库
            await self.vector_storage.upsert(chunks)

        for i in range(0, len(dataset), self.insert_batch_size):
            # 下面这一行直接把 slice 转成了 List[Dict] 对象
            batch = [dataset[j] for j in range(i, min(i + self.insert_batch_size, len(dataset)))]
            await process_chunk(batch)
        
        # 6. 注册文档
        # self.doc_registry.register_document(doc_id, dataset_name, chunk_ids)
        
        logger.info(f"Ingested document {doc_id} with {len(chunk_ids)} chunks")
    
    
    async def retrieve(self, query: List[Dict[str, str]], top_k: int = 5) -> list[DataChunk]:
        """执行检索流程"""
        # 1. 查询向量化
        # 2. 向量搜索
        # 3. 结果排序
        query_index = await self.embedding_service.async_embed_batch(query)
        results = await self.vector_storage.search(query_index[0], top_k=top_k)
        return results
        
    async def rerank(self, query: str, chunks: list[DataChunk]) -> list[DataChunk]:
        """可选：使用重排模型优化结果（Cohere/BGE-reranker）"""
        pass


    async def delete_document(self, doc_id: str) -> None:
        """删除文档（级联删除分块）"""
        # 1. 从向量库删除
        await self.vector_storage.delete_by_doc_id(doc_id)
        
        # 2. 从注册表删除
        # self.doc_registry.delete_document(doc_id)
        
        logger.info(f"Deleted document {doc_id}")
    
    async def shutdown(self) -> None:
        """优雅关闭（确保数据持久化）"""
        await self.vector_storage.save_to_disk()
        logger.info("RAG system shutdown complete")

    def delete_rag(self):
        # TODO: delete workspace folder
        os.delete(self.workspace)
        logger.info("Deleted RAG workspace")

async def main():
    
    embedding_service = OpenaiEmbeddingService()
    llm_service = VLMService()
    RAG = MMRAG(
        embedding_service=embedding_service,
        llm_service=llm_service,
        workspace="./workspace",
    )
    # await RAG.ainit_rag()
    print(RAG.vector_storage.client.get_collection_stats(RAG.vector_storage.collection_name)['row_count'])
    query = [{"text": "ICA-derived 3D model"}]
    results = await RAG.retrieve(query, top_k=5)
    for r in results:
        print(f'score: {r.metadata.get("score")}: {r.content}...')
    
    # test recall@5 and MRR on 200
    dataset_size = 200
    dataset = MedMaxDataset("datasets/preprocessed_datasets/medmax", dataset_size)
    recall_5_score = 0
    MRR_score = 0
    for i in range(len(dataset)):
        item = dataset[i]
        query = item['question']
        results = await RAG.retrieve(query, top_k=5)
        recall_score = 0
        local_mrr_score = 0
        for local_index, r in enumerate(results):
            if r.chunk_id == item['chunk_id']:
                recall_score = 1
                local_mrr_score = 1/(local_index+1)
        recall_5_score += recall_score
        MRR_score += local_mrr_score
    print(f"recall@5: {recall_5_score / dataset_size}, MRR: {MRR_score / dataset_size}")

if __name__ == "__main__":
    asyncio.run(main())
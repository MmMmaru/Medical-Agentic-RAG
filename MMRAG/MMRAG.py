from typing import Any, dict, list, Optional
import os
import json
from datasets import load_dataset
from torch.utils.data import Dataset, Subset
import asyncio
import shutil
import sys
if __name__ == "__main__":
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data.pmc_oa import PMCOADataset
from MMRAG.model_service.embedding_service import EmbeddingService, OpenaiEmbeddingService
from MMRAG.model_service.rerank_service import OpenAIRerankerService
from MMRAG.model_service.vlm_service import VLMService
from MMRAG.DB.milvus_vectorDB import MilvusVectorStorage
from MMRAG.DB.bm25_storage import BM25Storage
from MMRAG.retrieval_fusion import rrf_fuse
from MMRAG.utils import logger, compute_mdhash_id
from MMRAG.base import DataChunk, dict_to_datachunk

class MMRAG:
    """
    Docstring for MMRAG
    1、支持文件持久化
    2、支持多模态数据存储与检索（多路检索，图片文本分开然后进行关联合并）
    3、支持向量化存储与检索
    """
    def __init__(
        self,
        embedding_service: OpenaiEmbeddingService,
        rerank_service: OpenAIRerankerService,
        llm_service: VLMService,
        workspace: str = "./workspace",
    ):
        self.workspace = workspace
        self.embedding_service = embedding_service
        self.rerank_service = rerank_service
        self.llm_service = llm_service
        self.insert_batch_size = 256
        self.init_RAG()
    
    def init_RAG(self):
        
        # 初始化持久化组件
        self.vector_storage = MilvusVectorStorage(
            workspace=self.workspace,
            embedding_dim=self.embedding_service.embedding_dim,
            auto_save=True,
            save_interval=50
        )

        # 初始化BM25存储
        self.bm25_storage = BM25Storage(
            workspace=self.workspace,
            k1=1.5,
            b=0.75
        )

    def clear_RAG(self):
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)

    async def aquery(query: str, top_k: int = 5) -> list[dict[str, Any]]:
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

    async def ainit_rag(self, dataset_list: list[Dataset]):
        for dataset in dataset_list:
            await self.ingest_dataset(dataset)
    
    async def ingest_dataset(self, dataset: Dataset, dataset_size: int = None, filter: str = None) -> None:
        """文档入库（带持久化）"""
        # 1. 生成文档ID
        dataset_name = dataset.dataset_name
        if dataset_size:
            # dataset = dataset[:dataset_size]
            dataset = Subset(dataset, range(dataset_size))
        # 2. 处理文档
        # dataset = self._process_dataset(dataset, dataset_size)
        doc_id = compute_mdhash_id(dataset_name, prefix="doc_")
        chunk_ids = []

        async def process_chunk(data: list[dict]):
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
            
            # 5. 存储到向量库和BM25索引
            await self.vector_storage.upsert(chunks)
            await self.bm25_storage.upsert(chunks)

        for i in range(0, len(dataset), self.insert_batch_size):
            # 下面这一行直接把 slice 转成了 list[dict] 对象
            # batch = dataset[i:i+self.insert_batch_size]
            batch = [dataset[j] for j in range(i, min(i + self.insert_batch_size, len(dataset)))]
            await process_chunk(batch)
        
        # 6. 注册文档
        # self.doc_registry.register_document(doc_id, dataset_name, chunk_ids)
        
        logger.info(f"Ingested document {doc_id} with {len(chunk_ids)} chunks")
    

    async def naive_retrieve(self, query: list[dict], top_k: int = 5) -> list[DataChunk]:
        """执行检索流程"""
        # 1. 查询向量化
        # 2. 向量搜索
        # 3. 结果排序
        query_index = await self.embedding_service.async_embed_batch(query)
        results = await self.vector_storage.search(query_index[0], top_k=top_k)
        return [DataChunk(**r) for r in results] # TODO

    async def hybrid_retrieve(
        self,
        query_text: str,
        query_image_paths: Optional[list[str]] = None,
        top_k: int = 10,
        fusion_k: float = 60.0,
        initial_top_k: int = 100
    ) -> list[DataChunk]:
        """
        混合检索：结合Dense向量检索和BM25稀疏检索，使用RRF融合结果

        Args:
            query_text: 查询文本
            query_image_paths: 可选的查询图像路径列表
            top_k: 最终返回的结果数量
            fusion_k: RRF融合常数（默认60）
            initial_top_k: 从每种检索方式获取的初始结果数量

        Returns:
            List[DataChunk]: 融合后的检索结果，包含完整的文档内容
        """
        # 1. 准备查询
        query = {"text": query_text}
        if query_image_paths:
            query["image_paths"] = query_image_paths

        # 2. Dense检索（向量检索）
        query_vectors = await self.embedding_service.async_embed_batch([query])
        dense_results = await self.vector_storage.search(query_vectors[0], top_k=initial_top_k)

        # 转换为 (chunk_id, score) 格式
        dense_scores = [
            (chunk.chunk_id, chunk.metadata.get("score", 0.0))
            for chunk in dense_results
            if chunk.chunk_id
        ]

        # 3. Sparse检索（BM25）
        bm25_results = await self.bm25_storage.search(query_text, top_k=initial_top_k)
        # BM25结果已经是 (chunk_id, score) 格式

        # 4. RRF融合
        fused_results = rrf_fuse(
            dense_results=dense_scores,
            sparse_results=bm25_results,
            k=fusion_k,
            top_k=top_k
        )

        # 5. 获取完整的文档内容
        # 从dense_results构建chunk_id到DataChunk的映射
        chunk_map = {chunk.chunk_id: chunk for chunk in dense_results}

        # 对于BM25独有的结果，需要从存储中获取
        final_results = []
        for chunk_id, rrf_score in fused_results:
            if chunk_id in chunk_map:
                chunk = chunk_map[chunk_id]
                # 更新metadata中的RRF分数
                chunk.metadata["rrf_score"] = rrf_score
                chunk.metadata["fusion_method"] = "rrf"
                final_results.append(chunk)
            else:
                # 尝试从BM25存储获取原始文本
                text = self.bm25_storage.get_document(chunk_id)
                if text:
                    chunk = DataChunk(
                        chunk_id=chunk_id,
                        content=text,
                        metadata={"rrf_score": rrf_score, "fusion_method": "rrf", "source": "bm25"}
                    )
                    final_results.append(chunk)

        logger.info(f"Hybrid retrieval: dense={len(dense_scores)}, sparse={len(bm25_results)}, "
                   f"fused={len(final_results)}")

        return final_results
    
    async def rerank(self, query: str, chunks: list[DataChunk]) -> list[DataChunk]:
        """可选：使用重排模型优化结果（Cohere/BGE-reranker）"""
        chunks = [chunk.to_dict() for chunk in chunks ]
        scores = await self.rerank_service.async_generate_batch([query], chunks)
        scores = scores[0]
        sorted_chunks = [DataChunk(**chunk) for _, chunk in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
        return sorted_chunks

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
        await self.bm25_storage.persist()
        logger.info("RAG system shutdown complete")

async def main():
    # # first init
    # workspace = "workspace"
    # if os.path.exists(workspace):
    #     shutil.rmtree(workspace)
    # os.makedirs(workspace)

    embedding_service = OpenaiEmbeddingService()
    llm_service = VLMService()
    RAG = MMRAG(
        embedding_service=embedding_service,
        llm_service=llm_service,
        workspace="./workspace",
    )
    # dataset_list = [
    #     PMCOADataset("pmc-oa")
    # ]
    # await RAG.ainit_rag(dataset_list)
    print(RAG.vector_storage.client.get_collection_stats(RAG.vector_storage.collection_name)['row_count'])
    query = [{"text": "ICA-derived 3D model"}]
    results = await RAG.retrieve(query, top_k=5)
    for r in results:
        print(f'score: {r.metadata.get("score")}: {r.content}')
    
    # test recall@5 and MRR on 200
    dataset_size = 400
    dataset = PMCOADataset("pmc-oa", dataset_size)
    recall_1_score, recall_5_score, recall_10_score = 0,0,0
    MRR_score = 0
    for i in range(len(dataset)):
        item = dataset[i]
        query = item['question']
        query = [{'text': query}]
        results = await RAG.retrieve(query, top_k=10)
        local_mrr_score = 0
        for local_index, r in enumerate(results):
            if r.chunk_id == item['chunk_id']:
                if local_index == 0:
                    recall_1_score += 1
                if local_index < 5:
                    recall_5_score += 1
                if local_index < 10:
                    recall_10_score += 1
                local_mrr_score = 1/(local_index+1)
        MRR_score += local_mrr_score
    print(f"""
            recall@1: {recall_1_score / dataset_size}, 
            recall@5: {recall_5_score / dataset_size},
            recall@10: {recall_10_score / dataset_size},
            MRR: {MRR_score / dataset_size}""")

if __name__ == "__main__":
    asyncio.run(main())
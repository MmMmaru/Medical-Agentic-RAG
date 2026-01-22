class Retriever:
    """检索协调器"""
    
    def __init__(self, vector_storage: VectorStorage, embedding_service: EmbeddingService):
        pass
        
    async def retrieve(self, query: str, top_k: int = 5) -> list[TextChunk]:
        """执行检索流程"""
        # 1. 查询向量化
        # 2. 向量搜索
        # 3. 结果排序
        
    async def rerank(self, query: str, chunks: list[TextChunk]) -> list[TextChunk]:
        """可选：使用重排模型优化结果（Cohere/BGE-reranker）"""
        pass
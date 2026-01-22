
class DocumentProcessor:
    """文档处理核心组件"""
    
    async def load_document(self, file_path: str) -> str:
        """加载文档内容，支持 txt/pdf/docx/md"""
        
    async def split_chunks(self, content: str, chunk_size: int = 512) -> list[TextChunk]:
        """将文档切分为固定大小的文本块"""
        
    async def process(self, file_path: str) -> list[TextChunk]:
        """完整处理流程：加载 -> 分块 -> 返回"""
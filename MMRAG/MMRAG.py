from typing import Any, Dict, List

class MMRAG():
    """
    Docstring for MMRAG
    1、支持文件持久化
    2、支持多模态数据存储与检索（多路检索，图片文本分开然后进行关联合并）
    3、支持向量化存储与检索
    """
    async def aquery(query: str, image_paths: list[str] | None = None, top_k: int = 5) -> List[Dict[str, Any]]:
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
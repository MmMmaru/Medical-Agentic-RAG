import openai
import asyncio
import httpx
import sys
import os
if __name__ == "__main__":
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from MMRAG.utils import encode_image_paths_to_base64

DEFAULT_RERANKER_INSTRUCTION="""

"""

class OpenAIRerankerService:
    """OpenAI VLM 服务封装"""
    def __init__(self, model_name: str, api_key: str="EMPTY", url="http://localhost:8002/v1/rerank", instruction=DEFAULT_RERANKER_INSTRUCTION):
        self.url = url
        self.model_name = model_name
        self.client = httpx.AsyncClient()

    async def async_generate_batch(self, querys: list[dict],docs: list[list[dict]], temperature: float = 0.7) -> list[list[int]]:
        # doc: [[{"content": str, "image_paths": []}]]
        # query: [{"text": str}] or [{"image": <encoded image>}]

        docs_formatted=[]
        for documents in docs:
            documents_formatted = []
            for d in documents:
                contents = []
                content = d.get("content", None)
                if content:
                    contents.append({"text": d['content']})
                image_paths = d.get("image_paths", None)
                if image_paths:
                    contents += encode_image_paths_to_base64(d['image_paths'])
                documents_formatted.append({"content": contents})
            docs_formatted.append(documents_formatted)
        tasks = []
        for query, documents in zip(querys, docs_formatted):
            tasks.append(self.client.post(
                self.url,
                json={
                    "model": self.model_name,
                    "query": query.get('text',''),
                    "documents": documents # recieve {"content": list[dict]}
                }
            ))
        responses = await asyncio.gather(*tasks)
        all_scores = []
        # TODO: need test
        for response_docs in responses:
            all_scores.append(response_docs.json())
        return all_scores


if __name__ == "__main__":
    queries = [
        {"text": "A woman playing with her dog on a beach at sunset."},
    ]

    documents = [
        {"content": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
        {"image_paths": ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"]},
        {"content": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", 
        "image_paths": ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"]}
    ]
    reranker_service = OpenAIRerankerService("Qwen/Qwen3-VL-Reranker-2B")
    results = asyncio.run(reranker_service.async_generate_batch(
        queries,
        [documents]
    ))
    print(results)
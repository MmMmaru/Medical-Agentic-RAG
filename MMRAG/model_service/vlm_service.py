import openai
from abc import ABC, abstractmethod
import asyncio

class VLMService:
    """LLM 调用封装"""
    async def async_generate_batch(self, prompt: str, temperature: float = 0.7) -> str:
        """生成回答"""
        raise NotImplementedError("Subclasses must implement this method.")

class OpenAIVLMService(VLMService):
    """OpenAI VLM 服务封装"""
    def __init__(self, model_name: str, api_key: str="EMPTY", url="http://localhost:8000/v1"):
        self.client = openai.AsyncOpenAI(
            base_url=url,
            api_key=api_key
        )
        self.model_name = model_name
        openai.api_key = api_key

    async def async_generate_batch(self, contents: list[list[dict]], temperature: float = 0.7) -> list[str]:
        "contents: [[{'text': str}, ..]]"
        tasks = [self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=temperature
        ) for content in contents]
        results = await asyncio.gather(*tasks)
        return [result.choices[0].message.content.strip() for result in results]

if __name__ == "__main__":
    contents = [
    [
        {
            "type": "image_url",
            "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
        },
        {
            "type": "text",
            "text": "Describe the image in detail."
        }
    ],
    [
        {
            "type": "image_url",
            "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
        },
        {
            "type": "text",
            "text": "Describe the image in detail."
        }
    ],
    ]
    vlm_service = OpenAIVLMService("Qwen3-VL-4B-Instruct", api_key="EMPTY", url="http://localhost:8000/v1")
    responses = asyncio.run(vlm_service.async_generate_batch(contents))
    for resp in responses:
        print("="*50)
        print(resp)
    

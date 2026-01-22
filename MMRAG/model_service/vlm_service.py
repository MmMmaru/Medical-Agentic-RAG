import openai

class LLMService:
    """LLM 调用封装"""
    
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """生成回答"""
        
    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """流式生成"""
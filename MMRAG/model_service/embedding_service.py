import numpy as np
from typing import List, Dict, Any
from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image
from dotenv import load_dotenv
from dataclasses import dataclass
import openai
from openai.types.create_embedding_response import CreateEmbeddingResponse
import base64
import io
from PIL import Image
import requests
import asyncio

import sys
import os
# 获取当前文件所在目录的父目录的父目录（即项目根目录）
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..utils import logger

load_dotenv()
class EmbeddingService:
    embedding_dim: int

    def embed(self, inputs: List[Dict[str, Any]]) -> List[float]:
        raise NotImplementedError

    async def async_embed_batch(self, inputs: List[Dict[str, Any]], batch_size: int = 128) -> List[List[float]]:
        raise NotImplementedError

@dataclass
class VllmEmbeddingService(EmbeddingService):
    embedding_dim: int = 2048

    max_model_len: int = 4096
    max_num_seqs: int = 128
    tensor_parallel_size = 2
    gpu_memory_utilization = 0.8
    model_path = os.getcwd()+"/.cache/modelscope/models/qwen/Qwen3-VL-Embedding-2B"
    def __init__(self):

        engine_args = EngineArgs(
            model=self.model_path,
            runner="pooling",
            dtype="auto",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            kv_cache_dtype="auto",
        )
        self.llm = LLM(**vars(engine_args))
        logger.info(f"✅ vLLM model loaded from {self.model_path}")

    def embed(self, inputs: List[Dict[str, Any]]) -> List[np.ndarray]:
        'e.g. [{"test", "A woman playing with her dog on a beach at sunset."}, {"image": "https://..."}]'
        vllm_inputs = [self._prepare_vllm_inputs(inp) for inp in inputs]
        outputs = self.llm.embed(vllm_inputs)
        embeddings = [output.outputs.embedding for output in outputs]
        return embeddings
    
    async def async_embed_batch(self, inputs: List[Dict[str, Any]], batch_size: int = 128) -> List[np.ndarray]:
        all_embeddings = []
        formatted_inputs = [self._prepare_vllm_inputs(inp) for inp in inputs]
        for i in range(0, len(inputs), batch_size):
            batch_inputs = formatted_inputs[i:i+batch_size]
            outputs = self.llm.embed(batch_inputs) # List[EmbedOutput]
            embeddings = [output.outputs.embedding for output in outputs]
            all_embeddings.extend(embeddings)
        return all_embeddings
    
    def _format_input_to_conversation(self, input_dict: Dict[str, Any], instruction: str = "Represent the user's input.") -> List[Dict]:
        """
        format input to conversation format
        deal with image path
        Args:
            input_dict: Dict with 'text' and optional 'image'
        """
        content = []
        
        text = input_dict.get('text')
        image = input_dict.get('image')
        
        if image:
            image_content = None
            if isinstance(image, str):
                if image.startswith(('http', 'https', 'oss')):
                    image_content = image
                else:
                    abs_image_path = os.path.abspath(image)
                    image_content = 'file://' + abs_image_path
            else:
                image_content = image
            
            if image_content:
                content.append({
                    'type': 'image', 
                    'image': image_content,
                })
        
        if text:
            content.append({'type': 'text', 'text': text})
        
        if not content:
            content.append({'type': 'text', 'text': ""})
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content}
        ]
        
        return conversation

    def _prepare_vllm_inputs(self, input_dict: Dict[str, Any], instruction: str = "Represent the user's input.") -> Dict[str, Any]:
        """
        prepare vllm input format
        Args:
            input_dict: Dict with 'text' and optional 'image'
                e.g. {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.",
                     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
            instruction: Instruction for the model (for qwen3-vl-embedding)
        """
        text = input_dict.get('text')
        image = input_dict.get('image')
        
        conversation = self._format_input_to_conversation(input_dict, instruction)
        
        prompt_text = self.llm.llm_engine.tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False # 是否需要assistant的提示符
        )
        
        multi_modal_data = None
        if image:
            if isinstance(image, str):
                if image.startswith(('http', 'https', 'oss')):
                    try:
                        image_obj = fetch_image(image)
                        multi_modal_data = {"image": image_obj}
                    except Exception as e:
                        print(f"Warning: Failed to fetch image {image}: {e}")
                else:
                    abs_image_path = os.path.abspath(image)
                    if os.path.exists(abs_image_path):
                        from PIL import Image
                        image_obj = Image.open(abs_image_path)
                        multi_modal_data = {"image": image_obj}
                    else:
                        print(f"Warning: Image file not found: {abs_image_path}")
            else:
                multi_modal_data = {"image": image}
        
        result = {
            "prompt": prompt_text,
            "multi_modal_data": multi_modal_data
        }
        return result

@dataclass
class OpenaiEmbeddingService(EmbeddingService):
    base_url: str = "http://localhost:8001/v1"
    embedding_dim: int = 2048

    def __init__(self, model_name: str = "Qwen3-VL-Embedding-2B", api_key: str = None):
        self.model_name = model_name
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            base_url=self.base_url,
        )

    async def async_embed_batch(self, inputs: List[Dict[str, Any]], batch_size=1) -> List[List[float]]:
        """
        Args:
            inputs: List of Dicts with 'content' and optional 'images'
            batch_size: OpenAI API does not support batch, so this is ignored
        Returns:
            List of float embeddings
        """
        all_embeddings = []
        formatted_inputs = [self._format_input_to_conversation(inp) for inp in inputs]
        tasks = [
            self.openai_client.post(
                "/embeddings",
                cast_to=CreateEmbeddingResponse,
                body={
                    "messages": messages,
                    "model": self.model_name,
                    "encoding_format": "float",
                },
            ) for messages in formatted_inputs
        ]
        results = await asyncio.gather(*tasks)
        for response in results:
            all_embeddings.append(response.data[0].embedding)
        return all_embeddings
    
    def _format_input_to_conversation(self, input_dict: Dict[str, Any], instruction: str = "Represent the user's input.") -> List[Dict]:
        """
        format input to conversation format
        deal with image path
        Args:
            input_dict: Dict with 'text' and optional 'image'
        """
        content = []
        
        text = input_dict.get('content') or input_dict.get('text')
        images = input_dict.get('image_paths') or input_dict.get('image')
        images = images if isinstance(images, list) else [images] if images else []

        for image in images:
            image_content = None
            if isinstance(image, str):
                if image.startswith(('http', 'https', 'oss')):
                    response = requests.get(image)
                    image_str = base64.b64encode(response.content).decode("utf-8")
                    image_content = f"data:image/jpeg;base64,{image_str}"
                elif os.path.exists(image):
                    abs_image_path = os.path.abspath(image)
                    mime_type = "image/png" if abs_image_path.lower().endswith(".png") else "image/jpeg"
                    with open(abs_image_path, "rb") as img_file:
                        img_str = base64.b64encode(img_file.read()).decode("utf-8")
                    image_content = f"data:{mime_type};base64,{img_str}"
                elif isinstance(image, Image.Image):
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    image_content = f"data:image/jpeg;base64,{img_str}"
            else:
                raise ValueError("Unsupported image format")
            
            if image_content:
                content.append({
                    'type': 'image_url', 
                    'image_url': {'url': image_content},
                })
        
        if text:
            content.append({'type': 'text', 'text': text})
        
        if not content:
            content.append({'type': 'text', 'text': ""}) 
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content}
        ]
        
        return conversation

async def main():

    # Define a list of query texts
    queries = [
        {"text": "A woman playing with her dog on a beach at sunset."},
        {"text": "Pet owner training dog outdoors near water."},
        {"text": "Woman surfing on waves during a sunny day."},
        {"text": "City skyline view from a high-rise building at night."}
    ]

    # Define a list of document texts and images
    documents = [
        {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
        {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
        {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
    ]

    openai_embedding  = OpenaiEmbeddingService()
    all_inputs = queries + documents
    embeddings = await openai_embedding.async_embed_batch(all_inputs)
    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Each embedding dimension: {len(embeddings[0])}")
    np_embedding = np.array(embeddings)
    query_embeddings = np_embedding[:len(queries)]
    doc_embeddings = np_embedding[len(queries):]
    similarity_scores = query_embeddings @ doc_embeddings.T # (4,3)
    print("\nSimilarity Scores:")
    print(similarity_scores.tolist())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
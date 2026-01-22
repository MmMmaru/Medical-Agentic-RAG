import argparse
import numpy as np
import os
from typing import List, Dict, Any
from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image
import os
from dotenv import load_dotenv
from dataclasses import dataclass

import sys
import os
# 获取当前文件所在目录的父目录的父目录（即项目根目录）
sys.path.append(os.getcwd())

from MMRAG.utils import logger
from MMRAG.base import EmbeddingFunc


load_dotenv()
class VllmEmbedding(EmbeddingFunc):
    max_model_len: int = 8192
    def __init__(self, model_path: str, dtype: str = "float16", tensor_parallel_size: int = 4, gpu_memory_utilization: float = 0.8):
        self.max_model_len: int = 8192

        engine_args = EngineArgs(
            model=model_path,
            runner="pooling",
            dtype="auto",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=self.max_model_len,
            kv_cache_dtype="auto",
        )
        self.llm = LLM(**vars(engine_args))
        logger.info(f"✅ vLLM model loaded from {model_path}")

    def embed(self, inputs: List[Dict[str, Any]]) -> List[np.ndarray]:
        vllm_inputs = [self._prepare_vllm_inputs(inp) for inp in inputs]
        outputs = self.llm.embed(vllm_inputs)
        embeddings = [output.outputs.embedding for output in outputs]
        return embeddings
    
    def embed_batch(self, inputs: List[Dict[str, Any]], batch_size: int = 8) -> List[np.ndarray]:
        all_embeddings = []
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_embeddings = self.embed(batch_inputs)
            all_embeddings.extend(batch_embeddings)
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

model_path = os.getcwd()+"/.cache/modelscope/models/qwen/Qwen3-VL-Embedding-2B"

if __name__ == "__main__":
    vllm_embedding  = VllmEmbedding(model_path=model_path)
    all_inputs = queries + documents
    embeddings = vllm_embedding.embed(all_inputs)
    print(f"Generated {len(embeddings)} embeddings.")
    print(f"Each embedding dimension: {len(embeddings[0])}")
    np_embedding = np.array(embeddings)
    query_embeddings = np_embedding[:len(queries)]
    doc_embeddings = np_embedding[len(queries):]
    similarity_scores = query_embeddings @ doc_embeddings.T # (4,3)
    print("\nSimilarity Scores:")
    print(similarity_scores.tolist())
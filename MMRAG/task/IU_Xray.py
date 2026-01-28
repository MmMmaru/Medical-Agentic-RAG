from datasets import load_dataset, load_from_disk
import os
from torch.utils.data import Dataset
import asyncio
from typing import List, Dict, Any
import sys

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_service.vlm_service import OpenAIVLMService
from utils import compute_mdhash_id, extract_catogorical_answer, logger, encode_image_paths_to_base64

class IUXrayDataset(Dataset):
    def __init__(self, dataset_name, vlm_service, dataset_size=None):
        super().__init__()
        self.vlm_service = vlm_service
        self.dataset_name = dataset_name
        self.dataset_path = f"./datasets/processed_datasets/{self.dataset_name}"
        if os.path.exists(self.dataset_path):
            logger.info(f"Loading processed dataset from {self.dataset_path}.")
            self.dataset = load_from_disk(self.dataset_path)
        else:
            self.dataset = self.process_dataset(dataset_size=dataset_size)
    
    def process_dataset(self, dataset_size=None, rewrite=True):
        """将数据集处理为统一格式，改写内容，保存dataset
        对VQA数据集增加content，对report类型数据集增加question
        keys: 'question', 'image_paths', 'content', 'index', 'chunk_id', 'dataset_id', 'answer', 'answer_labal', 'key_words', 'source'
        """
        dataset = load_dataset(self.dataset_name, split="test") # load from network
        if dataset_size is not None:
            dataset = dataset.select(range(min(dataset_size, len(dataset))))
        workspace_folder = self.dataset_path
        os.makedirs(os.path.join(workspace_folder, "images"), exist_ok=True)
        client = self.vlm_service
        sem = asyncio.Semaphore(64)  # 限制并发数量，防止过多请求导致问题
        async def process_example(idx, ex):
            # 处理单个样本
            async with sem:
                image_paths = []
                chunk_id = compute_mdhash_id(content, "chunk_")
                for i, image in enumerate(ex['images']):
                    image_path = os.path.join(workspace_folder, "images", f"pmcvqa_{idx}_img{i}.jpeg")
                    image.save(image_path)
                    image_paths.append(image_path)
                
                if rewrite:
                    from task.common import REWRITE_CONTENT_PROMPT
                    image_content = encode_image_paths_to_base64(image_paths)
                    prompt_text = REWRITE_CONTENT_PROMPT.format(question=ex["question"])
                    messages = image_content + [{"type": "text", "text": prompt_text}]
                    response = await client.async_generate_batch([messages], temperature=0.7)
                    content = response[0].strip()
                else:
                    content = ex['answer'].strip()
                    
                new_example = {
                    "question": ex['question'],
                    "image_paths": image_paths,
                    "content": content,
                    "index": idx,
                    "chunk_id": chunk_id,
                    "dataset_id": self.dataset_name,
                    "answer": ex['answer'],
                    "answer_label": ex['answer_label'],
                    "key_words": [],
                    "source": self.dataset_name
                }
                return new_example
            
        async def run_all():
            return await asyncio.gather(
                *[process_example(idx, ex) for idx, ex in enumerate(dataset)]
            )
        
        results = asyncio.run(
            run_all()
        )
        
        from datasets import Dataset
        dataset = Dataset.from_list(results)
        logger.info(f"processed dataset{self.dataset_name} with {len(dataset)} examples.")
        dataset.save_to_disk(self.dataset_path)
        logger.info(f"saved processed dataset to {self.dataset_path}.")
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def evaluate(self, index, conversation):
        assistant_message = conversation['messages'][-1]
        last_text_part = assistant_message['content'][-1]['text'] # this contains the final answer in GSM8K
        pred_answer = extract_catogorical_answer(last_text_part)
        gold_answer = self.dataset[index]['answer_labal']
        return int(pred_answer == gold_answer)

if __name__ == "__main__":
    openai_service = OpenAIVLMService(model_name="Qwen3-VL-4B-Instruct", api_key="EMPTY", url="http://localhost:8000/v1")
    dataset = IUXrayDataset("UCSC-VLAA/MedVLThinker-Eval", openai_service, 10)
    dataset.process_dataset()

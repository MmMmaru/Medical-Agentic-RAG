from datasets import load_dataset, load_from_disk
import os
from utils import compute_mdhash_id
from model_service.vlm_service import OpenAIVLMService
from torch.utils.data import Dataset
from utils import extract_catogorical_answer, logger, encode_image_paths_to_base64
import asyncio
from typing import List, Dict, Any


class PmcVQADataset(Dataset):
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
        dataset = load_dataset(self.dataset_name, split=" ") # load from network
        if dataset_size is not None:
            dataset = dataset.select(range(min(dataset_size, len(dataset))))
        workspace_folder = self.dataset_path
        os.makedirs(os.path.join(workspace_folder, "images"), exist_ok=True)
        client = self.vlm_service
        
        async def process_example(idx, ex):
            # 处理单个样本
            image_paths = []
            for i, image in enumerate(ex['images']):
                image_path = os.path.join(workspace_folder, "images", f"{ex['chunk_id']}_img{i}.jpeg")
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
                
            chunk_id = compute_mdhash_id(content, "chunk_")
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
        
        results = []
        results = asyncio.run(
            asyncio.gather(
            *[process_example(idx, ex) for idx, ex in enumerate(dataset)]
        )
        )
        
        from datasets import Dataset
        dataset = Dataset.from_list(results)
        logger.info(f"processed dataset{self.dataset_name} with {len(dataset)} examples.")
        dataset.save_to_disk(self.dataset_path)
        logger.info(f"saved processed dataset to {self.dataset_path}.")
        return dataset
    
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
    openai_service = OpenAIVLMService(model_name="google/medgemma-1.5-5b-it", api_key="EMPTY", url="http://localhost:8000/v1")
    dataset = PmcVQADataset("UCSC-VLAA/MedVLThinker-Eval", openai_service, 10)

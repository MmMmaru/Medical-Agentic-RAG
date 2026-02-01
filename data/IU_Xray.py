# iu_xary

# sample
# {"id": "CXR2384_IM-0942", "report": "The heart size and pulmonary vascularity appear within normal limits. A large hiatal hernia is noted. The lungs are free of focal airspace disease. No pneumothorax or pleural effusion is seen. Degenerative changes are present in the spine.", "image_path": ["CXR2384_IM-0942/0.png", "CXR2384_IM-0942/1.png"], "split": "train"}, {"id": "CXR2926_IM-1328", "report": "Cardiac and mediastinal contours are within normal limits. The lungs are clear. Bony structures are intact.", "image_path": ["CXR2926_IM-1328/0.png", "CXR2926_IM-1328/1.png"], "split": "train"}, {"id": "CXR1451_IM-0291", "report": "Left lower lobe calcified granuloma. Heart size normal. No pleural effusion or pneumothorax. Mild medial right atelectasis. Mild emphysema.", "image_path": ["CXR1451_IM-0291/0.png", "CXR1451_IM-0291/1.png"], "split": "train"}

from datasets import load_dataset, load_from_disk
import os
from torch.utils.data import Dataset
import asyncio
from typing import List, Dict, Any
import sys

if __name__ == "__main__":
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data.common import REWRITE_CONTENT_QUESTION_PROMPT
from MMRAG.model_service.vlm_service import OpenAIVLMService
from MMRAG.utils import compute_mdhash_id, extract_catogorical_answer, logger, encode_image_paths_to_base64

class MedMaxDataset(Dataset):
    def __init__(self, dataset_name, dataset_size=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_path = f"./datasets/processed_datasets/{self.dataset_name}"
        if os.path.exists(self.dataset_path):
            logger.info(f"Loading processed dataset from {self.dataset_path}.")
            self.dataset = load_from_disk(self.dataset_path)
        else:
            self.dataset = None
    
    @staticmethod
    def process_dataset(dataset_name, output_path, vlm_service, root_path, dataset_size=None, rewrite=True):
        """将数据集处理为统一格式，改写内容，保存dataset
        keys: 'question', 'image_paths', 'content', 'index', 'chunk_id', 'dataset_id', 'answer', 'answer_label', 'key_words', 'source'
        """
        # Load from Hugging Face
        dataset = load_dataset(dataset_name, split=f"train[:{dataset_size}]")
        
        # Filter by credential if needed - usually we want 'no' for public processing
        # dataset = dataset.filter(lambda x: x['credential'] == 'no')
        
        if dataset_size is not None:
            dataset = dataset.select(range(min(dataset_size, len(dataset))))
        
        workspace_folder = output_path
        os.makedirs(os.path.join(workspace_folder, "images"), exist_ok=True)
        client = vlm_service
        sem = asyncio.Semaphore(64)

        async def process_example(idx, ex):
            async with sem:
                image_paths = []
                for img_path in ex['image_path']:
                    image_paths.append(os.path.join(root_path, ))
            
        async def run_all():
            return await asyncio.gather(
                *[process_example(idx, ex) for idx, ex in enumerate(dataset)]
            )
        
        results = asyncio.run(run_all())
        
        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_list(results)
        logger.info(f"processed dataset {dataset_name} with {len(dataset)} examples.")
        dataset.save_to_disk(output_path)
        logger.info(f"saved processed dataset to {output_path}.")
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def evaluate(self, item, conversation):
        assistant_message = conversation['messages'][-1]
        last_text_part = assistant_message['content'][-1]['text'] 
        pred_answer = extract_catogorical_answer(last_text_part)
        gold_answer = item['answer_label']
        return int(pred_answer == gold_answer)

if __name__ == "__main__":
    openai_service = OpenAIVLMService(model_name="Qwen3-VL-4B-Instruct", api_key="EMPTY", url="http://localhost:8000/v1")
    output_path = "datasets/preprocessed_datasets/medmax"
    MedMaxDataset.process_dataset("datasets/medmax", output_path, openai_service, "datasets/medmax", dataset_size=100)
# mint-medmax/medmax_data
# Each dataset instance includes:

# text: Instruction, context, and the expected response (can be purely textual or multimodal).
# tokens: Tokenized representations for text and images (credentialed entries have no pre-included tokens, users need to download images and tokenize them).
# image_path: References to corresponding image files.
# task: The type of biomedical task (e.g., VQA, Image Generation, Report Understanding).
# source: Data source origin.
# credential: Access level ('yes'/'no') indicating if special credentials are required.


from datasets import load_dataset, load_from_disk
import os
from torch.utils.data import Dataset
import asyncio
from typing import List, Dict, Any
import sys

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        dataset = load_dataset(dataset_name, split="train") 
        
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
                if ex['credential'] == "no":
                    image_paths = []
                    
                    # 1. Handle image_path
                    raw_image_paths = ex.get('image_path', [])
                    if isinstance(raw_image_paths, str):
                        raw_image_paths = [raw_image_paths]
                    
                    # Note: In HF datasets, images are often pre-loaded as PIL objects in a specific key
                    # but if we follow the comment strictly, we use 'image_path'.
                    # We assume 'images' might still be available or we need to open from 'image_path'.
                    if 'images' in ex: # Usual HF pattern
                        for i, image in enumerate(ex['images']):
                            if image:
                                if image.mode != 'RGB':
                                    image = image.convert('RGB')
                                image_path = os.path.join(workspace_folder, "images", f"medmax_{idx}_img{i}.jpeg")
                                image.save(image_path)
                                image_paths.append(image_path)
                    elif raw_image_paths: # Use path if images not in-memory
                        image_paths = []
                        for i, img_p in enumerate(raw_image_paths):
                            image_path = os.path.join(root_path, img_p)
                            if os.path.exists(image_path):
                                image_paths.append(image_path)
                    if image_paths is None:
                        return None

                    # 2. Handle text (Instruction, context, and expected response)
                    full_text = ex.get('text', '').strip()
                    question = full_text
                    answer = ""
                    
                    # 3. Rewrite content or use answer
                    if rewrite:
                        image_content = encode_image_paths_to_base64(image_paths)
                        prompt_text = REWRITE_CONTENT_QUESTION_PROMPT.format(question=question)
                        messages = image_content + [{"type": "text", "text": prompt_text}]
                        response = await client.async_generate_batch([messages], temperature=0.7)
                        content = response[0].strip()
                        chunk_id = compute_mdhash_id(content, "chunk_")
                        
                        # 4. Extract answer label for evaluation
                        # If it's multiple choice, the answer might be "A" or "A. text"
                        answer_label = extract_catogorical_answer(answer) if answer else ""
                        if not answer_label and answer:
                            answer_label = answer[0] if answer[0] in "ABCDE" else answer

                        new_example = {
                            "question": content['question'],
                            "image_paths": image_paths,
                            "content": content['content'],
                            "index": idx,
                            "chunk_id": chunk_id,
                            "dataset_id": dataset_name,
                            "answer": answer,
                            "answer_label": answer_label,
                            "key_words": content['key_words'],
                            "source": ex.get('source', dataset_name),
                            "task": ex.get('task', ''),
                            "credential": ex.get('credential', 'no')
                        }
                    else:
                        content = answer if answer else question
                    
                else:
                    new_example = None
                return new_example
            
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
    MedMaxDataset.process_dataset("mint-medmax/medmax_data", output_path, openai_service, "datasets/medmax")
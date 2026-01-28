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
from model_service.vlm_service import OpenAIVLMService
from utils import compute_mdhash_id, extract_catogorical_answer, logger, encode_image_paths_to_base64

class MedMaxDataset(Dataset):
    def __init__(self, dataset_name, vlm_service, dataset_size=None):
        super().__init__()
        self.vlm_service = vlm_service
        self.dataset_name = dataset_name
        self.dataset_path = f"./datasets/processed_datasets/{self.dataset_name}"
        if os.path.exists(self.dataset_path):
            logger.info(f"Loading processed dataset from {self.dataset_path}.")
            self.dataset = load_from_disk(self.dataset_path)
        else:
            self.dataset = None
    
    def process_dataset(self, dataset_size=None, rewrite=True):
        """将数据集处理为统一格式，改写内容，保存dataset
        keys: 'question', 'image_paths', 'content', 'index', 'chunk_id', 'dataset_id', 'answer', 'answer_label', 'key_words', 'source'
        """
        # Load from Hugging Face
        dataset = load_dataset("mint-medmax/medmax_data", split="test") 
        
        # Filter by credential if needed - usually we want 'no' for public processing
        # dataset = dataset.filter(lambda x: x['credential'] == 'no')
        
        if dataset_size is not None:
            dataset = dataset.select(range(min(dataset_size, len(dataset))))
        
        workspace_folder = self.dataset_path
        os.makedirs(os.path.join(workspace_folder, "images"), exist_ok=True)
        client = self.vlm_service
        sem = asyncio.Semaphore(64)

        async def process_example(idx, ex):
            async with sem:
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
                    import shutil
                    for i, img_p in enumerate(raw_image_paths):
                        if os.path.exists(img_p):
                            dest_path = os.path.join(workspace_folder, "images", f"medmax_{idx}_img{i}.jpeg")
                            shutil.copy(img_p, dest_path)
                            image_paths.append(dest_path)

                # 2. Handle text (Instruction, context, and expected response)
                full_text = ex.get('text', '')
                question = full_text
                answer = ""
                
                # Split text into question and answer
                # Common patterns in MedMax or similar instruction datasets
                if "### Response:" in full_text:
                    question, answer = full_text.split("### Response:", 1)
                elif "Assistant:" in full_text:
                    question, answer = full_text.split("Assistant:", 1)
                elif "\nAnswer:" in full_text:
                    question, answer = full_text.split("\nAnswer:", 1)
                
                question = question.strip()
                answer = answer.strip()

                # 3. Rewrite content or use answer
                if rewrite:
                    from task.common import REWRITE_CONTENT_PROMPT
                    if image_paths:
                        image_content = encode_image_paths_to_base64(image_paths)
                        prompt_text = REWRITE_CONTENT_PROMPT.format(question=question)
                        messages = image_content + [{"type": "text", "text": prompt_text}]
                        response = await client.async_generate_batch([messages], temperature=0.7)
                        content = response[0].strip()
                    else:
                         content = question
                else:
                    content = answer if answer else question
                
                chunk_id = compute_mdhash_id(content, "chunk_")
                
                # 4. Extract answer label for evaluation
                # If it's multiple choice, the answer might be "A" or "A. text"
                answer_label = extract_catogorical_answer(answer) if answer else ""
                if not answer_label and answer:
                    answer_label = answer[0] if answer[0] in "ABCDE" else answer

                new_example = {
                    "question": question,
                    "image_paths": image_paths,
                    "content": content,
                    "index": idx,
                    "chunk_id": chunk_id,
                    "dataset_id": self.dataset_name,
                    "answer": answer,
                    "answer_label": answer_label,
                    "key_words": [],
                    "source": ex.get('source', self.dataset_name),
                    "task": ex.get('task', ''),
                    "credential": ex.get('credential', 'no')
                }
                return new_example
            
        async def run_all():
            return await asyncio.gather(
                *[process_example(idx, ex) for idx, ex in enumerate(dataset)]
            )
        
        results = asyncio.run(run_all())
        
        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_list(results)
        logger.info(f"processed dataset {self.dataset_name} with {len(dataset)} examples.")
        dataset.save_to_disk(self.dataset_path)
        logger.info(f"saved processed dataset to {self.dataset_path}.")
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def evaluate(self, index, conversation):
        assistant_message = conversation['messages'][-1]
        last_text_part = assistant_message['content'][-1]['text'] 
        pred_answer = extract_catogorical_answer(last_text_part)
        gold_answer = self.dataset[index]['answer_label']
        return int(pred_answer == gold_answer)

if __name__ == "__main__":
    openai_service = OpenAIVLMService(model_name="Qwen3-VL-4B-Instruct", api_key="EMPTY", url="http://localhost:8000/v1")
    dataset = MedMaxDataset("medmax", openai_service, 10)
    dataset.process_dataset()
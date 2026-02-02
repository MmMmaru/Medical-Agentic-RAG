# pmc-oa dataset

# sample
# {
#     "image": "PMC212319_Fig3_4.jpg",
#     "caption": "A. Real time image of the translocation of ARF1-GFP to the plasma membrane ...",
# }

from datasets import load_dataset, load_from_disk
import os
from torch.utils.data import Dataset
import asyncio
from typing import List, Dict, Any
import sys
import json

if __name__ == "__main__":
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data.common import REWRITE_QUESTION_PROMPT
from MMRAG.model_service.vlm_service import OpenAIVLMService
from MMRAG.utils import compute_mdhash_id, extract_catogorical_answer, logger, encode_image_paths_to_base64


class PMCOADataset(Dataset):
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
        keys: 'question', 'image_paths', 'content', 'index', 'chunk_id', 'dataset_id', 'answer', 'answer_label', 'key_words', 'source', 'task', 'credential'
        """
        # Load from Hugging Face
        dataset = load_dataset('json', dataset_name, split=f"train[:{dataset_size}]")

        if dataset_size is not None:
            dataset = dataset.select(range(min(dataset_size, len(dataset))))

        workspace_folder = output_path
        os.makedirs(os.path.join(workspace_folder, "images"), exist_ok=True)
        client = vlm_service
        sem = asyncio.Semaphore(64)

        async def process_example(idx, ex):
            async with sem:
                image_paths = []

                # 1. Handle image - can be PIL object or path string
                raw_image = ex.get('image', None)

                if raw_image is not None:
                    # Check if image is a PIL Image object
                    if hasattr(raw_image, 'mode') and hasattr(raw_image, 'save'):
                        # It's a PIL Image
                        if raw_image.mode != 'RGB':
                            raw_image = raw_image.convert('RGB')
                        image_path = os.path.join(workspace_folder, "images", f"pmcoa_{idx}.jpeg")
                        raw_image.save(image_path)
                        image_paths.append(image_path)
                    elif isinstance(raw_image, str):
                        # It's a path string, join with root_path
                        image_path = os.path.join(root_path, raw_image)
                        if os.path.exists(image_path):
                            image_paths.append(image_path)

                if len(image_paths) == 0:
                    return None

                # 2. Handle caption as answer and build question
                caption = ex.get('caption', '').strip()
                question = "What is shown in this medical figure?"
                answer = caption

                # 3. Rewrite content or use original
                if rewrite:
                    image_content = encode_image_paths_to_base64(image_paths)
                    content_text = caption
                    prompt_text = REWRITE_QUESTION_PROMPT.format(content=content_text)
                    messages = image_content + [{"type": "text", "text": prompt_text}]
                    response = await client.async_generate_batch([messages], temperature=0.7)
                    response_text = response[0].strip()

                    # Parse JSON response
                    try:
                        # Try to extract JSON from response
                        json_match = response_text
                        if '```json' in response_text:
                            json_match = response_text.split('```json')[1].split('```')[0].strip()
                        elif '```' in response_text:
                            json_match = response_text.split('```')[1].split('```')[0].strip()

                        content = json.loads(json_match)
                    except (json.JSONDecodeError, IndexError) as e:
                        logger.warning(f"Failed to parse JSON response for idx {idx}: {e}")
                        # Fallback to original content
                        content = {
                            "question": question,
                            "content": caption,
                            "key_words": []
                        }

                    chunk_id = compute_mdhash_id(content.get('content', caption), "chunk_")

                    # 4. Extract answer label for evaluation
                    answer_label = extract_catogorical_answer(answer) if answer else ""

                    new_example = {
                        "question": content.get('question', question),
                        "image_paths": image_paths,
                        "content": content.get('content', caption),
                        "index": idx,
                        "chunk_id": chunk_id,
                        "dataset_id": dataset_name,
                        "answer": answer,
                        "answer_label": answer_label,
                        "key_words": content.get('key_words', []),
                        "source": "pmc_oa",
                        "task": "medical_figure_captioning",
                        "credential": "no"
                    }
                else:
                    chunk_id = compute_mdhash_id(caption, "chunk_")
                    new_example = {
                        "question": question,
                        "image_paths": image_paths,
                        "content": caption,
                        "index": idx,
                        "chunk_id": chunk_id,
                        "dataset_id": dataset_name,
                        "answer": answer,
                        "answer_label": "",
                        "key_words": [],
                        "source": "pmc_oa",
                        "task": "medical_figure_captioning",
                        "credential": "no"
                    }
                return new_example

        async def run_all():
            return await asyncio.gather(
                *[process_example(idx, ex) for idx, ex in enumerate(dataset)]
            )

        results = asyncio.run(run_all())

        # Filter out None results
        results = [r for r in results if r is not None]

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
        """Evaluate the prediction against the ground truth answer.

        For PMC-OA dataset, we compare the generated caption with the reference caption.
        Returns 1 if they match (or contain similar key information), 0 otherwise.
        """
        assistant_message = conversation['messages'][-1]
        last_text_part = assistant_message['content'][-1]['text']
        pred_answer = last_text_part.strip()
        gold_answer = item['answer']

        # For captioning task, we check if the prediction contains key information
        # Simple exact match for now - can be extended with semantic similarity
        return int(pred_answer.lower().strip() == gold_answer.lower().strip())


if __name__ == "__main__":
    openai_service = OpenAIVLMService(model_name="Qwen3-VL-4B-Instruct", api_key="EMPTY", url="http://localhost:8000/v1")
    output_path = "datasets/preprocessed_datasets/pmcoa"
    PMCOADataset.process_dataset("datasets/pmcoa", output_path, openai_service, "datasets/pmcoa", dataset_size=100)

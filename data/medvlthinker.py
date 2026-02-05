# MedVLThinker-Eval dataset
# HuggingFace dataset: UCSC-VLAA/MedVLThinker-Eval
# Use this dataset for evaluation
#
# Original data format keys:
# - images: list[PIL.Image.Image]
# - question: str
# - options: str (e.g., "A. option1\nB. option2\nC. option3\nD. option4")
# - answer_label: str (e.g., "A", "B", "C", "D")
# - answer: str (full answer text)
# - dataset_name: str
# this dataset don't need process
from datasets import load_dataset, load_from_disk
import os
from torch.utils.data import Dataset, Subset
import asyncio
from typing import List, Dict, Any
import sys
import json
import re

if __name__ == "__main__":
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from data.common import REWRITE_CONTENT_PROMPT
from MMRAG.model_service.vlm_service import OpenAIVLMService
from MMRAG.utils import compute_mdhash_id, extract_catogorical_answer, logger, encode_image_paths_to_base64


class MedVLThinkerEval(Dataset):
    """
    MedVLThinker-Eval dataset for multi-modal medical VQA evaluation.

    This dataset contains medical visual question answering examples with multiple choice options.
    Data source: HuggingFace dataset "UCSC-VLAA/MedVLThinker-Eval"
    """

    def __init__(self, dataset_name: str, dataset_size: int = None):
        """
        Initialize the MedVLThinkerEval dataset.

        Args:
            dataset_name: Name of the specific sub-dataset to load (e.g., "MMMU", "MedXpertQA", etc.)
                         This corresponds to the 'dataset_name' field in the original data.
            dataset_size: Optional limit on the number of examples to load.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_path = f"./datasets/preprocessed_datasets/medvlthinker_{self.dataset_name}"

        if os.path.exists(self.dataset_path):
            logger.info(f"Loading processed dataset from {self.dataset_path}.")
            self.dataset = load_from_disk(self.dataset_path)
            if dataset_size:
                self.dataset = Subset(self.dataset, range(min(dataset_size, len(self.dataset))))
        else:
            self.dataset = load_dataset("UCSC-VLAA/MedVLThinker-Eval", split="test").filter(lambda x: x['dataset_name'] == dataset_name)

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a single example from the dataset.

        Args:
            index: Index of the example to retrieve.

        Returns:
            A dictionary containing the example data with keys:
            - question: str
            - image_paths: List[str]
            - content: str (rewritten medical case description)
            - index: int
            - chunk_id: str
            - dataset_id: str
            - answer: str (full answer text)
            - answer_label: str (e.g., "A", "B", "C", "D")
            - key_words: List[str]
            - source: str
            - task: str
            - credential: str
            - options: str (original options string)
        """
        return self.dataset[index]

    def evaluate(self, item: Dict[str, Any], conversation: Dict[str, Any]) -> int:
        """
        Evaluate the prediction against the ground truth answer.

        For MedVLThinker-Eval, we compare the predicted answer label (A/B/C/D)
        with the correct answer label.

        Args:
            item: The ground truth item from the dataset, containing 'answer_label' and 'options'.
            conversation: The model's response conversation, containing the predicted answer.

        Returns:
            1 if the prediction matches the ground truth, 0 otherwise.
        """
        # Extract the predicted answer from the conversation
        assistant_message = conversation['messages'][-1]
        last_text_part = assistant_message['content'][-1]['text']

        # Extract the predicted answer label (A/B/C/D)
        pred_answer = extract_catogorical_answer(last_text_part)

        # Get the ground truth answer label
        gold_answer = item.get('answer_label', '')

        return int(pred_answer == gold_answer)

    @staticmethod
    def parse_options(options_str: str) -> Dict[str, str]:
        """
        Parse the options string into a dictionary.

        Args:
            options_str: String containing options, e.g., "A. option1\nB. option2\nC. option3\nD. option4"

        Returns:
            Dictionary mapping option labels to option text.
        """
        options = {}
        if not options_str:
            return options

        # Split by newline and parse each option
        lines = options_str.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Match pattern like "A. option text" or "A) option text"
            match = re.match(r'^([A-E])[\.\)\s]+(.+)$', line)
            if match:
                label = match.group(1)
                text = match.group(2).strip()
                options[label] = text

        return options

    @staticmethod
    def process_dataset(
        dataset_name: str,
        output_path: str,
        vlm_service: OpenAIVLMService,
        dataset_size: int = None,
        rewrite: bool = True
    ) -> None:
        """
        Process the MedVLThinker-Eval dataset and save it to disk.

        This method:
        1. Loads the dataset from HuggingFace
        2. Filters by dataset_name if specified
        3. Rewrites questions into medical case descriptions using VLM
        4. Saves the processed dataset to the specified output path

        Args:
            dataset_name: Name of the specific sub-dataset to process (e.g., "MMMU", "MedXpertQA")
                         Use "all" to process all sub-datasets.
            output_path: Path where the processed dataset will be saved.
            vlm_service: VLM service instance for rewriting content.
            dataset_size: Optional limit on the number of examples to process.
            rewrite: Whether to rewrite questions into medical case descriptions.

        Output format keys:
            - question: str
            - image_paths: List[str]
            - content: str (rewritten medical case description)
            - index: int
            - chunk_id: str
            - dataset_id: str
            - answer: str
            - answer_label: str
            - key_words: List[str]
            - source: str
            - task: str
            - credential: str
            - options: str (original options)
        """
        # Load the full dataset from HuggingFace
        logger.info("Loading MedVLThinker-Eval dataset from HuggingFace...")
        full_dataset = load_dataset("UCSC-VLAA/MedVLThinker-Eval", split="test")

        # Filter by dataset_name if not "all"
        if dataset_name != "all":
            filtered_dataset = full_dataset.filter(lambda x: x.get('dataset_name') == dataset_name)
            logger.info(f"Filtered dataset '{dataset_name}' with {len(filtered_dataset)} examples.")
        else:
            filtered_dataset = full_dataset
            logger.info(f"Using all datasets with {len(filtered_dataset)} examples.")

        # Limit dataset size if specified
        if dataset_size is not None:
            filtered_dataset = filtered_dataset.select(range(min(dataset_size, len(filtered_dataset))))
            logger.info(f"Limited to {len(filtered_dataset)} examples.")

        # Create output directories
        workspace_folder = output_path
        os.makedirs(os.path.join(workspace_folder, "images"), exist_ok=True)

        client = vlm_service
        sem = asyncio.Semaphore(64)  # Limit concurrent requests

        async def process_example(idx: int, ex: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single example."""
            async with sem:
                # 1. Handle images - save PIL images to disk
                image_paths = []
                images = ex.get('images', [])

                if not isinstance(images, list):
                    images = [images] if images else []

                for i, image in enumerate(images):
                    if image is not None and hasattr(image, 'mode') and hasattr(image, 'save'):
                        # It's a PIL Image
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        image_path = os.path.join(workspace_folder, "images", f"medvlthinker_{idx}_img{i}.jpeg")
                        image.save(image_path)
                        image_paths.append(image_path)

                if len(image_paths) == 0:
                    logger.warning(f"No valid images for example {idx}, skipping.")
                    return None

                # 2. Extract question and options
                question = ex.get('question', '').strip()
                options_str = ex.get('options', '')
                answer = ex.get('answer', '').strip()
                answer_label = ex.get('answer_label', '').strip()

                # 3. Rewrite content using VLM service
                if rewrite:
                    try:
                        image_content = encode_image_paths_to_base64(image_paths)
                        prompt_text = REWRITE_CONTENT_PROMPT.format(question=question)
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

                            content_data = json.loads(json_match)
                            content = content_data.get('content', answer)
                            key_words = content_data.get('key_words', [])
                        except (json.JSONDecodeError, IndexError) as e:
                            logger.warning(f"Failed to parse JSON response for idx {idx}: {e}")
                            # Fallback to original answer
                            content = answer
                            key_words = []

                    except Exception as e:
                        logger.warning(f"Error rewriting content for idx {idx}: {e}")
                        content = answer
                        key_words = []
                else:
                    content = answer
                    key_words = []

                # 4. Compute chunk_id
                chunk_id = compute_mdhash_id(content, "chunk_")

                # 5. Build the processed example
                new_example = {
                    "question": question,
                    "image_paths": image_paths,
                    "content": content,
                    "index": idx,
                    "chunk_id": chunk_id,
                    "dataset_id": dataset_name,
                    "answer": answer,
                    "answer_label": answer_label,
                    "key_words": key_words if isinstance(key_words, list) else [],
                    "source": "medvlthinker",
                    "task": "medical_vqa",
                    "credential": "no",
                    "options": options_str
                }

                return new_example

        async def run_all():
            """Process all examples concurrently."""
            return await asyncio.gather(
                *[process_example(idx, ex) for idx, ex in enumerate(filtered_dataset)]
            )

        # Run the async processing
        results = asyncio.run(run_all())

        # Filter out None results (failed examples)
        results = [r for r in results if r is not None]

        # Convert to HuggingFace Dataset and save
        from datasets import Dataset as HFDataset
        dataset = HFDataset.from_list(results)
        logger.info(f"Processed dataset '{dataset_name}' with {len(dataset)} examples.")
        dataset.save_to_disk(output_path)
        logger.info(f"Saved processed dataset to {output_path}.")


if __name__ == "__main__":
    # Example usage
    openai_service = OpenAIVLMService(
        model_name="Qwen3-VL-4B-Instruct",
        api_key="EMPTY",
        url="http://localhost:8000/v1"
    )

    # Process a specific sub-dataset (e.g., "MMMU", "MedXpertQA", "VQA-RAD", etc.)
    output_path = "datasets/preprocessed_datasets/medvlthinker_MMMU"
    # MedVLThinkerEval.process_dataset(
    #     dataset_name="MMMU",
    #     output_path=output_path,
    #     vlm_service=openai_service,
    #     dataset_size=100
    # )

    # Load and test the processed dataset
    dataset = MedVLThinkerEval("MMMU-medical")
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        print(f"First example: {dataset[0]}")

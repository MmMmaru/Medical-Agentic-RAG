# harvard fair VL Med dataset
# sample
# {
#     "id": "data_00001",
#     "image_path": "slo_fundus_00001.jpg",
#     "filename": "data_00001.npz",
#     "report": "The 56 y/o female patient has optic nerve head drusen and narrow angles in both eyes, and a history of basilar artery aneurysms. No evidence of glaucoma mentioned.",
#     "age": 56.56,
#     "gender": "female",
#     "race": "black",
#     "ethnicity": "non-hispanic",
#     "language": "english",
#     "maritalstatus": "single",
#     "note": "ms. PERSON is a 56 yo woman presenting to establish care. per previous optometry note: 1. optic nerve head drusen, ou - longstanding - baseline visual field DATE_TIME reveals inferior arcuate od, inferior nasal and inferior temporal defects left eye hvf DATE_TIME: inferior nasal step ou, not very reliable 2. h/o basilar artery aneurysms - followed by neurologist at brigham every year, per pt; next exam in DATE_TIME - aneurysms at basilar artery tip and right a1 aneurysm - no neuro defects on visual field 3. narrow angles, ou recommend lpi od first, then os risks, benefits, and alternatives to surgery were discussed with the patient, including the potential risk of infection, bleeding, loss of vision, loss of eye, need for further surgery or laser, retinal detachment, change in glasses and reading glasses. no guarantees given. questions answered.? PERSON, pgy3 i saw and evaluated this patient and discussed the case as appropriate with the resident/fellow. i have reviewed the resident/fellow's notes and made any necessary changes. PERSON, md, facs",
#     "gpt4_summary": "The 56 y/o female patient has optic nerve head drusen and narrow angles in both eyes, and a history of basilar artery aneurysms. No evidence of glaucoma mentioned.",
#     "glaucoma": "yes",
#     "use": "training"
# }


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

class HarvardFairVLMedDataset(Dataset):
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
        """将Harvard FairVLMed数据集处理为统一格式，改写内容，保存dataset
        keys: 'question', 'image_paths', 'content', 'index', 'chunk_id', 'dataset_id', 'answer', 'answer_label', 'key_words', 'source', 'task', 'credential'
        """
        # Load from Hugging Face or local path
        dataset = load_dataset('json', dataset_name, split="train")

        if dataset_size is not None:
            dataset = dataset.select(range(min(dataset_size, len(dataset))))

        os.makedirs(output_path, exist_ok=True)
        client = vlm_service
        sem = asyncio.Semaphore(64)

        async def process_example(idx, ex):
            async with sem:
                image_paths = []

                # 1. Handle image_path - Harvard数据集图像已在本地，直接使用拼接后的路径
                raw_image_path = ex.get('image_path', '')
                if isinstance(raw_image_path, str) and raw_image_path:
                    # 将相对路径与root_path拼接
                    image_path = os.path.join(root_path, raw_image_path)
                    if os.path.exists(image_path):
                        image_paths.append(image_path)

                if len(image_paths) == 0:
                    return None

                # 2. Handle text - 基于report构建question，使用report或gpt4_summary作为answer
                report = ex.get('gpt4_summary', '').strip()
                gpt4_summary = ex.get('gpt4_summary', '').strip()

                # 构建问题：基于report创建医疗分析问题
                question = "Please analyze this retinal image and provide a diagnosis."
                if report:
                    # 可以基于report内容构建更具体的问题
                    question = f"Based on this retinal image, please analyze the patient's condition. {report[:100]}..."

                # 使用report或gpt4_summary作为answer
                answer = gpt4_summary if gpt4_summary else report
                from data.common import REWRITE_QUESTION_PROMPT
                # 3. Rewrite content or use original
                if rewrite:
                    image_content = encode_image_paths_to_base64(image_paths)
                    prompt_text = REWRITE_QUESTION_PROMPT.format(content=report)
                    messages = image_content + [{"type": "text", "text": prompt_text}]
                    response = await client.async_generate_batch([messages], temperature=0.7)
                    content = response[0].strip()
                    chunk_id = compute_mdhash_id(content, "chunk_")

                    # 4. Extract answer label for evaluation
                    # Harvard数据集不是多选题，answer_label直接使用answer
                    answer_label = extract_catogorical_answer(answer) if answer else ""

                    new_example = {
                        "question": question,
                        "image_paths": image_paths,
                        "content": content,
                        "index": idx,
                        "chunk_id": chunk_id,
                        "dataset_id": dataset_name,
                        "answer": answer,
                        "answer_label": answer_label,
                        "key_words": [],
                        "source": "Harvard-FairVLMed",
                        "task": "retinal_image_analysis",
                        "credential": "no"
                    }
                else:
                    content = report if report else gpt4_summary
                    chunk_id = compute_mdhash_id(content, "chunk_")
                    answer_label = extract_catogorical_answer(answer) if answer else ""

                    new_example = {
                        "question": question,
                        "image_paths": image_paths,
                        "content": content,
                        "index": idx,
                        "chunk_id": chunk_id,
                        "dataset_id": dataset_name,
                        "answer": answer,
                        "answer_label": answer_label,
                        "key_words": [],
                        "source": "Harvard-FairVLMed",
                        "task": "retinal_image_analysis",
                        "credential": "no"
                    }
                return new_example

        async def run_all():
            return await asyncio.gather(
                *[process_example(idx, ex) for idx, ex in enumerate(dataset)]
            )

        results = asyncio.run(run_all())
        # Filter out None values
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
        assistant_message = conversation['messages'][-1]
        last_text_part = assistant_message['content'][-1]['text']
        pred_answer = extract_catogorical_answer(last_text_part)
        gold_answer = item['answer_label']
        return int(pred_answer == gold_answer)

if __name__ == "__main__":
    openai_service = OpenAIVLMService(model_name="Qwen3-VL-4B-Instruct", api_key="EMPTY", url="http://localhost:8000/v1")
    output_path = "datasets/preprocessed_datasets/Harvard-FairVLMed"
    dataset_path = "datasets/Harvard_FairVLMed"
    # Harvard FairVLMed数据集在HuggingFace上的名称: "Harvard-FairVLMed/FairVLMed"
    HarvardFairVLMedDataset.process_dataset(dataset_path, output_path, openai_service, "datasets/Harvard-FairVLMed", dataset_size=100)
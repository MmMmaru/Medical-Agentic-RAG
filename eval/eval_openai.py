#!/usr/bin/env python
"""
run_med_vqa.py

Query a running vLLM OpenAI‑compatible server with Qwen‑2.5‑VL
(or another VLM) and evaluate predictions on a medical VQA dataset.

Usage example:
python run_med_vqa.py \
    --dataset-name AdaptLLM/biomed-VQA-benchmark \
    --subset PMC-VQA \
    --split test \
    --base-url http://localhost:8000/v1 \
    --model qwen2.5-vl-7b-instruct \
    --out-file preds_pmc_test.json
"""
import dotenv

dotenv.load_dotenv(override=True)
import base64
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import click
import openai
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import asyncio


# ------------------------- helper functions ------------------------- #
def pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode PIL image to base64 string suitable for OpenAI vision input."""
    buffer = io.BytesIO()
    # ensure 3‑channel input; Qwen handles PNG/JPEG
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buffer, format=fmt)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def build_messages(
    image_b64: str,
    question: str,
    options: Dict[str, str],
    system_prompt: str,
    cot: bool,
) -> List[Dict[str, Any]]:
    """Construct OpenAI vision chat messages with CoT prompt."""
    prompt_lines = [
        """You should provide your thoughts within <think> </think> tags, then answer with just one of the options below within <answer> </answer> tags (For example, if the question is \n’Is the earth flat?\n A: Yes\nB: No’, you should answer with <think>...</think> <answer>B: No</answer>)."""
    ]
    qtxt = f"Question: {question}\nOptions:{options}"
    if cot:
        qtxt += "\n\nLet's think step by step."

    return [
        # {
        #     "role": "system",
        #     "content": system_prompt
        #     or "You are ChatGPT, a large language model, assisting with medical vision‑language reasoning.",
        # },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": "\n".join(prompt_lines) + "\n\n" + qtxt},
            ],
        },
    ]


def extract_answer(text: str) -> str:
    """Return A/B/C/D if found in model output, else ''."""
    content_match = re.search(r"<answer>(.*?)</answer>", text)
    given_answer = content_match.group(1).strip() if content_match else text.strip()
    final_answer = re.search(r"\b([A-E])\b", given_answer)
    return final_answer.group(1) if final_answer else ""

async def main():
    
    dataset_name: str = "UCSC-VLAA/MedVLThinker-Eval"
    subset: str = None
    split: str = "test"
    base_url: str = "http://localhost:8000/v1"
    model_name: str = "Qwen3-VL-4B-Instruct"
    out_file: str = "preds.json"
    batch_size: int = 128
    cot: bool = True
    dataset_size = 100
    num_proc: int = 16

    # Configure OpenAI client for local vLLM server
    openai_api_key = "EMPTY"
    openai_api_base = base_url

    client = openai.AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # Load dataset
    ds = load_dataset(dataset_name, subset)[split]
    if dataset_size is not None:
        ds = ds.select(range(int(dataset_size)))
    # TODO: resize image if needed

    ds = ds.map(
        resize_image_to_qwen2_5_vl,
        num_proc=num_proc,
        desc="Resizing images to fit Qwen2.5-VL requirements",
    )
    print(f"Loaded {len(ds)} examples from {dataset_name}.")
    async def process_example(idx: int, ex: dict) -> dict:
        img: Image.Image = ex["images"][0]  # PIL Image
        b64_img = pil_to_base64(img)

        messages = build_messages(
            image_b64=b64_img,
            question=ex["question"],
            options=ex["options"],
            system_prompt="",
            cot=cot,
        )

        resp = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=2048,
        )
        output_text = resp.choices[0].message.content.strip()

        pred_letter = extract_answer(output_text)
        correct = pred_letter == ex["answer_label"]

        return {
            "id": idx,
            "question": ex["question"],
            "options": ex["options"],
            "response": output_text,
            "pred_letter": pred_letter,
            "answer_label": ex["answer_label"],
            "correct": correct,
        }

    results = []
    results = await asyncio.gather(
        *[process_example(idx, ex) for idx, ex in enumerate(ds)]
    )

    # Save JSON lines (one dict per row)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(results)
    acc = sum(r["correct"] for r in results) / total
    print(f"\nSaved {total} records to {out_file}. Accuracy: {acc:.2%}")


# https://github.com/Yuxiang-Lai117/Med-R1/blob/53e46ba24e04d7d7705db4750551786a93e96493/src/r1-v/local_scripts/prepare_hf_data.py#L144
def resize_image_to_qwen2_5_vl(example):
    # for Qwen2-VL-2B's processor requirement
    # Assuming the image is in a format that can be checked for dimensions
    # You might need to adjust this depending on how the image is stored in your dataset
    image = example["images"][0]
    if image.height < 28 or image.width < 28:
        image = resize_shortest_dim(image, target_size=28)

    if image.height > 1024 or image.width > 1024:
        image = resize_longest_dim(image, target_size=1024)
    example["images"][0] = image

    return example


def resize_shortest_dim(image, target_size=28):
    """Resize the shortest dimension of the image to the specified size."""
    width, height = image.size
    if width < height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    return image.resize((new_width, new_height), Image.BILINEAR)


# resize the longest dim to 384
def resize_longest_dim(image, target_size=1024):
    width, height = image.size
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    return image.resize((new_width, new_height), Image.BILINEAR)


if __name__ == "__main__":
    asyncio.run(main())

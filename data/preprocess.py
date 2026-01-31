# preprocess_vqa_dataset.py

import os
import argparse
from datasets import load_dataset, Dataset
from model_service.vlm_service import OpenAIVLMService
from utils import compute_mdhash_id, logger

def save_image_and_get_path(image, chunk_id, img_idx, images_dir):
    """保存单张图像并返回路径"""
    image_path = os.path.join(images_dir, f"{chunk_id}_img{img_idx}.jpeg")
    image.save(image_path, format="JPEG", quality=95)
    return image_path

def main(args):
    # 创建输出目录
    output_dir = os.path.join("./datasets/processed_datasets", args.dataset_name.replace("/", "_"))
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    logger.info(f"Loading raw dataset: {args.dataset_name}")
    raw_dataset = load_dataset(args.dataset_name, split="train")

    # 如果指定了大小，就截断
    if args.max_samples is not None:
        raw_dataset = raw_dataset.select(range(min(args.max_samples, len(raw_dataset))))

    # 初始化 VLM 服务（仅在需要生成答案时使用）
    vlm_service = None
    if args.use_vlm_for_answer:
        logger.info("Initializing VLM service...")
        vlm_service = OpenAIVLMService(
            model_name=args.model_name,
            api_key=args.api_key,
            url=args.vlm_url
        )

    processed_examples = []

    for idx, example in enumerate(raw_dataset):
        logger.info(f"Processing sample {idx + 1}/{len(raw_dataset)}")

        question = example["question"]
        images = example["images"]  # 假设是 PIL.Image 列表

        # === 步骤1: 获取答案 ===
        gold_answer = example.get("answer", "").strip()
        if args.use_vlm_for_answer or not gold_answer:
            # 如果没有标准答案，或强制用 VLM 生成
            try:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        *[{"type": "image"} for _ in images]
                    ]
                }]
                response = vlm_service.generate(messages=messages, images=images)
                pred_answer = response.strip()
                answer_to_use = pred_answer
            except Exception as e:
                logger.error(f"Failed to generate answer for sample {idx}: {e}")
                answer_to_use = ""
        else:
            answer_to_use = gold_answer

        # === 步骤2: 构造 content ===
        content = f"{question}<image>\nAnswer:{answer_to_use}"

        # === 步骤3: 生成唯一 ID ===
        chunk_id = compute_mdhash_id(content, prefix="chunk_")

        # === 步骤4: 保存图像 ===
        image_paths = []
        for img_idx, img in enumerate(images):
            path = save_image_and_get_path(img, chunk_id, img_idx, images_dir)
            image_paths.append(path)

        # === 步骤5: 构建新样本 ===
        new_example = {
            "question": question,
            "image_paths": image_paths,
            "content": content,
            "index": idx,
            "chunk_id": chunk_id,
            "dataset_id": args.dataset_name,
            "answer": answer_to_use,
            "answer_label": gold_answer if gold_answer else answer_to_use,  # 优先用原始标签
            "key_words": [],  # 可按需填充
            "source": args.dataset_name
        }
        processed_examples.append(new_example)

    # 转为 Dataset 并保存
    processed_dataset = Dataset.from_list(processed_examples)
    processed_dataset.save_to_disk(output_dir)
    logger.info(f"✅ Processed dataset saved to: {output_dir}")
    logger.info(f"   Total samples: {len(processed_dataset)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess VQA dataset with VLM.")
    parser.add_argument("--dataset_name", type=str, required=True, help="HuggingFace dataset name, e.g., UCSC-VLAA/MedVLThinker-Eval")
    parser.add_argument("--use_vlm_for_answer", action="store_true", help="Use VLM to generate answers (if no ground truth)")
    parser.add_argument("--model_name", type=str, default="google/medgemma-1.5-5b-it")
    parser.add_argument("--vlm_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples to process")
    
    args = parser.parse_args()
    main(args)
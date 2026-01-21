# login huggingface
import re
import torch
import dotenv
import time
dotenv.load_dotenv()

huggingface_token = os.getenv("HF_TOKEN")
from huggingface_hub import login
login(token=huggingface_token)  # 替换为你的实际 token

def load_dataset(file_path = "datasets/MedQA/data_clean/questions/Mainland/train.jsonl"):
    import json
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): 
                data_list.append(json.loads(line.strip()))
    print(len(data_list))
    print(data_list[:5])
    return data_list

def load_model():
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.float16)
    model.eval()
    print("✅ processor model loaded")
    return processor, model, device

def logger(content):
    print(content)
def format_messages(sample, include_options: bool = True) -> str:
    """
    Args:
      sample: Dict
    将 MedQA 样本转换为 Zero-shot Prompt。
    可以根据你的微调格式修改此模板。
    """
    question = sample['question']
    options = sample['options']
    image_path = sample.get('image', None)
    
    # 构建选项文本
    options_text = ""
    for key, value in sorted(options.items()):
        options_text += f"{key}. {value}\n"
    
    # Prompt 模板
    prompt = (
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n"
        f"Answer the question by selecting the correct option letter (A, B, C, D, or E).\n"
        f"Answer:"
    )
    if image_path:
        messages = [
            {
                "role":"user",
                "content":[
                    {
                        "type":"image",
                        "url": image_path
                    },
                    {
                        "type":"text",
                        "text": prompt
                    }
                ]
            }
        ]
    else:
        messages = [
            {
                "role":"user",
                "content":[
                    {
                        "type":"text",
                        "text":prompt
                    }
                ]
            }
        ]

    return messages
def extract_option(response: str) -> str:
    """
    使用正则表达式从模型输出中提取选项 (A-E)。
    策略：优先匹配 'Answer: A'，如果失败则匹配最后一个出现的选项字母。
    """
    # 1. 强匹配模式：Answer is A / The answer is A
    pattern_strong = r"(?:Answer|Option)\s*(?:is|:)?\s*([A-E])\b"
    match = re.search(pattern_strong, response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # 2. 弱匹配模式：如果模型只输出了 "A" 或者结尾是 "A"
    # 查找最后出现的单个大写字母 A-E，且周围没有其他字母
    matches = re.findall(r"\b([A-E])\b", response)
    if matches:
        return matches[-1].upper() # 取最后一个提到的选项通常比较稳健
    
    return "UNKNOWN" # 无法提取

data_list = load_dataset("datasets/MedQA/data_clean/questions/Mainland/test.jsonl")
processor, model, device = load_model()
score = 0

for step, data in enumerate(data_list[:100]):
    messages = format_messages(data)
    inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    start = time.time()
    generated_ids = model.generate(
        **inputs,
        do_sample=False,              # greedy decode
        max_new_tokens=512
    )
    end = time.time()
    generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
    output_text = processor.batch_decode(
       generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
    response_option = extract_option(output_text[0])
    if response_option == data["answer_idx"]:
        score += 1

    logger(f'{step}/{len(data_list)}, response:{output_text}, ground_truth:{data["answer_idx"]}: {data["answer"]}\n')
    throughput = len(generated_ids[0]) / (end - start)
    logger(f'Throughput: {throughput:.2f} tokens/sec\n')
    
logger("="*70)
acc = score/100
print(acc)

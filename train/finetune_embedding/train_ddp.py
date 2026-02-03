import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import functools
import argparse
from peft import LoraConfig, PeftModel, get_peft_model
from datasets import load_from_disk
from ...data.medmax import MedMaxDataset
from qwen3_vl_embedding import Qwen3VLForEmbedding, Qwen3VLEmbeddingProcessor

# from MMRAG.utils import logger # Avoid import issues if not set up

def cleanup():
    """Destroy distributed process group"""
    dist.destroy_process_group()

def load_lora_model(model_name, peft_config=None):
    model = Qwen3VLForEmbedding.from_pretrained(model_name)
    processor = Qwen3VLEmbeddingProcessor(model_name)
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
    print()
    return model, processor

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        self.delta = 0.1

    def forward(self, query_embeds, doc_embeds):
        """
        Args:
            query_embeds: (batch_size, dim)
            doc_embeds: (batch_size, dim)
        """
        B,D = query_embeds.shape
        # Positive logits: (batch_size, 1)
        # l_pos = (query_embeds * pos_embeds).sum(dim=-1, keepdim=True) / self.temperature
        
        # In-batch negatives calculation
        # Similarity matrix: (batch_size, batch_size)
        sim_matrix = torch.matmul(query_embeds, doc_embeds.t())
        
        logits = torch.diag(sim_matrix, 0)
        sim_matrix[torch.arange(B), torch.arange(B)] = 0
        hard_negatives = torch.topk(sim_matrix, 10, dim=-1)[0]
        hard_negatives = torch.where(hard_negatives > logits + self.delta, hard_negatives, 0)

        exp_logits = torch.exp(logits/self.temperature)
        denominator = torch.exp(hard_negatives/self.temperature).sum(-1) + exp_logits
        infoNCE_loss = -torch.mean(torch.log(exp_logits/denominator))
            
        return infoNCE_loss

class MixedModalDataset(Dataset):
    def __init__(self, data_path, processor):
        self.processor = processor
        self.dataset = load_from_disk(data_path) # 
        self.system_prompt = "You are an helpful assistant"

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item # Dict[str, str]

def collate_fn(batch, processor):
    
    doc_items = [{
        "text": item['content'],
        "image": item["image_paths"]
    }for item in batch]

    query_items = [{
        "text": item['question']
    }for item in batch]
    doc_inputs = processor.process(doc_items)
    query_inputs = processor.process(query_items)

    return {
        "query": query_inputs,
        "doc": doc_inputs
    }

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, default="Qwen/Qwen3-VL-Embedding-2B", help="path or name to model")
    parser.add_argument('--train_data', required=True, type=str, help="path to tokenized train data")
    parser.add_argument('--val_data', default="", type=str, help="path to tokenized val data")

    parser.add_argument('--ckpt', required=True, type=str, help="finetuning checkpoint")
    parser.add_argument('--ds', required=True, type=str, help="deepspeed config")
    parser.add_argument('--output_dir', required=True, type=str, help="output directory")

    parser.add_argument('--resume', default=False, action='store_true', help='resume from the last checkpoint')         

    parser.add_argument('--lora', default=False, action='store_true', help='lora finetuning')         
    parser.add_argument('--lora_r', type=int, default=16, help="lora r")         
    parser.add_argument('--lora_alpha', type=int, default=16, help="lora alpha")         
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="lora dropout")         

    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate")         
    parser.add_argument('--epoch', type=int, default=1, help="num of epochs")         
    parser.add_argument('--grad_acc', type=int, default=1, help="gradient accumulation steps")         
    parser.add_argument('--steps', type=int, default=-1, help="num of training steps")         
    parser.add_argument('--bs', type=int, default=1, help="per device batch size")         
    parser.add_argument('--save_strategy', default="no", type=str, choices=["no", "epoch", "steps"], help="save strategy")
    parser.add_argument('--save_steps', type=float, default=0.1, help="save checkpoints every save_steps of the training run")         
    parser.add_argument('--logging_steps', type=float, default=0.001, help="logging at every logging_steps of the total training steps")         
    parser.add_argument('--eval_steps', type=float, default=0.1, help="evaluate at every evaluation steps of the total training steps")         
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help="warmup ratio")         
    parser.add_argument('--lr_scheduler', default="cosine", type=str, help="lr scheduler")

    parser.add_argument('--bf16', default=False, action='store_true', help='bf16')         
    parser.add_argument('--fp16', default=False, action='store_true', help='fp16')         

    parser.add_argument('--wandb', default=False, action='store_true', help='wandb or not')         
    parser.add_argument('--wandb_entity', default="", type=str, help='wandb entity')         
    parser.add_argument('--name', default="", type=str, help="wandb experiment name")

    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    peft_config = None
    if args.lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj","gate_proj"],
            task_type="CAUSAL_LM",
        )
    # 1. Model & Processor
    model, processor = load_lora_model(args.model_name, peft_config)

    # 2. FSDP Wrap
    # Identify transformer layers for auto wrap
    # For Qwen2VL, the layer class is usually Qwen2VLDecoderLayer
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer
    
    model.to(device)
    model = torch.compile(model)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = ContrastiveLoss().to(device)
    
    # 3. Data
    dataset = MixedModalDataset(args.train_data, processor)
    sampler = DistributedSampler(dataset, rank=rank)
    dataloader = DataLoader(
        dataset, 
        batch_size=64, 
        sampler=sampler, 
        collate_fn=functools.partial(collate_fn, processor=processor)
    )

    # 4. Training Loop
    model.train()
    num_epochs = 3
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for step, inputs in enumerate(dataloader):
            optimizer.zero_grad()
            query_inputs = {k: v.to(device) for k,v in inputs['query'].items()}
            doc_inputs = {k: v.to(device) for k,v in inputs['doc'].items()}
            # Forward
            # Note: We need to run forward twice: once for queries, once for docs
            query_embed = model(**query_inputs)
            doc_embed = model(**doc_inputs)
            
            loss = criterion(query_embed, doc_embed)
            
            loss.backward()
            optimizer.step()
            
            if rank == 0 and step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    # Save
    if rank == 0:
        print("Saving model...")
        output_dir = "./checkpoint"
        model.save_pretrained(output_dir)
        # torch.save(model.module.state_dict(), "checkpoint/checkpoint.pt")
        # Save cpu_state...
        
        # 模型加载流程
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # from peft import PeftModel

        # # 1. 加载原始基座模型
        # base_model_path = "path/to/base_model" # 例如 meta-llama/Llama-2-7b-hf
        # model = AutoModelForCausalLM.from_pretrained(base_model_path)
        # tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # # 2. 加载你保存的 LoRA 权重
        # lora_weights_path = "./my_lora_weights"
        # model = PeftModel.from_pretrained(model, lora_weights_path)

        # 现在可以使用 model 进行推理了
    cleanup()

if __name__ == "__main__":
    main()
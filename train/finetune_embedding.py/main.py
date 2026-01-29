import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AutoProcessor, Qwen2VLModel, Qwen2VLConfig
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    CPUOffload,
    MixedPrecision,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
import functools

from qwen3_vl_embedding import Qwen3VLEmbedder

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from MMRAG.utils import logger # Avoid import issues if not set up

def setup():
    """Initialize distributed environment"""
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

def cleanup():
    """Destroy distributed process group"""
    dist.destroy_process_group()

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, query_embeds, pos_embeds, neg_embeds=None):
        """
        Args:
            query_embeds: (batch_size, dim)
            pos_embeds: (batch_size, dim)
            neg_embeds: (batch_size, num_neg, dim) or None
        """
        # Normalize
        query_embeds = torch.nn.functional.normalize(query_embeds, p=2, dim=-1)
        pos_embeds = torch.nn.functional.normalize(pos_embeds, p=2, dim=-1)
        
        # Positive logits: (batch_size, 1)
        # l_pos = (query_embeds * pos_embeds).sum(dim=-1, keepdim=True) / self.temperature
        
        # In-batch negatives calculation
        # Similarity matrix: (batch_size, batch_size)
        sim_matrix = torch.matmul(query_embeds, pos_embeds.t()) / self.temperature
        
        labels = torch.arange(query_embeds.size(0), device=query_embeds.device)
        
        # If we have hard negatives
        if neg_embeds is not None:
            # neg_embeds: (batch_size, num_neg, dim)
            # l_neg = (query_embeds.unsqueeze(1) * neg_embeds).sum(dim=-1) / self.temperature
            # logits = torch.cat([l_pos, l_neg], dim=1)
            # labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            pass # TODO: Implement explicit hard negative logic merging with in-batch
            
        return self.cross_entropy(sim_matrix, labels)

class MixedModalDataset(Dataset):
    def __init__(self, data_path, processor):
        self.data = [] # Load from data_path
        self.processor = processor
        # Placeholder data generation
        for _ in range(100):
            self.data.append({
                "query_text": "Describe this chest x-ray.",
                "pos_text": "The chest x-ray shows clear lungs.",
                "pos_image": None # Optional path
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Process query
        query_text = item["query_text"]
        # Process positive (text + optional image)
        pos_text = item["pos_text"]
        
        # We need to process independently? Or return raw texts to be collated?
        # Better to return raw and collate in batch
        return item

def collate_fn(batch, processor):
    # Prepare queries
    queries = [b['query_text'] for b in batch]
    query_inputs = processor(text=queries, padding=True, return_tensors="pt")
    
    # Prepare positives
    pos_texts = [b['pos_text'] for b in batch]
    # Handle images if present...
    pos_inputs = processor(text=pos_texts, padding=True, return_tensors="pt")
    
    return query_inputs, pos_inputs

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, default="Qwen/Qwen3-VL-Embedding-2B", help="path to model")
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

    setup()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    model_path = os.path.expanduser("~/.cache/modelscope/models/qwen/Qwen3-VL-Embedding-2B")
    # Fallback to local path if needed or handle download
    if not os.path.exists(model_path):
        # assume it's in current dir for demo or pass via env
        model_path = "Qwen/Qwen2-VL-2B-Instruct" # Placeholder for actual name

    if rank == 0:
        print(f"Loading model from {model_path}")

    # 1. Model & Processor
    model = Qwen3VLEmbedder(args.model_path)
    
    # 2. FSDP Wrap
    # Identify transformer layers for auto wrap
    # For Qwen2VL, the layer class is usually Qwen2VLDecoderLayer
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer
    
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2VLDecoderLayer},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD, 
        limit_all_gathers=True,
    )

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = ContrastiveLoss()

    # 3. Data
    dataset = MixedModalDataset("./datasets/processed_datasets", processor)
    sampler = DistributedSampler(dataset, rank=rank)
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        sampler=sampler, 
        collate_fn=functools.partial(collate_fn, processor=processor)
    )

    # 4. Training Loop
    model.train()
    num_epochs = 3
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for step, (query_inputs, pos_inputs) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Move to device
            q_ids = query_inputs['input_ids'].to(local_rank)
            q_mask = query_inputs['attention_mask'].to(local_rank)
            
            p_ids = pos_inputs['input_ids'].to(local_rank)
            p_mask = pos_inputs['attention_mask'].to(local_rank)
            
            # Forward
            # Note: We need to run forward twice: once for queries, once for docs
            q_embeds = model(q_ids, q_mask)
            p_embeds = model(p_ids, p_mask)
            
            loss = criterion(q_embeds, p_embeds)
            
            loss.backward()
            optimizer.step()
            
            if rank == 0 and step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    # Save
    if rank == 0:
        print("Saving model...")
        save_policy = StateDictType.FULL_STATE_DICT
        with FSDP.state_dict_type(model, save_policy):
            cpu_state = model.state_dict()
        # Save cpu_state...
        
    cleanup()

if __name__ == "__main__":
    main()
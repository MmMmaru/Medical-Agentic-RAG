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

class QwenVLEmbeddingWrapper(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # Load config first to check
        self.model = Qwen2VLModel.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, pixel_values=None, image_grid_thw=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw
        )
        # Use last hidden state of the last token (EOS) or mean pooling
        # Assuming the last token is EOS or representative
        last_hidden_state = outputs.last_hidden_state
        
        # Extract embedding from the last token
        # input_ids shape: (batch, seq_len)
        # Gather the last non-padding token
        # For simplicity, if left-padded or right-padded, we need attention mask
        # Here we assume standard right-padding, so we take the last token that is attended
        
        # Check if eos_token is at the end?
        # Let's do Mean Pooling masked by attention mask for robust embedding
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        
        return embedding

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
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = QwenVLEmbeddingWrapper(model_path)
    
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
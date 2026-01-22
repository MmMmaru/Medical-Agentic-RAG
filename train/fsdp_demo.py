import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

# FSDP 核心组件
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, # 对应 ZeRO 的不同阶段
    CPUOffload,       # 是否把参数卸载到 CPU
)
# 自动包裹策略 (非常重要，否则 FSDP 退化为 DDP)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

def setup():
    """初始化分布式环境 (NCCL)"""
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

def cleanup():
    """销毁进程组"""
    dist.destroy_process_group()

def get_model():
    """构建一个玩具 Transformer 模型"""
    # 这里用一个小模型演示，实际中这通常是 Llama 或 Qwen
    model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=2)
    return model

def main():
    setup()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # 1. 定义模型
    model = get_model()

    # 2. 定义 FSDP 的包裹策略 (关键步骤)
    # FSDP 需要知道怎么切分模型。最佳实践是按层切分。
    # 这里简单的使用“基于参数量”的策略：超过 20000 个参数的层就被单独切分
    my_auto_wrap_policy = lambda module, recurse, nonwrapped_child: size_based_auto_wrap_policy(
        module, recurse, nonwrapped_child, min_num_params=20000
    )

    # 3. 使用 FSDP 包裹模型
    # 这步之后，模型在显存里已经是“碎片化”的状态了
    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        # ShardingStrategy.FULL_SHARD = ZeRO-3 (参数、梯度、优化器全切)
        # ShardingStrategy.SHARD_GRAD_OP = ZeRO-2 (只切梯度和优化器)
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        # 如果显存极度紧张，可以开启 CPU Offload (速度变慢，显存无限)
        cpu_offload=CPUOffload(offload_params=False) 
    )

    # 4. 定义优化器
    # 注意：一定要在模型被 FSDP 包裹之后再定义优化器！
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 5. 模拟训练循环
    # 构造假数据
    bs = 16
    seq_len = 32
    src = torch.rand(seq_len, bs, 512).to(local_rank)
    tgt = torch.rand(seq_len, bs, 512).to(local_rank)

    print(f"Rank {rank} start training...")

    for step in range(10):
        optimizer.zero_grad()
        
        # Forward: FSDP 会自动执行 All-Gather 把需要的层拼回来，算完再丢掉
        output = model(src, tgt)
        
        loss = output.mean()
        
        # Backward: FSDP 自动处理梯度的 Reduce-Scatter
        loss.backward()
        
        optimizer.step()

        if rank == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    main()
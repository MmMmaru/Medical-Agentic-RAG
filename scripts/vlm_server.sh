# with 5090
export CUDA_VISIBLE_DEVICES=0
vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --uvicorn-log-level info \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.4 \

# vllm serve google/medgemma-1.5-4b-it \
#     --host 127.0.0.1 \
#     --port 8000 \
#     --uvicorn-log-level info \
#     --dtype float32 \
#     --tensor-parallel-size 2 \
#     --pipeline-parallel-size 3 \
#     --max-model-len 4096 \
#     --max-num-seqs 128 \
#     --gpu-memory-utilization 0.9 \
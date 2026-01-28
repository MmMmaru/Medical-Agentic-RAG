export CUDA_VISIBLE_DEVICES=0
vllm serve .cache/modelscope/models/qwen/Qwen3-VL-Embedding-2B \
    --host 127.0.0.1 \
    --port 8001 \
    --uvicorn-log-level info \
    --dtype float16 \
    --runner pooling \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.8 \
    --served-model-name Qwen3-VL-Embedding-2B
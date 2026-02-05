export CUDA_VISIBLE_DEVICES=0
vllm serve Qwen/Qwen3-VL-Embedding-2B \
    --host 127.0.0.1 \
    --port 8001 \
    --uvicorn-log-level info \
    --dtype bfloat16 \
    --runner pooling \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.2 \

vllm serve Qwen/Qwen3-VL-Reranker-2B \
    --host 127.0.0.1 \
    --port 8002 \
    --uvicorn-log-level info \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.2 \
    --runner pooling \
    --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
    --chat-template examples/pooling/score/template/qwen3_vl_reranker.jinja

vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --uvicorn-log-level info \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.4 \
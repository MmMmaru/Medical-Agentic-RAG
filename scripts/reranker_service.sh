export CUDA_VISIBLE_DEVICES=0
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
    --chat-template scripts/template/qwen3_vl_reranker.jinja
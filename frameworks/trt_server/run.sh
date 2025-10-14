# edit
MAX_SEQ_LEN=8192
MAX_BATCH_SIZE=64
TENSOR_PARALLEL_SIZE=2

# do not edit
MAX_NUM_TOKENS=2048
MODEL_NAME=/home/homework/models
PORT=8080

# export CUDA_VISIBLE_DEVICES=2,3  # 测试时指定一下GPU

NVTX_DISABLE=1 TRTLLM_ENABLE_PDL=1 trtllm-serve serve $MODEL_NAME \
  --host 0.0.0.0 --port $PORT \
  --backend pytorch --max_batch_size $MAX_BATCH_SIZE --max_seq_len $MAX_SEQ_LEN \
  --max_num_tokens $MAX_NUM_TOKENS --tp_size $TENSOR_PARALLEL_SIZE --pp_size 1 \
  --kv_cache_free_gpu_memory_fraction 0.75 \
  --extra_llm_api_options /home/homework/extra_config/default_ifb.yaml

# NVTX_DISABLE=1 防止内存泄漏

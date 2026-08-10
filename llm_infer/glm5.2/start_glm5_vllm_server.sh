#!/bin/bash

# ========== 配置区 ==========
IMAGE=quay.io/ascend/vllm-ascend:glm5.2-a3
NAME=vllm-glm52-w8a8
PORT=9527                  # 服务监听端口，可按需修改
# ===========================

# 如果已有同名容器，强制删除（避免冲突）
docker rm -f "$NAME" 2>/dev/null || true

# 启动容器（后台运行，不自动删除）
docker run -d \
  --name "$NAME" \
  --net=host \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci8 \
  --device /dev/davinci9 \
  --device /dev/davinci10 \
  --device /dev/davinci11 \
  --device /dev/davinci12 \
  --device /dev/davinci13 \
  --device /dev/davinci14 \
  --device /dev/davinci15 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /data/models/GLM-5.2-w8a8:/root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8 \
  -e HCCL_OP_EXPANSION_MODE=AIV \
  -e OMP_PROC_BIND=false \
  -e OMP_NUM_THREADS=1 \
  -e HCCL_BUFFSIZE=200 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ASCEND_BALANCE_SCHEDULING=1 \
  -e VLLM_ASCEND_ENABLE_MLAPO=1 \
  -e VLLM_VERSION=0.21.0 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  "$IMAGE" \
vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w8a8 \
    --host 0.0.0.0 --port "$PORT" \
    --served-model-name GLM-5.2-w8a8 \
    --data-parallel-size 2 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --seed 1024 \
    --max-num-seqs 48 \
    --max-model-len 65535 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.96 \
    --quantization ascend \
    --async-scheduling \
    --additional-config '{"enable_npugraph_ex": true,"fuse_muls_add":true,"multistream_overlap_shared_expert":true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp"}' \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --reasoning-parser glm45

echo ""
echo "✅ 容器 '$NAME' 已启动，vLLM 服务正在后台运行（端口: $PORT, max-num-seqs=8, max-model-len=737280）。"
echo "📋 查看实时日志： docker logs -f $NAME"
echo "🛑 停止服务：     docker stop $NAME"
echo "▶️  重新启动：     docker start $NAME"

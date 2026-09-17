#!/bin/bash

# ========== 配置区 ==========
IMAGE=quay.io/ascend/vllm-ascend:glm-5.3-flash-a3
NAME=vllm-glm5.3-flash
PORT=8077                  # 服务监听端口，可按需修改
# ===========================

# 如果已有同名容器，强制删除（避免冲突）
docker rm -f "$NAME" 2>/dev/null || true

# 启动容器（后台运行，不自动删除）
# 官方 glm-5.3-flash-a3 镜像已内置 custom_transformer 算子环境，无需手动 source set_env.bash，
# 直接以 vllm serve 作为容器启动命令（与 glm5.2 部署脚本一致）。
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
  -v /data/models/GLM-5.3-Flash-w8a8:/root/.cache/modelscope/hub/models/Eco-Tech/GLM-5.3-Flash-w8a8 \
  -e HCCL_OP_EXPANSION_MODE=AIV \
  -e OMP_PROC_BIND=false \
  -e OMP_NUM_THREADS=1 \
  -e HCCL_BUFFSIZE=400 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  "$IMAGE" \
vllm serve /root/.cache/modelscope/hub/models/Eco-Tech/GLM-5.3-Flash-w8a8 \
    --host 0.0.0.0 --port "$PORT" \
    --served-model-name GLM-5.3-Flash-w8a8 \
    --data-parallel-size 1 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --seed 1024 \
    --safetensors-load-strategy prefetch \
    --max-num-seqs 16 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 16384 \
    --trust-remote-code \
    --quantization ascend \
    --limit-mm-per-prompt '{"image": 1, "video": 0}' \
    --gpu-memory-utilization 0.90 \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,16,32,64,96,128]}' \
    --api-server-count 1 \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --reasoning-parser glm45

echo ""
echo "✅ 容器 '$NAME' 已启动，vLLM 服务正在后台运行（端口: $PORT, TP=16, DP=1, max-model-len=1048576）。"
echo "⚠️  1M 上下文：能跑通 ≠ 检索准确，建议长文本召回质量做实测；生产稳妥档可回退到 262144~524288。"
echo "📋 查看实时日志： docker logs -f $NAME"
echo "🛑 停止服务：     docker stop $NAME"
echo "▶️  重新启动：     docker start $NAME"

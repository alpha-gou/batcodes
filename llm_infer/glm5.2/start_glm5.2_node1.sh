#!/bin/bash

# ========== 配置区（请根据实际情况修改） ==========
IMAGE=quay.io/ascend/vllm-ascend:glm5.2-a3   # 镜像名称:tag（与 Node 0 相同）
NAME=vllm-glm5.2-node1                # 容器名称
PORT=8077                             # 监听端口（实际不对外暴露，但需与节点 0 一致）
MODEL_NAME=GLM-5.2-W8A8               # 模型名称
NIC_NAME="eth0"                  # 当前节点网络接口名
LOCAL_IP="192.168.XX.X2"             # 当前节点 IP
NODE0_IP="192.168.XX.X1"             # 主节点（节点 0）的 IP
# ==================================================

docker rm -f "$NAME" 2>/dev/null || true

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
  -v /data/models/GLM-5.2-w8a8:/root/.cache/modelscope/hub/models/vllm-ascend/GLM5.2-w8a8 \
  -e VLLM_VERSION=0.21.0 \
  -e HCCL_OP_EXPANSION_MODE="AIV" \
  -e VLLM_ASCEND_BALANCE_SCHEDULING=0 \
  -e HCCL_IF_IP="$LOCAL_IP" \
  -e GLOO_SOCKET_IFNAME="$NIC_NAME" \
  -e TP_SOCKET_IFNAME="$NIC_NAME" \
  -e HCCL_SOCKET_IFNAME="$NIC_NAME" \
  -e OMP_PROC_BIND=false \
  -e OMP_NUM_THREADS=1 \
  -e HCCL_BUFFSIZE=400 \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ASCEND_ENABLE_MLAPO=1 \
  -e VLLM_ASCEND_ENABLE_FLASHCOMM1=1 \
  -e ASCEND_LAUNCH_BLOCKING=0 \
  "$IMAGE" \
vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM5.2-w8a8 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --headless \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-rpc-port 12980 \
    --data-parallel-address "$NODE0_IP" \
    --tensor-parallel-size 16 \
    --seed 1024 \
    --served-model-name "$MODEL_NAME" \
    --max-num-seqs 48 \
    --max-model-len 131072 \
    --max-num-batched-tokens 4096 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --quantization ascend \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --async-scheduling \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --reasoning-parser glm45 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_npugraph_ex": true,"fuse_muls_add":true,"multistream_overlap_shared_expert":true}' \
    --speculative-config '{"num_speculative_tokens": 15, "method": "deepseek_mtp"}'

echo ""
echo "✅ 节点 1 容器 '$NAME' 已启动（headless 模式）"
echo "📋 查看日志： docker logs -f $NAME"
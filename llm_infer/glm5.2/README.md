# GLM-5.2 部署指南（华为 Atlas A3）

本目录包含在华为 Atlas A3（Ascend 910C，8 卡 × 128GB）机器上通过 **vLLM-Ascend** 部署 GLM-5.2-W8A8 模型的完整脚本，支持**单机部署**和**双机部署**两种模式。

> 参考资料：<https://ai.gitcode.com/Ascend-SACT/GLM5.2>

> [!WARNING]
> **⚠️ 启动参数必须包含以下三项，否则工具调用与推理能力将异常！**
>
> 在启动 vLLM 服务时，**务必**在启动命令中追加以下三个参数（脚本中已内置，请勿删除）：
>
> ```bash
> --tool-call-parser glm47 \
> --enable-auto-tool-choice \
> --reasoning-parser glm45 \
> ```
>
> | 参数 | 作用 |
> |------|------|
> | `--tool-call-parser glm47` | 指定 GLM-5.2 的工具调用解析器，缺失会导致 function call 解析失败 |
> | `--enable-auto-tool-choice` | 启用自动工具选择能力，缺失时模型无法自主决定调用工具 |
> | `--reasoning-parser glm45` | 指定 GLM-5.2 的推理链解析器，缺失会导致 thinking 内容输出异常 |
>
> **如需基于本目录脚本自行修改启动命令，请务必保留上述三个参数。**

---

## 目录结构

```
glm5.2/
├── README.md                     # 本文档
├── start_glm5_vllm_server.sh     # 单机部署脚本（16 卡）
├── start_glm5.2_node0.sh         # 双机部署 - 主节点（Node 0）
└── start_glm5.2_node1.sh         # 双机部署 - 从节点（Node 1，headless）
```

---

## 环境要求

| 项目 | 要求 |
|------|------|
| 硬件 | 华为 Atlas A3（Ascend 910C），每台 8 张 NPU 卡，每张 128GB 显存 |
| 操作系统 | Linux（推荐 EulerOS / Ubuntu 20.04+） |
| Docker | 20.10+ |
| Docker 镜像 | `quay.io/ascend/vllm-ascend:glm5.2-a3` |
| NPU 驱动 | 已安装并可通过 `npu-smi info` 正常查看设备 |
| 模型文件 | GLM-5.2-W8A8，存放于 `/data/models/GLM-5.2-w8a8` |

### 拉取镜像

```bash
docker pull quay.io/ascend/vllm-ascend:glm5.2-a3
```

脚本中已通过 `IMAGE=quay.io/ascend/vllm-ascend:glm5.2-a3` 指定镜像名称，无需手动替换。

### 模型下载

模型需提前下载至本机 `/data/models/GLM-5.2-w8a8` 目录。脚本中挂载路径为：

```
宿主机:  /data/models/GLM-5.2-w8a8
容器内:  /root/.cache/modelscope/hub/models/vllm-ascend/GLM5.2-w8a8   (单机脚本为 GLM-5.2-w8a8)
```

> **注意**：单机脚本与双机脚本的容器内路径略有不同，请保持与脚本一致，不要随意修改。

---

## 部署模式对比

| 参数 | 单机部署 | 双机部署 |
|------|---------|---------|
| 机器数量 | 1 台 | 2 台 |
| NPU 卡数 | 8 卡 | 16 卡（每台 8 卡） |
| Tensor Parallel | 8 | 16 |
| Data Parallel | 2 | 2 |
| 服务端口 | 9527 | 8077 |
| max-model-len | 65535 | 131072 |
| 投机解码 tokens | 3 | 5 |
| 最大并发序列数 | 48 | 48 |
| GPU 显存利用率 | 0.96 | 0.95 |

---

## 单机部署

适用于单台 Atlas A3（8 卡）的场景，使用脚本 `start_glm5_vllm_server.sh`。

### 启动步骤

```bash
# 1. 确认模型文件已就位
ls /data/models/GLM-5.2-w8a8

# 2. 确认 NPU 设备正常
npu-smi info

# 3. 启动服务
bash start_glm5_vllm_server.sh
```

### 关键参数说明

| 参数 | 值 | 说明 |
|------|----|------|
| `--tensor-parallel-size` | 8 | 将模型张量切分到 8 张 NPU 卡上 |
| `--data-parallel-size` | 2 | 2 组数据并行副本（8×2=16 卡） |
| `--enable-expert-parallel` | - | 启用 MoE 专家并行 |
| `--max-model-len` | 65535 | 最大上下文长度 |
| `--max-num-seqs` | 48 | 最大并发请求序列数 |
| `--gpu-memory-utilization` | 0.96 | KV Cache 显存使用比例 |
| `--quantization` | ascend | 使用 Ascend W8A8 量化 |
| `--speculative-config` | 3 tokens | 投机解码，加速推理 |
| `--tool-call-parser` | glm47 | **【必需】** GLM-5.2 工具调用解析器 |
| `--enable-auto-tool-choice` | - | **【必需】** 启用自动工具选择 |
| `--reasoning-parser` | glm45 | **【必需】** GLM-5.2 推理链解析器 |

> ⚠️ 上表中标注 **【必需】** 的三个参数对工具调用与推理能力至关重要，详见文档顶部的警告说明。启动命令中请勿遗漏。

### 环境变量说明

| 变量 | 值 | 说明 |
|------|----|------|
| `HCCL_OP_EXPANSION_MODE` | AIV | Ascend 通信算子展开模式 |
| `HCCL_BUFFSIZE` | 200 | HCCL 通信缓冲区大小（MB） |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | 1 | 启用均衡调度（单机开启） |
| `VLLM_ASCEND_ENABLE_MLAPO` | 1 | 启用 MLA 算子优化 |
| `VLLM_VERSION` | 0.21.0 | vLLM 版本号 |
| `VLLM_ENGINE_READY_TIMEOUT_S` | 7200 | 引擎就绪超时时间（秒），大模型加载较慢时可调大 |

---

## 双机部署

适用于两台 Atlas A3（共 16 卡）的场景，提供更大的上下文窗口（131072 tokens）和更高的吞吐量。使用脚本 `start_glm5.2_node0.sh`（主节点）和 `start_glm5.2_node1.sh`（从节点）。

### 网络要求

- 两台机器之间需要有高速互联网络（推荐 RoCE / InfiniBand）
- 两台机器的 NPU 需要通过 HCCN 配置好互联拓扑
- 确保两台机器的内网网卡可以互通，执行 `ip addr show | grep 'inet ' | grep -v 127.0.0.1` 可查看本机网卡名和内网 IP

### 启动步骤

**重要**：必须**先启动 Node 0，再启动 Node 1**。

#### 第一步：配置并启动主节点（Node 0）

在 Node 0 机器上执行前，修改 `start_glm5.2_node0.sh` 配置区：

```bash
IMAGE=quay.io/ascend/vllm-ascend:glm5.2-a3  # 镜像名称:tag
NAME=vllm-glm5.2-node0                       # 容器名称
PORT=8077                                    # 服务监听端口
MODEL_NAME=GLM-5.2-W8A8                      # 模型名称
NIC_NAME="eth0"                         # 网络接口名（内网网卡）
LOCAL_IP="192.168.XX.X1"                    # ★ 本机（Node 0）内网 IP
NODE0_IP="192.168.XX.X1"                    # ★ 主节点内网 IP（与 LOCAL_IP 相同）
```

> **获取网卡名和 IP**：执行 `ip addr show | grep 'inet ' | grep -v 127.0.0.1` 查看本机所有网卡及其内网 IP，根据输出填写 `NIC_NAME` 和 `LOCAL_IP`。

然后启动：

```bash
bash start_glm5.2_node0.sh
```

#### 第二步：配置并启动从节点（Node 1）

在 Node 1 机器上执行前，修改 `start_glm5.2_node1.sh` 配置区：

```bash
IMAGE=quay.io/ascend/vllm-ascend:glm5.2-a3  # 镜像名称:tag（与 Node 0 相同）
NAME=vllm-glm5.2-node1                # 容器名称
PORT=8077                             # 端口（需与 Node 0 一致）
MODEL_NAME=GLM-5.2-W8A8               # 模型名称
NIC_NAME="eth0"                  # 本机内网网卡名
LOCAL_IP="192.168.XX.X2"             # ★ 本机（Node 1）内网 IP
NODE0_IP="192.168.XX.X1"             # ★ 主节点（Node 0）内网 IP
```

> **获取网卡名和 IP**：执行 `ip addr show | grep 'inet ' | grep -v 127.0.0.1` 查看本机所有网卡及其内网 IP，根据输出填写 `NIC_NAME` 和 `LOCAL_IP`。`NODE0_IP` 填写主节点的内网 IP。

然后启动：

```bash
bash start_glm5.2_node1.sh
```

### 双机关键参数说明

| 参数 | Node 0 | Node 1 | 说明 |
|------|--------|--------|------|
| `--tensor-parallel-size` | 16 | 16 | 模型张量切分到 16 张卡（跨机，每机 8 卡） |
| `--data-parallel-size` | 2 | 2 | 2 组数据并行 |
| `--data-parallel-size-local` | 1 | 1 | 本节点承载 1 个 DP 副本 |
| `--data-parallel-address` | Node0 IP | Node0 IP | 主节点地址，用于 DP 通信 |
| `--data-parallel-rpc-port` | 12980 | 12980 | DP RPC 通信端口 |
| `--data-parallel-start-rank` | （无，默认 0） | 1 | Node 1 的 DP rank 从 1 开始 |
| `--headless` | 无 | 有 | Node 1 以 headless 模式运行，不暴露 API |
| `--max-model-len` | 131072 | 131072 | 双机 32 卡支持更长上下文 |
| `--tool-call-parser` | glm47 | glm47 | **【必需】** GLM-5.2 工具调用解析器 |
| `--enable-auto-tool-choice` | - | - | **【必需】** 启用自动工具选择 |
| `--reasoning-parser` | glm45 | glm45 | **【必需】** GLM-5.2 推理链解析器 |

> ⚠️ 上表中标注 **【必需】** 的三个参数对工具调用与推理能力至关重要，两个节点的脚本中均须保留，详见文档顶部的警告说明。

### 双机环境变量说明

| 变量 | 说明 |
|------|------|
| `HCCL_IF_IP` | 本机 IP，用于 HCCL 通信绑定 |
| `GLOO_SOCKET_IFNAME` | Gloo 后端通信网卡 |
| `TP_SOCKET_IFNAME` | Tensor Parallel 通信网卡 |
| `HCCL_SOCKET_IFNAME` | HCCL 集合通信网卡 |
| `HCCL_BUFFSIZE` | 400（双机互联带宽更大，缓冲区相应增大） |
| `VLLM_ASCEND_BALANCE_SCHEDULING` | 0（双机场景关闭均衡调度，由 DP 调度器管理） |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | 1（启用 FlashComm 优化通信） |

---

## API 调用示例

服务启动并加载完成后（可通过 `docker logs -f <容器名>` 观察日志），即可通过 OpenAI 兼容 API 进行调用。

### 查看模型列表

```bash
# 单机部署（端口 9527）
curl http://<HOST_IP>:9527/v1/models

# 双机部署（端口 8077，访问 Node 0）
curl http://<NODE0_IP>:8077/v1/models
```

### 对话补全

```bash
curl http://<HOST_IP>:<PORT>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.2-w8a8",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "请用一句话介绍华为昇腾。"}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### Python 调用（openai 库）

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<HOST_IP>:<PORT>/v1",
    api_key="not-needed",   # vLLM 本地部署无需真实 key
)

response = client.chat.completions.create(
    model="GLM-5.2-w8a8",
    messages=[
        {"role": "user", "content": "你好，请做个自我介绍。"}
    ],
    max_tokens=512,
)

print(response.choices[0].message.content)
```

---

## 常用运维命令

```bash
# 查看容器实时日志
docker logs -f vllm-glm52-w8a8          # 单机
docker logs -f vllm-glm5.2-node0        # 双机 Node 0
docker logs -f vllm-glm5.2-node1        # 双机 Node 1

# 停止服务
docker stop <容器名>

# 重新启动
docker start <容器名>

# 强制删除并重建（脚本内置逻辑）
docker rm -f <容器名>

# 查看 NPU 使用率
npu-smi info

# 进入容器排查
docker exec -it <容器名> bash
```

---

## 常见问题

### 1. 模型加载超时

大模型加载时间较长，单机脚本已设置 `VLLM_ENGINE_READY_TIMEOUT_S=7200`（2 小时）。如果双机部署也遇到超时，可在两个脚本中都添加该环境变量。

### 2. NPU 设备挂载失败

确保 `/dev/davinci*` 设备存在且 NPU 驱动正常运行：

```bash
ls /dev/davinci*
npu-smi info
```

### 3. 双机互联失败

- 检查两台机器网络是否互通：`ping <对方IP>`
- 确认 HCCN 互联拓扑已配置：`hccn_tool -i <card_id> -link_status -g`
- 确认网卡名和内网 IP 正确：执行 `ip addr show | grep 'inet ' | grep -v 127.0.0.1`

### 4. 端口被占用

修改脚本中的 `PORT` 变量，选择未被占用的端口。双机部署时两个节点的端口需保持一致。

### 5. 显存不足（OOM）

可适当降低 `--gpu-memory-utilization` 或 `--max-model-len`，也可以减少 `--max-num-seqs` 来降低并发显存占用。

---

## Ascend 910C 设备说明

Atlas A3 搭载的 **Ascend 910C** 采用双 die 封装，每张物理卡对应 2 个 davinci 设备。因此：

| 物理卡数 | davinci 设备数 | 说明 |
|---------|---------------|------|
| 8 张 910C | `/dev/davinci0` ~ `/dev/davinci15`（共 16 个） | 单机脚本挂载 16 个设备 |
| 2 × 8 张 910C | 共 32 个 davinci 设备 | 双机脚本每台挂载 16 个设备 |

这也解释了脚本中并行参数的设置逻辑：

- **单机**：`--tensor-parallel-size 8` + `--data-parallel-size 2` → 8×2=16 个 davinci 设备（对应 8 张物理卡）
- **双机**：`--tensor-parallel-size 16` + `--data-parallel-size 2` → 16×2=32 个 davinci 设备（对应 16 张物理卡）

> 以上为 vLLM-Ascend 的逻辑视角，**无需修改脚本**，8 张 910C 的机器直接使用即可。

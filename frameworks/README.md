# 部署框架模板

本目录收录各类**大模型/业务服务部署的通用框架模板**，可在具体项目中复制修改后使用。

> 与 `llm_infer/` 目录的区别：`llm_infer/` 是针对特定模型+硬件的完整部署方案，本目录提供的是**可复用的框架脚手架**。

## 目录结构

```
frameworks/
├── python_server/   # Python HTTP 服务框架（自研 winrain 框架）
├── trt_server/      # TensorRT-LLM 服务模板（NVIDIA GPU）
└── vllm_server/     # vLLM 服务模板（NVIDIA GPU）
```

## python_server：Python HTTP 服务框架

基于自研的 `winrain` 框架构建的 HTTP 服务，用于 AI 内容生成类业务（如视频脚本生产、题目解析等）。

**核心特点：**
- 基于自定义 `WinRain` 框架封装 HTTP 路由
- 支持数据队列（DataQueue）驱动的多阶段流水线
- 提供 debug / ship / 生产 多套环境接口
- 支持 Gunicorn 部署

**目录说明：**

| 文件/目录 | 说明 |
|-----------|------|
| `my_app.py` | 服务主入口，定义各 HTTP 接口 |
| `winrain/` | 自研 HTTP 框架核心 |
| `utils/` | 工具函数（数据处理、Prompt 模板、格式校验等） |
| `conf/` | 服务配置（Gunicorn、应用配置等） |
| `start.sh` | 服务启动脚本 |
| `Dockerfile` | 容器化构建文件 |

**启动方式：**

```bash
bash start.sh
```

## trt_server：TensorRT-LLM 服务模板

基于 TensorRT-LLM 的高性能推理服务模板，适用于 NVIDIA GPU 环境。

**核心特点：**
- 基于 `trtllm-serve` 命令启动 OpenAI 兼容 API
- 支持 PyTorch 后端，可配置 TP/PP 并行度
- 内置 patch 文件可自定义/修复框架行为
- 支持 extra_config 注入额外配置（如 In-flight Batching）

**使用方法：**

1. 将模型文件放入 `models/` 目录
2. 根据需要修改 `run.sh` 中的参数（`MAX_SEQ_LEN`、`MAX_BATCH_SIZE`、`TENSOR_PARALLEL_SIZE`）
3. 构建并运行 Docker 容器

**关键参数（run.sh）：**

| 参数 | 说明 |
|------|------|
| `MAX_SEQ_LEN` | 最大序列长度 |
| `MAX_BATCH_SIZE` | 最大 batch 大小 |
| `TENSOR_PARALLEL_SIZE` | Tensor Parallel 卡数 |
| `MAX_NUM_TOKENS` | 最大 token 数 |

**patch 说明：**

`patch/` 目录包含对 TensorRT-LLM 源码的补丁文件，在构建 Docker 镜像时自动替换：
- `openai_protocol.py` — OpenAI API 协议适配
- `openai_server.py` — OpenAI 服务端点定制
- `llm.py` — LLM API 层定制
- `modeling_utils.py` — 模型加载工具函数

## vllm_server：vLLM 服务模板

基于 vLLM 的通用推理服务模板，适用于 NVIDIA GPU 环境。

**核心特点：**
- 基于 `vllm.entrypoints.openai.api_server` 启动 OpenAI 兼容 API
- Docker 容器化部署
- 参数可在 `run.sh` 中直接修改

**使用方法：**

1. 将模型文件放入容器内 `/home/homework/models/` 路径
2. 修改 `run.sh` 中的参数（模型名、端口、GPU 数量等）
3. 构建并运行 Docker 容器

**关键参数（run.sh）：**

| 参数 | 说明 |
|------|------|
| `model_path` | 模型文件路径 |
| `model_name` | 对外暴露的模型名称 |
| `port` | 服务监听端口 |
| `gpu_num` | Tensor Parallel GPU 数量 |
| `max_num_seqs` | 最大并发序列数 |
| `gpu_memory_utilization` | GPU 显存利用率 |

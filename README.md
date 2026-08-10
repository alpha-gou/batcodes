# 个人常用脚本（batcodes）

个人开发与运维中积累的脚本集合，覆盖大模型训练、推理部署、服务框架、工具脚本等场景。

## 目录结构

```
batcodes/
├── llm_infer/       # 大模型推理部署（特定模型 + 硬件的完整部署方案）
├── llm_training/    # 大模型训练（微调/训练脚本与配置）
├── frameworks/      # 部署框架模板（可复用的服务脚手架）
├── script/          # 日常脚本（模型调用、数据处理、环境安装）
├── tools/           # 可安装的 Python 工具包
└── config/          # 个人环境配置备份
```

## 各目录说明

### [llm_infer/](./llm_infer/) — 大模型推理部署

针对特定模型与硬件平台的**生产环境部署脚本**。

| 子目录 | 说明 |
|--------|------|
| `glm5.2/` | GLM-5.2 (W8A8) 在华为 Atlas A3 (Ascend 910C) 上的 vLLM-Ascend 部署，支持单机/双机 |

### [llm_training/](./llm_training/) — 大模型训练

大模型微调与训练的脚本与配置，支持 SFT / DPO / PT / RM 等训练范式。

| 子目录 | 说明 |
|--------|------|
| `accelerate/rm/` | HuggingFace Accelerate + DeepSpeed ZeRO-3 训练 Reward Model |
| `llama_factory/` | 基于 LLaMA-Factory 的全参数分布式训练（单机/多机） |

### [frameworks/](./frameworks/) — 部署框架模板

可复用的服务部署脚手架，在具体项目中复制修改后使用。

| 子目录 | 说明 |
|--------|------|
| `python_server/` | 基于自研 winrain 框架的 Python HTTP 服务（AI 内容生成类业务） |
| `trt_server/` | TensorRT-LLM 推理服务模板（NVIDIA GPU） |
| `vllm_server/` | vLLM 推理服务模板（NVIDIA GPU） |

> `frameworks/` vs `llm_infer/`：`frameworks/` 是通用框架模板，`llm_infer/` 是针对特定模型+硬件的完整部署方案。

### [script/](./script/) — 日常脚本

零散的日常使用脚本，包括模型推理请求、数据批处理、vLLM 环境搭建等。

### [tools/](./tools/) — Python 工具包

可通过 `pip install .` 安装的工具包。

| 子目录 | 说明 |
|--------|------|
| `multi_thread/` | 多线程并发请求工具包（含令牌桶限速、CSV 输出） |
| `homemade/` | 命令行小工具集（批量重命名 `bfrename`、批量创建文件 `mkfs` 等） |
| `bf16_to_block_fp8/` | BF16 → Block-wise FP8 模型量化工具 |
| `bash_scripts/` | 服务器 Bash 配置（alias 等） |

### [config/](./config/) — 个人环境配置备份

个人开发环境的配置备份，详见 [config/README.md](./config/README.md)。

| 文件 | 说明 |
|------|------|
| `com.googlecode.iterm2.plist` | iTerm2 配置（通过硬链接同步） |
| `Default (OSX).sublime-keymap` | Sublime Text macOS 快捷键配置 |
| `.zshrc` | Zsh 配置（主要含 alias 配置） |

## 快速开始

### 安装工具包

```bash
# 安装多线程并发请求工具
cd tools/multi_thread && pip install .

# 安装命令行小工具集
cd tools/homemade && pip install .
```

### 使用模型部署脚本

参考具体部署方案的 README：
- [GLM-5.2 部署（Atlas A3）](./llm_infer/glm5.2/README.md)
- [vLLM 服务模板](./frameworks/vllm_server/)
- [TensorRT-LLM 服务模板](./frameworks/trt_server/)

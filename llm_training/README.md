# 大模型训练

本目录收录大模型微调/训练相关的脚本与配置，支持 SFT、DPO、PT、RM（Reward Model）等多种训练范式。

## 目录结构

```
llm_training/
├── accelerate/          # HuggingFace Accelerate 训练脚本
│   └── rm/              # Reward Model 训练（分类任务形式）
│       ├── run_accelerate_Yi.sh              # 启动训练主脚本
│       ├── run_Yi_classification_no_trainer.py  # 训练代码（基于 no_trainer 模式）
│       ├── ds_stage3_offload.yaml            # DeepSpeed ZeRO-3 + CPU Offload 配置
│       ├── data_preprocess_for_classification.py  # 数据预处理脚本
│       ├── run_data_preprocess.sh            # 数据预处理启动脚本
│       ├── run_validation.sh                 # 验证集评测脚本
│       └── run_test_on_gpu.py                # GPU 推理测试
│
└── llama_factory/       # LLaMA Factory 框架训练脚本
    ├── install_llama_factory.sh   # LLaMA Factory 安装指南（Docker 方式）
    ├── pack.sh                    # 打包脚本
    ├── old/                       # 旧版脚本归档
    │   ├── data_preprocess.yaml   # 数据预处理配置
    │   ├── run_data_preprocess.sh # 数据预处理启动脚本
    │   ├── start_pt.sh            # PT（预训练）启动脚本
    │   └── stop_pt.sh             # 停止训练脚本
    └── project_template/          # 项目模板（新训练项目可复制此目录使用）
        ├── train.sh               # 分布式训练启动主脚本（NCCL + torchrun）
        ├── run_lmfc_train.sh      # LLaMA Factory 训练封装脚本
        ├── qwen2_full_sft_ds3.yaml  # Qwen2 SFT 训练配置（DeepSpeed ZeRO-3）
        ├── qwen2_full_dpo_ds3.yaml  # Qwen2 DPO 训练配置
        ├── qwen2_full_pt_ds3.yaml   # Qwen2 PT（预训练）配置
        ├── ds_z3_config.json      # DeepSpeed ZeRO-3 配置文件
        ├── single_machine_util.sh # 单机训练辅助脚本
        ├── multi_machine_util.sh  # 多机训练辅助脚本
        ├── change_ds.sh           # 切换 DeepSpeed 配置的脚本
        ├── iplist                 # 多机 IP 列表模板
        └── data/dataset_info.json # 数据集注册信息
```

## accelerate/rm：Reward Model 训练

使用 HuggingFace Accelerate + DeepSpeed ZeRO-3 训练 Reward Model（以分类任务形式实现打分）。

**典型流程：**

```bash
# 1. 数据预处理
bash run_data_preprocess.sh

# 2. 启动训练（8 卡 + DeepSpeed ZeRO-3 + CPU Offload）
bash run_accelerate_Yi.sh

# 3. 验证/评测
bash run_validation.sh
```

**关键配置：**

| 参数 | 说明 |
|------|------|
| DeepSpeed 配置 | `ds_stage3_offload.yaml`（ZeRO-3 + CPU Offload） |
| NCCL 通信 | 默认绑定 `eth0`，启用 InfiniBand（`mlx5_bond_*`） |
| 实验追踪 | 接入 W&B（Weights & Biases） |

## llama_factory：LLaMA Factory 框架

基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 的全参数微调方案，支持 SFT / DPO / PT，单机与多机分布式训练。

**使用方式：**

```bash
# 1. 安装 LLaMA Factory（参考 install_llama_factory.sh）
bash install_llama_factory.sh

# 2. 复制 project_template 为新项目目录
cp -r project_template my_project
cd my_project

# 3. 修改 train.sh 中的 MASTER_ADDR、NNODES、RANK 等参数
# 4. 选择训练阶段配置文件（sft / dpo / pt）
# 5. 启动训练
bash train.sh
```

**训练阶段配置文件：**

| 文件 | 训练阶段 |
|------|---------|
| `qwen2_full_sft_ds3.yaml` | Supervised Fine-Tuning（SFT） |
| `qwen2_full_dpo_ds3.yaml` | Direct Preference Optimization（DPO） |
| `qwen2_full_pt_ds3.yaml` | Pre-Training（PT） |

**多机训练注意事项：**
- 在 `train.sh` 中修改 `NNODES`（机器数）、`MASTER_ADDR`（主节点 IP）、`RANK`（当前节点编号）
- `iplist` 文件记录所有参与训练的机器 IP
- `multi_machine_util.sh` / `single_machine_util.sh` 辅助分发与启停

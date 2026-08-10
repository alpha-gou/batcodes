# 大模型推理部署

本目录收录针对不同硬件平台和模型的**生产环境部署脚本**，涵盖容器化部署、分布式并行、推理加速等方案。

> 与 `frameworks/` 目录的区别：`frameworks/` 提供通用的部署框架模板，本目录则针对**特定模型 + 特定硬件**给出可直接运行的完整部署方案。

## 目录结构

```
llm_infer/
└── glm5.2/       # GLM-5.2 (W8A8) 在华为 Atlas A3 上的 vLLM-Ascend 部署
```

## 已支持的部署方案

### GLM-5.2（华为 Atlas A3 / Ascend 910C）

| 项目 | 说明 |
|------|------|
| 模型 | GLM-5.2-W8A8（Ascend 量化版） |
| 硬件 | 华为 Atlas A3（Ascend 910C，8 卡 × 128GB） |
| 框架 | vLLM-Ascend |
| 部署模式 | 单机（8 卡）/ 双机（16 卡） |
| 详见 | [glm5.2/README.md](./glm5.2/README.md) |

> ⚠️ GLM-5.2 部署时启动命令**必须**包含 `--tool-call-parser glm47`、`--enable-auto-tool-choice`、`--reasoning-parser glm45` 三个参数，详见子目录文档。

## 新增部署方案

如需添加新的模型/硬件部署方案，建议按如下结构创建子目录：

```
llm_infer/
└── <模型名>/
    ├── README.md              # 部署说明文档
    ├── start_<模式1>.sh       # 启动脚本
    └── start_<模式2>.sh       # 其他部署模式的脚本
```

README 建议包含：环境要求、镜像拉取、模型下载、启动步骤、关键参数说明、常见问题排查。

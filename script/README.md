# 脚本工具集

本目录收录日常开发中使用的**零散脚本**，包括模型服务调用、数据批处理、环境安装等。

## 脚本清单

### 模型推理请求脚本

这些脚本通常基于 `tools/multi_thread` 提供的多线程请求框架，用于批量调用已部署的模型服务。

| 脚本 | 说明 |
|------|------|
| `request_qwen_vllm.py` | 批量请求 Qwen 模型（vLLM 服务），用于脚本评分任务 |
| `request_qwen_triton.py` | 批量请求 Qwen 模型（Triton Inference Server） |
| `request_xgv6_vllm.py` | 批量请求 XG V6 模型（vLLM 服务） |
| `request_xgv6_triton.py` | 批量请求 XG V6 模型（Triton 服务） |
| `request_qwq_infomiss.py` | 请求 QwQ 模型，用于信息缺失检测 |
| `request_ae.py` | 请求 Answer Equivalence（答案等价）服务 |
| `request_ae_plus.py` | 答案等价服务的增强版实现（含 QwQ 评判逻辑） |
| `request_rm3.py` | 请求 RM3 Reward Model 服务 |
| `request_question_answer_for_image_tid.py` | 根据题目 ID 获取图片题目的问题与答案 |

### vLLM 服务启停脚本

| 脚本 | 说明 |
|------|------|
| `run_vllm_server.sh` | vLLM 服务启动脚本合集，包含单机/多实例/多 GPU 等多种启动方式 |
| `install_vllm.sh` | vLLM 环境安装指南（Docker + pip），含 H20 GPU 特殊版本说明 |

### 数据处理

| 文件 | 说明 |
|------|------|
| `data_processing.ipynb` | 数据处理 Jupyter Notebook（数据清洗、分析等） |

## 使用说明

大部分请求脚本继承自 `tools/multi_thread` 的 `MultiThreadRequester`，使用前需先安装该包：

```bash
cd ../tools/multi_thread
pip install .
```

各脚本通常**不应直接运行**，需要根据实际的服务地址、模型路径、数据文件等参数进行修改后再使用。

# 常用工具

本目录收录日常开发中使用的 Python 工具包和脚本，安装后可直接在命令行中使用。

## 目录结构

```
tools/
├── multi_thread/        # 多线程并发请求工具包
├── homemade/            # 命令行小工具集
├── bf16_to_block_fp8/   # BF16 → Block-wise FP8 量化工具
└── bash_scripts/        # 服务器 Bash 配置脚本
```

## multi_thread：多线程并发请求工具包

提供多线程 HTTP 请求、异步数据处理、令牌桶限速等能力，是 `script/` 目录下多数 `request_*.py` 脚本的依赖。

**安装：**

```bash
cd multi_thread
pip install .
```

**提供的类：**

| 类名 | 说明 |
|------|------|
| `MultiThreadRequester` | 多线程同步请求器，支持 CSV 输出、重试、限速（令牌桶） |
| `AsyncRequestClient` | 异步客户端（基于 `asyncio`） |
| `AsyncDataProcessor` | 异步数据处理管道 |

**使用示例：**

```python
from multi_thread import MultiThreadRequester

class MyRequester(MultiThreadRequester):
    def request_main(self, data):
        # 实现单条数据的处理逻辑
        ...

# 批量请求，结果自动写入 CSV
requester = MyRequester(...)
requester.run(data_list)
```

**详见** [examples/](./multi_thread/examples/) 目录中的示例脚本。

## homemade：命令行小工具集

**安装：**

```bash
cd homemade
pip install .
```

**提供的 CLI 命令：**

| 命令 | 模块 | 说明 |
|------|------|------|
| `mkfs` | `mkfiles` | 批量创建文件 |
| `bfrename` | `batch_rename` | 批量重命名文件 |
| `nameplus` | `nameplus` | 文件名增强/规范化 |
| `replacewd` | `replace_words` | 批量文本替换 |

## bf16_to_block_fp8：BF16 → Block-wise FP8 量化工具

将 BF16 精度的模型权重转换为 Block-wise FP8 格式，用于支持 FP8 推理加速。

**使用：**

```bash
python convert.py \
    -i /path/to/input/bf16/model \
    -o /path/to/output/fp8/model \
    -n [GPU_COUNT]
```

## bash_scripts：服务器配置脚本

| 文件 | 说明 |
|------|------|
| `bashrc_wn` | 自定义 Bash 配置文件（alias 等），复制到服务器 `~/.bashrc` 即可使用 |
| `reset_conda.sh` | Conda 环境重置脚本 |

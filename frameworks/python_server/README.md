# Python HTTP 服务框架（winrain）

基于自研 `winrain` 框架构建的 HTTP 服务，主要用于 AI 内容生成类业务（如视频脚本生产、题目解析等）。

## 目录结构

```
python_server/
├── my_app.py          # 服务主入口，定义各 HTTP 接口
├── winrain/           # 自研 HTTP 框架核心
│   ├── winrain.py     # 框架主体
│   └── winlogger.py   # 日志模块
├── utils/             # 工具函数
│   ├── processor.py   # 业务处理器
│   ├── prompts.py     # Prompt 模板
│   ├── qwen_prompts.py # Qwen 模型 Prompt 模板
│   ├── dataqueue.py   # 数据队列客户端
│   ├── data_structure.py  # 数据结构定义
│   ├── format_check.py    # 数据格式校验
│   └── const.py       # 常量定义
├── conf/              # 服务配置
│   ├── app.config.yaml    # 应用配置
│   ├── gunicorn.conf      # Gunicorn 配置
│   └── jad.conf           # 调度配置
├── start.sh           # 服务启动脚本
├── Dockerfile         # 容器化构建文件
└── requirements.txt   # Python 依赖
```

## 主要接口

| 路径 | 说明 |
|------|------|
| `/ready` | 健康检查 |
| `/modelVersion` | 获取模型版本 |
| `/generate_main` | 脚本生产主接口（结果入生产队列） |
| `/debug_main` | 调试用主接口（结果入 debug 队列） |
| `/generate_main_ship` | Ship 测试环境接口 |
| `/script_filter` | 仅走挑选流程（跳过改写阶段） |
| `/chinese_jx` | 语文解析接口 |

## 启动方式

```bash
bash start.sh
```

## 数据流转

服务采用队列驱动的多阶段流水线架构：请求进入后写入数据队列（DataQueue），由下游消费者依次处理各阶段任务，最终通过 callback 回写结果。

# 示例代码，不要直接运行

# 基于已有image创建vllm服务容器
docker run --shm-size 128g -m 2010g -it \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --gpus all --ipc=host --network=host --privileged \
    -v /data/strategy_protect/workspace/:/workspace/ \
    -d --name wn_vllm c6b1ebae6927

# pip镜像设置
docker exec wn_vllm bash -c \
    "mkdir ~/.pip && echo -e \"[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple \" > ~/.pip/pip.conf "

# 安装vllm
docker exec wn_vllm bash -c \
    "pip install vllm==0.3.1 transformers==4.37.2 torch==2.1.2"

# H20显卡需要安装如下版本
docker exec wn_vllm bash -c \
    "pip install torch==2.4.1 nvidia-cublas-cu12==12.8.3.14 transformers==4.46.3 vllm==0.6.3"


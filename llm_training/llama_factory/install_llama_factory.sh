# 示例代码，不要直接运行

cd ../third_party
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 基于已有的image直接安装（推荐）
docker run --shm-size 128g -m 2010g -it \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --gpus all \
    --ipc=host \
    --network=host \
    --privileged \
    -v /data/strategy_protect/workspace/:/workspace/ \
    -d --name wn_lmfc \
    c6b1ebae6927

# pip添加清华源
docker exec wn_lmfc bash \
    -c "mkdir ~/.pip && echo -e \"[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple \" > ~/.pip/pip.conf "


# 安装LLaMA-Factory
docker exec wn_lmfc bash \
    -c "cd /workspace/experiment/LLaMA-Factory; pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  -e \".[torch,metrics]\" "

# 或者进入docker后运行
cd /workspace/experiment/LLaMA-Factory
pip install -e ".[torch,metrics]" -i https://pypi.tuna.tsinghua.edu.cn/simple


# 参考: https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md
# 使用他们的docker
docker build -f ./docker/docker-cuda/Dockerfile \
    --build-arg INSTALL_BNB=false \
    --build-arg INSTALL_VLLM=false \
    --build-arg INSTALL_DEEPSPEED=false \
    --build-arg INSTALL_FLASHATTN=false \
    --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    -t llamafactory:latest .


docker run -dit \
    -v /data5/strategy_protect/workspace:/workspace \
    --shm-size 128g -m 2048g \
    --gpus=all \
    --ipc=host \
    --network=host \
    --privileged \
    --name wn_lmfc \
    llamafactory:latest

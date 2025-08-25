#!/bin/bash

# 公共参数
CURRENT_FOLDER=$(basename "$PWD")
PROJECT_PATH="/workspace/ningwang/lmfc/${CURRENT_FOLDER}"
PORT="8986"


# 单服务
# PROJECT_PATH="/workspace/ningwang/xg"

# SERVED_MODEL_NAME="xg_model"
# MODEL_PATH="${PROJECT_PATH}/output/checkpoint-400"
# # MODEL_PATH="${PROJECT_PATH}/fp_res/Qwen2.5-Math-72B-Instruct-aivideo_dpo_9w_v0-2-step_300"

# # SERVED_MODEL_NAME="qwen_score"
# # MODEL_PATH="/workspace/models/Qwen2.5-72B-Instruct"

# LOG_FILE="$PROJECT_PATH/log/vllm_8986_log.txt"
# docker exec wn_vllm bash -c \
#     "cd ${PROJECT_PATH} && CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 -m vllm.entrypoints.openai.api_server \
#     --model ${MODEL_PATH} --dtype auto --tensor-parallel-size 4 \
#     --gpu-memory-utilization 0.98 --disable-custom-all-reduce --enforce-eager \
#     --host 0.0.0.0 --port ${PORT} --trust-remote-code \
#     --served-model-name ${SERVED_MODEL_NAME} > ${LOG_FILE} 2>&1 &"

# 多服务
for i in {0..3}; do
    # 动态计算参数
    CUDA_DEVICES="$((i*2)),$((i*2+1))"
    PORT=$((8980 + i))
    LOG_FILE="${PROJECT_PATH}/log/${PORT}_server.log"

    # 需修改参数
    # MODEL_PATH="${PROJECT_PATH}/output/checkpoint-$((20 + i*10))"
    MODEL_PATH="${PROJECT_PATH}/output/checkpoint-$((100 + i*100))"
    # MODEL_PATH="${PROJECT_PATH}/fp_res/Qwen2.5-Math-72B-Instruct-aivideo_dpo_9w_v0-2-step_300"
    SERVED_MODEL_NAME="xg_model"

    # 执行docker命令
    docker exec wn_vllm bash -c \
        "cd ${PROJECT_PATH} && CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} nohup python3 -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} --dtype auto  --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.98 --disable-custom-all-reduce --enforce-eager \
        --host 0.0.0.0 --port ${PORT} --trust-remote-code \
        --served-model-name ${SERVED_MODEL_NAME} > ${LOG_FILE} 2>&1 &"
done


# 停止服务
docker exec wn_vllm bash -c \
    'pid_list=($(ps -ef | grep "[p]ython" | awk "{print \$2}")); \
    [ ${#pid_list[@]} -gt 0 ] && kill -9 ${pid_list[@]}'


# docker exec llama_factory bash -c \
#     'pid_list=($(ps -ef | grep "[p]ython" | awk "{print \$2}")); \
#     [ ${#pid_list[@]} -gt 0 ] && kill -9 ${pid_list[@]}'

# for i in {150..300..50}; do
#     python format_check.py test/v8_72b_res/xg_v8.0_sft_72b_st$i.csv xg_res
# done


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen2.5-72B-Instruct --dtype auto --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.98 --disable-custom-all-reduce --enforce-eager \
    --host 0.0.0.0 --port 8986 --trust-remote-code \
    --served-model-name xg_model > log/vllm_8986_log.txt 2>&1 &

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m vllm.entrypoints.openai.api_server \
#     --model fpres/Qwen2.5-Math-72B-Instruct-aivideo_dpo_9w_v0-2-step_300 --dtype auto --tensor-parallel-size 8 \
#     --gpu-memory-utilization 0.98 --disable-custom-all-reduce --enforce-eager \
#     --host 0.0.0.0 --port 8986 --trust-remote-code \
#     --served-model-name xg_model

export VLLM_WORKER_MULTIPROC_METHOD=spawn   # 使用魔塔默认的vllm的环境时要加上这个
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/QwQ-32B --dtype auto --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.98 --disable-custom-all-reduce --enforce-eager \
    --host 0.0.0.0 --port 8986 --trust-remote-code \
    --served-model-name gscore > logs/vllm_8986_log.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python3 -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen3-32B --dtype auto --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.98 --disable-custom-all-reduce --enforce-eager \
    --host 0.0.0.0 --port 8987 --trust-remote-code \
    --served-model-name gscore > logs/vllm_8987_log.txt 2>&1 &
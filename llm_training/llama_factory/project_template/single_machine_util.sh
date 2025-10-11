#!/bin/bash
# 单机训练控制脚本
DOCKER_NAME="wn_lmfc"
DOCKER_ROOT_PATH="/workspace/ningwang/lmfc"
CURRENT_FOLDER=$(basename "$PWD")
LOG_FILE="log/v8_1_log.txt"


# 启动训练
docker exec ${DOCKER_NAME} bash -c \
  "cd ${DOCKER_ROOT_PATH}/${CURRENT_FOLDER}; nohup bash run_lmfc_train.sh > ${LOG_FILE} 2>&1 &"


# 停止训练
# docker exec ${DOCKER_NAME} bash -c \
#     " ps -ef | grep llamafactory-cli | awk '{print $2}' | xargs kill -9 "


# 重启docker
# docker stop ${DOCKER_NAME}
# docker start ${DOCKER_NAME}


# stop
# parallel_ssh iplist "docker exec llama_factory bash -c \" ps -ef | grep llamafactory-cli | awk '{print $2}' | xargs kill -9 \""
# parallel_ssh iplist "docker exec llama_factory bash -c \"pid=\$(ps -ef | grep '[l]lamafactory-cli' | awk '{print \$2}'); [ ! -z \"\$pid\" ] && kill -9 \$pid\""
# docker exec ${DOCKER_NAME} bash -c \
#     "pid=\$(ps -ef | grep 'python' | awk '{print \$2}'); [ ! -z \"\$pid\" ] && kill -9 \$pid"


# pid=$(ps -ef | grep 'python' | awk '{print $2}'); [ ! -z "$pid" ] && kill -9 $pid

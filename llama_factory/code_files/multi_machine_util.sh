#!/bin/bash
# 多机训练控制脚本
# 1. 修改: iplist ; change_ds.sh ; train.sh ; .yaml ;
# 2. 运行: sh change_ds.sh
# 3. 修改并运行此脚本

# stop
# parallel_ssh iplist "docker exec llama_factory bash -c \" ps -ef | grep llamafactory-cli | awk '{print $2}' | xargs kill -9 \""
# parallel_ssh iplist "docker exec llama_factory bash -c \"pid=\$(ps -ef | grep '[l]lamafactory-cli' | awk '{print \$2}'); [ ! -z \"\$pid\" ] && kill -9 \$pid\""

# parallel_ssh iplist  "docker exec llama_factory bash -c \"pid=\$(ps -ef | grep 'python' | awk '{print \$2}'); [ ! -z \"\$pid\" ] && kill -9 \$pid\""
parallel_ssh iplist "docker exec llama_factory pkill -f 'python'"


# 重启docker
# parallel_ssh iplist "docker stop llama_factory"
# parallel_ssh iplist "docker start llama_factory"

# 启动训练
# cli_ssh iplist "docker exec llama_factory bash -c \"cd /workspace/ningwang/lmfc/xg_v8_qwen2.5_72b_math_sft_data4_16w; nohup bash train.sh > log/v8_3_log.txt 2>&1 & \""



# pid=$(ps -ef | grep 'python' | awk '{print $2}'); [ ! -z "$pid" ] && kill -9 $pid

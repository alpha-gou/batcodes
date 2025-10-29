#!/bin/sh

cd /home/homework/


echo "启动 vllm 服务..."
export CUDA_VISIBLE_DEVICES=0,1,2,3
model_path="/home/homework/models/"
model_name="qwen3"
port=8080
gpu_num=4
max_num_seqs=128


python3 -m vllm.entrypoints.openai.api_server \
    --model $model_path \
    --served-model-name $model_name \
    --port $port \
    --tensor-parallel-size $gpu_num \
    --max-num-seqs $max_num_seqs \
    --gpu-memory-utilization=0.9 \
    --enforce-eager \
    --disable-custom-all-reduce

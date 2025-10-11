#!/bin/bash

# 读取的服务器列表文件名
server_list_file="iplist"

# 要修改的文件路径和文件名
current_folder=$(basename "$PWD")
exp_name=$current_folder
dir_to_modify="/data/strategy_protect/workspace/ningwang/lmfc"
file_to_modify="$dir_to_modify/$exp_name/train.sh"

parallel_scp_h2c iplist iplist "$dir_to_modify/${exp_name}/"
parallel_scp_h2c iplist qwen2_full_sft_ds3.yaml "$dir_to_modify/${exp_name}/"
parallel_scp_h2c iplist train.sh "$dir_to_modify/${exp_name}/"
parallel_scp_h2c iplist ./data/ "$dir_to_modify/${exp_name}/"

# 读取服务器列表文件中的所有行，并将每行保存到数组servers中
readarray -t servers < "$server_list_file"

# 遍历服务器数组，登录到每台服务器并修改文件中的相应行
for server in "${servers[@]}"; do
    # 获取服务器编号，从0开始
    server_number=$(grep -n "$server" "$server_list_file" | cut -d ":" -f 1 | awk '{print $1 - 1}')
    echo " "
    echo $server
    echo $server_number

    # 使用ssh登录到服务器并修改文件中的相应行
    modify_res=$(ssh $server "sed -i 's/RANK=0/RANK=$server_number/' '$file_to_modify'")

    echo "Server $server: machine_rank in $file_to_modify modified."
done

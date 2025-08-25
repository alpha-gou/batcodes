export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset NCCL_SOCKET_NTHREADS
unset NCCL_NSOCKS_PERTHREAD
unset NCCL_SOCKET_IFNAME
unset NCCL_IB_DISABLE
unset NCCL_MAX_NRINGS
unset NCCL_P2P_DISABLE
unset NCCL_NET_GDR_LEVEL
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_NET_GDR_LEVEL=3
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export WANDB_BASE_URL="http://172.29.213.162:8900"
# export WANDB_DISABLED=true

wandb_api_key="local-7fdf70a47979d344ad6550e1032ba39b48cc110f"


# 需修改配置
exp_name="xg_v8.8_qwen_72b_sft_data4.1_15w"
stage="sft"
# stage="dpo"


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FORCE_TORCHRUN=1 WANDB_API_KEY=${wandb_api_key} WANDB_PROJECT=${exp_name} \
llamafactory-cli train qwen2_full_${stage}_ds3.yaml 


# 依赖环境：
# pip install deepspeed==0.15.4 -i https://pypi.tuna.tsinghua.edu.cn/simple


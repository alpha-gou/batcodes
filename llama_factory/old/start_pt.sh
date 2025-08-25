### 训练任务

current_folder=$(basename "$PWD")
exp_name=$current_folder
# exp_name="Qwen2.5-72B-Instruct-lf_chin_all_clm_3455w_v1_3"

wandb_url="http://172.29.213.162:8900"
wandb_api_key="local-7fdf70a47979d344ad6550e1032ba39b48cc110f"

### 登陆wandb
# parallel_ssh iplist "docker exec llama_factory bash -c \"python3 report_to_wandb.py\""
### 启动预训练
parallel_ssh iplist "docker exec llama_factory bash -c \"rm -rf /root/.cache/torch_extensions/; cd /workspace/experiment/${exp_name}; nohup bash train_pt.sh > log.${exp_name} 2>&1 &\""

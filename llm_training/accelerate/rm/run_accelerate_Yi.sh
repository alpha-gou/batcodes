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

experiment_name="scoring_v4.5_qwen32b_da3"
accelerate launch --config_file ds_stage3_offload.yaml run_Yi_classification_no_trainer.py \
  --train_file "/workspace/data/test_clm.csv" \
  --validation_file "/workspace/data/test_clm.csv" \
  --model_name_or_path "/workspace/models/Qwen2.5-32B-Instruct" \
  --data_cache_dir "dataset/traindata_4.5_da_qwen32b" \
  --experiment_name "$experiment_name" \
  --output_dir "model_res/$experiment_name" \
  --with_tracking \
  --report_to "wandb" \
  --checkpointing_steps 5000 \
  --checkpointing_autosave_threshold 0.0001 \
  --checkpointing_autosave_interval 100000 \
  --pad_to_max_length \
  --block_size 3000 \
  --per_device_train_batch_size 10 \
  --per_device_eval_batch_size 10 \
  --gradient_accumulation_steps 1 \
  --preprocessing_num_workers 20 \
  --num_train_epochs 100 \
  --learning_rate 2e-6

  # --resume_from_checkpoint "model_res/scoring_v4.1_qwen32b_da/step_304" \
  # --model_name_or_path "/workspace/models/Yi-1.5-34B-math_all_clm_10000w_v0-1-step_27600" \

  # --exp_iplist "./iplist" \
  # --just_valid \
  # --extern_eval_file "/workspace/data/eval_all_20230609.csv" \

# experiment_name="scoring_yi_34b_math_v10_rebuild"
# accelerate launch --config_file ds_stage3_offload.yaml run_Yi_classification_no_trainer.py \
#   --train_file "/workspace/data/test_clm.csv" \
#   --validation_file "/workspace/data/test_clm.csv" \
#   --model_name_or_path "/workspace/models/Yi-1.5-34B-math_all_clm_10000w_v0-1-step_27600" \
#   --data_cache_dir "dataset/traindata_v10_rebuild" \
#   --experiment_name "$experiment_name" \
#   --output_dir "model_res/$experiment_name" \
#   --with_tracking \
#   --report_to "wandb" \
#   --checkpointing_steps 200 \
#   --checkpointing_autosave_threshold 0.0001 \
#   --checkpointing_autosave_interval 100000 \
#   --pad_to_max_length \
#   --block_size 1500 \
#   --per_device_train_batch_size 18 \
#   --per_device_eval_batch_size 18 \
#   --gradient_accumulation_steps 4 \
#   --preprocessing_num_workers 20 \
#   --num_train_epochs 1 \
#   --learning_rate 5e-6


# experiment_name="scoring-Yi-1.5-34b-math_v3_nomark"

# accelerate launch --config_file ds_stage3_offload.yaml run_Yi_classification_no_trainer.py \
#   --train_file "/workspace/data/test_clm.csv" \
#   --validation_file "/workspace/data/test_clm.csv" \
#   --data_cache_dir "dataset/traindata_v3_nomark" \
#   --model_name_or_path "/workspace/models/Yi-1.5-34B-math_all_clm_10000w_v0-1-step_27600" \
#   --resume_from_checkpoint "/workspace/ningwang/scoring/model_res/scoring-Yi-1.5-34b-math_v3_nomark/step_200" \
#   --experiment_name "$experiment_name" \
#   --output_dir "model_res/$experiment_name" \
#   --with_tracking \
#   --report_to "wandb" \
#   --checkpointing_steps 200 \
#   --checkpointing_autosave_threshold 0.0001 \
#   --checkpointing_autosave_interval 100000 \
#   --block_size 4096 \
#   --per_device_train_batch_size 6 \
#   --per_device_eval_batch_size 6 \
#   --gradient_accumulation_steps 2 \
#   --preprocessing_num_workers 20 \
#   --num_train_epochs 1 \
#   --learning_rate 5e-6

# version_name="validate_500"
# python3 ../data_preprocess_for_classification.py \
#   --train_file "data/validate_500.csv" \
#   --validation_file "data/validate_500.csv" \
#   --model_name_or_path "/workspace/models/Yi-1.5-34B-math_all_clm_10000w_v0-1-step_27600" \
#   --data_cache_dir "cache/${version_name}" \
#   --block_size 2048 \
#   --process_type "Yi" \
#   --preprocessing_num_workers 100

# python3 validation_preprocess

# version_name="test_o1_yi34b"

version_name="review_1218_human_yi"
python3 ../data_preprocess_for_classification.py \
  --train_file "data/xg_1218_for_scoring.csv" \
  --validation_file "data/review_1218_4scoring_human.csv" \
  --model_name_or_path "../model_res/fp_res/scoring-Yi-1.5-34b-math_v7_npy_step_800" \
  --data_cache_dir "cache/${version_name}" \
  --block_size 3000 \
  --process_type "Yi" \
  --preprocessing_num_workers 100

  # --model_name_or_path "/workspace/models/Yi-1.5-34B-math_all_clm_10000w_v0-1-step_27600" \
  # --model_name_or_path "/workspace/models/Qwen2.5-32B-Instruct" \
  # --validation_file "../dataset/scoring_model_test_data_v7_npy_1k.csv" \


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


# experiment_name="sg_v4_step_800_ut1"
# experiment_name="sg_v1_step_1800"
# experiment_name="review1218_qw4.5_st313"
experiment_name="review_1218_human_yi"


accelerate launch --config_file ds_stage3_offload.yaml run_test_on_gpu.py \
  --train_file "/workspace/data/test_clm.csv" \
  --validation_file "/workspace/data/test_clm.csv" \
  --data_cache_dir "cache/${version_name}" \
  --model_name_or_path "../model_res/fp_res/scoring-Yi-1.5-34b-math_v7_npy_step_800" \
  --just_valid \
  --experiment_name "$experiment_name" \
  --output_dir "val_res/$experiment_name" \
  --with_tracking \
  --report_to "wandb" \
  --checkpointing_steps 200 \
  --checkpointing_autosave_threshold 0.8 \
  --checkpointing_autosave_interval 3600 \
  --block_size 3000 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --preprocessing_num_workers 50 \
  --pad_to_max_length \
  --num_train_epochs 1 \
  --learning_rate 5e-6

  # --model_name_or_path "../model_res/qwen32_scoring/scoring_v4.5_qwen32b_st80" \
  # --exp_iplist "./iplist" \
  # --pad_to_max_length \
  # --extern_eval_file "/workspace/data/eval_all_20230609.csv" \


# python df_dorp_dumplicates.py "val_res/${experiment_name}/val_res.csv"

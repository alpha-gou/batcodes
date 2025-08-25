# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# python3 data_preprocess_for_classification.py \
#   --train_file "train_data.csv" \
#   --model_name_or_path "/workspace/models/Yi-1.5-9B/" \
#   --data_cache_dir "/workspace/data/Yi-1.5-9b_train_prm_math_v0/" \
#   --block_size 2048 \
#   --process_type "Yi" \
#   --preprocessing_num_workers 100

# python3 data_preprocess_for_classification.py \
#   --train_file "dataset/scoring_train_data_v10_rebuild_200w.csv" \
#   --validation_file "validation/data/ground_truth_validation_df_404.csv" \
#   --model_name_or_path "/workspace/models/Qwen2.5-Math-72B-Instruct" \
#   --data_cache_dir "dataset/traindata_v13_clean" \
#   --block_size 1500 \
#   --process_type "Yi" \
#   --preprocessing_num_workers 100

# python3 data_preprocess_for_classification.py \
#   --train_file "dataset/scoring_train_data_v10_rebuild_200w.csv" \
#   --validation_file "dataset/scoring_validate_data_v10_rebuild_1k.csv" \
#   --model_name_or_path "/workspace/models/Qwen2-72b-math_all_dpo_60w_v0-2-step_500" \
#   --data_cache_dir "dataset/traindata_v11_qw2rb" \
#   --block_size 1500 \
#   --process_type "Yi" \
#   --preprocessing_num_workers 100


python3 data_preprocess_for_classification.py \
  --train_file "dataset/train_data_v4.4_2.5k.csv" \
  --validation_file "dataset/test_data_v4.csv" \
  --model_name_or_path "/workspace/models/Qwen2.5-32B-Instruct" \
  --data_cache_dir "dataset/traindata_4.4_da_qwen32b" \
  --block_size 3000 \
  --process_type "Yi" \
  --preprocessing_num_workers 100

  # --validation_file "validation/data/ground_truth_validation_df_404.csv" \


# python3 data_preprocess_for_classification.py \
#   --train_file "dataset/scoring_model_validation_data_1k.csv" \
#   --validation_file "dataset/scoring_model_train_data_1k.csv"\
#   --model_name_or_path "/workspace/models/Yi-1.5-34B-math_all_clm_10000w_v0-1-step_27600" \
#   --data_cache_dir "dataset/testdata_test_1k" \
#   --block_size 4096 \
#   --process_type "Yi" \
#   --preprocessing_num_workers 100



##################
#  debug data    #
##################
# python3 data_preprocess_for_classification.py \
#   --train_file "dataset/debug_data2.csv" \
#   --validation_file "dataset/debug_data2.csv" \
#   --model_name_or_path "/workspace/models/Yi-1.5-34B-math_all_clm_10000w_v0-1-step_27600" \
#   --data_cache_dir "dataset/debug_data2" \
#   --block_size 4096 \
#   --process_type "Yi" \
#   --preprocessing_num_workers 100


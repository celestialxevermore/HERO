# train_epochs=100
# master_port=01028
# num_process=1

# Define arrays for trend and seasonality lengths
# for pred_len in 96 192 336 720; do
#     for trend_length in 24 48 96 192; do
#         for seasonal_length in 24 48 96 192; do
#             CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 --master_port=29500 run.py \
#                 --task_name long_term_forecast \
#                 --is_training 1 \
#                 --data_path ETTh1.csv \
#                 --model_id ETTh1_512_${pred_len} \
#                 --model S2IPLLM \
#                 --data ETTh1 \
#                 --number_variable 7 \
#                 --features M \
#                 --seq_len 512 \
#                 --label_len 0 \
#                 --pred_len ${pred_len} \
#                 --enc_in 7 \
#                 --dec_in 7 \
#                 --c_out 7 \
#                 --des 'Exp' \
#                 --itr 1 \
#                 --d_model 768 \
#                 --learning_rate 0.0001 \
#                 --patch_size 16 \
#                 --stride 8 \
#                 --add_prompt 1 \
#                 --prompt_length 4 \
#                 --batch_size 128 \
#                 --sim_coef -0.05 \
#                 --pool_size 1000 \
#                 --percent 100 \
#                 --trend_length ${trend_length} \
#                 --seasonal_length ${seasonal_length} \
#                 --train_epochs 100 \
#                 --exp_info "ETTh1_experiments(20240826)" \
#                 --exp_protocol "Multi_ETTh1_512_${pred_len}_trend:${trend_length}_seasonality:${seasonal_length}" \
#                 --use_multi_gpu

            
#             sleep 5
#         done
#     done
# done

# for pred_len in 96 192 336 720; do
#     for trend_length in 24 48 96 192; do
#         for seasonal_length in 24 48 96 192; do
#             CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 --master_port=29500 run.py \
#                 --task_name long_term_forecast \
#                 --is_training 1 \
#                 --data_path ETTh2.csv \
#                 --model_id ETTh2_512_${pred_len} \
#                 --model S2IPLLM \
#                 --data ETTh2 \
#                 --number_variable 7 \
#                 --features M \
#                 --seq_len 512 \
#                 --label_len 0 \
#                 --pred_len ${pred_len} \
#                 --enc_in 7 \
#                 --dec_in 7 \
#                 --c_out 7 \
#                 --des 'Exp' \
#                 --itr 1 \
#                 --d_model 768 \
#                 --learning_rate 0.0001 \
#                 --patch_size 16 \
#                 --stride 8 \
#                 --add_prompt 1 \
#                 --prompt_length 8 \
#                 --batch_size 128 \
#                 --sim_coef -0.05 \
#                 --pool_size 1000 \
#                 --percent 100 \
#                 --trend_length ${trend_length} \
#                 --seasonal_length ${seasonal_length} \
#                 --train_epochs 100 \
#                 --exp_info "ETTh2_experiemnts(20240826)" \
#                 --exp_protocol "Multi_ETTh2_512_${pred_len}_trend:${trend_length}_seasonality:${seasonal_length}" \
#                 --use_multi_gpu

#              v
#             sleep 5
#         done
#     done
# done





CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 --master_port=29500 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model S2IPLLM \
  --data ETTh1 \
  --number_variable 7 \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 768 \
  --learning_rate 0.0001 \
  --patch_size 16 \
  --stride 8 \
  --add_prompt 1 \
  --prompt_length 4 \
  --batch_size 128 \
  --sim_coef -0.05 \
  --pool_size  1000 \
  --percent 100 \
  --trend_length 96 \
  --seasonal_length 96 \
  --train_epochs 100 \
  --exp_info "ETTh1_experiments(20240826)" \
  --exp_protocol "Multi_ETTh1_512_96" \
  --use_multi_gpu

# CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 --master_port=29500 run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_512_192 \
#   --model S2IPLLM \
#   --data ETTh1 \
#   --number_variable 7 \
#   --features M \
#   --seq_len 512 \
#   --label_len 0 \
#   --pred_len 192 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 768 \
#   --learning_rate 0.0001 \
#   --patch_size 16 \
#   --stride 8 \
#   --add_prompt 1 \
#   --prompt_length 4 \
#   --batch_size 128 \
#   --sim_coef -0.05 \
#   --pool_size  1000 \
#   --percent 100 \
#   --trend_length 96 \
#   --seasonal_length 96 \
#   --train_epochs 100 \
#   --exp_info "ETTh1_experiments(20240825)_OPTIMAL_SETTING" \
#   --exp_protocol "Multi_ETTh1_512_192" \
#   --use_multi_gpu

# CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=1 --master_port=29500 run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_512_336 \
#   --model S2IPLLM \
#   --data ETTh1 \
#   --number_variable 7 \
#   --features M \
#   --seq_len 512 \
#   --label_len 0 \
#   --pred_len 336 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 768 \
#   --learning_rate 0.0001 \
#   --patch_size 16 \
#   --stride 8 \
#   --add_prompt 1 \
#   --prompt_length 8 \
#   --batch_size 128 \
#   --sim_coef -0.01 \
#   --pool_size  1000 \
#   --percent 100 \
#   --trend_length 24 \
#   --seasonal_length 24 \
#   --train_epochs 100 \
#   --exp_info "ETTh1_experiments(20240825)_OPTIMAL_SETTING" \
#   --exp_protocol "Multi_ETTh1_512_336" \
#   --use_multi_gpu

# CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 --master_port=29500 run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_512_720 \
#   --model S2IPLLM \
#   --data ETTh1 \
#   --number_variable 7 \
#   --features M \
#   --seq_len 512 \
#   --label_len 0 \
#   --pred_len 720 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 768 \
#   --learning_rate 0.001 \
#   --patch_size 16 \
#   --stride 8 \
#   --add_prompt 1 \
#   --prompt_length 4 \
#   --batch_size 128 \
#   --sim_coef -0.01 \
#   --pool_size  1000 \
#   --percent 100 \
#   --trend_length 192 \
#   --seasonal_length 48 \
#   --train_epochs 100 \
#   --exp_info "ETTh1_experiments(20240825)_OPTIMAL_SETTING" \
#   --exp_protocol "Multi_ETTh1_512_720" \
#   --use_multi_gpu

# CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 --master_port=29500 run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_512_96 \
#   --model S2IPLLM \
#   --data ETTh2 \
#   --number_variable 7 \
#   --features M \
#   --seq_len 512 \
#   --label_len 0 \
#   --pred_len 96 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 768 \
#   --learning_rate 0.0001 \
#   --patch_size 16 \
#   --stride 8 \
#   --add_prompt 1 \
#   --prompt_length 16 \
#   --batch_size 128 \
#   --sim_coef -0.05 \
#   --pool_size  1000 \
#   --percent 100 \
#   --trend_length 96 \
#   --seasonal_length 96 \
#   --train_epochs 100 \
#   --exp_info "ETTh2_experiments(20240825)_OPTIMAL_SETTING" \
#   --exp_protocol "Multi_ETTh2_512_96" \
#   --use_multi_gpu

# CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 --master_port=29500 run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_512_192 \
#   --model S2IPLLM \
#   --data ETTh2 \
#   --number_variable 7 \
#   --features M \
#   --seq_len 512 \
#   --label_len 0 \
#   --pred_len 192 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 768 \
#   --learning_rate 0.0001 \
#   --patch_size 16 \
#   --stride 8 \
#   --add_prompt 1 \
#   --prompt_length 4 \
#   --batch_size 128 \
#   --sim_coef -0.05 \
#   --pool_size  1000 \
#   --percent 100 \
#   --trend_length 96 \
#   --seasonal_length 12 \
#   --train_epochs 100 \
#   --exp_info "ETTh2_experiments(20240825)_OPTIMAL_SETTING" \
#   --exp_protocol "Multi_ETTh2_512_192" \
#   --use_multi_gpu

# CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 --master_port=29500 run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_512_336 \
#   --model S2IPLLM \
#   --data ETTh2 \
#   --number_variable 7 \
#   --features M \
#   --seq_len 512 \
#   --label_len 0 \
#   --pred_len 336 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 768 \
#   --learning_rate 0.0001 \
#   --patch_size 16 \
#   --stride 8 \
#   --add_prompt 1 \
#   --prompt_length 8 \
#   --batch_size 128 \
#   --sim_coef -0.05 \
#   --pool_size  1000 \
#   --percent 100 \
#   --trend_length 96 \
#   --seasonal_length 12 \
#   --train_epochs 100 \
#   --exp_info "ETTh2_experiments(20240825)_OPTIMAL_SETTING" \
#   --exp_protocol "Multi_ETTh2_512_336" \
#   --use_multi_gpu

# CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 --master_port=29500 run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_512_720 \
#   --model S2IPLLM \
#   --data ETTh2 \
#   --number_variable 7 \
#   --features M \
#   --seq_len 512 \
#   --label_len 0 \
#   --pred_len 720 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 768 \
#   --learning_rate 0.0001 \
#   --patch_size 16 \
#   --stride 8 \
#   --add_prompt 1 \
#   --prompt_length 2 \
#   --batch_size 128 \
#   --sim_coef -0.01 \
#   --pool_size  1000 \
#   --percent 100 \
#   --trend_length 24 \
#   --seasonal_length 24 \
#   --train_epochs 100 \
#   --exp_info "ETTh2_experiments(20240825)_OPTIMAL_SETTING" \
#   --exp_protocol "Multi_ETTh2_512_720_192_96" \
#   --use_multi_gpu

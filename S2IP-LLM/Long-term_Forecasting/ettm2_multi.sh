CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29500 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_96 \
  --model S2IPLLM \
  --data ETTm2 \
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
  --pool_size 1000 \
  --percent 100 \
  --trend_length 24 \
  --seasonal_length 24 \
  --train_epochs 1 \
  --exp_info "ETTm2_experiments(20240822)" \
  --exp_protocol "Multi_ETTm2_512_96" \
  --use_multi_gpu



CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29500 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_192 \
  --model S2IPLLM \
  --data ETTm2 \
  --number_variable 7 \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 192 \
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
  --prompt_length 8 \
  --batch_size 128 \
  --sim_coef -0.05 \
  --pool_size 1000 \
  --percent 100 \
  --trend_length 192 \
  --seasonal_length 48 \
  --train_epochs 1 \
  --exp_info "ETTm2_experiments(20240822)" \
  --exp_protocol "Multi_ETTm2_512_192" \
  --use_multi_gpu




CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29500 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_336 \
  --model S2IPLLM \
  --data ETTm2 \
  --number_variable 7 \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 336 \
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
  --prompt_length 8 \
  --batch_size 128 \
  --sim_coef -0.05 \
  --pool_size 1000 \
  --percent 100 \
  --trend_length 192 \
  --seasonal_length 96 \
  --train_epochs 1 \
  --exp_info "ETTm2_experiments(20240822)" \
  --exp_protocol "Multi_ETTm2_512_336" \
  --use_multi_gpu




CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29500 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --data_path ETTm2.csv \
  --model_id ETTm2_512_720 \
  --model S2IPLLM \
  --data ETTm2 \
  --number_variable 7 \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 720 \
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
  --prompt_length 8 \
  --batch_size 128 \
  --sim_coef -0.05 \
  --pool_size 1000 \
  --percent 100 \
  --trend_length 192 \
  --seasonal_length 96 \
  --train_epochs 1 \
  --exp_info "ETTm2_experiments(20240822)" \
  --exp_protocol "Multi_ETTm2_512_720" \
  --use_multi_gpu
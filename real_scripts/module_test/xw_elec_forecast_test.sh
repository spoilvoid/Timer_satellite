#!/bin/sh


# model_name=Timer_multivariate
# seq_len=672
# label_len=576
# input_len=96
# pred_len=96
# output_len=96
# patch_len=96
# ckpt_path=checkpoints/Timer_multivariate_forecast_xw_elec_d1024_n8_l8/checkpoint.pth
# data=multivariate

# # test
# torchrun --nnodes=1 --nproc_per_node=2 run.py \
#   --task_name forecast \
#   --is_finetuning 0 \
#   --seed 1 \
#   --ckpt_path $ckpt_path \
#   --root_path ./dataset/xw/elec \
#   --data $data \
#   --model_id forecast_multivariate_with_covariate \
#   --model $model_name \
#   --seq_len $seq_len \
#   --label_len $label_len \
#   --input_len $input_len \
#   --pred_len $pred_len \
#   --output_len $output_len \
#   --use_norm \
#   --covariate \
#   --n_pred_vars 15 \
#   --e_layers 8 \
#   --factor 3 \
#   --des 'Exp' \
#   --d_model 1024 \
#   --d_ff 2048 \
#   --batch_size 16 \
#   --learning_rate 3e-5 \
#   --finetune_epochs 1000 \
#   --num_workers 4 \
#   --patch_len $patch_len \
#   --train_test 0 \
#   --itr 1 \
#   --gpu 0 \
#   --test_version test \
#   --use_multi_gpu






model_name=Timer_multivariate
seq_len=672
label_len=576
input_len=96
pred_len=96
output_len=256
patch_len=96
ckpt_path=checkpoints/xw_elec_forecast_d1024_n8_l8_pruned_without_train/checkpoint.pth
data=multivariate

# test
torchrun --nnodes=1 --nproc_per_node=2 run.py \
  --task_name forecast \
  --is_finetuning 0 \
  --seed 1 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/xw/elec \
  --data $data \
  --model_id forecast_multivariate_with_covariate \
  --model $model_name \
  --seq_len $seq_len \
  --label_len $label_len \
  --input_len $input_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --use_norm \
  --covariate \
  --n_pred_vars 15 \
  --e_layers 8 \
  --factor 3 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --finetune_epochs 1000 \
  --num_workers 4 \
  --patch_len $patch_len \
  --train_test 0 \
  --itr 1 \
  --gpu 0 \
  --test_version predict \
  --use_multi_gpu

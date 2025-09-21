#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

model_name=Timer_multivariate
ckpt_path=checkpoints/xw_elec_forecast_d1024_n8_l8/checkpoint.pth
d_model=1024
d_ff=2048
e_layers=8
seq_len=672
patch_len=96
data=multivariate

# set mask rate of imputation
for subset_rand_ratio in 1
do
  # set mask rate of imputation
  for mask_rate in 0.25 0.5
  do
  python -u run.py \
    --task_name imputation \
    --is_finetuning 1 \
    --seed 1 \
    --ckpt_path $ckpt_path \
    --root_path ./dataset/xw/elec \
    --data $data \
    --model_id Timer_multivariate_imputation \
    --model $model_name \
    --subset_rand_ratio $subset_rand_ratio \
    --mask_rate $mask_rate \
    --use_norm \
    --covariate \
    --n_pred_vars 15 \
    --seq_len $seq_len \
    --input_len 0 \
    --output_len 0 \
    --patch_len $patch_len \
    --e_layers $e_layers \
    --factor 3 \
    --train_test 0 \
    --batch_size 4 \
    --d_model $d_model \
    --d_ff $d_ff \
    --des 'Exp' \
    --itr 1 \
    --use_ims \
    --learning_rate 0.001 \
    --train_epochs 100
  done
done
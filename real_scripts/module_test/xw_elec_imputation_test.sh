#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

model_name=Timer_multivariate
ckpt_path=checkpoints/module_test/xw_elec_imputation_finetune_ckpt/checkpoint.pth
d_model=256
d_ff=512
e_layers=4
seq_len=192
patch_len=24
data=multivariate

# set mask rate of imputation
for mask_rate in 0.125
do
python -u run.py \
  --task_name imputation \
  --is_finetuning 0 \
  --seed 1 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/xw/elec_v2 \
  --data $data \
  --model_id Timer_multivariate_imputation \
  --model $model_name \
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
  --test_version test
done

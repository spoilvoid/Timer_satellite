#!/bin/sh

model_name=Timer_multivariate
seq_len=672
label_len=576
input_len=96
pred_len=96
output_len=96
patch_len=96
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
data=multivariate

# train
torchrun --nnodes=1 --nproc_per_node=2 run.py \
  --task_name forecast \
  --is_finetuning 1 \
  --seed 1 \
  --ckpt_path $ckpt_path \
  --root_path ./dataset/xw/elec \
  --data $data \
  --model_id Timer_multivariate_forecast \
  --model $model_name \
  --seq_len $seq_len \
  --label_len $label_len \
  --input_len $input_len \
  --pred_len $pred_len \
  --output_len $output_len \
  --use_norm \
  --e_layers 8 \
  --factor 3 \
  --des 'Exp' \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 48 \
  --learning_rate 1e-4 \
  --finetune_epochs 10 \
  --num_workers 4 \
  --patch_len $patch_len \
  --train_test 0 \
  --itr 1 \
  --gpu 0 \
  --use_ims \
  --use_multi_gpu

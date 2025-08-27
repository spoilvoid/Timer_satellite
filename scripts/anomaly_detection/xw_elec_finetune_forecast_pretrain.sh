#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

model_name=Timer_multivariate
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
seq_len=768
d_model=1024
d_ff=2048
e_layers=8
patch_len=96
dataset_dir="./dataset/xw/elec"
data_type=multivariate_anomaly

# ergodic datasets
python3 -u run.py \
  --task_name anomaly_detection \
  --is_finetuning 1 \
  --ckpt_path $ckpt_path \
  --root_path $dataset_dir \
  --data $data_type \
  --model_id Timer_multivariate_anomaly_detection \
  --model $model_name \
  --seq_len $seq_len \
  --patch_len $patch_len \
  --use_norm \
  --covariate \
  --n_pred_vars 15 \
  --d_model $d_model \
  --d_ff $d_ff \
  --e_layers $e_layers \
  --train_test 0 \
  --batch_size 32 \
  --use_ims \
  --train_epochs 100
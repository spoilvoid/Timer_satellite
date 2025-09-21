#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

model_name=Timer_multivariate
ckpt_path=checkpoints/Timer_forecast_1.0.ckpt
seq_len=768
d_model=1024
d_ff=2048
e_layers=8
patch_len=96
dataset_dir="./dataset/xw/elec_v2"
data_type=multivariate_anomaly

for file_path in "$dataset_dir"/test/*
do
data=$(basename "$file_path")
python3 -u run.py \
  --task_name anomaly_detection \
  --is_finetuning 0 \
  --ckpt_path $ckpt_path \
  --root_path $dataset_dir \
  --data_path $data \
  --data $data_type \
  --model_id Timer_multivariate_anomaly_detection \
  --model $model_name \
  --seq_len $seq_len \
  --patch_len $patch_len \
  --use_norm \
  --covariate \
  --n_pred_vars 15 \
  --loss_threshold 10 \
  --d_model $d_model \
  --d_ff $d_ff \
  --e_layers $e_layers \
  --test_version test \
  --batch_size 128 \
  --use_ims \
  --train_epochs 10
done
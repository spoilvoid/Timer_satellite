#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

model_name=Timer_multivariate
ckpt_path=checkpoints/Timer_multivariate_anomaly_detection_multivariate_anomaly_d256_n8_l4_itr0_25-06-05_15-22-56/checkpoint.pth
seq_len=768
d_model=256
d_ff=512
e_layers=4
patch_len=96
dataset_dir="./dataset/xw/elec"
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
  --use_ims
done
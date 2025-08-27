#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

model_name=Timer_multivariate
ckpt_path=checkpoints/Timer_anomaly_detection_1.0.ckpt
seq_len=768
d_model=256
d_ff=512
e_layers=4
patch_len=96
subset_rand_ratio=0.01
dataset_dir="./dataset/D1/UCR_Anomaly_FullData"

# ergodic datasets
for file_path in "$dataset_dir"/*
do
data=$(basename "$file_path")
python -u run.py \
  --task_name anomaly_detection \
  --is_finetuning 1 \
  --root_path $dataset_dir \
  --data_path $data \
  --model_id UCRA_$data \
  --ckpt_path $ckpt_path \
  --model $model_name \
  --use_norm \
  --covariate \
  --data UCRA \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --d_model $d_model \
  --d_ff $d_ff \
  --patch_len $patch_len \
  --e_layers $e_layers \
  --train_test 0 \
  --batch_size 128 \
  --use_ims \
  --subset_rand_ratio $subset_rand_ratio \
  --train_epochs 1
done
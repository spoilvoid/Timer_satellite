python3 ./pretrain_data_gen/general_model_data_process.py \
  --input_dir /datasda1/wangweipeng/workspace/xw/data_yc/data \
  --param_dir ./pretrain_data_gen/params \
  --output_dir ./pretrain_data_gen/general_data_processed \
  --delta_t 60 \
  --n_offset 6 \
  --patch_time 90 \
  --density_threshold 0.1

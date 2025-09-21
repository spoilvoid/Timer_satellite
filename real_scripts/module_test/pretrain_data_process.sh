# 已完成
python3 ./pretrain_data_gen/general_model_data_process_single_variable.py \
  --input_dir /datasda1/wangweipeng/workspace/xw/data_yc/data \
  --param_dir ./pretrain_data_gen/params \
  --output_dir ./pretrain_data_gen/general_data_processed_v2 \
  --resume \
  --delta_t 60 \
  --n_offset 6 \
  --num_workers 8 \
  --patch_time 90 \
  --min_point_num 768 \
  --density_threshold 0.01

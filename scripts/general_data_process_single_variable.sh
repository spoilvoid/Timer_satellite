python3 ./pretrain_data_gen/general_model_data_process_single_variable.py \
  --input_dir ./pretrain_data_gen/general_data \
  --output_dir ./pretrain_data_gen/general_data_processed \
  --delta_t 60 \
  --n_offset 6 \
  --num_workers 8 \
  --patch_time 90 \
  --min_point_num 0 \
  --density_threshold 0.0

python3 enc_dec_paillier.py \
    --task key_gen \
    --key_size 2048 \
    --key_path ./keys

python3 enc_dec_paillier.py \
    --task enc \
    --key_path ./keys \
    --input_dir ./dataset/xw/elec/trainval \
    --output_dir ./dataset/xw/elec/trainval_enc

python3 enc_dec_paillier.py \
    --task dec \
    --key_path ./keys \
    --input_dir ./dataset/xw/elec/trainval_enc \
    --output_dir ./dataset/xw/elec/trainval_dec
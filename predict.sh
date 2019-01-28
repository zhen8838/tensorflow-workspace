# !/bin/bash

python3     predict.py \
            --model_path "pretrained/mobilenetv1_flowers.pb" \
            --image_dir "data" \
            --image_size 224  \
            --class_num 5
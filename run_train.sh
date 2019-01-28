#!/bin/bash
python3 train.py    --pretrained_model "pretrained/mobilenetv1_1.0.pkl" \
                    --logs_base_dir "backup_classifier" \
                    --gpus 0 \
                    --data_dir "/media/zqh/Datas/DataSet/flower_photos" \
                    --max_nrof_epochs 5 \
                    --batch_size 32 \
                    --image_size 224 \
                    --class_num 5 \
                    --keep_probability 1.0 \
                    --weight_decay 0.0 \
                    --optimizer "ADAM" \
                    --init_learning_rate 0.0006 \
                    --learning_rate_decay_epochs 20 \
                    --learning_rate_decay_factor 0.9 \
                    --seed 3 
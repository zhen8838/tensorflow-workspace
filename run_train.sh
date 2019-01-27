#!/bin/bash
python3 train.py    --logs_base_dir "backup_classifier" \
                    --models_base_dir "backup_classifier" \
                    --gpu_memory_fraction 1.0 \
                    --gpus 2 \
                    --class_num_changed False \
                    --data_dir /media/zqh/Datas/DataSet/flower_photos \
                    --max_nrof_epochs 1 \
                    --batch_size 32 \
                    --image_size 224 \
                    --class_num 5 \
                    --keep_probability 1.0 \
                    --weight_decay 0.0 \
                    --optimizer "ADAM" \
                    --learning_rate 0.0005 \
                    --learning_rate_decay_epochs 10 \
                    --learning_rate_decay_factor 0.9 \
                    --moving_average_decay 0.9999 \
                    --seed 666 \
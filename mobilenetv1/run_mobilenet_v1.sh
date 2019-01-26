#!/bin/bash

python3 train_softmax.py \
--model_def models.mobilenet_v1 \
--data_dir /media/zqh/Datas/DataSet/flower_photos \
--gpu_memory_fraction=0.85 \
--gpus 1 \
--image_size 224 \
--logs_base_dir backup_classifier \
--models_base_dir backup_classifier \
--batch_size 32 \
--epoch_size 50 \
--learning_rate 0.0004 \
--max_nrof_epochs 1 \
--class_num 5 \
--use_fixed_image_standardization \
--optimizer ADAM \
--learning_rate_schedule_file data/learning_rate.txt \
--keep_probability 1.0 

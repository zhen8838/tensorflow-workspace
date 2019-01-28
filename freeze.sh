# !/bin/bash
python3 freeze_graph.py --class_num 5 \
                        "backup_classifier/20190128-210009" \
                        "pretrained/mobilenetv1_flowers.pb"
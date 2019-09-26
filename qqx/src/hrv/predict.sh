#!/bin/sh


# 100Hz
python test.py \
    -id hrv-100-12-resnet-ensemble \
    -n resnet \
    -s ../../data/testA_txt \
    -v ../../data/testA_hrv \
    -m ../../models/hrv-100-12-resnet \
    -t ../../data/hf_round1_subA.txt \
    -o ../../outputs/hrv-100-12-resnet \
    -d 100 \
    -g 0

# 200Hz
# python test.py \
#     -id hrv-200-12-resnet-ensemble \
#     -n resnet \
#     -s ../../data/testA_txt \
#     -v ../../data/testA_hrv \
#     -m ../../models/hrv-200-12-resnet \
#     -t ../../data/hf_round1_subA.txt \
#     -o ../../outputs/hrv-200-12-resnet \
#     -d 200 \
#     -g 0
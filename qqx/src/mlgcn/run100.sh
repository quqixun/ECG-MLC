#!/bin/sh


# 100Hz
python train.py \
    -s ../../data/train_npy_12 \
    -v ../../data/train_hrv \
    -l ../../data/train.csv \
    -m ../../models \
    -f ./params.json \
    -p graph-100-12-resnet \
    -c 5 \
    -g 0

#!/bin/sh


# 200Hz
python train.py \
    -s ../../data/train_200Hz \
    -v ../../data/train_hrv \
    -l ../../data/train.csv \
    -m ../../models \
    -f ./params.json \
    -p hrv-200-12-resnet \
    -c 5 \
    -g 1

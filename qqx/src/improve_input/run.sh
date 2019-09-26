#!/bin/sh

# python train.py \
#     -s ../../data/train_npy_12 \
#     -l ../../data/train.csv \
#     -m ../../models \
#     -f ./params.json \
#     -p i2-resnet \
#     -c 5 \
#     -g 1

python train.py \
    -s ../../data/train_npy_12 \
    -l ../../data/train.csv \
    -m ../../models \
    -f ./params.json \
    -p i2-12-resnet \
    -c 5 \
    -g 0
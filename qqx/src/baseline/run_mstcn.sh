#!/bin/sh

python train.py \
    -s ../../data/train_npy \
    -l ../../data/train.csv \
    -m ../../models \
    -f ./params.json \
    -p baseline-mstcn \
    -c 5 \
    -g 1

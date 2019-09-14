#!/bin/sh

python train.py \
    -s ../../data/train_npy \
    -l ../../data/train.csv \
    -m ../../models \
    -f ./params.json \
    -p baseline-tcn \
    -c 5 \
    -g 1

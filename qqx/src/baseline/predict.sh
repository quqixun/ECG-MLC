#!/bin/sh

python test.py \
    -id baseline-resnet \
    -s ../../data/testA_txt \
    -m ../../models/baseline-resnet/1/model.pth \
    -t ../../data/hf_round1_subA.txt \
    -o ../../outputs/baseline-resnet \
    -g 0

python test.py \
    -id baseline-resnet-ensemble \
    -s ../../data/testA_txt \
    -m ../../models/baseline-resnet \
    -t ../../data/hf_round1_subA.txt \
    -o ../../outputs/baseline-resnet \
    -g 0

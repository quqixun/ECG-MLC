#!/bin/sh


python test.py \
    -id st-12-resnet-ensemble \
    -n resnet \
    -s ../../data/testA_txt \
    -m ../../models/st-12-resnet \
    -t ../../data/hf_round1_subA.txt \
    -o ../../outputs/st-12-resnet \
    -g 0
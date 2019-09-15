#!/bin/sh

# python test.py \
#     -id baseline-resnet \
#     -n reesnet \
#     -s ../../data/testA_txt \
#     -m ../../models/baseline-resnet/1/model.pth \
#     -t ../../data/hf_round1_subA.txt \
#     -o ../../outputs/baseline-resnet \
#     -g 0

# python test.py \
#     -id baseline-resnet-ensemble \
#     -n reesnet \
#     -s ../../data/testA_txt \
#     -m ../../models/baseline-resnet \
#     -t ../../data/hf_round1_subA.txt \
#     -o ../../outputs/baseline-resnet \
#     -g 0

python test.py \
    -id baseline-tcn \
    -n tcn \
    -s ../../data/testA_txt \
    -m ../../models/baseline-tcn/2/model.pth \
    -t ../../data/hf_round1_subA.txt \
    -o ../../outputs/baseline-tcn \
    -g 0

python test.py \
    -id baseline-tcn-ensemble \
    -n tcn \
    -s ../../data/testA_txt \
    -m ../../models/baseline-tcn \
    -t ../../data/hf_round1_subA.txt \
    -o ../../outputs/baseline-tcn \
    -g 0

python test.py \
    -id baseline-mstcn \
    -n mstcn \
    -s ../../data/testA_txt \
    -m ../../models/baseline-mstcn/3/model.pth \
    -t ../../data/hf_round1_subA.txt \
    -o ../../outputs/baseline-mstcn \
    -g 0

python test.py \
    -id baseline-mstcn-ensemble \
    -n mstcn \
    -s ../../data/testA_txt \
    -m ../../models/baseline-mstcn \
    -t ../../data/hf_round1_subA.txt \
    -o ../../outputs/baseline-mstcn \
    -g 0

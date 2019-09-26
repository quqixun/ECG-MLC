#!/bin/sh


# python test.py \
#     -id i2-resnet-ensemble \
#     -n resnet \
#     -s ../../data/testA_txt \
#     -m ../../models/i2-resnet \
#     -t ../../data/hf_round1_subA.txt \
#     -o ../../outputs/i2-resnet \
#     -g 0

# python test.py \
#     -id i2-12-resnet-ensemble \
#     -s ../../data/testA_txt \
#     -m ../../models/i2-12-resnet \
#     -t ../../data/hf_round1_subA.txt \
#     -o ../../outputs/i2-12-resnet \
#     -g 0

python test.py \
    -id i2-12-resnet-correct-ensemble \
    -s ../../data/testA_txt \
    -m ../../models/i2-12-resnet \
    -t ../../data/hf_round1_subA.txt \
    -o ../../outputs/i2-12-resnet \
    -g 0
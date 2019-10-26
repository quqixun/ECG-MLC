#!/bin/sh

# Testing model
# Ensemble 5 predictions
python ecg_test.py \
    -s /tcdata/hf_round2_testA \
    -m ./user_data/models2 \
    -t /tcdata/hf_round2_subA.txt \
    -a ./data/hf_round2_arrythmia.txt \
    -g 0

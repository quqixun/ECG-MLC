#!/bin/sh

# Preprocessing
# -1- Convert labels of train and testA to csv
# -2- Merge labels of train and testA
# -3- Remove duplicates in train and testA
python ./prep/ecg_prep.py \
    -a ../data/hf_round1_arrythmia.txt \
    -tt ../data/hf_round1_label.txt \
    -at ../data/hefei_round1_ansA_20191008.txt \
    -td ../data/hf_round1_train/train \
    -ad ../data/hf_round1_testA/testA \
    -o ../user_data

# Training model
# 5 Cross Validation
# This step may take long time
python ./train/ecg_train.py \
    -l ../user_data/train_testA_noDup.csv \
    -m ../user_data/models \
    -c 5 \
    -g 0

# Testing model
# Ensemble 5 predictions
python ./test/ecg_test.py \
    -s ../data/hf_round1_testB_noDup_rename/testB_noDup_rename \
    -m ../user_data/models \
    -t ../data/hf_round1_subB_noDup_rename.txt \
    -a ../data/hf_round1_arrythmia.txt \
    -o ../prediction_result \
    -g 0

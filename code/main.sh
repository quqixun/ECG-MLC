#!/bin/sh

# Preprocessing
# -1- Unzip data into ../user_data
# -2- Convert labels of train and testA to csv
# -3- Merge labels of train and testA
# -4- Remove duplicates in train and testA
python ./prep/ecg_prep.py \
    -a ../data/hf_round1_arrythmia.txt \
    -tt ../data/hf_round1_label.txt \
    -at ../data/hefei_round1_ansA_20191008.txt \
    -tz ../data/hf_round1_train.zip \
    -az ../data/hf_round1_testA.zip \
    -bz ../data/hf_round1_testB_noDup_rename.zip \
    -o ../user_data

# Training model
# 5 Cross Validation
python ./train/ecg_train.py \
    -l ../user_data/train_testA_noDup.csv \
    -m ../user_data/models \
    -c 5 \
    -g 0

# Testing model
# Ensemble 5 predictions
python ./test/ecg_test.py \
    -s ../user_data/testB_noDup_rename \
    -m ../user_data/models \
    -t ../data/hf_round1_subB_noDup_rename.txt \
    -a ../data/hf_round1_arrythmia.txt \
    -o ../prediction_result \
    -g 0

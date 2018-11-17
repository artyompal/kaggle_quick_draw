#!/bin/bash

./predict_train.sh \
    predict_resnet_64x64.py \
    ../models/train_2b_resnet34_64x64/train_2b_resnet34_64x64_best_model.pk \
    ../predicts/train_resnet34_64x64 \
    ../data/train_raw/


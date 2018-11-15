#!/bin/bash

./predict_train.sh \
    predict_mobilenet_64x64.py \
    ../models/train_2a_mobilenet_64x64/train_2a_mobilenet_64x64_best_model.pk \
    ../predicts/train_mobilenet_64x64 \
    ../data/train_raw/


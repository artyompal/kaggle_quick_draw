#!/bin/bash

./predict_train.sh \
    predict_mobilenet_128x128.py \
    ../models/train_1h_mobilenet_128x128/train_1h_mobilenet_128x128_best_model.pk \
    ../predicts/train_mobilenet_128x128 \
    ../data/train_raw/


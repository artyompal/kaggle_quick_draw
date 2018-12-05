#!/bin/bash

for pred in *.npy
do
    dst="${pred%%.*}"
    dst="$dst.csv"

    echo "predicting from $pred to $dst"
    ./blend2.py $dst $pred
done


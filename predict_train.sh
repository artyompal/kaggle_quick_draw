#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "usage: $0 <predict_script.py> /path/to/model <dst_dir> <source_dir>"
    exit
fi

PREDICT_SCRIPT=$1
MODEL_PATH=$2
DEST_DIR=$3
SOURCE_FILES="$4/*.csv"

for csv in $(ls $SOURCE_FILES | LC_ALL=C sort)
do
    dst="$(basename -- $csv)"
    dst="${dst%%.*}"
    dst="$DEST_DIR/$dst.npz"

    echo "predicting from $csv to $dst"
    "./$PREDICT_SCRIPT" $dst $MODEL_PATH $csv
done


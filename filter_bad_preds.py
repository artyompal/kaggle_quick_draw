#!/usr/bin/python3.6
""" Takes prediction as a series of numpy arrays, generates a submission csv file. """

import os, sys
from typing import *
from glob import glob
import re, inspect
import numpy as np,  pandas as pd
from tqdm import tqdm

NpArray = Any

def dprint(*args: Any) -> None:
    frame = inspect.currentframe().f_back           # type: ignore
    with open(inspect.getframeinfo(frame).filename) as f:
        line = f.readlines()[frame.f_lineno - 1]    # type: ignore

    m = re.match(r'\s*dprint\((.*)\)\s*', line)
    if m:
        print(m.group(1), *args)
    else:
        assert False

if __name__ == "__main__":
    '''
    Algorithm:
    for every csv:
        load both predicts
        find wrong predicts for each of them
        join those sets
        filter dataframe using this set
        write csv
    '''

    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <destination_dir> <prediction_dir1> ...")
        sys.exit(0)

    destination_dir = sys.argv[1]
    source_dir = "../data/train_raw"
    os.makedirs(destination_dir, exist_ok=True)

    classes = sorted([os.path.basename(f)[:-4] for f in glob("../data/train_raw/*.csv")])

    for train_path in sorted(glob(os.path.join(source_dir, "*.csv"))):
        print("processing", train_path)
        name = os.path.splitext(os.path.split(train_path)[1])[0]

        train_csv = pd.read_csv(train_path)
        train_csv["word"] = train_csv["word"].apply(lambda s: s.replace(' ', '_'))
        ground_truth = train_csv["word"].values
        filter = np.zeros(train_csv.shape[0], dtype=bool)

        for pred_dir in sys.argv[2:]:
            print("using predict", pred_dir)
            pred_path = os.path.join(pred_dir, name + ".npz")
            pred = np.load(pred_path)

            pred_indices = pred["pred_indices"]
            pred_indices = pred_indices[:, 0]   # take top predict
            dprint(pred_indices)

            predicts = np.array([classes[p] for p in pred_indices])
            print(predicts.shape)

            dprint(predicts)
            dprint(ground_truth)

            filter = filter | (predicts != ground_truth)

        print("total", filter.shape[0], "wrong", np.sum(filter))
        filtered_csv = train_csv[filter]
        filtered_csv.to_csv(os.path.join(destination_dir, name + ".csv"))

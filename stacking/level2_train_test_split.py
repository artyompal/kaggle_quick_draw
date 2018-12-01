#!/usr/bin/python3.6
""" Converts the data into a format, suitable for training. """

import multiprocessing, os, sys
from functools import partial
from glob import glob
from typing import *

import numpy as np, pandas as pd
from tqdm import tqdm

TRAIN_IMGS_PER_CLASS = 400
VAL_IMGS_PER_CLASS = 100

if __name__ == "__main__":
    big_val = pd.read_csv("val_pavel.csv")
    small_val = pd.read_csv("val_common_simple.csv")

    train_level2 = []

    difference = big_val[~big_val.key_id.isin(small_val.key_id)]
    for word, df in tqdm(difference.groupby("word")):
        train_level2.append(df.iloc[:TRAIN_IMGS_PER_CLASS])

    train_df = pd.concat(train_level2)
    train_df.to_csv("level2_train.csv", index=False)

    val_df = big_val[~big_val.key_id.isin(train_df.key_id)]
    print(val_df.shape)
    val_df.to_csv("level2_val.csv", index=False)

    train_indices = big_val.index[big_val.key_id.isin(train_df.key_id)]
    val_indices = big_val.index[big_val.key_id.isin(val_df.key_id)]
    print(train_indices)
    print(val_indices)

    np.save("level2_train_indices", train_indices)
    np.save("level2_val_indices", val_indices)

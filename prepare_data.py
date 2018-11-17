#!/usr/bin/python3.6
""" Converts the data into a format, suitable for training. """

import multiprocessing, os, sys
from functools import partial
from glob import glob
from typing import *

import numpy as np, pandas as pd
from tqdm import tqdm

VALIDATION_FRACTION = 30    # 1 sample out of N sample is in validation dataset
SAMPLES_PER_EPOCH   = 2000  # number of samples per epoch per class
VAL_SAMPLES_PER_EPOCH = 100
SEED = 7


def split_csv(dest_train: str, dest_val: str, csv: str) -> None:
    # print("processing", csv)
    df = pd.read_csv(csv)

    np.random.seed(SEED)
    val_indices = np.random.choice(df.shape[0], df.shape[0] // VALIDATION_FRACTION,
                                   replace=False)
    val_df = df[df.index.isin(val_indices)]
    train_df = df[~df.index.isin(val_indices)]
    assert(val_df.shape[0] + train_df.shape[0] == df.shape[0])

    # write val
    basename = os.path.basename(csv)
    val_df.to_csv(os.path.join(dest_val, basename), index=False)

    # write train by chunks
    basename = os.path.splitext(basename)[0]
    for ofs in range(0, train_df.shape[0], SAMPLES_PER_EPOCH):
        part = train_df[ofs: ofs + SAMPLES_PER_EPOCH]
        i = ofs // SAMPLES_PER_EPOCH
        part.to_csv(os.path.join(dest_train, f"{basename}_part_{i:02d}.csv"),
                    index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <dest_train_dir> <dest_val_dir> <source_data_dir>")
        sys.exit()

    pool = multiprocessing.Pool(processes=16)
    split_func = partial(split_csv, sys.argv[1], sys.argv[2])

    file_list = glob(os.path.join(sys.argv[3], "*.csv"))
    print(len(list(tqdm(pool.imap_unordered(split_func, file_list), total=len(file_list)))))

#     val = pd.concat([pd.read_csv(csv)[:VAL_SAMPLES_PER_EPOCH] for csv in
#                      tqdm(glob(os.path.join(sys.argv[2], "*.csv")))])
#     val.to_csv("../data/validation.csv", index=False)


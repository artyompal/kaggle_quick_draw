#!/usr/bin/python3.6
""" Converts the data into a format, suitable for training. """

import multiprocessing, os, sys
from functools import partial
from glob import glob
from typing import *

import numpy as np, pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <dest_file> <dest_val_dir>")
        sys.exit()

    val = pd.concat([pd.read_csv(csv) for csv in tqdm(glob(os.path.join(sys.argv[2], "*.csv")))])
    val.to_csv(sys.argv[1], index=False)


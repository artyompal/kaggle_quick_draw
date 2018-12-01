#!/usr/bin/python3.6
""" Converts the data into a format, suitable for training. """

import multiprocessing, os, sys
from functools import partial
from glob import glob
from typing import *

import numpy as np, pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} <res> <csv1> <csv2>")
        sys.exit()

    df1 = pd.read_csv(sys.argv[2])
    df2 = pd.read_csv(sys.argv[3])

    df = pd.merge(df1, df2, how='inner', on=['drawing', 'word', 'recognized',
                                             'countrycode'])
    df.drop(columns=["timestamp_x", "timestamp_y"], inplace=True)

    # 97% of data have at least 10 samples
    df = df.groupby('word').head(10)

    df.to_csv(sys.argv[1], index=False)

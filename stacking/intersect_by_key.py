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

    df = pd.merge(df1, df2, how='inner', on='key_id')
    df.drop(columns=['timestamp', 'countrycode_y', 'drawing_y', 'recognized_y', 'word_y'],
            inplace=True)
    df.rename(columns={'drawing_x': 'drawing', 'word_x': 'word', 'recognized_x':
                       'recognized', 'countrycode_x': 'countrycode'}, inplace=True)
    df.to_csv(sys.argv[1], index=False)

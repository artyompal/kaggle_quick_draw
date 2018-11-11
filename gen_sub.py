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
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <submission.csv> <predictions.npz>")
        sys.exit(0)

    predictions = sys.argv[2]
    submission = sys.argv[1]
    data = np.load(predictions)

    pred_indices = data["pred_indices"]
    confidences = data["pred_confs"]

    print("pred_indices", pred_indices.shape)
    print(pred_indices)
    print("confidences", confidences.shape)
    print(confidences)

    # get list of classes
    classes = [os.path.basename(f) for f in glob("../data/train_simplified/*.csv")]
    classes = sorted([s[:-4].replace(' ', '_') for s in classes])

    # get list of test samples
    test_samples = pd.read_csv("../data/test_simplified.csv")["key_id"].values

    # get predictions
    predicts = []
    for indices in pred_indices:
        pred = " ".join([classes[i] for i in indices[:3]])
        predicts.append(pred)

    sub = pd.DataFrame({"key_id": test_samples, "word": predicts})
    sub.to_csv(submission, index=False)

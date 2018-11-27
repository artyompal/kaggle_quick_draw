#!/usr/bin/python3.6
""" Takes prediction as a series of numpy arrays, generates a submission csv file. """

import os, sys
from typing import *
from glob import glob
from collections import defaultdict
import re, inspect
import numpy as np,  pandas as pd
from tqdm import tqdm
import torch

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

def load_prediction(filename: str) -> Tuple[NpArray, NpArray]:
    """ Reads predicts from file. """
    preds = np.load(filename)

    if filename.startswith("seres"):
        preds = torch.nn.functional.softmax(torch.as_tensor(preds)).numpy()

    print("preds", preds.shape)
    print(preds)
    return preds

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <submission.csv> <prediction1.npz> ...")
        sys.exit(0)

    submission = sys.argv[1]
    predicts = load_prediction(sys.argv[2])

    for filename in tqdm(sys.argv[3:]):
        predicts += load_prediction(filename)

    # we don't have to normalize, just take top-3 predictions
    predicted_classes = np.argsort(predicts)
#     dprint(predicted_classes.shape)
#     dprint(np.argmax(predicts[0]))
#     dprint(np.amax(predicts[0]))

    predicted_classes = predicted_classes[:, -1:-4:-1]
#     dprint(predicted_classes.shape)
#     dprint(predicted_classes[0, 0])
#     dprint(predicts[predicted_classes[0, 0], 0])


    # get list of classes
    classes = [os.path.basename(f) for f in glob("../data/train_simplified/*.csv")]
    classes = sorted([s[:-4].replace(' ', '_') for s in classes])

    # get list of test samples
    test_samples = pd.read_csv("../data/test_raw.csv")["key_id"].values

    # get predictions
    lines = []
    for indices in predicted_classes:
        pred = " ".join([classes[i] for i in indices[:3]])
        lines.append(pred)

    sub = pd.DataFrame({"key_id": test_samples, "word": lines})
    sub.to_csv(submission, index=False)

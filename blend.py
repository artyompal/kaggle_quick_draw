#!/usr/bin/python3.6
""" Takes prediction as a series of numpy arrays, generates a submission csv file. """

import os, sys
from typing import *
from glob import glob
from collections import defaultdict
import re, inspect
import numpy as np,  pandas as pd
from tqdm import tqdm

NpArray = Any
TOP_K = 20

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
    data = np.load(filename)
    pred_indices = data["pred_indices"]
    confidences = data["pred_confs"]

    print("pred_indices", pred_indices.shape)
    print(pred_indices)
    print("confidences", confidences.shape)
    print(confidences)

    return pred_indices, confidences

def combine_predictions(indices1: NpArray, confs1: NpArray, indices2: NpArray,
                        confs2: NpArray) -> Tuple[NpArray, NpArray]:
    """ Joins two predictions, returns sorted top-3 results in every row """
    dprint(indices1.shape)
    dprint(indices2.shape)
    assert indices1.shape == indices2.shape
    assert confs1.shape == confs2.shape

    merged_indices = []
    merged_confs = []

    for idx1, conf1, idx2, conf2 in tqdm(zip(indices1, confs1, indices2, confs2),
                                         total=indices1.shape[0]):
        items: DefaultDict[int, float] = defaultdict(float)

        for i, c in zip(idx1, conf1):
            items[i] += c

        for i, c in zip(idx2, conf2):
            items[i] += c

        indices = sorted(items.keys(), key=lambda i: -items[i])
        confs = [items[i] for i in indices]

        merged_indices.append(indices[:TOP_K])
        merged_confs.append(confs[:TOP_K])

    return np.array(merged_indices), np.array(merged_confs)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <submission.csv> <prediction1.npz> ...")
        sys.exit(0)

    submission = sys.argv[1]
    predicts = load_prediction(sys.argv[2])

    for filename in sys.argv[3:]:
        another = load_prediction(filename)
        predicts = combine_predictions(predicts[0], predicts[1], another[0], another[1])

    # we don't have to normalize, just return top-3 predictions
    predicted_classes = predicts[0]

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

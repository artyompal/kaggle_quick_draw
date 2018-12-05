#!/usr/bin/python3.6
""" Converts the data into a format, suitable for training. """

import multiprocessing, os, sys
from functools import partial
from glob import glob
from typing import *

import numpy as np, pandas as pd
from tqdm import tqdm

from metrics import mapk
import torch


def get_classes_list() -> List[str]:
    classes = [os.path.basename(f)[:-4] for f in glob("../../data/train_raw/*.csv")]
    return sorted(classes, key=lambda s: s.lower())

if __name__ == '__main__':
    test_paths = sorted(glob('new/*test*.npy'))
    print(test_paths)
    train_paths = [s.replace('test', 'train') for s in test_paths]

    train_df = pd.read_csv('val_pavel.csv')
    # val_df = pd.read_csv('level2_val.csv')
    classes = get_classes_list()
    class2idx = {c.replace('_', ' '): i for i, c in enumerate(classes)}

    test_predicts = [np.load(pred) for pred in test_paths]
    train_predicts = np.array([np.load(pred) for pred in train_paths])
    print("train_predicts", train_predicts.shape)

    # train_idx = np.load('level2_train_indices.npy')
    # val_idx =  np.load('level2_val_indices.npy')
    # train_predicts = all_predicts[:, train_idx, :]
    # val_predicts = all_predicts[:, val_idx, :]
    # print("train_predicts", train_predicts.shape)
    # print("val_predicts", val_predicts.shape)

    train_targets = train_df.word.apply(lambda s: class2idx[s]).values
    print("train_targets", train_targets.shape)
    # val_targets = val_df.word.apply(lambda s: class2idx[s]).values
    # print("val_targets", val_targets.shape)

    ###########################################################################
    # Search
    ###########################################################################
    def loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        weights[1] = weights[0] = 0
        weights[3:5] = weights[5]
        weights[7:] = weights[6]

        weights /= np.sum(weights)
        # print("weights", weights)
        final_predict = np.zeros_like(train_predicts[0])

        for weight, prediction in zip(weights, train_predicts):
            # print("weight", weight, "prediction", prediction)
            final_predict += weight * prediction

        # print("final_predict", final_predict)
        # print("train_targets", train_targets.shape)
        score = mapk(torch.tensor(final_predict), torch.tensor(train_targets))
        print("score", score)
        return score

    best_score = 0
    best_weights = np.zeros(train_predicts.shape[0])
    start_weights = np.array([0.03061995, 0.03061995, 0.33040306, 0.04477581, 0.04477581,
                              0.2565899, 0.06555388, 0.06555388, 0.06555388, 0.06555388])

    while True:
        weights = start_weights + np.random.rand(train_predicts.shape[0]) * 0.1 - 0.05
        weights = np.clip(weights, 0, 1)
        weights /= np.sum(weights)

        score = loss_func(weights)
        if score > best_score:
            best_score,best_weights = score, weights
            print("better score:", best_score)
            print("best_weights", best_weights)


    ###########################################################################
    # Generate results
    ###########################################################################

    best_weights /= sum(best_weights)
    final_predict = np.zeros_like(test_predicts[0])

    for weight, prediction in zip(best_weights, test_predicts):
        final_predict += weight * prediction

    # pred_indices = np.argsort(final_predict, axis=1)[:, 3]
    # print(pred_indices.shape)
    # print(pred_indices)
    predicts = []

    for predict in final_predict:
        predict = np.argsort(-predict)[:3]

        pred = " ".join([classes[i] for i in predict])
        predicts.append(pred)

    test_samples = pd.read_csv("../../data/test_raw.csv")["key_id"].values
    sub = pd.DataFrame({"key_id": test_samples, "word": predicts})
    sub.to_csv("weighted_blend.csv", index=False)

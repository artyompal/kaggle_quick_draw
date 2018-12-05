#!/usr/bin/python3.6
""" Converts the data into a format, suitable for training. """

import multiprocessing, os, sys
from functools import partial
from glob import glob
from typing import *

import numpy as np, pandas as pd
from tqdm import tqdm

from metrics import mapk
from scipy.optimize import minimize
import torch


def get_classes_list() -> List[str]:
    classes = [os.path.basename(f)[:-4] for f in glob("../../data/train_raw/*.csv")]
    return sorted(classes, key=lambda s: s.lower())

if __name__ == '__main__':
    test_paths = sorted(glob('new/*test*.npy'))
    train_paths = [s.replace('test', 'train') for s in test_paths]

    train_df = pd.read_csv('level2_train.csv')
    val_df = pd.read_csv('level2_val.csv')
    classes = get_classes_list()
    class2idx = {c.replace('_', ' '): i for i, c in enumerate(classes)}

    test_predicts = [np.load(pred) for pred in test_paths]
    all_predicts = np.array([np.load(pred) for pred in train_paths])
    print("all_predicts", all_predicts.shape)

    train_idx = np.load('level2_train_indices.npy')
    val_idx =  np.load('level2_val_indices.npy')
    train_predicts = all_predicts[:, train_idx, :]
    val_predicts = all_predicts[:, val_idx, :]
    print("train_predicts", train_predicts.shape)
    print("val_predicts", val_predicts.shape)

    train_targets = train_df.word.apply(lambda s: class2idx[s]).values
    val_targets = val_df.word.apply(lambda s: class2idx[s]).values
    print("train_targets", train_targets.shape)
    print("val_targets", val_targets.shape)

    def loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        weights /= np.sum(weights)
        print("weights", weights)
        final_predict = np.zeros_like(train_predicts[0])

        for weight, prediction in zip(weights, train_predicts):
            # print("weight", weight, "prediction", prediction)
            final_predict += weight * prediction

        # print("final_predict", final_predict)
        # print("train_targets", train_targets.shape)
        score = -mapk(torch.tensor(final_predict), torch.tensor(train_targets))
        print("score", score)
        return score

    # the algorithms need a starting value, right not we chose 0.5 for all weights
    # its better to choose many random starting points and run minimize a few times
    starting_values = [1 / len(train_predicts)] * len(train_predicts)

    # adding constraints  and a different solver as suggested by user 16universe
    # https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
    # cons = ({'type': 'eq','fun': lambda w: 1 - sum(w)})

    # our weights are bound between 0 and 1
    bounds = [(0, 1)] * len(train_predicts)

    res = minimize(loss_func, starting_values, method='Nelder-Mead', bounds=bounds)
                   # options={'disp': False, 'maxiter': 100000})
                   # ) #, constraints=cons)

    best_score = res['fun']
    best_weights = res['x']
    print("best_score", best_score, "best_weights", best_weights)

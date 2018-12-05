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
    test_paths = list(glob('new/*test*.npy'))
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

#     ###########################################################################
#     # Search
#     ###########################################################################
#     def loss_func(weights):
#         ''' scipy minimize will pass the weights as a numpy array '''
#         weights /= np.sum(weights)
#         # print("weights", weights)
#         final_predict = np.zeros_like(train_predicts[0])
#
#         for weight, prediction in zip(weights, train_predicts):
#             # print("weight", weight, "prediction", prediction)
#             final_predict += weight * prediction
#
#         # print("final_predict", final_predict)
#         # print("train_targets", train_targets.shape)
#         score = mapk(torch.tensor(final_predict), torch.tensor(train_targets))
#         print("score", score)
#         return score
#
#     best_score = 0
#     best_weights = np.zeros(train_predicts.shape[0])
#
#     while True:
#         weights = np.random.rand(train_predicts.shape[0])
#         weights /= np.sum(weights)
#
#         score = loss_func(weights)
#         if score > best_score:
#             best_score,best_weights = score, weights
#             print("better score:", best_score)
#             print("best_weights", best_weights)


    ###########################################################################
    # Generate results
    ###########################################################################

    # best_weights = np.array([0.02998098, 0.11336356, 0.06901118, 0.14185379, 0.04836411,
    #                          0.03454324, 0.08777448, 0.04318226, 0.35650676, 0.07541965])
    # best_weights = np.array([0.00138377, 0.00138377, 0.28841285, 0.16506287, 0.16506287,
    #                         0.10249708, 0.0690492, 0.0690492, 0.0690492, 0.0690492])

    best_weights = np.array([0, 0, 0.37989285, 0, 0, 0.3718497, 0.06206436, 0.06206436, 0.06206436, 0.06206436])

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

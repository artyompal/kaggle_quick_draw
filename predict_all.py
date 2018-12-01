#!/usr/bin/python3.6
""" Takes prediction as a series of numpy arrays, generates a submission csv file. """

import os, sys, subprocess
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
    dprint(data.shape)
    return data

def get_model_params(model_name: str) -> Tuple[str, str]:
    """ Returns two strings: data loader name and input image resolution. """
    model_name = model_name.split('_[')[0]
    if model_name.endswith("_best"):
        model_name = model_name[:-5]

    assert os.path.exists(model_name + ".py")
    data_loader, resolution = None, None
    with open(model_name + ".py") as f:
        for line in f:
            # cfg.DATASET.DATA_LOADER = 'data_loader_v1'
            if line.find("cfg.DATASET.DATA_LOADER") != -1:
                data_loader = line.split('=')[1].strip()
                data_loader = data_loader[1: -1]

            # opt.MODEL.IMAGE_SIZE = 224
            if line.find("opt.MODEL.IMAGE_SIZE =") != -1:
                resolution = line.split("=")[1].strip()

    if data_loader is None:
        print("could not parse model", model_name, "data loader not detected")
        assert False

    if resolution is None:
        print("could not parse model", model_name, "resolution not detected")
        assert False

    print(f"data_loader='{data_loader}', resolution='{resolution}'")
    return data_loader, resolution

def predict(model: str, dest: str, csv: str) -> None:
    if not os.path.exists(dest):
        ret = subprocess.run(["./predict_pretrainedmodels.py", dest,
                              model, csv, resolution, data_loader])
        if ret.returncode:
            print("subprocess failed")
            sys.exit()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <model1.npy> ...")
        sys.exit(0)

    print("predicting on test")

    for model in sys.argv[1:]:
        model_name = os.path.splitext(os.path.basename(model))[0]
        print("\nprocessing model", model_name)

        # parse a py file to get these options
        data_loader, resolution = get_model_params(model_name)
        use_simple_data = data_loader == "data_loader_v1"

        pred_test = f"../output/pred_test_{model_name}.npy"
        source = "test_simplified" if use_simple_data else "test_raw"
        predict(model, pred_test, f"../data/{source}.csv")

    print("predicting on validation")

    for model in sys.argv[1:]:
        model_name = os.path.splitext(os.path.basename(model))[0]
        print("\nprocessing model", model_name)

        # parse a py file to get these options
        data_loader, resolution = get_model_params(model_name)
        use_simple_data = data_loader == "data_loader_v1"

        pred_train = f"../output/pred_train_{model_name}.npy"
        source = "simple" if use_simple_data else "full"
        predict(model, pred_train, f"stacking/val_common_{source}.csv")


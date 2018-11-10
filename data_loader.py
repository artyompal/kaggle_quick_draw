""" Data loaders for training & validation. """
import os, pickle

from glob import glob
from typing import *

import numpy as np, pandas as pd

import torch.utils.data as data
from PIL import Image
from tqdm import tqdm


def get_file_list(root: str) -> List[str]:
    return sorted(glob(os.path.join(root, 'train_simplified/*.csv')))

def 

class DatasetFolder(data.Dataset):
    def __init__(self, root, transform, num_classes, conf=None):
        # self.samples = get_file_list(root)
        # self.classes = load_targets(root, self.samples, conf)
        # self.num_classes = num_classes
        #
        # assert(len(self.samples) > 0)
        # assert(len(self.classes) > 0)
        #
        # self.root = root
        # self.loader = default_loader
        # self.transform = transform

    def __getitem__(self, index: int):
        """ Returns: tuple (sample, target) """
        path = self.samples[index]
        target = np.zeros(self.num_classes, dtype=np.float32)

        for cls_ in self.classes[index]:
            target[cls_] = 1.0

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

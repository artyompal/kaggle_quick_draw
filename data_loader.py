""" Data loaders for training & validation. """
import os, pickle
from collections import defaultdict
from glob import glob
from typing import *

import numpy as np, pandas as pd
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

NpArray = Any

def get_file_table(root: str) -> DefaultDict[str, List[str]]:
    res: DefaultDict[str, List[str]] = defaultdict(list)

    for name in sorted(glob(os.path.join(root, '*.csv'))):
        basename = os.path.basename(name)[:-4]
        class_ = basename.split("_part_")[0]
        res[class_].append(name)

    return res

class DatasetFolder(data.Dataset):
    def __init__(self, root: str, transform: Any, num_classes: int, val: bool) -> None:
        self.file_table = get_file_table(root)
        self.transform = transform
        self.num_classes = num_classes
        self.validation = val

        classes = sorted(self.file_table.keys())
        self.class2idx = {cls_: idx for idx, cls_ in enumerate(classes)}

        assert(len(self.class2idx) == num_classes)
        assert(len(self.file_table) == num_classes)

    def start_new_epoch(self) -> None:
        print("preparing data for a new epoch...")
        samples = []

        for name in tqdm(self.file_table):
            if not self.validation:
                samples.append(pd.read_csv(np.random.choice(self.file_table[name])))
            else:
                samples.append(pd.read_csv(self.file_table[name][0]))

        samples_df = pd.concat(samples)
        self.samples = samples_df["drawing"].values
        self.targets = [self.class2idx[c] for c in samples_df["word"].values]
        print("done")

    def _create_image(self, strokes: str) -> NpArray:
        in_strokes = eval(strokes)
        fig, ax = plt.subplots()

        for x,y in in_strokes:
            ax.plot(x, y, linewidth=12)

        ax.axis('off')
        fig.canvas.draw()
        x = np.array(fig.canvas.renderer._renderer)
        plt.close(fig)
        print("x", x.shape)
        return Image.fromarray(x)

    def __getitem__(self, index: int) -> Tuple[NpArray, NpArray]:
        """ Returns: tuple (sample, target) """
        target = np.zeros(self.num_classes, dtype=np.float32)
        target[self.targets[index]] = 1

        sample = self._create_image(self.samples[index])
        # print(f"getitem index={index} target={self.targets[index]} sample={sample}")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

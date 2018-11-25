""" Data loaders for training & validation.
Uses raw datasets. They have 3 channels of denormalized data. """
import json, multiprocessing, os, pickle
from collections import defaultdict
from glob import glob
from typing import *
from functools import partial

import numpy as np, pandas as pd
import torch.utils.data as data
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt

NpArray = Any

SAVE_DEBUG_IMAGES = False
DEBUG_SHOW = False
MAX_VAL_SAMPLES = 200

def get_file_table(root: str) -> DefaultDict[str, List[str]]:
    res: DefaultDict[str, List[str]] = defaultdict(list)

    for name in sorted(glob(os.path.join(root, '*.csv'))):
        basename = os.path.basename(name)[:-4]
        class_ = basename.split("_part_")[0]
        res[class_].append(name)

    return res

class DatasetFolder(data.Dataset):
    def __init__(self, root: str, transform: Any, num_classes: int, mode: str,
                 image_size: int) -> None:
        print("created data loader", os.path.basename(__file__))

        self.transform = transform
        self.num_classes = num_classes
        self.mode = mode
        self.epoch = 0
        self.image_size = image_size

        countries = pd.read_csv("../data/countries.csv")
        self.country_data = {s[1]: int(255 * (s[5] / 360 + 0.5)) for _, s in countries.iterrows()}

        if mode != "test":
            self.file_table = get_file_table(root)
            classes = sorted(self.file_table.keys())
            self.class2idx = {cls_: idx for idx, cls_ in enumerate(classes)}

            assert(len(self.class2idx) == num_classes)
            assert(len(self.file_table) == num_classes)
        else:
            samples = pd.read_csv(root)
            self.samples = samples["drawing"].values
            self.countries = samples["countrycode"].values

    def start_new_epoch(self) -> None:
        print("preparing data for a new epoch...")
        samples = []

        for name in tqdm(self.file_table):
            files = self.file_table[name]
            if self.mode == "train":
                samples.append(pd.read_csv(files[self.epoch % len(files)]))
            else:
                samples.append(pd.read_csv(files[0])[:MAX_VAL_SAMPLES])

        samples_df = pd.concat(samples)
        self.samples = samples_df["drawing"].values
        self.targets = [self.class2idx[c] for c in samples_df["word"].values]
        self.countries = samples_df["countrycode"].values
        self.epoch += 1
        print("done")

    def _create_image(self, strokes_str: str, idx: int, country: str) -> NpArray:
        strokes: List[List[List[float]]] = json.loads(strokes_str)
        L = len(strokes)

        country_code = self.country_data[country] if country in self.country_data else 0

        im = Image.new('RGB', (self.image_size, self.image_size))
        draw = ImageDraw.Draw(im)

        min_x = min_y = min_t = +100500.0
        max_x = max_y = max_t = -100500.0
        max_dim = self.image_size - 1

        for stroke in strokes:
            min_x = min(min_x, min(stroke[0]))
            min_y = min(min_y, min(stroke[1]))

            max_x = max(max_x, max(stroke[0]))
            max_y = max(max_y, max(stroke[1]))

        range_x = max(1, max_x - min_x)
        range_y = max(1, max_y - min_y)

        for i, stroke in enumerate(strokes):
            stroke_num = 255 * i // L
            # min_local_t, max_local_t = min(stroke[2]), max(stroke[2])
            # time_range = max_local_t - min_local_t
            # time_range = max(1, time_range)

            prev_x = (stroke[0][0] - min_x) * max_dim // range_x
            prev_y = (stroke[1][0] - min_y) * max_dim // range_y

            for i in range(1, len(stroke[0])):
                x = (stroke[0][i] - min_x) * max_dim // range_x
                y = (stroke[1][i] - min_y) * max_dim // range_y

                draw.line([prev_x, prev_y, x, y],
                          fill=(stroke_num, 255 - stroke_num, 0))
                prev_x, prev_y = x, y

        if SAVE_DEBUG_IMAGES:
            basename = os.path.splitext(os.path.basename(__file__))[0]
            path = f"../output/{basename}"
            os.makedirs(path, exist_ok=True)
            im.save(os.path.join(path, f"{idx:06d}.jpg"))

        if DEBUG_SHOW:
            plt.imshow(im)
            plt.show()

        return im

    def __getitem__(self, index: int) -> Tuple[NpArray, Optional[NpArray]]:
        """ Returns: tuple (sample, target) """
        sample = self._create_image(self.samples[index], index, self.countries[index])

        if self.transform is not None:
            sample = self.transform(sample)

        if self.mode == "test":
            return sample
        else:
            return sample, self.targets[index]

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

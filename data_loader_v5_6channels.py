""" Data loaders for training & validation.
This model uses 6 channels of denormalized data. """
import json, multiprocessing, os, pickle
from collections import defaultdict
from glob import glob
from typing import *

import numpy as np, pandas as pd
import torch.utils.data as data
import torch

from PIL import Image, ImageDraw
from tqdm import tqdm

NpArray = Any

SAVE_DEBUG_IMAGES = False
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

        if mode != "test":
            self.file_table = get_file_table(root)
            classes = sorted(self.file_table.keys())
            self.class2idx = {cls_: idx for idx, cls_ in enumerate(classes)}

            assert(len(self.class2idx) == num_classes)
            assert(len(self.file_table) == num_classes)
        else:
            samples = pd.read_csv(root)
            self.samples = samples["drawing"].values

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
        self.epoch += 1
        print("done")

    def _create_image(self, strokes_str: str, idx: int) -> NpArray:
        strokes: List[List[List[float]]] = json.loads(strokes_str)
        L = len(strokes)

        im = Image.new('RGB', (256, 256))
        draw = ImageDraw.Draw(im)

        second = len(strokes) / 3
        third = 2 * len(strokes) / 3

        for i, line in enumerate(strokes):
            col = int(127 * i / L) + 128
            points = list(zip(line[0], line[1]))

            red = col if i < second else 0
            green = col if i >= second and i < third else 0
            blue = col if i >= third else 0

            draw.line(points, fill=(red, green, blue))

        im2 = Image.new('RGB', (256, 256))
        draw2 = ImageDraw.Draw(im2)

        for i, line in enumerate(strokes):
            col = int(255 * i / L) + 128
            points = list(zip(line[0], line[1]))

            red = col if i < third else 0
            green = col if i >= second else 0

            draw2.line(points, fill=(red, green, col))

        if SAVE_DEBUG_IMAGES:
            basename = os.path.splitext(os.path.basename(__file__))[0]
            path = f"../output/{basename}"
            os.makedirs(path, exist_ok=True)
            im.save(os.path.join(path, f"{idx:06d}_1.jpg"))
            im2.save(os.path.join(path, f"{idx:06d}_2.jpg"))

        # print("im", np.asarray(im).shape)
        # print("im2", np.asarray(im2).shape)
        res = np.concatenate([im, im2], axis=2)
        # print("res", res.shape)

        # normalize it so it has zero mean, identity std
        res -= 128
        res = res.astype(float)
        res /= 128.0
        res = res[16:-16, 16:-16, :]
        res = np.swapaxes(res, 0, -1)
        # print("res", res.shape, res.dtype)

        res = torch.tensor(res, dtype=torch.float32)
        return res

    def __getitem__(self, index: int) -> Tuple[NpArray, Optional[NpArray]]:
        """ Returns: tuple (sample, target) """
        sample = self._create_image(self.samples[index], index)

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

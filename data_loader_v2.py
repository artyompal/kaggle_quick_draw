""" Data loaders for training & validation. """
import json, multiprocessing, os, pickle
from collections import defaultdict
from glob import glob
from typing import *
from functools import partial

import numpy as np, pandas as pd
import torch.utils.data as data
from PIL import Image, ImageDraw
from tqdm import tqdm

NpArray = Any

SAVE_DEBUG_IMAGES = True
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

    def _load_csv(self, name: str) -> Any:
        files = self.file_table[name]
        if self.mode == "train":
            return pd.read_csv(files[self.epoch % len(files)])
        else:
            return pd.read_csv(files[0])[:MAX_VAL_SAMPLES]

    def start_new_epoch(self) -> None:
        print("preparing data for a new epoch...")
        load_func = partial(DatasetFolder._load_csv, self)
        pool = multiprocessing.Pool()
        samples = list(tqdm(pool.imap(load_func, self.file_table), total=len(self.file_table)))
        pool.close()
        pool.terminate()

        samples_df = pd.concat(samples)
        self.samples = samples_df["drawing"].values
        self.targets = [self.class2idx[c] for c in samples_df["word"].values]
        self.epoch += 1
        print("done")

    def _create_image(self, strokes: str, idx: int) -> NpArray:
        lines: List[List[List[float]]] = json.loads(strokes)
        L = len(lines)

        im = Image.new('RGB', (256, 256))
        draw = ImageDraw.Draw(im)

        for i, line in enumerate(lines):
            stroke_num = 255 * i // L
            start_time, end_time = min(line[2]), max(line[2])
            time_range = end_time - start_time

            for i in range(1, len(line)):
                time_code = (line[2][i] - start_time) * 255 // time_range
                draw.line([line[0][i-1], line[1][i-1], line[0][i], line[1][i]],
                          width=2, fill=(stroke_num, time_code, 128))

        if SAVE_DEBUG_IMAGES:
            im.save(f"../output/debug_images_v2/{idx:06d}.jpg")

        return im.resize((self.image_size, self.image_size), Image.LANCZOS)

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

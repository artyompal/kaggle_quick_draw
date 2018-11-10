""" Data loaders for training & validation. """
import os, pickle

from glob import glob
from typing import *

import numpy as np, pandas as pd

import torch.utils.data as data
from PIL import Image
from tqdm import tqdm


def get_file_list(root: str) -> List[str]:
    cache_path = f'../output/image_list.pkl'
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    file_list = sorted(glob(os.path.join(root, 'train/*.jpg')))
    with open(cache_path, "wb") as f:
        pickle.dump(file_list, f)

    return file_list

def load_targets(root: str, files: List[str], conf: Optional[float] = None
                 ) -> List[List[int]]:
    """ Loads target classes info. Optionally uses machine-generated labels. """
    if conf is None:
        cache_path = f'../output/image_classes.pkl'
    else:
        cache_path = f'../output/image_classes_thresh_{conf}.pkl'

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print('loading targets')
    classes = pd.read_csv(os.path.join(root, 'classes-trainable.csv'))['label_code'].values
    cls2idx = {class_: i for i, class_ in enumerate(classes)}

    labels = pd.read_csv(os.path.join(root, "train_human_labels.csv"))

    if conf is not None:
        print("human labels:", labels.shape)
        machine_labels = pd.read_csv(os.path.join(root, "train_machine_labels.csv"))
        machine_labels = machine_labels[machine_labels.Confidence >= conf]
        labels = pd.concat([labels, machine_labels])
        print("expanded labels:", labels.shape)

    img2label = dict()

    for id, table in tqdm(labels.groupby("ImageID")):
        class_labels = [cls2idx[label] for label in table["LabelName"]
                        if label in cls2idx]
        img2label[id] = class_labels

    targets = [img2label[os.path.basename(file)[:-4]] for file in files]
    del img2label

    with open(cache_path, "wb") as f:
        pickle.dump(targets, f)

    return targets


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DatasetFolder(data.Dataset):
    """A generic data loader.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, transform, num_classes, conf=None):
        self.samples = get_file_list(root)
        self.classes = load_targets(root, self.samples, conf)
        self.num_classes = num_classes

        assert(len(self.samples) > 0)
        assert(len(self.classes) > 0)

        self.root = root
        self.loader = default_loader
        self.transform = transform

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

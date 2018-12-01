#!/usr/bin/python3.6

import os, sys
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import logging
import numpy as np
import random
import time
import datetime
import pprint
from easydict import EasyDict as edict
import pandas as pd
from tqdm import tqdm

import pretrainedmodels
from utils import create_logger

from data_loader import get_data_loader

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

cudnn.benchmark = True

timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

if len(sys.argv) != 6:
    print(f'usage: {sys.argv[0]} <predict.npz> /path/to/model.pk /path/to/test/ <resolution> <data_loader>')
    sys.exit()


cfg = edict()

cfg.ROOT_DIR = ".."
cfg.EXPERIMENT_DIR = osp.join(cfg.ROOT_DIR, 'models')
if not osp.exists(cfg.EXPERIMENT_DIR):
    os.makedirs(cfg.EXPERIMENT_DIR)

cfg.DATASET = edict()
cfg.DATASET.NUM_CLASSES = 340


opt = edict()

opt.MODEL = edict()
opt.MODEL.PRETRAINED = True
opt.MODEL.IMAGE_SIZE = int(sys.argv[4])
opt.MODEL.INPUT_SIZE = int(sys.argv[4])

opt.EXPERIMENT = edict()
opt.EXPERIMENT.CODENAME = 'predict'
opt.EXPERIMENT.TASK = 'test'
opt.EXPERIMENT.DIR = osp.join(cfg.EXPERIMENT_DIR, opt.EXPERIMENT.CODENAME)

opt.LOG = edict()
opt.LOG.LOG_FILE = osp.join(opt.EXPERIMENT.DIR, f'log_{opt.EXPERIMENT.TASK}.txt')

opt.TEST = edict()
opt.TEST.CHECKPOINT = sys.argv[2]
opt.TEST.WORKERS = 12
opt.TEST.BATCH_SIZE = 256
opt.TEST.OUTPUT = sys.argv[1]


if not osp.exists(opt.EXPERIMENT.DIR):
    os.makedirs(opt.EXPERIMENT.DIR)

logger = create_logger(opt.LOG.LOG_FILE)
logger.info('\n\nOptions:')
logger.info(pprint.pformat(opt))


DATA_INFO = cfg.DATASET

# Data-loader of testing set
transform_test = transforms.Compose([
#     transforms.Resize((opt.MODEL.IMAGE_SIZE)),
#     transforms.CenterCrop(opt.MODEL.INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                          std = [ 0.229, 0.224, 0.225 ]),
])

test_dataset = get_data_loader(sys.argv[5], sys.argv[3], transform_test,
                               DATA_INFO.NUM_CLASSES, "test",
                               opt.MODEL.INPUT_SIZE)
logger.info(f'{len(test_dataset)} images are found for test')

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.TEST.BATCH_SIZE, shuffle=False, num_workers=opt.TEST.WORKERS)

last_checkpoint = torch.load(opt.TEST.CHECKPOINT)
opt.MODEL.ARCH = last_checkpoint['arch']


# create model
logger.info(f'using pre-trained model {opt.MODEL.ARCH}')
model = pretrainedmodels.__dict__[opt.MODEL.ARCH](num_classes=1000, pretrained='imagenet')
assert(opt.MODEL.INPUT_SIZE % 32 == 0)

if opt.MODEL.ARCH.startswith('resnet'):
    model.avgpool = nn.AvgPool2d(opt.MODEL.INPUT_SIZE // 32, stride=1)
    model.last_linear = nn.Linear(model.last_linear.in_features, DATA_INFO.NUM_CLASSES)
elif opt.MODEL.ARCH.startswith('dpn'):
    model.last_linear = nn.Conv2d(model.last_linear.in_channels, DATA_INFO.NUM_CLASSES, kernel_size=1, bias=True)
    model.test_time_pool = False
else:
    model.avgpool = nn.AvgPool2d(opt.MODEL.INPUT_SIZE // 32, stride=1)
    model.last_linear = nn.Linear(model.last_linear.in_features, DATA_INFO.NUM_CLASSES)

model = torch.nn.DataParallel(model).cuda()


model.module.load_state_dict(last_checkpoint['state_dict'])
logger.info(f"Checkpoint '{opt.TEST.CHECKPOINT}' was loaded.")

last_epoch = last_checkpoint['epoch']
softmax = torch.nn.Softmax(dim=1).cuda()

# pred_indices = []
# pred_scores = []
pred_confs = []

# model.eval()

with torch.no_grad():
    for input in tqdm(test_loader):
        output = model(input)
        top_scores, top_indices = torch.topk(output, k=20)
        top_indices = top_indices.data.cpu().numpy()
        top_scores = top_scores.data.cpu().numpy()

        confs = softmax(output)
#         top_confs, _ = torch.topk(confs, k=20)
#         top_confs = top_confs.data.cpu().numpy()
#
#         pred_indices.append(top_indices)
#         pred_scores.append(top_scores)
        pred_confs.append(confs)

# pred_indices = np.concatenate(pred_indices)
# pred_scores = np.concatenate(pred_scores)
pred_confs = np.concatenate(pred_confs)

# np.savez(opt.TEST.OUTPUT, pred_indices=pred_indices, pred_scores=pred_scores, pred_confs=pred_confs, checkpoint=opt.TEST.CHECKPOINT)
np.save(opt.TEST.OUTPUT, pred_confs)
logger.info(f"Results were saved to {opt.TEST.OUTPUT}")

#!/usr/bin/python3.6
""" Trains a model. """

import os
import os.path as osp
import sys

from typing import *

import torch
import torch.nn as nn
import torch.optim as optim
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
import PIL

import pretrainedmodels
from utils import create_logger, AverageMeter, accuracy
import torchsummary
from cosine_scheduler import CosineLRWithRestarts

from data_loader import get_data_loader

cudnn.benchmark = True
timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

if len(sys.argv) != 3:
    print(f'usage: {sys.argv[0]} <model> <data_loader>')
    sys.exit()


cfg = edict()

cfg.ROOT_DIR = ".."
cfg.EXPERIMENT_DIR = osp.join(cfg.ROOT_DIR, 'models')
if not osp.exists(cfg.EXPERIMENT_DIR):
    os.makedirs(cfg.EXPERIMENT_DIR)

cfg.DATASET = edict()
cfg.DATASET.TRAIN_DIR = osp.join(cfg.ROOT_DIR, 'data/train_full')
cfg.DATASET.VAL_DIR = osp.join(cfg.ROOT_DIR, 'data/val_full')
cfg.DATASET.NUM_CLASSES = 340


opt = edict()

opt.MODEL = edict()
opt.MODEL.IMAGE_SIZE = 224
opt.MODEL.INPUT_SIZE = 224

opt.EXPERIMENT = edict()
opt.EXPERIMENT.CODENAME = 'validation'
opt.EXPERIMENT.DIR = osp.join(cfg.EXPERIMENT_DIR, opt.EXPERIMENT.CODENAME)

opt.LOG = edict()
opt.LOG.LOG_FILE = osp.join(opt.EXPERIMENT.DIR, 'log_validation.txt')

opt.TRAIN = edict()
opt.TRAIN.BATCH_SIZE = 130
opt.TRAIN.SHUFFLE = True
opt.TRAIN.WORKERS = 12
opt.TRAIN.PRINT_FREQ = 20
opt.TRAIN.SEED = 7
opt.TRAIN.LEARNING_RATE = 1e-4
opt.TRAIN.EPOCHS = 1000
opt.TRAIN.VAL_SUFFIX = '7'
opt.TRAIN.SAVE_FREQ = 1
opt.TRAIN.STEPS_PER_EPOCH = 7000
opt.TRAIN.RESUME = sys.argv[1]

opt.TRAIN.COSINE = edict()
opt.TRAIN.COSINE.PERIOD = 32
opt.TRAIN.COSINE.COEFF = 1.2

opt.VALID = edict()
opt.VALID.BATCH_SIZE = 256

if opt.TRAIN.SEED is None:
    opt.TRAIN.SEED = int(time.time())

random.seed(opt.TRAIN.SEED)
torch.manual_seed(opt.TRAIN.SEED)
torch.cuda.manual_seed(opt.TRAIN.SEED)


if not osp.exists(opt.EXPERIMENT.DIR):
    os.makedirs(opt.EXPERIMENT.DIR)


logger = create_logger(opt.LOG.LOG_FILE)
logger.info('Options:')
logger.info(pprint.pformat(opt))

msg = f'Use time as random seed: {opt.TRAIN.SEED}'
logger.info(msg)


DATA_INFO = cfg.DATASET

# data-loader for validation set
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                          std = [ 0.229, 0.224, 0.225 ]),
])


val_dataset = get_data_loader(sys.argv[2], DATA_INFO.VAL_DIR, transform_val,
                            DATA_INFO.NUM_CLASSES, "val",
                            opt.MODEL.IMAGE_SIZE)

test_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=opt.VALID.BATCH_SIZE, shuffle=False, num_workers=opt.TRAIN.WORKERS)


# create model
last_checkpoint = torch.load(opt.TRAIN.RESUME)
model_arch = last_checkpoint['arch']
logger.info(f"using a model {model_arch}")
model = pretrainedmodels.__dict__[model_arch](pretrained='imagenet')

assert(opt.MODEL.INPUT_SIZE % 32 == 0)
model.avgpool = nn.AvgPool2d(opt.MODEL.INPUT_SIZE // 32, stride=1)
model.last_linear = nn.Linear(model.last_linear.in_features, DATA_INFO.NUM_CLASSES)
model = torch.nn.DataParallel(model).cuda()

model.module.load_state_dict(last_checkpoint['state_dict'])
logger.info(f"Checkpoint {opt.TRAIN.RESUME} was loaded.")

epoch = last_checkpoint['epoch']
logger.info(f"Validating after epoch {last_checkpoint['epoch']}")


train_losses: List[float] = []
train_top1: List[float] = []
train_top3: List[float] = []
val_losses: List[float] = []
val_top1: List[float] = []
val_top3: List[float] = []

def save_checkpoint(state: Dict[str, Any], filename: str) -> None:
    torch.save(state, osp.join(opt.EXPERIMENT.DIR, filename))
    logger.info(f'A snapshot was saved to {filename}')

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):
            target = target.cuda(async=True)

            # compute output
            output = model(input_)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, (1, 3))
            losses.update(loss.data.item(), input_.size(0))
            top1.update(prec1.item(), input_.size(0))
            top3.update(prec3.item(), input_.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.TRAIN.PRINT_FREQ == 0:
                logger.info(f'Test {epoch} [{i}/{len(val_loader)}]\t'
                            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                            f'Prec@1 {top1.val:.3f} ({top1.avg:.5f})\t'
                            f'Prec@3 {top3.val:.3f} ({top3.avg:.5f})')

    logger.info(f'MAP@3 {top3.avg:.5f}')

    val_losses.append(losses.avg)
    val_top1.append(top1.avg)
    val_top3.append(top3.avg)

    return top3.avg

criterion = nn.CrossEntropyLoss()

best_map3 = 0
best_epoch = 0

val_dataset.start_new_epoch()
logger.info(f'{len(val_dataset)} images are found for validation')

try:
    map3 = validate(test_loader, model, criterion)
except KeyboardInterrupt:
    logger.info("iterrupted")

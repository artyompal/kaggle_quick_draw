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
from data_loader import get_data_loader
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


cudnn.benchmark = True


cfg = edict()

cfg.ROOT_DIR = ".."
cfg.EXPERIMENT_DIR = osp.join(cfg.ROOT_DIR, 'models')
if not osp.exists(cfg.EXPERIMENT_DIR):
    os.makedirs(cfg.EXPERIMENT_DIR)

cfg.DATASET = edict()
cfg.DATASET.TRAIN_DIR = osp.join(cfg.ROOT_DIR, 'data/train_simple')
cfg.DATASET.VAL_DIR = osp.join(cfg.ROOT_DIR, 'data/val_simple')
cfg.DATASET.NUM_CLASSES = 340
cfg.DATASET.DATA_LOADER = 'data_loader_v1'


opt = edict()

opt.MODEL = edict()
opt.MODEL.ARCH = 'resnet50'
opt.MODEL.IMAGE_SIZE = 128
opt.MODEL.INPUT_SIZE = 128

opt.EXPERIMENT = edict()
opt.EXPERIMENT.CODENAME = os.path.splitext(os.path.basename(__file__))[0]
opt.EXPERIMENT.TASK = 'finetune'
opt.EXPERIMENT.DIR = osp.join(cfg.EXPERIMENT_DIR, opt.EXPERIMENT.CODENAME)

opt.LOG = edict()
opt.LOG.LOG_FILE = osp.join(opt.EXPERIMENT.DIR, f'log_{opt.EXPERIMENT.TASK}.txt')

opt.TRAIN = edict()
opt.TRAIN.BATCH_SIZE = 256
opt.TRAIN.SHUFFLE = True
opt.TRAIN.WORKERS = 12
opt.TRAIN.PRINT_FREQ = 20
opt.TRAIN.SEED = 7
opt.TRAIN.LEARNING_RATE = 3e-4
opt.TRAIN.EPOCHS = 1000
opt.TRAIN.VAL_SUFFIX = '7'
opt.TRAIN.SAVE_FREQ = 1
opt.TRAIN.STEPS_PER_EPOCH = 7000
opt.TRAIN.RESUME = None if len(sys.argv) == 1 else sys.argv[1]

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

# Data-loader of training set
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                          std = [ 0.229, 0.224, 0.225 ]),
])

# Data-loader of testing set
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                          std = [ 0.229, 0.224, 0.225 ]),
])


train_dataset = get_data_loader(DATA_INFO.DATA_LOADER, DATA_INFO.TRAIN_DIR,
                                transform_train, DATA_INFO.NUM_CLASSES, "train",
                                opt.MODEL.IMAGE_SIZE)
val_dataset = get_data_loader(DATA_INFO.DATA_LOADER, DATA_INFO.VAL_DIR,
                              transform_val, DATA_INFO.NUM_CLASSES, "val",
                              opt.MODEL.IMAGE_SIZE)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=opt.TRAIN.SHUFFLE, num_workers=opt.TRAIN.WORKERS)

test_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=opt.VALID.BATCH_SIZE, shuffle=False, num_workers=opt.TRAIN.WORKERS)


# create model
logger.info(f"using pre-trained model {opt.MODEL.ARCH}")
model = pretrainedmodels.__dict__[opt.MODEL.ARCH](pretrained='imagenet')

assert(opt.MODEL.INPUT_SIZE % 32 == 0)
model.avgpool = nn.AvgPool2d(opt.MODEL.INPUT_SIZE // 32, stride=1)
model.last_linear = nn.Linear(model.last_linear.in_features, DATA_INFO.NUM_CLASSES)
model = torch.nn.DataParallel(model).cuda()

if torch.cuda.device_count() == 1:
    torchsummary.summary(model, (3, opt.MODEL.INPUT_SIZE, opt.MODEL.INPUT_SIZE))

optimizer = optim.Adam(model.module.parameters(), opt.TRAIN.LEARNING_RATE)
lr_scheduler = CosineLRWithRestarts(optimizer, opt.TRAIN.BATCH_SIZE,
    opt.TRAIN.BATCH_SIZE * opt.TRAIN.STEPS_PER_EPOCH,
    restart_period=opt.TRAIN.COSINE.PERIOD, t_mult=opt.TRAIN.COSINE.COEFF)

if opt.TRAIN.RESUME is None:
    last_epoch = 0
    logger.info(f"Training will start from epoch {last_epoch+1}")

else:
    last_checkpoint = torch.load(opt.TRAIN.RESUME)
    assert(last_checkpoint['arch']==opt.MODEL.ARCH)
    model.module.load_state_dict(last_checkpoint['state_dict'])
    optimizer.load_state_dict(last_checkpoint['optimizer'])
    logger.info(f"Checkpoint {opt.TRAIN.RESUME} was loaded.")

    last_epoch = last_checkpoint['epoch']
    logger.info(f"Training will be resumed from epoch {last_checkpoint['epoch']}")


train_losses: List[float] = []
train_top1: List[float] = []
train_top3: List[float] = []
val_losses: List[float] = []
val_top1: List[float] = []
val_top3: List[float] = []

def save_checkpoint(state: Dict[str, Any], filename: str) -> None:
    torch.save(state, osp.join(opt.EXPERIMENT.DIR, filename))
    logger.info(f'A snapshot was saved to {filename}')

def train(train_loader, model, criterion, optimizer, epoch):
    logger.info(f'Epoch {epoch}')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    print("total batches:", len(train_loader))
    num_steps = min(len(train_loader), opt.TRAIN.STEPS_PER_EPOCH)

    end = time.time()
    for i, (input_, target) in enumerate(train_loader):
        if i >= opt.TRAIN.STEPS_PER_EPOCH:
            break

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)

        # compute output
        output = model(input_)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, (1, 3))
        losses.update(loss.data.item(), input_.size(0))
        top1.update(prec1.item(), input_.size(0))
        top3.update(prec3.item(), input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.batch_step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.TRAIN.PRINT_FREQ == 0:
            logger.info(f'{epoch} [{i}/{num_steps}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Prec@1 {top1.val:.4f} ({top1.avg:.4f})\t'
                        f'Prec@3 {top3.val:.4f} ({top3.avg:.4f})')

    train_losses.append(losses.avg)
    train_top1.append(top1.avg)
    train_top3.append(top3.avg)

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
                            f'Prec@1 {top1.val:.4f} ({top1.avg:.4f})\t'
                            f'Prec@3 {top3.val:.4f} ({top3.avg:.4f})')

    logger.info(f' * MAP@3 {top3.avg:.4f}')

    val_losses.append(losses.avg)
    val_top1.append(top1.avg)
    val_top3.append(top3.avg)

    return top3.avg

def set_lr(lr: float) -> None:
    for param_group in optimizer.param_groups:
       param_group['lr'] = lr

def read_lr() -> None:
    for param_group in optimizer.param_groups:
       lr = float(param_group['lr'])
       logger.info(f"learning rate: {lr}")
       return


criterion = nn.CrossEntropyLoss()
set_lr(opt.TRAIN.LEARNING_RATE)

best_map3 = 0
best_epoch = 0

val_dataset.start_new_epoch()
logger.info(f'{len(val_dataset)} images are found for validation')

for epoch in range(last_epoch+1, opt.TRAIN.EPOCHS+1):
    logger.info('-'*50)
    train_dataset.start_new_epoch()
    logger.info(f'{len(train_dataset)} images are found for train')

    lr_scheduler.step()
    read_lr()

    train(train_loader, model, criterion, optimizer, epoch)
    map3 = validate(test_loader, model, criterion)
    is_best = map3 > best_map3
    best_map3 = max(map3, best_map3)
    if is_best:
        best_epoch = epoch

    data_to_save = {
        'epoch': epoch,
        'arch': opt.MODEL.ARCH,
        'state_dict': model.module.state_dict(),
        'best_map3': best_map3,
        'map3': map3,
        'optimizer' : optimizer.state_dict(),
    }

    if epoch % opt.TRAIN.SAVE_FREQ == 0:
        save_checkpoint(data_to_save, f'{opt.EXPERIMENT.CODENAME}_[{epoch:03d}]_{map3:.04f}.pk')

    if is_best:
        save_checkpoint(data_to_save, f'{opt.EXPERIMENT.CODENAME}_best_[{epoch:03d}]_{map3:.04f}.pk')

logger.info(f'Best MAP@3 for single crop: {best_map3:.04f}')

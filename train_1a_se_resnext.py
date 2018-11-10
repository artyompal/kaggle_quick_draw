#!/usr/bin/python3.6

import os
import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from data_loader import DatasetFolder
import logging
import numpy as np
import random
import time
import datetime
import pprint
from easydict import EasyDict as edict
import PIL

import pretrainedmodels
from utils import cfg, create_logger, AverageMeter, F_score


print(pretrainedmodels.model_names)

cudnn.benchmark = True

timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

opt = edict()

opt.MODEL = edict()
opt.MODEL.ARCH = 'se_resnext101_32x4d'
opt.MODEL.PRETRAINED = True
opt.MODEL.IMAGE_SIZE = 256
opt.MODEL.INPUT_SIZE = 224 # crop size

opt.EXPERIMENT = edict()
opt.EXPERIMENT.CODENAME = os.path.splitext(os.path.basename(__file__))[0]
opt.EXPERIMENT.TASK = 'finetune'
opt.EXPERIMENT.DIR = osp.join(cfg.EXPERIMENT_DIR, opt.EXPERIMENT.CODENAME)

opt.LOG = edict()
opt.LOG.LOG_FILE = osp.join(opt.EXPERIMENT.DIR, 'log_{}.txt'.format(opt.EXPERIMENT.TASK))

opt.TRAIN = edict()
opt.TRAIN.BATCH_SIZE = 32
opt.TRAIN.SHUFFLE = True
opt.TRAIN.WORKERS = 12
opt.TRAIN.PRINT_FREQ = 20
opt.TRAIN.SEED = None
opt.TRAIN.LEARNING_RATE = 1e-4
opt.TRAIN.LR_GAMMA = 0.5
opt.TRAIN.LR_MILESTONES = [4, 6, 8, 10, 12, 14]
opt.TRAIN.EPOCHS = 15
opt.TRAIN.VAL_SUFFIX = '7'
opt.TRAIN.SAVE_FREQ = 1
opt.TRAIN.STEPS_PER_EPOCH = 7000
opt.TRAIN.RESUME = None if len(sys.argv) == 1 else sys.argv[1]

opt.VALID = edict()
opt.VALID.STEPS_PER_EPOCH = 2000

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

msg = 'Use time as random seed: {}'.format(opt.TRAIN.SEED)
logger.info(msg)


DATA_INFO = cfg.DATASET

# Data-loader of training set
transform_train = transforms.Compose([
    transforms.Resize((opt.MODEL.IMAGE_SIZE)), # smaller edge
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=20, scale=(0.8, 1.2), shear=10, resample=PIL.Image.BILINEAR),
    transforms.RandomCrop(opt.MODEL.INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                          std = [ 0.229, 0.224, 0.225 ]),
])

# Data-loader of testing set
transform_val = transforms.Compose([
    transforms.Resize((opt.MODEL.IMAGE_SIZE)),
    transforms.CenterCrop(opt.MODEL.INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                          std = [ 0.229, 0.224, 0.225 ]),
])


train_dataset = DatasetFolder(DATA_INFO.ROOT_DIR, transform_train,
                              DATA_INFO.NUM_CLASSES, conf=0.69)
val_dataset = DatasetFolder(DATA_INFO.ROOT_DIR, transform_val,
                            DATA_INFO.NUM_CLASSES, conf=0.69)

logger.info('{} images are found for train_val'.format(len(train_dataset.samples)))

train_set = [(img, target) for (img, target) in zip(train_dataset.samples, train_dataset.classes)
         if not img[-5] in opt.TRAIN.VAL_SUFFIX]
logger.info('{} images are used to train'.format(len(train_set)))
val_set = [(img, target) for (img, target) in zip(train_dataset.samples, train_dataset.classes)
           if img[-5] in opt.TRAIN.VAL_SUFFIX]
logger.info('{} images are used to val'.format(len(val_set)))

train_dataset.samples = [img for img, target in train_set]
train_dataset.classes = [target for img, target in train_set]
val_dataset.samples = [img for img, target in val_set]
val_dataset.classes = [target for img, target in val_set]


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=opt.TRAIN.SHUFFLE, num_workers=opt.TRAIN.WORKERS)

test_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=opt.TRAIN.BATCH_SIZE, shuffle=False, num_workers=opt.TRAIN.WORKERS)


# create model
if opt.MODEL.PRETRAINED:
    logger.info("=> using pre-trained model '{}'".format(opt.MODEL.ARCH ))
    model = pretrainedmodels.__dict__[opt.MODEL.ARCH](pretrained='imagenet')
else:
    raise NotImplementedError


if opt.MODEL.ARCH.startswith('resnet'):
    assert(opt.MODEL.INPUT_SIZE % 32 == 0)
    model.avgpool = nn.AvgPool2d(opt.MODEL.INPUT_SIZE // 32, stride=1)
    model.fc = nn.Linear(model.fc.in_features, DATA_INFO.NUM_CLASSES)
    model = torch.nn.DataParallel(model).cuda()
else:
    assert(opt.MODEL.INPUT_SIZE % 32 == 0)
    model.avgpool = nn.AvgPool2d(opt.MODEL.INPUT_SIZE // 32, stride=1)
    model.last_linear = nn.Linear(model.last_linear.in_features, DATA_INFO.NUM_CLASSES)
    model = torch.nn.DataParallel(model).cuda()


optimizer = optim.Adam(model.module.parameters(), opt.TRAIN.LEARNING_RATE)
lr_scheduler = MultiStepLR(optimizer, opt.TRAIN.LR_MILESTONES, gamma=opt.TRAIN.LR_GAMMA, last_epoch=-1)

if opt.TRAIN.RESUME is None:
    last_epoch = 0
    logger.info("Training will start from epoch {}".format(last_epoch+1))

else:
    last_checkpoint = torch.load(opt.TRAIN.RESUME)
    assert(last_checkpoint['arch']==opt.MODEL.ARCH)
    model.module.load_state_dict(last_checkpoint['state_dict'])
    optimizer.load_state_dict(last_checkpoint['optimizer'])
    logger.info("Checkpoint '{}' was loaded.".format(opt.TRAIN.RESUME))

    last_epoch = last_checkpoint['epoch']
    logger.info("Training will be resumed from epoch {}".format(last_checkpoint['epoch']))


train_losses = []
train_f2 = []
test_losses = []
test_f2 = []

def save_checkpoint(state, filename='checkpoint.pk'):
    torch.save(state, osp.join(opt.EXPERIMENT.DIR, filename))
    logger.info('A snapshot was saved to {}.'.format(filename))

def train(train_loader, model, criterion, optimizer, epoch):
    logger.info('Epoch {}'.format(epoch))
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    f2 = AverageMeter()

    # switch to train mode
    model.train()

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
        f2_score = F_score(output.data, target)
        losses.update(loss.data.item(), input_.size(0))
        f2.update(f2_score.item(), input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.TRAIN.PRINT_FREQ == 0:
            logger.info('[{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'F2 {f2.val:.3f} ({f2.avg:.3f})'.format(
                        epoch, i, opt.TRAIN.STEPS_PER_EPOCH, batch_time=batch_time,
                        data_time=data_time, loss=losses, f2=f2))

    train_losses.append(losses.avg)
    train_f2.append(f2.avg)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    f2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input_, target) in enumerate(val_loader):
            if i >= opt.VALID.STEPS_PER_EPOCH:
                break

            target = target.cuda(async=True)

            # compute output
            output = model(input_)
            loss = criterion(output, target)

            # measure accuracy and record loss
            f2_score = F_score(output.data, target)
            losses.update(loss.data.item(), input_.size(0))
            f2.update(f2_score.item(), input_.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.TRAIN.PRINT_FREQ == 0:
                logger.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'F2 {f2.val:.3f} ({f2.avg:.3f})\t'.format(
                            i, opt.VALID.STEPS_PER_EPOCH, batch_time=batch_time,
                            loss=losses, f2=f2))


    logger.info(' * F2 {f2.avg:.3f}'.format(f2=f2))

    test_losses.append(losses.avg)
    test_f2.append(f2.avg)

    return f2.avg

criterion = nn.BCEWithLogitsLoss()

best_f2 = 0
best_epoch = 0

for epoch in range(last_epoch+1, opt.TRAIN.EPOCHS+1):
    logger.info('-'*50)
    lr_scheduler.step(epoch)
    logger.info('lr: {}'.format(lr_scheduler.get_lr()))

    train(train_loader, model, criterion, optimizer, epoch)
    f2 = validate(test_loader, model, criterion)
    is_best = f2 > best_f2
    best_f2 = max(f2, best_f2)
    if is_best:
        best_epoch = epoch

    if epoch % opt.TRAIN.SAVE_FREQ == 0:
        save_checkpoint({
            'epoch': epoch,
            'arch': opt.MODEL.ARCH,
            'state_dict': model.module.state_dict(),
            'best_f2': best_f2,
            'f2': f2,
            'optimizer' : optimizer.state_dict(),
        }, '{}_[{}]_{:.03f}.pk'.format(opt.MODEL.ARCH, epoch, f2))

    if is_best:
        save_checkpoint({
            'epoch': epoch,
            'arch': opt.MODEL.ARCH,
            'state_dict': model.module.state_dict(),
            'best_f2': best_f2,
            'f2': f2,
            'optimizer' : optimizer.state_dict(),
        }, 'best_model.pk')


logger.info('Best F2-score for single crop: {:.05f}%'.format(best_f2))

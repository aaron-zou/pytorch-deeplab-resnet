#!/usr/bin/env python
from __future__ import print_function

import os
import random

import cv2
import numpy as np
import torch
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from docopt import docopt
from PIL import Image
from torch.autograd import Variable

import deeplab.resnet as resnet
from deeplab.datasets import Mode, SegmentationDataset

DOCSTR = """Train ResNet-DeepLab on VOC12 (scenes) in pytorch using MS-COCO
pretrained initialization.

Usage:
    train.py [options]

Options:
    -h, --help                  Print this message
    --GTpath=<str>              Ground truth path prefix [default: data/gt/]
    --IMpath=<str>              Sketch images path prefix [default: data/img/]
    --GText=<str>               Ground truth path extension [default: .png]
    --IMext=<str>               Sketch images path extension [default: .jpg]
    --NoLabels=<int>            The number of different labels in training data
                                VOC has 21 labels, including background
                                [default: 21]
    --LISTpath=<str>            Input image number list file
                                [default: data/train_aug.txt]
    --lr=<float>                Learning Rate [default: 0.00025]
    -i, --iterSize=<int>        Num iters to accumulate gradients over
                                [default: 10]
    --wtDecay=<float>           Weight decay during training [default: 0.0005]
    --gpu=<int>                 GPU number [default: 0]
    --maxIter=<int>             Maximum number of iterations [default: 20000]
    --outputPrefix=<str>        Prefix for snapshot output file
                                [default: VOC12_scenes]
    --imagenet                  Set to use ImageNet pretraining, otherwise
                                default to MS-COCO.
"""


def calc_loss(output, labels, gpu):
    """
    Calculate cross-entropy loss for model output and label.
    """
    loss = 0
    criterion = nn.CrossEntropyLoss().cuda()
    for i in range(len(output)):
        label = labels[i].squeeze(0)
        label = Variable(label.long()).cuda(gpu)
        loss += criterion(output[i], label)
    return loss


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1 - float(iter) / max_iter)**(power))


def get_model(num_labels, gpu):
    # Always use ImageNet-trained model
    model = resnet.getDeepLabV2FromResNet(num_labels)
    return model.float().eval().cuda(gpu)


def make_optimizer(model, lr, weight_decay):
    # This has the parameters for the network besides the last classification
    # layer and any batchnorm layers (requires_grad set to False in model)
    requires_grad_params = (p[1] for p in model.Scale.named_parameters()
                            if "layer5" not in p[0] and p[1].requires_grad)
    classification_params = model.Scale.layer5.parameters()
    return optim.SGD([
        {'params': requires_grad_params, 'lr': lr},
        {'params': classification_params, 'lr': 10*lr}],
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay)


def main():
    args = docopt(DOCSTR)
    print(args)

    gpu = int(args['--gpu'])
    outputPrefix = args['--outputPrefix']

    if not os.path.exists('snapshots'):
        os.makedirs('snapshots')

    # Set up data-related operations
    dataset = SegmentationDataset(Mode.TRAIN, args['--LISTpath'],
                                  args["--IMpath"], args["--GTpath"],
                                  args["--IMext"], args["--GText"])
    dataloader = DataLoader(dataset, shuffle=True,
                            num_workers=multiprocessing.cpu_count())

    # Retrieve model
    model = get_model(int(args['--NoLabels']), gpu)

    # Training parameters
    max_iter = int(args['--maxIter'])
    weight_decay = float(args['--wtDecay'])
    base_lr = float(args['--lr'])
    iter_size = int(args['--iterSize'])

    # Set up optimizer
    optimizer = make_optimizer(model, base_lr, weight_decay)
    optimizer.zero_grad()

    iteration = 0
    while iteration <= max_iter:
        for image, label in dataloader:
            output = model(Variable(image).cuda(gpu))
            loss = calc_loss(output, label, gpu) / iter_size
            loss.backward()

            if iteration % 1 == 0:
                print('{}/{} done, loss = {}'.format(
                    iteration, max_iter, iter_size * (loss.data.cpu().numpy())))

            if iteration % iter_size == 0:
                optimizer.step()
                lr = lr_poly(base_lr, iteration, max_iter, 0.9)
                print('(poly lr) learning rate = {}'.format(lr))
                optimizer = make_optimizer(model, lr, weight_decay)
                optimizer.zero_grad()

            if iteration % 5000 == 0 and iteration != 0:
                print('saving snapshot {}'.format(iteration))
                torch.save(model.state_dict(), os.path.join(
                    'snapshots', '{}_{}.pth'.format(outputPrefix, iteration)))

            iteration += 1

            if iteration > max_iter:
                break


if __name__ == "__main__":
    main()

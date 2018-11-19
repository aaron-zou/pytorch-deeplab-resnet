#!/usr/bin/env python2
from __future__ import print_function

import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from docopt import docopt
from PIL import Image
from torch.autograd import Variable

import resnet

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
    --NoLabels=<int>            The number of different labels in training data,
                                VOC has 21 labels, including background [default: 21]
    --LISTpath=<str>            Input image number list file
                                [default: data/list/train_aug.txt]
    --lr=<float>                Learning Rate [default: 0.00025]
    -i, --iterSize=<int>        Num iters to accumulate gradients over [default: 10]
    --wtDecay=<float>           Weight decay during training [default: 0.0005]
    --gpu0=<int>                GPU number [default: 0]
    --maxIter=<int>             Maximum number of iterations [default: 20000]
    --outputFn=<str>            Prefix for snapshot output file [default: VOC21_scenes]
"""


def outS(i):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    j = int(i)
    j = (j+1) // 2
    j = int(np.ceil((j+1) / 2.0))
    j = (j+1) // 2
    return j


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list


def chunker(seq, size):
    return (seq[pos:pos+size] for pos in xrange(0, len(seq), size))


def resize_label_batch(label, size):
    label_resized = np.zeros((size, size, 1, label.shape[3]))
    interp = nn.UpsamplingNearest2d(size=(size, size))
    labelVar = Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(
        labelVar).data.numpy().transpose(2, 3, 1, 0)
    return label_resized


def flip(I, flip_p):
    if flip_p > 0.5:
        return np.fliplr(I)
    else:
        return I


def scale_im(img_temp, scale):
    new_dims = (int(img_temp.shape[0]*scale), int(img_temp.shape[1]*scale))
    return cv2.resize(img_temp, new_dims).astype(float)


def scale_gt(img_temp, scale):
    new_dims = (int(img_temp.shape[0]*scale), int(img_temp.shape[1]*scale))
    return cv2.resize(img_temp, new_dims, interpolation=cv2.INTER_NEAREST).astype(float)


def get_data_from_chunk_v2(chunk, gt_path, img_path, gt_ext, img_ext):
    # random.uniform(0.5,1.5) does not fit in a Titan X with the present
    # version of pytorch, so we random scaling in the range (0.5,1.3),
    # different than caffe implementation in that caffe used only 4 fixed
    # scales. Refer to README
    scale = random.uniform(0.5, 1.1)
    dim = int(scale * 321)
    images = np.zeros((dim, dim, 3, len(chunk)))
    gt = np.zeros((dim, dim, 1, len(chunk)))
    for i, piece in enumerate(chunk):
        flip_p = random.uniform(0, 1)
        img_temp = cv2.imread(os.path.join(
            img_path, piece + img_ext)).astype(float)
        img_temp = cv2.resize(img_temp, (321, 321)).astype(float)
        img_temp = scale_im(img_temp, scale)
        img_temp[:, :, 0] = img_temp[:, :, 0] - 104.008
        img_temp[:, :, 1] = img_temp[:, :, 1] - 116.669
        img_temp[:, :, 2] = img_temp[:, :, 2] - 122.675
        img_temp = flip(img_temp, flip_p)
        images[:, :, :, i] = img_temp

        gt_temp = cv2.imread(os.path.join(gt_path, piece + gt_ext))[:, :, 0]
        gt_temp[gt_temp == 255] = 0
        gt_temp = cv2.resize(gt_temp, (321, 321),
                             interpolation=cv2.INTER_NEAREST)
        gt_temp = scale_gt(gt_temp, scale)
        gt_temp = flip(gt_temp, flip_p)
        gt[:, :, 0, i] = gt_temp
        a = outS(321*scale)  # 41
        b = outS((321*0.5)*scale+1)  # 21
    labels = [resize_label_batch(gt, i) for i in [a, a, b, a]]
    images = images.transpose((3, 2, 0, 1))
    images = torch.from_numpy(images).float()
    return images, labels


def loss_calc(out, label, gpu0):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label[:, :, 0, :].transpose(2, 0, 1)
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)
    m = nn.LogSoftmax()
    criterion = nn.NLLLoss()
    out = m(out)
    return criterion(out, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1 - float(iter) / max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = []

    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    b = []
    b.append(model.Scale.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def get_model(num_labels, pretraining_init_path, gpu0):
    # model = resnet.getDeepLabV2(num_labels, pretraining_init_path)
    model = resnet.getDeepLabV2FromResNet(num_labels)
    model.float()
    model.eval()  # use_global_stats = True
    return model.cuda(gpu0)


def main():
    args = docopt(DOCSTR, version='v0.1')
    print(args)

    gt_path = args['--GTpath']
    img_path = args['--IMpath']
    gt_ext = args['--GText']
    img_ext = args['--IMext']
    gpu0 = int(args['--gpu0'])
    outputFn = args['--outputFn']

    if not os.path.exists('data/snapshots'):
        os.makedirs('data/snapshots')

    model = get_model(int(args['--NoLabels']),
                      'data/MS_DeepLab_resnet_pretrained_COCO_init.pth', gpu0)

    # Read training parameters
    max_iter = int(args['--maxIter'])
    batch_size = 1
    weight_decay = float(args['--wtDecay'])
    base_lr = float(args['--lr'])

    # Construct file path lists
    img_list = read_file(args['--LISTpath'])
    data_list = []
    while len(data_list) < max_iter:
        np.random.shuffle(img_list)
        data_list.extend(img_list)

    # criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
    optimizer = optim.SGD([
        {'params': get_1x_lr_params_NOscale(model), 'lr': base_lr},
        {'params': get_10x_lr_params(model), 'lr': 10*base_lr}],
        lr=base_lr,
        momentum=0.9,
        weight_decay=weight_decay)

    optimizer.zero_grad()
    data_gen = chunker(data_list, batch_size)

    for iteration in range(max_iter+1):
        chunk = data_gen.next()

        images, label = get_data_from_chunk_v2(
            chunk, gt_path, img_path, gt_ext, img_ext)
        images = Variable(images).cuda(gpu0)

        out = model(images)
        loss = loss_calc(out[0], label[0], gpu0)
        iter_size = int(args['--iterSize'])
        for i in range(len(out)-1):
            loss = loss + loss_calc(out[i+1], label[i+1], gpu0)
        loss = loss / iter_size
        loss.backward()

        if iteration % 1 == 0:
            print('iter = {} of {} completed, loss = {}'.format(
                iteration, max_iter, iter_size * (loss.data.cpu().numpy())))

        if iteration % iter_size == 0:
            optimizer.step()
            lr_ = lr_poly(base_lr, iteration, max_iter, 0.9)
            print('(poly lr policy) learning rate: {}'.format(lr_))
            optimizer = optim.SGD([
                {'params': get_1x_lr_params_NOscale(model), 'lr': lr_},
                {'params': get_10x_lr_params(model), 'lr': 10*lr_}],
                lr=lr_,
                momentum=0.9,
                weight_decay=weight_decay)
            optimizer.zero_grad()

        if iteration % 5000 == 0 and iteration != 0:
            print('Taking snapshot {}'.format(iteration))
            torch.save(model.state_dict(), os.path.join(
                'data/snapshots', '{}_{}.pth'.format(outputFn, iteration)))


if __name__ == "__main__":
    main()

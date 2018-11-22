#!/usr/bin/env python
from __future__ import print_function

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from docopt import docopt
from PIL import Image
from torch.autograd import Variable

import deeplab_resnet
import matplotlib.pyplot as plt

DOCSTR = """Evaluate ResNet-DeepLab trained on scenes (VOC2012), a total of 21
labels including background.

Usage:
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 View outputs of each sketch
    --snapFolder=<str>          Snapshot folder [default: data/snapshots]
    --snapPath=<str>            If set, run on a specific snapshot in the folder.
                                [default: VOC21_scenes_20000.pth]
    --valPath=<str>             Path to file list of validation images.
                                [default: data/list/val.txt]
    --testGTpath=<str>          Ground truth path prefix [default: data/gt/]
    --testIMpath=<str>          Sketch images path prefix [default: data/img/]
    --GText=<str>               Ground truth path extension [default: .png]
    --IMext=<str>               Sketch image path extension [default: .jpg]
    --NoLabels=<int>            The number of different labels in training data,
                                VOC has 21 labels, including background
                                [default: 21]
    --gpu0=<int>                GPU number [default: 0]
"""


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def get_iou(pred, gt, max_label):
    if pred.shape != gt.shape:
        raise RuntimeError(
            'pred shape: {}, gt shape: {}'.format(pred.shape, gt.shape))
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    count = np.zeros((max_label+1,))

    for j in range(max_label+1):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:]) / float(len(np.unique(gt)))

    return Aiou


def main():
    args = docopt(DOCSTR, version='v0.1')
    print(args)

    gpu0 = int(args['--gpu0'])
    snapFolder = args['--snapFolder']
    snapPath = args['--snapPath']
    im_path = args['--testIMpath']
    gt_path = args['--testGTpath']
    im_ext = args['--IMext']
    gt_ext = args['--GText']
    max_label = int(args['--NoLabels'])-1  # labels from 0,1, ... 20 (for VOC)

    model = deeplab_resnet.Res_Deeplab(int(args['--NoLabels']))
    model = model.eval().cuda(gpu0)

    with open(args['--valPath'], 'r') as f:
        img_list = f.readlines()

    for snapshot in os.listdir(snapFolder):
        if snapPath and snapshot != snapPath:
            continue

        path = os.path.join(snapFolder, snapshot)
        print('Processing snapshot: {}'.format(snapshot))
        if not os.path.isfile(path):
            continue

        saved_state_dict = torch.load(path)
        model.load_state_dict(saved_state_dict)

        hist = np.zeros((max_label+1, max_label+1))
        pytorch_list = []

        print('Processing {} images'.format(len(img_list)))
        for i, filename in enumerate(img_list):
            print('({}/{})'.format(i, len(img_list)))
            img = np.zeros((513, 513, 3))
            img_temp = cv2.imread(os.path.join(
                im_path, filename[:-1] + im_ext)).astype(float)
            img_original = img_temp

            img_temp[:, :, 0] = img_temp[:, :, 0] - 104.008
            img_temp[:, :, 1] = img_temp[:, :, 1] - 116.669
            img_temp[:, :, 2] = img_temp[:, :, 2] - 122.675
            img[:img_temp.shape[0], :img_temp.shape[1], :] = img_temp

            # gt = cv2.imread(os.path.join(gt_path, i[:-1] + '.png'), 0)
            # gt[gt==255] = 0
            gt = np.array(Image.open(os.path.join(
                gt_path, filename[:-1] + gt_ext)))

            input_image = torch.from_numpy(
                img[np.newaxis, :].transpose(0, 3, 1, 2)).float()

            output = model(Variable(input_image).cuda(gpu0))
            interp = nn.UpsamplingBilinear2d(size=(513, 513))
            output = interp(output[3]).cpu().data[0].numpy()
            output = output[:, :img_temp.shape[0], :img_temp.shape[1]]
            output = output.transpose(1, 2, 0)
            output = np.argmax(output, axis=2)

            if args['--visualize']:
                plt.subplot(3, 1, 1)
                plt.imshow(img_original)
                plt.subplot(3, 1, 2)
                plt.imshow(gt)
                plt.subplot(3, 1, 3)
                plt.imshow(output)
                plt.show()

            iou_pytorch = get_iou(output, gt, max_label)
            pytorch_list.append(iou_pytorch)
            hist += fast_hist(gt.flatten(), output.flatten(), max_label+1)

        miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        print('pytorch {}, mean iou = {}'.format(
            snapshot,  np.sum(miou) / len(miou)))


if __name__ == "__main__":
    main()

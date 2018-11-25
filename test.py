#!/usr/bin/env python
from __future__ import print_function

import os
from typing import Any, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from docopt import docopt
from PIL import Image
from torch.autograd import Variable

import deeplab.resnet as resnet
from deeplab.dataloaders import FusionSegDataloader, Mode, VOC12Dataloader
from deeplab.deeplab_resnet import MS_Deeplab, Res_Deeplab

DOCSTR = """Evaluate ResNet-DeepLab trained on scenes (VOC2012), a total of 21
labels including background.

Usage:
    test.py (voc | fusionseg) [options]

Options:
    -h, --help                  Print this message
    voc                         Test a model on VOC dataset.
    fusionseg                   Test a model on fusionseg dataset.
    --snapPath=<str>            Path to snapshot to evaluate.
    --valPath=<str>             Path to file list of validation images.
    --root=<str>                Root path prefix.
    --NoLabels=<int>            Number of different labels in training data,
                                including background [default: 21]
    --gpu=<int>                 GPU number [default: 0]
    --visualize                 Generate visualizations of model outputs.

"""


def fast_hist(ground_truth: np.ndarray, output: np.ndarray, num_labels: int) -> np.ndarray:
    valid_idx = (ground_truth >= 0) & (ground_truth < num_labels)
    gt_contrib = num_labels * ground_truth[valid_idx].astype(int)
    confusion_mat = np.bincount(
        gt_contrib + output[valid_idx], minlength=num_labels**2)
    return confusion_mat.reshape(num_labels, num_labels)


def get_model(num_labels: int, snapshot_path: str, gpu: int) -> MS_Deeplab:
    """Retrieve model with appropriate parameters"""
    return resnet.getDeepLabV2(num_labels, snapshot_path).eval().cuda(gpu)


def get_dataloader(args: Dict[str, str]) -> Any:
    """Construct appropriate dataloader"""
    if args["voc"]:
        return VOC12Dataloader(args["--root"], args["--valPath"], Mode.VAL)
    elif args["fusionseg"]:
        return FusionSegDataloader(args["--root"],
                                   args["--valPath"],
                                   binary_class=args["--NoLabels"] == 2,
                                   mode=Mode.VAL)
    else:
        raise NotImplementedError


def validate(model: MS_Deeplab, num_labels: int, gpu: int, args: Dict[str, str]) -> None:
    hist = np.zeros((num_labels, num_labels))
    for i, (img, gt) in enumerate(get_dataloader(args)):
        print('processing {}'.format(i))

        # Rescale ground truth to [0, 255] for histogram calculation
        gt = np.array(gt.squeeze() * 255, dtype="uint8")

        with torch.no_grad():
            # TODO: Rescale to 255 only for pretrained models trained w [0-255]
            output = model(Variable(img * 255).cuda(gpu))

        # Resize to match ground truth size and take highest probability label
        output = F.interpolate(
            output[3], (513, 513), mode='bilinear', align_corners=True).cpu()
        output = output.data[0].numpy()[:, :gt.shape[0], :gt.shape[1]]
        output = np.argmax(output, axis=0).astype(np.uint8)

        if args['--visualize']:
            plt.subplot(2, 1, 1)
            plt.imshow(gt)
            plt.subplot(2, 1, 2)
            plt.imshow(output)
            plt.show()

        hist += fast_hist(gt.flatten(), output.flatten(), num_labels)

    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('Mean iou = {}'.format(np.sum(miou) / len(miou)))


def main():
    args = docopt(DOCSTR)
    print(args)

    # Create model
    num_labels = int(args['--NoLabels'])
    gpu = int(args['--gpu'])
    model = get_model(num_labels, args['--snapPath'], gpu)

    # Select which validation to use
    validate(model, num_labels, gpu, args)


if __name__ == "__main__":
    main()

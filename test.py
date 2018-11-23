#!/usr/bin/env python
from __future__ import print_function

from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from docopt import docopt
from PIL import Image
from torch.autograd import Variable

from deeplab.dataloaders import FusionSegDataloader, Mode, VOC12Dataloader
from deeplab.deeplab_resnet import MS_Deeplab, Res_Deeplab

DOCSTR = """Evaluate ResNet-DeepLab trained on scenes (VOC2012), a total of 21
labels including background.

Usage:
    evalpyt.py (voc | fusionseg) [options]

Options:
    -h, --help                  Print this message
    voc                         Test a model on VOC dataset.
    fusionseg                   Test a model on fusionseg dataset.
    --visualize                 Generate visualizations of model outputs.
    --snapPath=<str>            Which snapshot to evaluate.
                                [default: ./data/snapshots/VOC21_scenes_20000.pth]
    --valPath=<str>             Path to file list of validation images.
                                [default: data/list/val.txt]
    --root=<str>                Root path prefix [default: data/voc2012]
    --NoLabels=<int>            The number of different labels in training data,
                                VOC has 21 labels, including background
                                [default: 21]
    --gpu=<int>                GPU number [default: 0]

"""


def fast_hist(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def get_model(num_labels: int, snapshot_path: str, gpu: int) -> MS_Deeplab:
    """Retrieve model with appropriate parameters"""
    model = Res_Deeplab(num_labels)
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    return model.eval().cuda(gpu)


def get_dataloader(args: Dict[str, str]) -> torch.utils.data.Dataloader:
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


def main():
    args = docopt(DOCSTR, version='v0.1')
    print(args)

    gpu = int(args['--gpu'])
    max_label = int(args['--NoLabels']) - 1  # labels from 0,1, ... 20 (for VOC)

    model = get_model(int(args['--NoLabels']), args['--snapPath'], gpu)

    hist = np.zeros((max_label+1, max_label+1))
    for i, (img, gt) in enumerate(get_dataloader(args)):
        print('processing {}'.format(i))

        output = model(Variable(img).cuda(gpu))

        output = F.interpolate(output[3], (513, 513)).cpu().data[0].numpy()
        output = output[:, :gt.shape[0], :gt.shape[1]]

        output = output.transpose(1, 2, 0)
        output = np.argmax(output, axis=2)

        if args['--visualize']:
            plt.subplot(2, 1, 1)
            plt.imshow(gt)
            plt.subplot(2, 1, 2)
            plt.imshow(output)
            plt.show()

        hist += fast_hist(gt.flatten(), output.flatten(), max_label+1)

    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('Mean iou={}'.format(np.sum(miou) / len(miou)))


if __name__ == "__main__":
    main()

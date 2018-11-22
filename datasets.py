#!/usr/bin/env python
from __future__ import print_function
import os
import os.path
import sys
import torch
import torch.utils.data as data
from PIL import Image, ImageDraw


class VOC2012Segmentation(data.Dataset):
    """
    VOC2012 segmentation dataset.
    """

    def __init__(self, root, image_set):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


class FusionSeg(data.Dataset):
    """
    FusionSeg segmentation dataset - processed and filtered ImageVID dataset.
    """

    def __init__(self, root, image_set):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

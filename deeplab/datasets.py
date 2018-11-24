from __future__ import print_function

import os
import numpy as np
from typing import Callable, Tuple

import torch
import torch.utils.data as data
from PIL import Image
import cv2


class SegmentationDataset(data.Dataset):
    """
    Generic segmentation Dataset. Note that joint transform will be run first.
    """

    def __init__(
        self,
        root: str,
        list_path: str,
        joint_transform: Callable = None,
        img_transform: Callable = None,
        gt_transform: Callable = None,
        img_dir: str = "img",
        gt_dir: str = "gt",
        img_ext: str = ".jpg",
        gt_ext: str = ".png"
    ) -> None:
        self.root = root
        with open(list_path, 'r') as f:
            self.filenames = f.readlines()
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.joint_transform = joint_transform
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_ext = img_ext
        self.gt_ext = gt_ext

    def __getitem__(self, idx: int) -> Tuple:
        filename = self.filenames[idx].rstrip()

        img_path = os.path.join(self.root, self.img_dir,
                                "{}{}".format(filename, self.img_ext))
        gt_path = os.path.join(self.root, self.gt_dir,
                               "{}{}".format(filename, self.gt_ext))

        # NOTE: Explicitly want to use BGR Pillow images to satisfy the
        # requirements for torchvision transforms and model input
        img = Image.fromarray(cv2.imread(img_path))
        gt = Image.fromarray(cv2.imread(gt_path, 0))

        if self.joint_transform is not None:
            img, gt = self.joint_transform(img, gt)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)

        return img, gt

    def __len__(self) -> int:
        return len(self.filenames)


class VOC12Dataset(SegmentationDataset):
    """
    VOC2012 augmented segmentation dataset. Note that joint transform will be
    run first.
    """

    def __init__(
        self,
        root: str,
        list_path: str,
        joint_transform: Callable = None,
        img_transform: Callable = None,
        gt_transform: Callable = None
    ) -> None:
        super(VOC12Dataset, self).__init__(
            root, list_path, joint_transform, img_transform, gt_transform)


class FusionSegDataset(SegmentationDataset):
    """
    FusionSeg segmentation dataset - processed and filtered ImageVID dataset.
    """

    def __init__(self,
                 root: str,
                 list_path: str,
                 joint_transform: Callable = None,
                 img_transform: Callable = None,
                 gt_transform: Callable = None,
                 binary_class: bool = True) -> None:
        if binary_class:
            gt_ext = "_gt.png"
        else:
            gt_ext = "_gt_class.png"
        super(FusionSegDataset, self).__init__(
            root, list_path, joint_transform, img_transform, gt_transform,
            img_dir="optical_flow", gt_dir="ground_truth", img_ext="_flow.png",
            gt_ext=gt_ext
        )

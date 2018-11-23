#!/usr/bin/env python
from __future__ import print_function

import os
from typing import Callable, Tuple

import torch
import torch.utils.data as data
from PIL import Image


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
        img = Image.open(os.path.join(self.root, self.img_dir,
                                      "{}{}".format(filename, self.img_ext)))
        gt = Image.open(os.path.join(self.root, self.gt_dir,
                                     "{}{}".format(filename, self.gt_ext)))
        if self.joint_transform is not None:
            img, gt = self.joint_transform(img, gt)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)

        return (img, gt)

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


if __name__ == "__main__":
    # Sanity check that the number of items matches as expected and we can open
    # an image
    def check_dataset(dataset: SegmentationDataset, num_elems: int) -> None:
        assert len(dataset) == num_elems, "Mismatch in number of elements"
        for img, gt in dataset:
            assert img is not None and gt is not None, "Error during reading"
            break

    voc_root = "/media/storage/data/voc2012"
    check_dataset(VOC12Dataset(voc_root, "./data/train_aug.txt"), 10582)
    check_dataset(VOC12Dataset(voc_root, "./data/val.txt"), 1449)

    fusionseg_root = "/media/storage/data/segmentation"
    check_dataset(FusionSegDataset(fusionseg_root, os.path.join(
        fusionseg_root, "images.txt")), 84929)
    check_dataset(FusionSegDataset(fusionseg_root, os.path.join(
        fusionseg_root, "images.txt"), binary_class=False), 84929)

    print('data is as expected for:\nvoc_root={}\nfusionseg_root={}'.format(
        voc_root, fusionseg_root))

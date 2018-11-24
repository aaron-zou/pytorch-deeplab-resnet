from __future__ import print_function

import enum
import multiprocessing
import os
import random
from typing import Tuple

import torch.utils.data as data
import torchvision.transforms.functional as F
from torchvision import transforms

from deeplab.datasets import FusionSegDataset, VOC12Dataset


@enum.unique
class Mode(enum.Enum):
    TRAIN = 0
    VAL = 1


class PadToTransform:
    """
    Zero-pads an image to be at minimum the passed-in size.
    """

    def __init__(self, target_size: Tuple[int, int]) -> None:
        self.target_height = target_size[0]
        self.target_width = target_size[1]

    def __call__(self, img):
        # NOTE: height and width are reversed between PIL.Image and target size
        im_width, im_height = img.size
        if im_width < self.target_width:
            # Pad the right
            img = F.pad(img, (0, 0, self.target_width - im_width, 0))
        if im_height < self.target_height:
            # Pad the bottom
            img = F.pad(img, (0, 0, 0, self.target_height - im_height))

        return img


class TrainJointTransform:
    """
    Preprocessing done jointly on input image and label image during training.
    """

    def __init__(self,
                 size: Tuple[int, int] = (321, 321),
                 flip_prob: float = 0.5) -> None:
        self.size = size
        self.flip_prob = flip_prob
        self.pad = PadToTransform(size)

    def __call__(self, img, gt):
        # Pad if necessary
        img = self.pad(img)
        gt = self.pad(gt)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=self.size)
        img = F.crop(img, i, j, h, w)
        gt = F.crop(gt, i, j, h, w)

        # Random flip
        if random.random() > self.flip_prob:
            img = F.hflip(img)
            gt = F.hflip(gt)

        return img, gt


TRANSFORMS = {
    Mode.TRAIN: {
        "joint_transform": TrainJointTransform(),
        "img_transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.481078, 0.407875, 0.457525], std=[1, 1, 1])
        ]),
        "gt_transform": transforms.ToTensor()
    },
    Mode.VAL: {
        "joint_transform": None,
        "img_transform": transforms.Compose([
            PadToTransform((513, 513)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.481078, 0.407875, 0.457525], std=[1, 1, 1])
        ]),
        "gt_transform": transforms.ToTensor()
    }
}


class VOC12Dataloader(data.DataLoader):

    def __init__(
            self,
            root: str,
            list_path: str,
            mode: Mode,
            num_workers: int = multiprocessing.cpu_count()) -> None:
        super(VOC12Dataloader, self).__init__(
            VOC12Dataset(
                root,
                list_path,
                TRANSFORMS[mode]["joint_transform"],
                TRANSFORMS[mode]["img_transform"],
                TRANSFORMS[mode]["gt_transform"]),
            num_workers=num_workers)


class FusionSegDataloader(data.DataLoader):

    def __init__(
            self,
            root: str,
            list_path: str,
            binary_class: bool,
            mode: Mode,
            num_workers: int = multiprocessing.cpu_count()) -> None:
        super(FusionSegDataloader, self).__init__(
            FusionSegDataset(root,
                             list_path,
                             TRANSFORMS[mode]["joint_transform"],
                             TRANSFORMS[mode]["img_transform"],
                             TRANSFORMS[mode]["gt_transform"],
                             binary_class),
            num_workers=num_workers)

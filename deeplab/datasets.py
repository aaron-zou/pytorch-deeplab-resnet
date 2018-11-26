from __future__ import print_function

import enum
import os
import numpy as np
import random
from typing import Callable, Tuple

import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
import cv2


@enum.unique
class Mode(enum.Enum):
    TRAIN = 0
    VAL = 1


def output_size(i):
    """Given shape of input image as i,i,3 in deeplab-resnet model, this function
    returns j such that the shape of output blob of is j,j,21 (21 in case of VOC)"""
    j = int(i)
    j = (j+1) // 2
    j = int(np.ceil((j+1) / 2.0))
    j = (j+1) // 2
    return j


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

    def __call__(self, img, gt):
        # Differing behavior for binary vs. multiclass case
        gt_raw = np.array(gt)
        if np.all(np.unique(gt_raw) == [0, 255]):
            gt_raw[gt_raw == 255] = 1
        else:
            gt_raw[gt_raw == 255] = 0
        gt = Image.fromarray(gt_raw)

        # Padding and random cropping
        scale = random.uniform(0.5, 1.1)
        scaled_size = (int(scale * self.size[0]), int(scale * self.size[1]))
        pad = PadToTransform(scaled_size)
        img = pad(img)
        gt = pad(gt)
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=scaled_size)
        img = F.crop(img, i, j, h, w)
        gt = F.crop(gt, i, j, h, w)

        # Random flip
        if random.random() > self.flip_prob:
            img = F.hflip(img)
            gt = F.hflip(gt)

        return img, gt


class ResizeLabelBatch:
    """
    Mirror the forward pass with labels for each separate interpolated output.
    """

    def __init__(self):
        pass

    def resize_batch(self, label, size):
        """Resize and convert to tensor"""
        resized = F.resize(label, size, interpolation=Image.NEAREST)
        return F.to_tensor(resized) * 255

    def __call__(self, gt):
        a = output_size(gt.size[0])
        b = output_size((gt.size[0] / 2) + 1)
        return [self.resize_batch(gt, size) for size in [a, a, b, a]]


TRANSFORMS = {
    Mode.TRAIN: {
        "joint_transform": TrainJointTransform(),
        "img_transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        ]),
        "gt_transform": ResizeLabelBatch()
    },
    Mode.VAL: {
        "joint_transform": None,
        "img_transform": transforms.Compose([
            PadToTransform((513, 513)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        ]),
        "gt_transform": transforms.ToTensor()
    }
}


class SegmentationDataset(data.Dataset):
    """
    Generic segmentation Dataset. Note that joint transform will be run first.
    """

    def __init__(
        self,
        mode: Mode,
        list_path: str,
        img_path: str,
        gt_path: str,
        img_ext: str,
        gt_ext: str
    ) -> None:
        with open(list_path, 'r') as f:
            self.filenames = f.readlines()
        self.img_transform = TRANSFORMS[mode]["img_transform"]
        self.gt_transform = TRANSFORMS[mode]["gt_transform"]
        self.joint_transform = TRANSFORMS[mode]["joint_transform"]
        self.img_path = img_path
        self.gt_path = gt_path
        self.img_ext = img_ext
        self.gt_ext = gt_ext

    def __getitem__(self, idx: int) -> Tuple:
        filename = self.filenames[idx].rstrip()

        img_path = os.path.join(
            self.img_path, "{}{}".format(filename, self.img_ext))
        gt_path = os.path.join(
            self.gt_path, "{}{}".format(filename, self.gt_ext))

        # NOTE: using Pillow RGB images
        img = Image.open(img_path)
        gt = Image.open(gt_path)

        if self.joint_transform is not None:
            img, gt = self.joint_transform(img, gt)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)

        return img, gt

    def __len__(self) -> int:
        return len(self.filenames)

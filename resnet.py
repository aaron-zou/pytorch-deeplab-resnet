"""
Library with helpers for converting between ResNet-101 classification and
segmentation networks (following DeepLab v2).
"""
import os
import torch
import torch.nn as nn
from deeplab_resnet import Classifier_Module, Res_Deeplab
from torchvision import models

PRETRAINED_DEEPLAB_PATH = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'data/MS_DeepLab_resnet_pretrained_COCO_init.pth')


def getResNet101(pretrained, num_categories=101):
    """Retrieve pretrained base ResNet-101 network"""
    model = models.resnet101(pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_categories)
    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model


def getDeepLabV2(num_categories=2, path=PRETRAINED_DEEPLAB_PATH):
    """Retrieve DeepLab v2 system"""
    model = Res_Deeplab(NoLabels=num_categories)
    if path is not None:
        saved_state_dict = torch.load(path)
        if num_categories != 21:
            for i in saved_state_dict:
                i_parts = i.split('.')
                if i_parts[1] == 'layer5':
                    saved_state_dict[i] = model.state_dict()[i]
        model.load_state_dict(saved_state_dict)
    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model


def getDeepLabV2FromResNet(num_categories=2):
    """Retrieve DeepLab v2 system from ResNet"""
    seg_model = getDeepLabV2(num_categories, None)
    toResNet = seg_model.Scale
    fromResNet = getResNet101(True, num_categories)
    _copyResNet(fromResNet, toResNet)
    return seg_model


def getResNetFromDeepLabV2(trainedStatePath, num_categories=2):
    """Load weights from *trained* DeepLab v2 network and transfer weights to
    a ResNet101 instance.
    """
    seg_model = getDeepLabV2(num_categories, trainedStatePath)
    return _makeTransferredResNet(seg_model, num_categories)


def _makeTransferredResNet(seg_model, num_categories=101):
    fromResNet = seg_model.Scale
    toResNet = getResNet101(True, num_categories)
    _copyResNet(fromResNet, toResNet)
    return toResNet


def _copyResNet(src, dest):
    # Perform a inner join, ignoring extra items in the destination module
    src_dict = src.state_dict()
    dest_dict = dest.state_dict()
    for key, value in src_dict.items():
        if key in dest_dict.keys():
            print("Copying: {}".format(key))
            dest_dict[key] = value

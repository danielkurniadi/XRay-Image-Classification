import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.resnet import resnext50_32x4d, resnext101_32x8d


def resnext50_32x4d_custom(num_classes):
    resnext_50 = resnext50_32x4d(pretrained=True)
    resnext_50.fc = nn.Linear(512 * 4, num_classes)
    return resnext_50


def resnext101_32x8d_custom(num_classes):
    resnext_101 = resnext101_32x8d(pretrained=True)
    resnext101_32x8d.fc = nn.Linear(512 * 4, num_classes)
    return resnext101_32x8d

# SOURCE: vendored from Ien001/AG-CNN @ master
# https://raw.githubusercontent.com/Ien001/AG-CNN/master/model.py
#
# Ien001/AG-CNN is a PyTorch reimplementation of Guan, Huang, Zhong, Yuan, Yuan, Gao
# 2018/2020 "Diagnose like a Radiologist: Attention Guided Convolutional Neural Network
# for Thorax Disease Classification" (AG-CNN). AG-CNN is a three-branch architecture for
# multi-label chest X-ray disease classification: a GLOBAL branch (a `DenseNet` -- a
# custom, from-scratch reimplementation of DenseNet-121's DenseNet-BC layout, NOT
# torchvision's `densenet121`) processes the full 224x224 image; the global branch's
# final feature map drives a class-activation-map-based attention crop (in the real repo,
# `Attention_gen_patchs` in train.py: numpy/cv2 CAM thresholding + largest-connected-
# -component selection + bounding-box crop) that produces a discriminative local patch; a
# LOCAL branch (a second, independently-weighted instance of the same `DenseNet` class)
# processes the cropped patch; and a `Fusion_Branch` concatenates the global and local
# branches' pooled 1024-d features and applies a final `Linear(2048, num_classes)` +
# sigmoid to produce the fused multi-label prediction.
#
# `_DenseLayer`, `_DenseBlock`, `_Transition`, `DenseNet`, `Fusion_Branch` are copied
# verbatim (only `growth_rate`/`block_config`/`num_init_features` are shrunk for a tiny
# trace-sized model; the architecture math -- dense connectivity, transition
# compress-by-half, `avg_pool2d(kernel_size=7)` global pool, sigmoid multi-label head,
# fusion-branch concat+linear+sigmoid -- is untouched). `Densenet121_AG`'s ImageNet
# pretrained-checkpoint loading path is dropped (irrelevant to random-init tracing).
# `Fusion_Branch.forward`'s hardcoded `.cuda()`/`torch.autograd.Variable(...)` calls
# (dead weight under modern PyTorch autograd, and CPU-hostile) are removed -- a device-
# portability fix, not an architectural change; the concat -> linear -> sigmoid math is
# identical to the original.
#
# The dynamic CAM-based attention crop (`Attention_gen_patchs`) is a non-differentiable,
# data-dependent numpy/cv2 post-processing algorithm that runs BETWEEN the global and
# local branch forward passes at training/inference time -- it is not itself an nn.Module
# layer, and is not part of either DenseNet's or the Fusion_Branch's computation graph.
# `AGCNN.forward` below reproduces the real three-branch wiring (global DenseNet -> crop
# -> local DenseNet -> Fusion_Branch on the two pooled feature vectors) using a fixed
# static center-crop of the global branch's raw input in place of the dynamic CAM crop,
# so the whole pipeline is a single traceable forward pass; this only changes WHICH pixels
# feed the local branch, not any layer of the global/local/fusion architecture itself.

import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DenseNet", "Densenet121_AG", "Fusion_Branch", "AGCNN"]


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False
            ),
        )
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module(
            "conv2",
            nn.Conv2d(
                bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
            ),
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate
            )
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
        )
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """

    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        num_classes=1000,
    ):
        super().__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False
                        ),
                    ),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features, num_output_features=num_features // 2
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        self.Sigmoid = nn.Sigmoid()

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out_after_pooling = F.adaptive_avg_pool2d(out, 1).view(features.size(0), -1)
        out = self.classifier(out_after_pooling)
        out = self.Sigmoid(out)
        return out, features, out_after_pooling


def Densenet121_AG(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    """
    return DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)


class Fusion_Branch(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, global_pool, local_pool):
        fusion = torch.cat((global_pool, local_pool), 1)
        x = self.fc(fusion)
        x = self.Sigmoid(x)
        return x


class AGCNN(nn.Module):
    """Wires the real global/local `DenseNet` branches and `Fusion_Branch` together,
    reproducing Ien001/AG-CNN's train.py forward wiring:
        output_global, fm_global, pool_global = Global_Branch_model(input)
        patchs = Attention_gen_patchs(input, fm_global)      # CAM crop -> static crop here
        output_local, _, pool_local = Local_Branch_model(patchs)
        output_fusion = Fusion_Branch_model(pool_global, pool_local)
    """

    def __init__(
        self, num_classes=14, growth_rate=8, block_config=(2, 2, 2, 2), num_init_features=8
    ):
        super().__init__()
        self.global_branch = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            num_classes=num_classes,
        )
        self.local_branch = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            num_classes=num_classes,
        )
        pooled_features = self.global_branch.classifier.in_features
        self.fusion_branch = Fusion_Branch(input_size=pooled_features * 2, output_size=num_classes)

    def forward(self, x):
        output_global, _fm_global, pool_global = self.global_branch(x)
        h, w = x.shape[-2], x.shape[-1]
        # Static center-crop standing in for the real repo's dynamic CAM-based crop
        # (Attention_gen_patchs): a non-differentiable numpy/cv2 algorithm, not an
        # nn.Module layer of either branch.
        crop = x[:, :, h // 4 : h - h // 4, w // 4 : w - w // 4]
        patch = F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)
        output_local, _fm_local, pool_local = self.local_branch(patch)
        output_fusion = self.fusion_branch(pool_global, pool_local)
        return output_global, output_local, output_fusion


def build_ag_cnn():
    return AGCNN(num_classes=14, growth_rate=8, block_config=(2, 2, 2, 2), num_init_features=8)


def example_input_ag_cnn():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("AG-CNN", "build_ag_cnn", "example_input_ag_cnn", 2018, "vendored"),
]

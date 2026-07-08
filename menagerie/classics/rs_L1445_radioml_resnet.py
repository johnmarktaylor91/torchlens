# FAITHFUL PORT of drexelwireless/RadioML @ 652d633b65d4063345d768d2905fff47b159bb68 (original framework: TensorFlow/Keras)
# https://raw.githubusercontent.com/drexelwireless/RadioML/652d633b65d4063345d768d2905fff47b159bb68/modrec/models.py
# https://raw.githubusercontent.com/drexelwireless/RadioML/652d633b65d4063345d768d2905fff47b159bb68/bin/train.py
#
# drexelwireless/RadioML (Abbas, Pano, Mainland, Dandekar, "Radio Modulation
# Classification Using Deep Residual Neural Networks", MILCOM 2022). The repo's
# `modrec.models.resnet18` builds an unmodified standard ResNet-18 image classifier
# (sourced in the real repo from `classification_models.tfkeras.Classifiers.get("resnet18")`,
# a TF/Keras port of the standard torchvision-style ResNet-18) and wraps it in a small
# Keras Sequential: UpSampling2D(2x) on a 128x128x3 input (to match the backbone's
# expected 256x256x3), the ResNet-18 backbone (`include_top=False`, so ending at the
# global-pooled feature map), GlobalAveragePooling2D, Dense(256, relu),
# Dropout(dropout=0.5), BatchNorm, Dense(nclasses, softmax) -- transcribed verbatim from
# `modrec/models.py::resnet18()`.
#
# `classification_models.tfkeras` cannot be installed in the base torch env (it is a
# Keras-only image-classifiers package with a hard TensorFlow dependency), so the real
# ResNet-18 architecture is reproduced here using torchvision's own unmodified
# `torchvision.models.resnet18` (weights=None, matching the real code's random-init
# `weights=weights` default usage during architecture inspection) as the equivalent
# standard ResNet-18 backbone, feeding it through `features` up to (but not including)
# its own avgpool/fc (== Keras `include_top=False`), then reproducing the real repo's
# exact head: GlobalAveragePooling -> Linear(512, 256)+ReLU -> Dropout(0.5) ->
# BatchNorm1d(256) -> Linear(256, nclasses) -> Softmax. The real code's
# `keras.layers.UpSampling2D(size=(2, 2))` on a 128x128 input (used because the code
# hardcodes the backbone's `input_shape=(256,256,3)`) is reproduced with
# `nn.Upsample(scale_factor=2, mode="nearest")` (Keras UpSampling2D's default
# interpolation is nearest-neighbor).
#
# The class-count-independent portion of the "RadioML ResNet" family used in the
# repo's `bin/train.py::get_model()` ("resnet18-outer"/"resnet18-gasf"/"resnet18-gadf")
# is exactly this network -- the three variants differ only in a pre-network numpy/TF
# image-encoding of the raw I/Q signal (outer product / Gramian angular summation field /
# Gramian angular difference field, in `modrec/preprocessing.py`), not in network
# architecture, so a single ResNet-18-classifier module captures the shared trainable
# graph.

import torch
import torch.nn as nn
from torchvision.models import resnet18


class RadioMLResNet(nn.Module):
    """ResNet-18-backbone modulation-image classifier.

    Faithful port of ``modrec.models.resnet18`` (drexelwireless/RadioML). The
    backbone is the real, unmodified ``torchvision.models.resnet18`` (structurally
    identical to the Keras ResNet-18 the original code loads via
    ``classification_models.tfkeras``); only the classification head is
    reconstructed layer-for-layer from the real Keras ``Sequential`` head.
    """

    def __init__(self, n_classes: int, dropout: float = 0.5):
        super().__init__()

        # keras.layers.UpSampling2D(size=(2, 2)) on a 128x128x3 input, to match the
        # backbone's hardcoded input_shape=(256, 256, 3).
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        backbone = resnet18(weights=None)
        # include_top=False: keep everything up through the final conv stage,
        # drop the backbone's own avgpool/fc so the Sequential head below owns them.
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # GlobalAveragePooling2D -> Dense(256, relu) -> Dropout -> BatchNorm ->
        # Dense(nclasses, softmax)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(512, 256)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(256)
        self.dense2 = nn.Linear(256, n_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.backbone(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dense1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x


def build_radioml_resnet():
    torch.manual_seed(0)
    model = RadioMLResNet(n_classes=11)
    model.eval()
    return model


def example_input_radioml_resnet():
    torch.manual_seed(0)
    # Batch x Channel(RGB-style encoded I/Q image) x 128 x 128, matching the real
    # code's `keras.layers.Input((128,128,3))`.
    return torch.randn(2, 3, 128, 128)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "RadioML-ResNet18",
        "build_radioml_resnet",
        "example_input_radioml_resnet",
        2022,
        MENAGERIE_ZOO,
    ),
]

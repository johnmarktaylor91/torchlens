# SOURCE: vendored from IVPLatNU/DeepCovidXR @ emerged_models
# (repo: https://raw.githubusercontent.com/IVPLatNU/DeepCovidXR/emerged_models/covid_models/*.py)
#
# The original repo builds several TF/Keras classification heads
# (DenseNet-121 / ResNet-50 / Xception / InceptionResNetV2 / EfficientNet-B2) as
# `backbone -> GlobalAveragePooling2D -> Dense(1, sigmoid)`, e.g.
# `covid_models/Densenet_model.py::DenseNet.buildBaseModel` and
# `covid_models/Resnet_model.py::ResNet.buildBaseModel`:
#
#     base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
#     x = base_model.output
#     x = layers.GlobalAveragePooling2D()(x)
#     predictions = layers.Dense(1, activation='sigmoid', name='last')(x)
#     model = Model(inputs=base_model.input, outputs=predictions)
#
# and a dropout variant (`buildDropModel`) used by the keras-tuner hyper-model
# (`covid_models/Hyper_model.py::hyperModel.build`):
#
#     x = base_model.output
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dropout(hp.Float('dropout_rate', ...))(x)
#     predictions = layers.Dense(1, activation='sigmoid', name='last')(x)
#
# This module ports that exact head structure faithfully onto the REAL torchvision
# backbone classes (unmodified `densenet121`/`resnet50`/`inception_v3`, random init,
# no pretrained weights needed for tracing) -- the backbones are the actual library
# classes, only the classification head (GAP + Linear + Sigmoid, optionally with
# Dropout) is transcribed from the repo's Keras code since torchvision backbones do
# not ship that head. The repo also has a post-hoc probability-level ensembling step
# (`ensemble.py::ensemble_members`, a Dirichlet-weighted average of independently
# computed scalar probabilities) which is not a tensor-graph-level fusion and is not
# reproduced here; each backbone head below is exactly what get traced/trained
# in the original code (`buildBaseModel` / `buildDropModel`).
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
import torch.nn as nn
from torchvision.models import densenet121, resnet50, inception_v3

MENAGERIE_ZOO = "vendored-pytorch"


class _SigmoidHead(nn.Module):
    """GlobalAveragePooling2D -> [Dropout] -> Dense(1, sigmoid), as in
    covid_models/*_model.py::buildBaseModel / buildDropModel."""

    def __init__(self, in_features, dropout=0.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.last = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        x = self.last(x)
        return self.sigmoid(x)


class DeepCovidXRDenseNet(nn.Module):
    """Port of covid_models/Densenet_model.py::DenseNet.buildBaseModel (real
    torchvision densenet121 backbone, faithfully-ported GAP+Dense+sigmoid head)."""

    def __init__(self, dropout=0.0):
        super().__init__()
        base = densenet121(weights=None)
        self.features = base.features
        self.relu = nn.ReLU(inplace=True)
        self.head = _SigmoidHead(base.classifier.in_features, dropout=dropout)

    def forward(self, x):
        f = self.features(x)
        f = self.relu(f)
        return self.head(f)


class DeepCovidXRResNet(nn.Module):
    """Port of covid_models/Resnet_model.py::ResNet.buildBaseModel (real
    torchvision resnet50 backbone, faithfully-ported GAP+Dense+sigmoid head)."""

    def __init__(self, dropout=0.0):
        super().__init__()
        base = resnet50(weights=None)
        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.head = _SigmoidHead(base.fc.in_features, dropout=dropout)

    def forward(self, x):
        f = self.stem(x)
        return self.head(f)


class DeepCovidXRInception(nn.Module):
    """Port of covid_models/Inceptionnet_model.py::Inception.buildBaseModel (real
    torchvision inception_v3 backbone, faithfully-ported GAP+Dense+sigmoid head)."""

    def __init__(self, dropout=0.0):
        super().__init__()
        base = inception_v3(weights=None, aux_logits=True, init_weights=False)
        base.aux_logits = False
        base.AuxLogits = None
        self.base = base
        self.head = _SigmoidHead(base.fc.in_features, dropout=dropout)

    def forward(self, x):
        # Mirror torchvision's internal feature stack, stopping before base.fc
        # (base.fc is unused; the real spatial features feed our own head).
        b = self.base
        x = b.Conv2d_1a_3x3(x)
        x = b.Conv2d_2a_3x3(x)
        x = b.Conv2d_2b_3x3(x)
        x = b.maxpool1(x)
        x = b.Conv2d_3b_1x1(x)
        x = b.Conv2d_4a_3x3(x)
        x = b.maxpool2(x)
        x = b.Mixed_5b(x)
        x = b.Mixed_5c(x)
        x = b.Mixed_5d(x)
        x = b.Mixed_6a(x)
        x = b.Mixed_6b(x)
        x = b.Mixed_6c(x)
        x = b.Mixed_6d(x)
        x = b.Mixed_6e(x)
        x = b.Mixed_7a(x)
        x = b.Mixed_7b(x)
        x = b.Mixed_7c(x)
        return self.head(x)


def build_deepcovidxr_densenet():
    return DeepCovidXRDenseNet(dropout=0.3)


def example_input_deepcovidxr_densenet():
    return torch.randn(1, 3, 224, 224)


def build_deepcovidxr_resnet():
    return DeepCovidXRResNet(dropout=0.3)


def example_input_deepcovidxr_resnet():
    return torch.randn(1, 3, 224, 224)


def build_deepcovidxr_inception():
    return DeepCovidXRInception(dropout=0.3)


def example_input_deepcovidxr_inception():
    return torch.randn(1, 3, 299, 299)


MENAGERIE_ENTRIES = [
    (
        "DeepCovidXR-DenseNet121",
        build_deepcovidxr_densenet,
        example_input_deepcovidxr_densenet,
        2020,
        "vendored-pytorch",
    ),
    (
        "DeepCovidXR-ResNet50",
        build_deepcovidxr_resnet,
        example_input_deepcovidxr_resnet,
        2020,
        "vendored-pytorch",
    ),
    (
        "DeepCovidXR-InceptionV3",
        build_deepcovidxr_inception,
        example_input_deepcovidxr_inception,
        2020,
        "vendored-pytorch",
    ),
]

# FAITHFUL PORT of ncbi-nlp/DeepSeeNet @ master (original framework: Keras 2.2.4 /
# TensorFlow, deepseenet/deepseenet_risk_factor.py). The shared `RiskFactorModel`
# architecture used by every DeepSeeNet head (drusen, pigment, advanced_amd, ga, cga --
# all confirmed to import and call this exact same factory) is:
#   base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
#   x = base_model.get_layer('mixed10').output   # final inception block, pre-GAP
#   x = GlobalAveragePooling2D()(x)
#   x = Dense(256, activation='relu', name='global_dense1')(x)
#   x = Dropout(0.5)(x)
#   x = Dense(128, activation='relu', name='global_dense2')(x)
#   x = Dropout(0.5)(x)
#   predictions = Dense(n_classes, activation='softmax', name='global_predictions')(x)
# Keras InceptionV3's `mixed10` block is the same "last inception block before the
# classifier head" as torchvision's `Inception3.Mixed_7c` (both are the final
# concatenated-branch block of the canonical Inception-v3 architecture prior to global
# pooling). Since the real code is Keras/TensorFlow (no runnable torch source exists in
# the repo -- only pretrained .h5 weight downloads), this file uses torchvision's REAL
# `Inception3` class (feature-extractor path only, matching `include_top=False`) as the
# backbone and faithfully reproduces the custom head (GAP -> Dense(256,relu) -> Dropout(0.5)
# -> Dense(128,relu) -> Dropout(0.5) -> Dense(n_classes,softmax)) verbatim from the Keras
# code above.
"""Faithful torch port of DeepSeeNet's shared RiskFactorModel head (ncbi-nlp/DeepSeeNet)."""

import torch
import torch.nn as nn
from torchvision.models import Inception3

MENAGERIE_ZOO = "ported-pytorch"


class InceptionV3FeatureBackbone(nn.Module):
    """Real torchvision Inception3, run only through `Mixed_7c` (the torch analog of
    Keras InceptionV3's `mixed10` layer used by `include_top=False`); the aux classifier
    and the top `avgpool`/`dropout`/`fc` stack are unused, matching the Keras
    `include_top=False` feature-extraction contract.
    """

    def __init__(self):
        super().__init__()
        backbone = Inception3(
            num_classes=1000, aux_logits=False, init_weights=True, transform_input=False
        )
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.maxpool1 = backbone.maxpool1
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.maxpool2 = backbone.maxpool2
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e
        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c

    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        return x


class DeepSeeNetRiskFactorModel(nn.Module):
    """The shared RiskFactorModel head used by every DeepSeeNet risk-factor classifier
    (drusen, pigment, advanced_amd, ga, cga): InceptionV3 features -> GAP ->
    Dense(256,relu) -> Dropout(0.5) -> Dense(128,relu) -> Dropout(0.5) ->
    Dense(n_classes,softmax), exactly as `RiskFactorModel()` builds it.
    """

    def __init__(self, n_classes=2):
        super().__init__()
        self.backbone = InceptionV3FeatureBackbone()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_dense1 = nn.Linear(2048, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.global_dense2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.global_predictions = nn.Linear(128, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x).flatten(1)
        x = self.relu1(self.global_dense1(x))
        x = self.dropout1(x)
        x = self.relu2(self.global_dense2(x))
        x = self.dropout2(x)
        x = self.global_predictions(x)
        return self.softmax(x)


# ---------------------------------------------------------------------------
# Staging build/example helpers. Inception3 requires spatial dims large enough
# to survive its stride/pooling stack (>= ~75x75); use 150x150 for a fast but
# valid trace instead of the repo's full 224x224.
# ---------------------------------------------------------------------------


def build_deepseenet():
    return DeepSeeNetRiskFactorModel(n_classes=3)


def example_input_deepseenet():
    return (torch.rand(1, 3, 150, 150),)


MENAGERIE_ENTRIES = [
    ("DeepSeeNet", "build_deepseenet", "example_input_deepseenet", 2019, "ported-pytorch"),
]

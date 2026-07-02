# SOURCE: vendored from drkhan107/CoroNet @ master (Covid19_CXR_Classification_Code.ipynb)
#
# CoroNet (Khan, "CoroNet: A deep neural network for detection and diagnosis of
# COVID-19 from chest x-ray images", Computer Methods and Programs in Biomedicine, 2020).
# The official notebook builds the model in Keras/TensorFlow as:
#
#   conv_base = tensorflow.keras.applications.Xception(weights='imagenet',
#                                                        include_top=False,
#                                                        input_shape=(150, 150, 3))
#   conv_base.trainable = True
#   model = models.Sequential()
#   model.add(conv_base)
#   model.add(layers.Flatten())
#   model.add(layers.Dropout(0.5))
#   model.add(layers.Dense(256, activation='relu'))
#   model.add(layers.Dense(4, activation='softmax'))
#
# i.e. the real Chollet Xception backbone (entry/middle/exit separable-conv flow,
# unmodified) with a Flatten -> Dropout -> Dense(256) -> Dense(4) classification head
# for 4-class (COVID-19 / bacterial pneumonia / viral pneumonia / normal) chest X-ray
# classification. timm's `legacy_xception` is the same Chollet Xception architecture
# (installed base lib, unmodified -- only the imports/framework differ: torch instead
# of Keras). This module vendors the real backbone class via timm and re-expresses the
# real notebook's exact head topology (Flatten, Dropout(0.5), Linear+ReLU(256),
# Linear+Softmax(4)) in torch.

import timm
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class CoroNet(nn.Module):
    """CoroNet: Xception backbone (timm legacy_xception, unmodified) + the official
    notebook's Flatten -> Dropout(0.5) -> Dense(256, relu) -> Dense(4, softmax) head."""

    def __init__(self, num_classes: int = 4, input_size: int = 150):
        super().__init__()
        self.conv_base = timm.create_model(
            "legacy_xception",
            pretrained=False,
            num_classes=0,
            global_pool="",
        )
        with torch.no_grad():
            feat = self.conv_base(torch.zeros(1, 3, input_size, input_size))
        flat_dim = feat.numel()

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(flat_dim, 256)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_base(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x


def build_coronet() -> CoroNet:
    return CoroNet(num_classes=4, input_size=150).eval()


def example_input_coronet() -> torch.Tensor:
    return torch.randn(1, 3, 150, 150)


MENAGERIE_ENTRIES = [
    ("CoroNet", "build_coronet", "example_input_coronet", 2020, "vendored-pytorch"),
]

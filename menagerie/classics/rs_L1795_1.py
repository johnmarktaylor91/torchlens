# SOURCE: vendored from arnoweng/CheXNet @ master (model.py)
"""
CheXNet: DenseNet-121 fine-tuned for 14-disease multi-label chest X-ray
classification (Rajpurkar et al. 2017; reference PyTorch implementation
arnoweng/CheXNet). The architecture is the real torchvision DenseNet-121
backbone with its classifier head replaced by a Linear + Sigmoid multi-label
head, exactly as in the original model.py.
"""

import torch
import torch.nn as nn
import torchvision

MENAGERIE_ZOO = "vendored-pytorch"


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(weights=None)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


N_CLASSES = 14  # ChestX-ray14 disease labels


def build_chexnet():
    return DenseNet121(N_CLASSES)


def example_input_chexnet():
    return torch.randn(1, 3, 224, 224)


MENAGERIE_ENTRIES = [
    ("CheXNet", build_chexnet, example_input_chexnet, 2017, "vendored-pytorch"),
]

# SOURCE: vendored from KimiaLabMayo/KimiaNet @ main
# https://raw.githubusercontent.com/KimiaLabMayo/KimiaNet/main/KimiaNet_Feature_Extraction_Code_Samples/KimiaNet_PyTorch_Feature_Extraction.py
#
# KimiaNet -- Riasatian et al. 2021 (Medical Image Analysis) DenseNet-121 backbone
# fine-tuned on TCGA whole-slide histopathology patches for representation
# learning / retrieval. The real architecture is `torchvision.models.densenet121`
# with its `features` submodule followed by a global-average-pool, wrapped in the
# repo's own `fully_connected` head class (which returns both the pooled 1024-d
# feature embedding AND the classification logits) -- this `fully_connected`
# class is copied verbatim from the official feature-extraction script. No
# architectural code was rewritten; only the pretrained-weight loading /
# dataset / CUDA-device setup (irrelevant to the architecture) was dropped.
#
# Upstream license: KimiaLabMayo/KimiaNet (see repo for terms; weights under
# separate research-use agreement, not used here -- random init only).

import torch
import torch.nn as nn
import torchvision

MENAGERIE_ZOO = "vendored-pytorch"


class fully_connected(nn.Module):
    """docstring for BottleNeck"""

    def __init__(self, model, num_ftrs, num_classes):
        super(fully_connected, self).__init__()
        self.model = model
        self.fc_4 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        out_1 = x
        out_3 = self.fc_4(x)
        return out_1, out_3


def build_kimianet():
    model = torchvision.models.densenet121(weights=None)
    model.features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    num_ftrs = model.classifier.in_features
    model_final = fully_connected(model.features, num_ftrs, 30)
    model_final.eval()
    return model_final


def example_input_kimianet():
    return (torch.randn(1, 3, 224, 224),)


MENAGERIE_ENTRIES = [
    ("KimiaNet", "build_kimianet", "example_input_kimianet", 2021, "vendored-pytorch"),
]

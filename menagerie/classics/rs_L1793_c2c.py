# SOURCE: vendored from YashSharma/C2C @ d53b6350372117bf3e5251044a856e0251d810b2
# https://raw.githubusercontent.com/YashSharma/C2C/main/C2C/models/resnet.py
#
# Sharma, Shrivastava, Ehsan, Moskaluk, Syed, Brown 2021 (MIDL) "Cluster-to-Conquer: A
# Framework for End-to-End Multi-Instance Learning for Whole Slide Image Classification".
# `WSIClassifier` is the C2C forward-pass architecture: a ResNet-18 patch encoder (with
# BatchNorm re-initialized/frozen for cross-slide domain adaptation) feeds a small tail
# MLP producing a 64-d patch embedding, which is consumed by (a) a per-patch linear
# `patch_classifier` head and (b) a gated attention-pooling head (`attention` ->
# softmax over patches -> weighted-sum `M = A @ x`) that aggregates all patches of one
# whole-slide image into a single slide-level `classifier` prediction. The training-time
# k-means/faiss patch-clustering step (`C2C/cluster.py`) that decides WHICH patches are
# sampled per slide is a data-loading preprocessing stage, not part of the forward-pass
# graph, and is not vendored here. `WSIClassifier` is copied verbatim from the real
# `C2C/models/resnet.py` (only unused sibling classes `Enc`/`PatchClassifier`/`EncAttn`,
# which just slice `WSIClassifier`'s own submodules for inference-time reuse, are
# dropped as unnecessary duplication -- no architectural changes to `WSIClassifier`).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class WSIClassifier(nn.Module):
    def __init__(self, n_class=2, bn_track_running_stats=False):
        super(WSIClassifier, self).__init__()
        self.L = 64
        self.D = 32
        self.K = 1

        resnet = models.resnet18(weights=None)

        # Since patches in each batch belong to a WSI, switching off batch statistics tracking
        # Or reinitializing batch parameters and changing momentum for quick domain adoption
        if bn_track_running_stats:
            for modules in resnet.modules():
                if isinstance(modules, nn.BatchNorm2d):
                    modules.track_running_stats = False
        else:
            for modules in resnet.modules():
                if isinstance(modules, nn.BatchNorm2d):
                    modules.momentum = 0.9
                    modules.weight = nn.Parameter(torch.ones(modules.weight.shape))
                    modules.running_mean = torch.zeros(modules.weight.shape)
                    modules.bias = nn.Parameter(torch.zeros(modules.weight.shape))
                    modules.running_var = torch.ones(modules.weight.shape)

        modules = list(resnet.children())[:-1]
        self.resnet_head = nn.Sequential(*modules)
        self.resnet_tail = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, self.L), nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, n_class))
        self.patch_classifier = nn.Sequential(nn.Linear(self.L * self.K, n_class))

    def forward(self, x):
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        x = self.resnet_head(x)
        x = x.view(x.size(0), -1)
        x = self.resnet_tail(x)
        xp = self.patch_classifier(x)

        A_unnorm = self.attention(x)
        A = torch.transpose(A_unnorm, 1, 0)
        A = F.softmax(A, dim=1)

        M = torch.mm(A, x)
        Y_prob = self.classifier(M)
        return Y_prob, xp, A_unnorm


def build_c2c():
    return WSIClassifier(n_class=2, bn_track_running_stats=False)


def example_input_c2c():
    # One WSI's worth of sampled patches: (1, num_patches, 3, H, W).
    return torch.randn(1, 8, 3, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("C2C", "build_c2c", "example_input_c2c", 2021, "vendored"),
]

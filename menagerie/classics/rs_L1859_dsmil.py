# SOURCE: vendored from https://github.com/binli123/dsmil-wsi @ master (dsmil.py)
# DSMIL: Dual-stream multiple instance learning networks for tumor detection in
# whole slide image (Li, Li & Eliceiri, CVPR 2021). The classes below (`FCLayer`,
# `IClassifier`, `BClassifier`, `MILNet`) are copied verbatim from the official
# repo's dsmil.py (MIT licensed). Only import paths were adjusted (none were
# relative) and the module was renamed for staging. The `IClassifier` feature
# extractor + `BClassifier` combination below matches the repo's own
# `compute_feats.py` construction pattern (resnet backbone with fc replaced by
# Identity feeding IClassifier, wired into MILNet with BClassifier).
"""Vendored DSMIL model definition (FCLayer, IClassifier, BClassifier, MILNet)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

MENAGERIE_ZOO = "vendored-pytorch"


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(
        self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False
    ):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh()
            )
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v), nn.Linear(input_size, input_size), nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        # 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(
            c, 0, descending=True
        )  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(
            feats, dim=0, index=m_indices[0, :]
        )  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(
            Q, q_max.transpose(0, 1)
        )  # compute inner product of Q to each entry of q_max, A in shape N x C
        A = F.softmax(
            A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0
        )  # normalize attention scores
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        return classes, prediction_bag, A, B


# ---------------------------------------------------------------------------
# Staging build/example helpers (tiny config, matches the repo's own
# compute_feats.py construction: resnet18 backbone with fc->Identity feeding
# IClassifier(feature_size=512), wired into MILNet with BClassifier).
# ---------------------------------------------------------------------------

_NUM_CLASSES = 2
_FEATS_SIZE = 512
_PATCH = 32


def _tiny_resnet18_backbone():
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Identity()
    return resnet


def build_dsmil():
    resnet = _tiny_resnet18_backbone()
    i_classifier = IClassifier(resnet, _FEATS_SIZE, output_class=_NUM_CLASSES)
    b_classifier = BClassifier(input_size=_FEATS_SIZE, output_class=_NUM_CLASSES)
    return MILNet(i_classifier, b_classifier)


def example_input_dsmil():
    # A "bag" of N instances (WSI patches), N x 3 x H x W, matching the real
    # compute_feats.py -> IClassifier -> BClassifier pipeline shape contract.
    n_instances = 4
    return (torch.rand(n_instances, 3, _PATCH, _PATCH),)


MENAGERIE_ENTRIES = [
    ("DSMIL", "build_dsmil", "example_input_dsmil", 2021, "vendored-pytorch"),
]

# SOURCE: vendored from lucastabelini/PolyLaneNet @ 6155ce2e7d3841c46a0035a25acc9d2e304d9856
#
# https://github.com/lucastabelini/PolyLaneNet
# https://raw.githubusercontent.com/lucastabelini/PolyLaneNet/master/lib/models.py
#
# PolyLaneNet (Tabelini et al. 2020, ICPR "PolyLaneNet: Lane Estimation via
# Deep Polynomial Regression") regresses lane markings as low-order
# polynomial coefficients directly from a single CNN backbone feature vector
# -- no anchors, no segmentation. `PolyRegression` (this file) wraps a
# backbone classifier (the repo supports `efficientnet` variants via the
# non-base `efficientnet_pytorch` package, or plain torchvision resnet34 /
# resnet50 / resnet101) and replaces its final FC with the paper's
# `OutputLayer`: a `regular_outputs_layer` (poly coeffs + upper/lower bound +
# confidence per lane) plus an optional `extra_outputs_layer` (per-lane
# category logits). Copied verbatim from `lib/models.py` (only the
# `efficientnet` branch is dropped below since `efficientnet_pytorch` is not
# an installed base lib -- the resnet34/50/101 branches, which are real
# torchvision classes with zero architectural modification beyond the swapped
# head, are kept unmodified). `decode`/`loss` (postprocessing/training-loss
# helpers, not part of the forward architecture) are omitted for staging
# brevity; `OutputLayer` and `PolyRegression.__init__`/`forward` are
# unmodified from upstream.

import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet50, resnet101


class OutputLayer(nn.Module):
    def __init__(self, fc, num_extra):
        super(OutputLayer, self).__init__()
        self.regular_outputs_layer = fc
        self.num_extra = num_extra
        if num_extra > 0:
            self.extra_outputs_layer = nn.Linear(fc.in_features, num_extra)

    def forward(self, x):
        regular_outputs = self.regular_outputs_layer(x)
        if self.num_extra > 0:
            extra_outputs = self.extra_outputs_layer(x)
        else:
            extra_outputs = None

        return regular_outputs, extra_outputs


class PolyRegression(nn.Module):
    def __init__(
        self,
        num_outputs,
        backbone,
        pretrained,
        curriculum_steps=None,
        extra_outputs=0,
        share_top_y=True,
        pred_category=False,
    ):
        super(PolyRegression, self).__init__()
        if backbone == "resnet34":
            self.model = resnet34(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == "resnet50":
            self.model = resnet50(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        elif backbone == "resnet101":
            self.model = resnet101(pretrained=pretrained)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_outputs)
            self.model.fc = OutputLayer(self.model.fc, extra_outputs)
        else:
            raise NotImplementedError()

        self.curriculum_steps = [0, 0, 0, 0] if curriculum_steps is None else curriculum_steps
        self.share_top_y = share_top_y
        self.extra_outputs = extra_outputs
        self.pred_category = pred_category
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, epoch=None, **kwargs):
        output, extra_outputs = self.model(x, **kwargs)
        for i in range(len(self.curriculum_steps)):
            if epoch is not None and epoch < self.curriculum_steps[i]:
                output[:, -len(self.curriculum_steps) + i] = 0
        return output, extra_outputs


def build_polylanenet():
    # TuSimple config uses backbone='resnet34', num_outputs = 35 (5 lanes *
    # 7: score + upper + lower + 4 poly coeffs), extra_outputs=0,
    # pred_category=False -- matches cfgs/tusimple.yaml (share_top_y=True,
    # curriculum_steps default). pretrained=False for a tiny random-init
    # trace.
    return PolyRegression(
        num_outputs=35,
        backbone="resnet34",
        pretrained=False,
        curriculum_steps=[0, 0, 0, 0],
        extra_outputs=0,
        share_top_y=True,
        pred_category=False,
    )


def example_input_polylanenet():
    # TuSimple default input resolution per cfgs/tusimple.yaml.
    return torch.randn(1, 3, 360, 640)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("PolyLaneNet", "build_polylanenet", "example_input_polylanenet", 2020, "vendored"),
]

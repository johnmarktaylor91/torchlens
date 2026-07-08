# SOURCE: vendored from https://github.com/cszn/KAIR @ master (fc1732f4)
# (models/network_dncnn.py :: IRCNN + models/basicblock.py :: sequential)
#
# IRCNN: Learning Deep CNN Denoiser Prior for Image Restoration (Zhang, Zuo, Gu, Zhang,
# CVPR 2017). https://github.com/cszn/IRCNN
#
# The official cszn/IRCNN release ships the trained denoiser priors as MATLAB .mat
# weight files (no PyTorch model definition). The same author (Kai Zhang / cszn)
# publishes and maintains the real PyTorch re-implementation of this exact
# architecture in his companion KAIR toolkit (`models/network_dncnn.py::IRCNN`,
# consumed by `main_test_ircnn_denoiser.py`), which is what this file vendors: the
# 7-layer dilated (1,2,3,4,3,2,1) fully-convolutional residual denoiser (predicts the
# noise residual `n`, output is `x - n`), used as the CNN prior term inside the paper's
# HQS (half-quadratic splitting) plug-and-play image-restoration pipeline. Only the
# `sequential()` helper from `models/basicblock.py` was inlined verbatim so the module
# has no repo-relative imports; no architecture was altered.

from collections import OrderedDict

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# models/basicblock.py :: sequential
# ---------------------------------------------------------------------------
def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# ---------------------------------------------------------------------------
# models/network_dncnn.py :: IRCNN
# ---------------------------------------------------------------------------
class IRCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(IRCNN, self).__init__()
        L = []
        L.append(
            nn.Conv2d(
                in_channels=in_nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                bias=True,
            )
        )
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(
                in_channels=nc,
                out_channels=out_nc,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True,
            )
        )
        self.model = sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x - n


# ---------------------------------------------------------------------------
# Tiny random-init build/example for TorchLens tracing.
# ---------------------------------------------------------------------------
def build_ircnn():
    torch.manual_seed(0)
    model = IRCNN(in_nc=1, out_nc=1, nc=8)
    model.eval()
    return model


def example_input_ircnn():
    torch.manual_seed(0)
    return torch.randn(1, 1, 32, 32)


MENAGERIE_ENTRIES = [
    ("IRCNN", "build_ircnn", "example_input_ircnn", 2017, MENAGERIE_ZOO),
]

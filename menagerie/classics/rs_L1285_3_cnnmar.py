# FAITHFUL PORT of yanbozhang007/CNN-MAR @ master (original framework: MATLAB/MatConvNet)
# (cnnmar/initializeCNNMAR.m -> the CNN architecture; cnnmar/net_train.m -> the default
# hyperparameters nKernel=3, nFeature=32, nConv=5)
"""CNN-MAR: convolutional-neural-network-based metal artifact reduction in X-ray CT
(Zhang & Yu, "Convolutional Neural Network Based Metal Artifact Reduction in X-Ray Computed
Tomography", IEEE TMI 2018). Official repo: https://github.com/yanbozhang007/CNN-MAR
(``cnnmar/initializeCNNMAR.m`` @ master).

The official repo is pure MATLAB built on MatConvNet's ``simplenn`` layer stack (``vl_nnconv``,
``vl_nnrelu`` -- no Python/PyTorch code anywhere in the repo, no reasonably-installable base-env
equivalent), so this transcribes ``initializeCNNMAR.m`` FAITHFULLY into self-contained torch:
the real net is a plain feed-forward residual-artifact-regression CNN --
``nConv`` (default 5, ``net_train.m``) stacked ``Conv2d(nKernel x nKernel, same padding via
``pad=(nKernel-1)/2``) + ReLU`` blocks at constant width ``nFeature`` (default 32), taking a
multi-channel input stack of prior-correction images (``nMC`` channels -- multiple MAR
candidate reconstructions e.g. linear-interpolation, beam-hardening-correction, etc., stacked
as input planes) and mapping down to a single-channel prediction via one final
``Conv2d(nKernel x nKernel, same padding, out_channels=1)`` with NO trailing activation
(matches the MATLAB net's final ``'prediction'`` conv layer, which has no following ReLU in
``initializeCNNMAR.m``). Every conv layer's kernel size, feature width, padding, and the
conv/relu/.../conv layer ordering mirror the MATLAB layer list exactly; Xavier weight init
(``xavier.m`` in the repo) is preserved via PyTorch's Xavier-uniform initializer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class CNNMAR(nn.Module):
    """Faithful port of ``initializeCNNMAR(nPatch, nMC, nKernel, nFeature, nConv)``.

    Args:
        n_mc: number of input prior-correction channels stacked as input planes
            (MATLAB's ``nMC``, the 3rd dim of the ``[nPatch nPatch nMC 1]`` input blob).
        n_kernel: conv kernel size along x/y (must be odd, >= 3); MATLAB default 3.
        n_feature: conv feature width; MATLAB default 32.
        n_conv: number of conv layers, >= 2; MATLAB default 5 (this vendors the
            ``conv1 -> relu1 -> [conv_i -> relu_i for i in 2..nConv-1] -> prediction``
            structure of the source verbatim, including the "prediction" layer's lack of a
            trailing ReLU).
    """

    def __init__(self, n_mc=3, n_kernel=3, n_feature=32, n_conv=5):
        super().__init__()
        if n_conv < 2:
            raise ValueError("n_conv must be >= 2 (initializeCNNMAR.m requires nConv >= 2)")
        pad = (n_kernel - 1) // 2

        layers = []
        # conv1 + relu1
        conv1 = nn.Conv2d(n_mc, n_feature, kernel_size=n_kernel, padding=pad)
        nn.init.xavier_uniform_(conv1.weight)
        nn.init.zeros_(conv1.bias)
        layers += [conv1, nn.ReLU(inplace=True)]

        # conv2..conv(nConv-1) + relu2..relu(nConv-1)
        for _ in range(2, n_conv):
            conv_i = nn.Conv2d(n_feature, n_feature, kernel_size=n_kernel, padding=pad)
            nn.init.xavier_uniform_(conv_i.weight)
            nn.init.zeros_(conv_i.bias)
            layers += [conv_i, nn.ReLU(inplace=True)]

        # final "prediction" conv (stride 1, no activation -- matches source exactly)
        prediction = nn.Conv2d(n_feature, 1, kernel_size=n_kernel, stride=1, padding=pad)
        nn.init.xavier_uniform_(prediction.weight)
        nn.init.zeros_(prediction.bias)
        layers += [prediction]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Menagerie staging harness
# ---------------------------------------------------------------------------
def build_cnnmar():
    """Tiny-size CNN-MAR matching net_train.m's own default hyperparameters
    (nKernel=3, nFeature=32, nConv=5), only n_mc reduced from the repo's multi-prior-image
    stack down to a small representative channel count."""
    return CNNMAR(n_mc=3, n_kernel=3, n_feature=8, n_conv=5)


def example_input_cnnmar():
    torch.manual_seed(0)
    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "CNN-MAR Metal Artifact Reduction",
        "build_cnnmar",
        "example_input_cnnmar",
        2018,
        "ported-pytorch",
    ),
]

# SOURCE: vendored from js3611/Deep-MRI-Reconstruction @ d8a40efd892e57799c3413630e5cb92d5b035cf8
# (cascadenet_pytorch/model_pytorch.py::conv_block, ::DnCn;
#  cascadenet_pytorch/kspace_pytorch.py::data_consistency, ::DataConsistencyInKspace)
"""Deep Cascade of CNNs for MR image reconstruction (Schlemper et al., IEEE TMI 2018 /
"A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction",
also presented at IPMI 2017). Official repo: https://github.com/js3611/Deep-MRI-Reconstruction
(``cascadenet_pytorch/`` @ master).

``DnCn`` is the cascaded-CNN reconstruction network: a stack of ``nc`` refinement
blocks, each a small residual CNN (``conv_block``) followed by a k-space data-consistency
(DC) layer (``DataConsistencyInKspace``) that projects the current image estimate back
onto the acquired (undersampled) k-space samples. This is the paper's non-recurrent
"D5C5"-style cascade (as opposed to the repo's separate ``CRNN_MRI``/``BCRNNlayer``
recurrent variant for dynamic/temporal MRI, which is not vendored here).

One non-architectural adaptation was made so the class runs on this environment's torch
2.x: the repo's data-consistency layer calls the pre-1.8 ``torch.fft(x, 2, normalized=...)``
/ ``torch.ifft(x, 2, normalized=...)`` complex-as-real-pair API, which was removed from
torch. It is replaced here with the equivalent modern ``torch.fft.fft2`` / ``torch.fft.ifft2``
on proper complex tensors (view_as_complex/view_as_real around the same real/imaginary-pair
layout the repo used), applied with the same ``ortho`` normalization and to the same
(nx, ny) axes. No layer, channel count, or forward-pass control-flow was changed; the
network topology (conv blocks + interleaved DC layers) is exactly the repo's ``DnCn``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)


def relu():
    return nn.ReLU(inplace=True)


def conv_block(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl="lrelu", conv_dim=2, n_out=None):
    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        return conv(nf, nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == "relu" else lrelu

    layers = [conv_1, nll()]
    for _ in range(nd - 2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)


def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


class DataConsistencyInKspace(nn.Module):
    """Create data consistency operator.

    Warning: note that FFT2 is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.
    """

    def __init__(self, noise_lvl=None, norm="ortho"):
        super().__init__()
        self.normalized = norm == "ortho"
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    @staticmethod
    def _fft2_ri(x_ri, inverse, normalized):
        """Apply (i)fft2 to the last axis (real/imag pair) of an (..., 2) real tensor,
        over the two axes preceding it -- equivalent to the removed
        torch.fft(x, 2, normalized=...) / torch.ifft(x, 2, normalized=...) API."""
        x_complex = torch.view_as_complex(x_ri.contiguous())
        norm = "ortho" if normalized else "backward"
        if inverse:
            k_complex = torch.fft.ifft2(x_complex, dim=(-2, -1), norm=norm)
        else:
            k_complex = torch.fft.fft2(x_complex, dim=(-2, -1), norm=norm)
        return torch.view_as_real(k_complex)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        if x.dim() == 4:  # input is 2D
            x = x.permute(0, 2, 3, 1)
            k0 = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5:  # input is 3D
            x = x.permute(0, 4, 2, 3, 1)
            k0 = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        k = self._fft2_ri(x, inverse=False, normalized=self.normalized)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = self._fft2_ri(out, inverse=True, normalized=self.normalized)

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res


class DnCn(nn.Module):
    """Deep Cascade of CNNs: nc refinement blocks, each a residual conv_block
    followed by a k-space data-consistency projection."""

    def __init__(self, n_channels=2, nc=5, nd=5, **kwargs):
        super().__init__()
        self.nc = nc
        self.nd = nd
        conv_blocks = []
        dcs = []

        conv_layer = conv_block

        for _ in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs.append(DataConsistencyInKspace(norm="ortho"))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = nn.ModuleList(dcs)

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x = self.dcs[i].perform(x, k, m)

        return x


MENAGERIE_ZOO = "vendored-pytorch"


def build_deepmrirecon():
    return DnCn(n_channels=2, nc=3, nd=3, nf=8)


def example_input_deepmrirecon():
    x = torch.randn(1, 2, 16, 16)
    k = torch.randn(1, 2, 16, 16)
    m = torch.randint(0, 2, (1, 2, 16, 16)).float()
    return (x, k, m)


MENAGERIE_ENTRIES = [
    (
        "Deep MRI Reconstruction (DnCn)",
        "build_deepmrirecon",
        "example_input_deepmrirecon",
        2018,
        "vendored-pytorch",
    ),
]

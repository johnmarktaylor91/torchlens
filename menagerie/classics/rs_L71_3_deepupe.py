# SOURCE: vendored from hermosayhl/DeepUPE_pytorch @ main
# File: model.py
# https://github.com/hermosayhl/DeepUPE_pytorch
#
# hermosayhl/DeepUPE_pytorch is a complete unofficial PyTorch port of the
# official dvlab-research/DeepUPE (Underexposed Photo Enhancement Using
# Deep Illumination Estimation, CVPR 2019), whose official release
# (dvlab-research/DeepUPE) ships only a TensorFlow 1.x graph + custom CUDA
# op (main/hdrnet_ops.py, main/ops/bilateral_slice.cu.cc) and a frozen
# checkpoint -- no trainable TF1 source is runnable in this base env, and
# the custom CUDA op has no PyTorch equivalent to fall back on. This repo
# is a full from-scratch PyTorch reimplementation of the same
# HDRNet-based illumination-estimation architecture (Gharbi et al. "Deep
# Bilateral Learning for Real-Time Image Enhancement", used by DeepUPE as
# its illumination-map estimator), depending only on torch/torchvision/
# numpy -- all in the base env.
#
# Minimal changes from the original source:
#   - `train.py`'s hardcoded hyperparameters (opt.luma_bins=8,
#     opt.channel_multiplier=1, opt.spatial_bin=8, opt.batch_norm=True,
#     opt.low_size=256) are reproduced in a small `HDRNetParams`
#     dataclass to stand in for the ad-hoc `argparse.Namespace` the
#     original repo threads through as `params=`.
#   - Added build_deepupe()/example_input_deepupe() harness at the
#     bottom to construct a tiny instance (low_size shrunk to 64 to keep
#     the smoke trace small; the network is fully-convolutional /
#     size-agnostic aside from `spatial_bin` and `luma_bins`) and drive
#     the real HDRNet forward pass (lowres, fullres) -> enhanced image.
#
# Architecture (unmodified from source): HDRNet = Coeffs (splat conv
# tower -> global conv+FC tower -> local conv tower -> fused bilateral
# grid of per-pixel affine color-transform coefficients) + GuideNN
# (1x1-conv "guide map" predictor over the full-res image) + Slice
# (trilinear grid_sample lookup of per-pixel coefficients from the
# bilateral grid using the guide map) + ApplyCoeffs (applies the sliced
# 3x4 affine color matrix to the full-res input). This is exactly
# DeepUPE's illumination-estimation subnetwork.

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        inc,
        outc,
        kernel_size=3,
        padding=1,
        stride=1,
        use_bias=True,
        activation=nn.ReLU,
        batch_norm=False,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            int(inc), int(outc), kernel_size, padding=padding, stride=stride, bias=use_bias
        )
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        device = bilateral_grid.get_device()

        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3
        coeff = F.grid_sample(bilateral_grid, guidemap_guide)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):
        """
        Affine:
        r = a11*r + a12*g + a13*b + a14
        g = a21*r + a22*g + a23*b + a24
        ...
        """
        R = (
            torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True)
            + coeff[:, 3:4, :, :]
        )
        G = (
            torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True)
            + coeff[:, 7:8, :, :]
        )
        B = (
            torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True)
            + coeff[:, 11:12, :, :]
        )

        return torch.cat([R, G, B], dim=1)


class GuideNN(nn.Module):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params
        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=params.batch_norm)
        self.conv2 = ConvBlock(16, 1, kernel_size=1, padding=0, activation=nn.Tanh)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Coeffs(nn.Module):
    def __init__(self, nin=3, nout=4, params=None):
        super(Coeffs, self).__init__()
        self.params = params
        self.nin = nin
        self.nout = nout

        lb = params.luma_bins
        cm = params.channel_multiplier
        sb = params.spatial_bin
        bn = params.batch_norm
        nsize = params.low_size

        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(nsize / sb))
        self.splat_features = nn.ModuleList()
        prev_ch = nin
        for i in range(n_layers_splat):
            use_bn = bn if i > 0 else False
            self.splat_features.append(
                ConvBlock(prev_ch, cm * (2**i) * lb, 3, stride=2, batch_norm=use_bn)
            )
            prev_ch = splat_ch = cm * (2**i) * lb

        # global features
        n_layers_global = int(np.log2(sb / 4))
        self.global_features_conv = nn.ModuleList()
        self.global_features_fc = nn.ModuleList()
        for i in range(n_layers_global):
            self.global_features_conv.append(
                ConvBlock(prev_ch, cm * 8 * lb, 3, stride=2, batch_norm=bn)
            )
            prev_ch = cm * 8 * lb

        n_total = n_layers_splat + n_layers_global
        prev_ch = prev_ch * (nsize / 2**n_total) ** 2
        self.global_features_fc.append(FC(prev_ch, 32 * cm * lb, batch_norm=bn))
        self.global_features_fc.append(FC(32 * cm * lb, 16 * cm * lb, batch_norm=bn))
        self.global_features_fc.append(
            FC(16 * cm * lb, 8 * cm * lb, activation=None, batch_norm=bn)
        )

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(ConvBlock(splat_ch, 8 * cm * lb, 3, batch_norm=bn))
        self.local_features.append(
            ConvBlock(8 * cm * lb, 8 * cm * lb, 3, activation=None, use_bias=False)
        )

        # predicton
        self.conv_out = ConvBlock(8 * cm * lb, lb * nout * nin, 1, padding=0, activation=None)

    def forward(self, lowres_input):
        params = self.params
        bs = lowres_input.shape[0]
        lb = params.luma_bins
        cm = params.channel_multiplier
        sb = params.spatial_bin

        x = lowres_input
        for layer in self.splat_features:
            x = layer(x)
        splat_features = x

        for layer in self.global_features_conv:
            x = layer(x)
        x = x.view(bs, -1)
        for layer in self.global_features_fc:
            x = layer(x)
        global_features = x

        x = splat_features
        for layer in self.local_features:
            x = layer(x)
        local_features = x

        fusion_grid = local_features
        fusion_global = global_features.view(bs, 8 * cm * lb, 1, 1)
        fusion = self.relu(fusion_grid + fusion_global)

        x = self.conv_out(fusion)
        x = x.view(bs, self.nin * self.nout, lb, sb, sb)  # B x Coefs x Luma x Spatial x Spatial
        return x


class HDRNet(nn.Module):
    def __init__(self, params):
        super(HDRNet, self).__init__()
        self.coeffs = Coeffs(params=params)
        self.guide = GuideNN(params=params)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

    def forward(self, lowres, fullres):
        coeffs = self.coeffs(lowres)
        guide = self.guide(fullres)
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, fullres)
        return out


@dataclass
class HDRNetParams:
    luma_bins: int = 8
    channel_multiplier: int = 1
    spatial_bin: int = 8
    batch_norm: bool = True
    low_size: int = 64


def build_deepupe():
    return HDRNet(params=HDRNetParams())


def example_input_deepupe():
    # batch_size=2 (not 1): BatchNorm1d in the global-features FC tower
    # requires more than one sample per channel while in training mode.
    low_size = HDRNetParams().low_size
    lowres = torch.randn(2, 3, low_size, low_size)
    fullres = torch.randn(2, 3, low_size, low_size)
    return (lowres, fullres)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "DeepUPE",
        "build_deepupe",
        "example_input_deepupe",
        2019,
        "vendored",
    ),
]

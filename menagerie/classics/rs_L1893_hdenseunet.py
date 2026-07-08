# FAITHFUL PORT of https://github.com/xmengli/H-DenseUNet @ master (original framework: Keras 2.0.8 / TF1.x)
#
# Transcribed from the official repo's `hybridnet.py` (`dense_rnn_net`, `DenseUNet`,
# `DenseNet3D`, `conv_block`/`conv_block3d`, `dense_block`/`dense_block3d`,
# `transition_block`/`transition_block3d`) and `lib/custom_layers.py` (`Scale`, a
# learned per-channel affine gamma*x+beta used after every frozen BatchNorm, exactly
# as in the original DenseNet-Keras "BN+Scale" convention). The official repo ships a
# vendored Keras-2.0.8/TF1.x tree and cannot run in a modern torch env, so this module
# transcribes the real network layer-by-layer into self-contained torch (channels-first
# NCHW / NCDHW instead of Keras' channels-last NHWC / NDHWC).
#
# `dense_rnn_net` (the paper's actual "H-DenseUNet" hybrid model, TMI 2018,
# arxiv:1709.07330) is: a 2D DenseUNet (DenseNet-161 encoder + U-Net-style decoder)
# run independently on every overlapping 3-consecutive-slice window of the input CT/MR
# volume (each window predicts one center-slice segmentation + feature map); the
# per-slice 2D outputs are re-stacked along depth into pseudo-3D volumes; those are fed
# (concatenated with the original volume, scaled) into a 3D DenseNet (DenseNet-161-style
# 3D encoder + decoder); finally the 3D decoder features are summed with the restacked 2D
# features and passed through one more fusion Conv3D block + classifier. This fusion of
# an independently-supervised 2D branch and a hybridized 3D branch via feature summation
# IS the paper's core architectural contribution (not merely a data/objective change), so
# this is a faithful port, not a "real class already in a base lib" case.
#
# A community PyTorch port (ananyajana/HDenseUNet_pytorch) exists but only transcribes
# the 2D `DenseUNet` branch, not the 3D branch or the hybrid fusion -- it is not the real
# hybrid architecture, so it was not vendored as-is; this module independently transcribes
# the FULL hybrid net (2D + 3D + fusion) straight from the official Keras source.
"""Faithful torch port of H-DenseUNet's `dense_rnn_net` hybrid 2D+3D DenseUNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"

_EPS = 1.1e-5


class Scale(nn.Module):
    """lib/custom_layers.py::Scale -- learned per-channel affine y = x * gamma + beta,
    applied after every (frozen) BatchNorm, exactly as in the original DenseNet-Keras
    "BN+Scale" split (Keras BatchNorm here is trainable=False; the Scale layer supplies
    the learnable affine instead)."""

    def __init__(self, num_features: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1, -1] + [1] * (x.dim() - 2)
        return x * self.gamma.view(*shape) + self.beta.view(*shape)


# ---------------------------------------------------------------------------
# hybridnet.py :: 2D branch (conv_block / dense_block / transition_block / DenseUNet)
# ---------------------------------------------------------------------------


class ConvBlock2D(nn.Module):
    """hybridnet.py::conv_block -- BN+Scale+ReLU, 1x1 bottleneck conv, BN+Scale+ReLU,
    3x3 conv (with explicit zero-padding, matching Keras' `ZeroPadding2D` + valid-pad
    Conv2D idiom)."""

    def __init__(self, nb_inp_fea: int, growth_rate: int):
        super().__init__()
        inter_channel = growth_rate * 4
        self.bn1 = nn.BatchNorm2d(nb_inp_fea, eps=_EPS)
        self.scale1 = Scale(nb_inp_fea)
        self.conv1 = nn.Conv2d(nb_inp_fea, inter_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channel, eps=_EPS)
        self.scale2 = Scale(inter_channel)
        self.conv2 = nn.Conv2d(inter_channel, growth_rate, kernel_size=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.scale1(self.bn1(x)))
        x = self.conv1(x)
        x = F.relu(self.scale2(self.bn2(x)))
        x = F.pad(x, (1, 1, 1, 1))
        x = self.conv2(x)
        return x


class DenseBlock2D(nn.Module):
    """hybridnet.py::dense_block -- feature-concatenating chain of ConvBlock2D."""

    def __init__(self, nb_layers: int, nb_filter: int, growth_rate: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(nb_layers):
            self.layers.append(ConvBlock2D(nb_filter + i * growth_rate, growth_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = x
        for layer in self.layers:
            new_feat = layer(feats)
            feats = torch.cat([feats, new_feat], dim=1)
        return feats


class TransitionBlock2D(nn.Module):
    """hybridnet.py::transition_block -- BN+Scale+ReLU, 1x1 compress conv, 2x2 avg pool."""

    def __init__(self, nb_filter: int, compression: float):
        super().__init__()
        out_ch = int(nb_filter * compression)
        self.bn = nn.BatchNorm2d(nb_filter, eps=_EPS)
        self.scale = Scale(nb_filter)
        self.conv = nn.Conv2d(nb_filter, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.scale(self.bn(x)))
        x = self.conv(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class DenseUNet2D(nn.Module):
    """hybridnet.py::DenseUNet -- DenseNet-161 (growth_rate=48, layers=[6,12,36,24])
    stem+encoder with a symmetric U-Net-style upsampling decoder returning the
    penultimate decoder features `ac_up4` (fed into the fusion sum) alongside the
    per-slice classifier logits."""

    def __init__(self, in_channels: int = 3, growth_rate: int = 48, reduction: float = 0.5):
        super().__init__()
        nb_filter = 96
        nb_layers = [6, 12, 36, 24]
        compression = 1.0 - reduction

        self.conv1 = nn.Conv2d(in_channels, nb_filter, kernel_size=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(nb_filter, eps=_EPS)
        self.scale1 = Scale(nb_filter)

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        filt = nb_filter
        box_channels = []
        for block_idx in range(3):
            self.blocks.append(DenseBlock2D(nb_layers[block_idx], filt, growth_rate))
            filt += nb_layers[block_idx] * growth_rate
            box_channels.append(filt)  # box[block_idx+1] channels (post-block, pre-transition)
            self.transitions.append(TransitionBlock2D(filt, compression))
            filt = int(filt * compression)
        self.block4 = DenseBlock2D(nb_layers[3], filt, growth_rate)
        filt += nb_layers[3] * growth_rate
        self.nb_filter3 = box_channels[
            2
        ]  # box[3] channels: DenseBlock3's output, pre-trans3 (2112 @ defaults)
        self.bn_final = nn.BatchNorm2d(filt, eps=_EPS)
        self.scale_final = Scale(filt)

        self.conv_lat = nn.Conv2d(self.nb_filter3, 2208, kernel_size=1)

        self.conv_up0 = nn.Conv2d(2 * 2208, 768, kernel_size=3, padding=1)
        self.bn_up0 = nn.BatchNorm2d(768)
        self.conv_up1 = nn.Conv2d(2 * 768, 384, kernel_size=3, padding=1)
        self.bn_up1 = nn.BatchNorm2d(384)
        self.conv_up2 = nn.Conv2d(2 * 384, 96, kernel_size=3, padding=1)
        self.bn_up2 = nn.BatchNorm2d(96)
        self.conv_up3 = nn.Conv2d(2 * 96, 96, kernel_size=3, padding=1)
        self.bn_up3 = nn.BatchNorm2d(96)
        self.conv_up4 = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        self.bn_up4 = nn.BatchNorm2d(64)

        self.classifier = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x: torch.Tensor):
        box = []
        x = F.pad(x, (3, 3, 3, 3))
        x = self.conv1(x)
        x = F.relu(self.scale1(self.bn1(x)))
        box.append(x)
        x = F.pad(x, (1, 1, 1, 1))
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        for block, trans in zip(self.blocks, self.transitions):
            x = block(x)
            box.append(x)
            x = trans(x)

        x = self.block4(x)
        x = F.relu(self.scale_final(self.bn_final(x)))
        box.append(x)

        up0 = F.interpolate(x, scale_factor=2, mode="nearest")
        line0 = self.conv_lat(box[3])
        cat0 = torch.cat([line0, up0], dim=1)
        out = F.relu(self.bn_up0(self.conv_up0(cat0)))

        up1 = F.interpolate(out, scale_factor=2, mode="nearest")
        cat1 = torch.cat([box[2], up1], dim=1)
        out = F.relu(self.bn_up1(self.conv_up1(cat1)))

        up2 = F.interpolate(out, scale_factor=2, mode="nearest")
        cat2 = torch.cat([box[1], up2], dim=1)
        out = F.relu(self.bn_up2(self.conv_up2(cat2)))

        up3 = F.interpolate(out, scale_factor=2, mode="nearest")
        cat3 = torch.cat([box[0], up3], dim=1)
        out = F.relu(self.bn_up3(self.conv_up3(cat3)))

        up4 = F.interpolate(out, scale_factor=2, mode="nearest")
        conv_up4 = self.conv_up4(up4)
        ac_up4 = F.relu(self.bn_up4(conv_up4))

        classifer2d = self.classifier(ac_up4)
        return ac_up4, classifer2d


# ---------------------------------------------------------------------------
# hybridnet.py :: 3D branch (conv_block3d / dense_block3d / transition_block3d /
# DenseNet3D)
# ---------------------------------------------------------------------------


class ConvBlock3D(nn.Module):
    """hybridnet.py::conv_block3d -- 3D analogue of ConvBlock2D."""

    def __init__(self, nb_inp_fea: int, growth_rate: int):
        super().__init__()
        inter_channel = growth_rate * 4
        self.bn1 = nn.BatchNorm3d(nb_inp_fea, eps=_EPS)
        self.scale1 = Scale(nb_inp_fea)
        self.conv1 = nn.Conv3d(nb_inp_fea, inter_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(inter_channel, eps=_EPS)
        self.scale2 = Scale(inter_channel)
        self.conv2 = nn.Conv3d(inter_channel, growth_rate, kernel_size=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.scale1(self.bn1(x)))
        x = self.conv1(x)
        x = F.relu(self.scale2(self.bn2(x)))
        x = F.pad(x, (1, 1, 1, 1, 1, 1))
        x = self.conv2(x)
        return x


class DenseBlock3D(nn.Module):
    """hybridnet.py::dense_block3d."""

    def __init__(self, nb_layers: int, nb_filter: int, growth_rate: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(nb_layers):
            self.layers.append(ConvBlock3D(nb_filter + i * growth_rate, growth_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = x
        for layer in self.layers:
            new_feat = layer(feats)
            feats = torch.cat([feats, new_feat], dim=1)
        return feats


class TransitionBlock3D(nn.Module):
    """hybridnet.py::transition_block3d -- BN+Scale+ReLU, 1x1x1 compress conv,
    (2,2,1)-strided avg pool (depth axis untouched, matching the original's
    "keep-slice-count" pooling)."""

    def __init__(self, nb_filter: int, compression: float):
        super().__init__()
        out_ch = int(nb_filter * compression)
        self.bn = nn.BatchNorm3d(nb_filter, eps=_EPS)
        self.scale = Scale(nb_filter)
        self.conv = nn.Conv3d(nb_filter, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.scale(self.bn(x)))
        x = self.conv(x)
        x = F.avg_pool3d(x, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        return x


class DenseNet3D(nn.Module):
    """hybridnet.py::DenseNet3D -- DenseNet-161-style 3D encoder (growth_rate=32,
    layers=[3,4,12,8]) with a symmetric 3D upsampling decoder."""

    def __init__(self, in_channels: int, growth_rate: int = 32, reduction: float = 0.5):
        super().__init__()
        nb_filter = 96
        nb_layers = [3, 4, 12, 8]
        compression = 1.0 - reduction

        self.conv1 = nn.Conv3d(in_channels, nb_filter, kernel_size=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm3d(nb_filter, eps=_EPS)
        self.scale1 = Scale(nb_filter)

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        filt = nb_filter
        for block_idx in range(3):
            self.blocks.append(DenseBlock3D(nb_layers[block_idx], filt, growth_rate))
            filt += nb_layers[block_idx] * growth_rate
            self.transitions.append(TransitionBlock3D(filt, compression))
            filt = int(filt * compression)
        self.block4 = DenseBlock3D(nb_layers[3], filt, growth_rate)
        filt += nb_layers[3] * growth_rate
        self.bn_final = nn.BatchNorm3d(filt, eps=_EPS)
        self.scale_final = Scale(filt)

        self.conv_up0 = nn.Conv3d(filt, 504, kernel_size=3, padding=1)
        self.bn_up0 = nn.BatchNorm3d(504)
        self.conv_up1 = nn.Conv3d(504, 224, kernel_size=3, padding=1)
        self.bn_up1 = nn.BatchNorm3d(224)
        self.conv_up2 = nn.Conv3d(224, 192, kernel_size=3, padding=1)
        self.bn_up2 = nn.BatchNorm3d(192)
        self.conv_up3 = nn.Conv3d(192, 96, kernel_size=3, padding=1)
        self.bn_up3 = nn.BatchNorm3d(96)
        self.conv_up4 = nn.Conv3d(96, 64, kernel_size=3, padding=1)
        self.bn_up4 = nn.BatchNorm3d(64)

        self.classifier = nn.Conv3d(64, 3, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, (3, 3, 3, 3, 3, 3))
        x = self.conv1(x)
        x = F.relu(self.scale1(self.bn1(x)))
        x = F.pad(x, (1, 1, 1, 1, 1, 1))
        x = F.max_pool3d(x, kernel_size=3, stride=2)

        for block, trans in zip(self.blocks, self.transitions):
            x = block(x)
            x = trans(x)

        x = self.block4(x)
        x = F.relu(self.scale_final(self.bn_final(x)))

        up0 = F.interpolate(x, scale_factor=(2, 2, 1), mode="nearest")
        out = F.relu(self.bn_up0(self.conv_up0(up0)))

        up1 = F.interpolate(out, scale_factor=(2, 2, 1), mode="nearest")
        out = F.relu(self.bn_up1(self.conv_up1(up1)))

        up2 = F.interpolate(out, scale_factor=(2, 2, 1), mode="nearest")
        out = F.relu(self.bn_up2(self.conv_up2(up2)))

        up3 = F.interpolate(out, scale_factor=(2, 2, 2), mode="nearest")
        out = F.relu(self.bn_up3(self.conv_up3(up3)))

        up4 = F.interpolate(out, scale_factor=(2, 2, 2), mode="nearest")
        ac_up4 = F.relu(self.bn_up4(self.conv_up4(up4)))

        classifer3d = self.classifier(ac_up4)
        return ac_up4, classifer3d


# ---------------------------------------------------------------------------
# hybridnet.py :: dense_rnn_net -- the real end-to-end hybrid model. Slicing/
# restacking logic is transcribed directly from the official `slice`/`slice2d`/
# `slice_last` Lambda layers, translated from Keras NHWC/NDHWC to torch NCHW/NCDHW.
# ---------------------------------------------------------------------------


class HDenseUNet(nn.Module):
    """Faithful port of `dense_rnn_net`: independent 2D DenseUNet pass over every
    overlapping 3-slice window of the volume, restacked into a pseudo-3D volume,
    fused with a 3D DenseNet branch operating on [volume ; scaled 2D-response] via
    a final feature summation + Conv3D classifier head."""

    def __init__(self, input_cols: int = 8):
        super().__init__()
        if input_cols < 3:
            raise ValueError("dense_rnn_net requires input_cols >= 3 (3-slice windows)")
        self.input_cols = input_cols
        self.denseunet2d = DenseUNet2D(in_channels=3, reduction=0.5)
        self.densenet3d = DenseNet3D(
            in_channels=4, reduction=0.5
        )  # volume(1ch) + scaled 2D response(3ch)
        self.final_conv = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.3)
        self.final_bn = nn.BatchNorm3d(64)
        self.classifier = nn.Conv3d(64, 3, kernel_size=1, padding=0)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        # volume: (B, 1, H, W, D) matching the original Keras input_shape
        # (batch, input_size, input_size, input_cols, 1) reordered to torch NCHWD.
        b, _, h, w, d = volume.shape

        # --- build the (B*(D-2), 3, H, W) overlapping-3-slice-window batch, exactly
        # matching the official Lambda `slice` loop in `dense_rnn_net`. ---
        windows = []
        single = volume[:, :, :, :, 0:1]
        first = torch.cat([single, volume[:, :, :, :, 0:2]], dim=4)  # 3 slices: [0,0,1]
        windows.append(first)
        for i in range(d - 2):
            windows.append(volume[:, :, :, :, i : i + 3])
        final1 = volume[:, :, :, :, d - 2 : d]  # 2 slices: [D-2, D-1]
        final2 = volume[:, :, :, :, d - 1 : d]  # 1 slice: [D-1]
        final = torch.cat([final1, final2], dim=4)  # 3 slices: [D-2, D-1, D-1]
        windows.append(final)

        # each window: (B, 1, H, W, 3) -> (B, 3, H, W); stack along batch dim.
        input2d = torch.cat(
            [win.squeeze(1).permute(0, 3, 1, 2) for win in windows], dim=0
        )  # input_cols windows total, matching the original's per-slice loop count

        ac_up4_2d, classifer2d = self.denseunet2d(input2d)  # both: (n_windows*B, C, H, W)

        # --- restack the per-window 2D outputs back into pseudo-3D volumes,
        # matching `slice2d` + per-slice concat along the depth axis. ---
        n_windows = classifer2d.shape[0] // b
        res2d = classifer2d.view(n_windows, b, *classifer2d.shape[1:]).permute(
            1, 2, 3, 4, 0
        )  # (B,3,H,W,D)
        fea2d = ac_up4_2d.view(n_windows, b, *ac_up4_2d.shape[1:]).permute(
            1, 2, 3, 4, 0
        )  # (B,64,H,W,D)

        # --- 3D branch input: [original volume ; 250x-scaled 2D response],
        # matching `Lambda(lambda x: x * 250)` + concat along channel axis. ---
        res2d_input = res2d * 250.0
        input3d = torch.cat([volume, res2d_input], dim=1)  # (B, 1+3, H, W, D)

        fea3d, classifer3d = self.densenet3d(input3d)

        final = fea3d + fea2d
        final_conv = self.final_conv(final)
        final_conv = self.dropout(final_conv)
        final_ac = F.relu(self.final_bn(final_conv))
        classifer = self.classifier(final_ac)
        return classifer


# ---------------------------------------------------------------------------
# Staging build/example helpers (tiny spatial size + depth, scaled down from the
# repo's own 224x224xN training volumes for fast tracing).
# ---------------------------------------------------------------------------


def build_hdenseunet():
    return HDenseUNet(input_cols=4).eval()


def example_input_hdenseunet():
    batch = 1
    volume = torch.rand(batch, 1, 128, 128, 4)
    return (volume,)


MENAGERIE_ENTRIES = [
    ("H-DenseUNet", build_hdenseunet, example_input_hdenseunet, 2018, "ported-pytorch"),
]

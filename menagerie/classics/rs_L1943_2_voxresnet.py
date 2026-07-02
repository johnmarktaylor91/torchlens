# FAITHFUL PORT of Ryo-Ito/brain_segmentation @ master (original framework: Chainer)
# https://raw.githubusercontent.com/Ryo-Ito/brain_segmentation/master/model.py
#
# "VoxResNet: Deep Voxelwise Residual Networks for Brain Segmentation from 3D MR Images"
# (Chen et al., NeuroImage 2018). No maintained PyTorch source exists (the other public
# mirror, txin96/VoxResNet, is TensorLayer/TF1, not PyTorch either); this transcribes
# Ryo-Ito/brain_segmentation's Chainer `VoxResModule`/`VoxResNet` layer-for-layer into
# torch: same conv1a/bnorm1a/conv1b -> c1 side branch, bnorm1b/conv1c -> voxres2/voxres3 ->
# c2 side branch, bnorm3/conv4 -> voxres5/voxres6 -> c3 side branch, bnorm6/conv7 ->
# voxres8/voxres9 -> c4 side branch, then c1+c2+c3+c4 fused logits (VRN9, the paper's
# 9-conv-layer-deep architecture with 4 auxiliary classifiers), including the deliberately
# asymmetric side-branch upsampling kernel/stride/pad choices baked into the original
# DeconvolutionND calls (c1: k3 s1 p1 i.e. no spatial change; c2: k4 s2 p1; c3: k6 s4 p1;
# c4: k10 s8 p1) which are preserved via ConvTranspose3d with matching kernel/stride/
# padding. Chainer's `ConvolutionND(ndim=3, ...)` -> `nn.Conv3d`, `DeconvolutionND(ndim=3,
# ...)` -> `nn.ConvTranspose3d`, `L.BatchNormalization` -> `nn.BatchNorm3d`,
# `F.clipped_relu` (Chainer's relu6-family "clip to [0, z]", default z=20) -> a `ClippedReLU`
# module wrapping `torch.clamp(F.relu(x), max=20.0)`, and the final `F.softmax` (over the
# channel axis) -> `torch.softmax(..., dim=1)`. The `train=True` list-of-logits return path
# (used only for the repo's deep-supervision training loss) and the He-normal weight-init
# scale (`HeNormal(scale=0.01)`, a training-stability detail, not an architectural change)
# are omitted since this build is inference/tracing-only with default torch init.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class ClippedReLU(nn.Module):
    """Chainer's F.clipped_relu(x, z=20.0): relu(x) clamped to an upper bound."""

    def __init__(self, z: float = 20.0):
        super().__init__()
        self.z = z

    def forward(self, x):
        return torch.clamp(F.relu(x), max=self.z)


class VoxResModule(nn.Module):
    """
    Voxel Residual Module
    input
    BatchNormalization, ReLU
    Conv 64, 3x3x3
    BatchNormalization, ReLU
    Conv 64, 3x3x3
    output
    """

    def __init__(self):
        super().__init__()
        self.bnorm1 = nn.BatchNorm3d(64)
        self.conv1 = nn.Conv3d(64, 64, 3, padding=1)
        self.bnorm2 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 64, 3, padding=1)

    def forward(self, x):
        h = F.relu(self.bnorm1(x))
        h = self.conv1(h)
        h = F.relu(self.bnorm2(h))
        h = self.conv2(h)
        return h + x


class VoxResNet(nn.Module):
    """Voxel Residual Network"""

    def __init__(self, in_channels=1, n_classes=4):
        super().__init__()

        self.conv1a = nn.Conv3d(in_channels, 32, 3, padding=1)
        self.bnorm1a = nn.BatchNorm3d(32)
        self.conv1b = nn.Conv3d(32, 32, 3, padding=1)
        self.bnorm1b = nn.BatchNorm3d(32)
        self.conv1c = nn.Conv3d(32, 64, 3, stride=2, padding=1)
        self.voxres2 = VoxResModule()
        self.voxres3 = VoxResModule()
        self.bnorm3 = nn.BatchNorm3d(64)
        self.conv4 = nn.Conv3d(64, 64, 3, stride=2, padding=1)
        self.voxres5 = VoxResModule()
        self.voxres6 = VoxResModule()
        self.bnorm6 = nn.BatchNorm3d(64)
        self.conv7 = nn.Conv3d(64, 64, 3, stride=2, padding=1)
        self.voxres8 = VoxResModule()
        self.voxres9 = VoxResModule()
        self.clipped_relu = ClippedReLU()

        self.c1deconv = nn.ConvTranspose3d(32, 32, 3, stride=1, padding=1)
        self.c1conv = nn.Conv3d(32, n_classes, 3, padding=1)
        self.c2deconv = nn.ConvTranspose3d(64, 64, 4, stride=2, padding=1)
        self.c2conv = nn.Conv3d(64, n_classes, 3, padding=1)
        self.c3deconv = nn.ConvTranspose3d(64, 64, 6, stride=4, padding=1)
        self.c3conv = nn.Conv3d(64, n_classes, 3, padding=1)
        self.c4deconv = nn.ConvTranspose3d(64, 64, 10, stride=8, padding=1)
        self.c4conv = nn.Conv3d(64, n_classes, 3, padding=1)

    def forward(self, x, train=False):
        """
        calculate output of VoxResNet given input x

        Parameters
        ----------
        x : (batch_size, in_channels, xlen, ylen, zlen) tensor
            image to perform semantic segmentation

        Returns
        -------
        proba: (batch_size, n_classes, xlen, ylen, zlen) tensor
            probability of each voxel belonging each class
            elif train=True, returns list of logits
        """
        h = self.conv1a(x)
        h = F.relu(self.bnorm1a(h))
        h = self.conv1b(h)
        c1 = self.clipped_relu(self.c1deconv(h))
        c1 = self.c1conv(c1)

        h = F.relu(self.bnorm1b(h))
        h = self.conv1c(h)
        h = self.voxres2(h)
        h = self.voxres3(h)
        c2 = self.clipped_relu(self.c2deconv(h))
        c2 = self.c2conv(c2)

        h = F.relu(self.bnorm3(h))
        h = self.conv4(h)
        h = self.voxres5(h)
        h = self.voxres6(h)
        c3 = self.clipped_relu(self.c3deconv(h))
        c3 = self.c3conv(c3)

        h = F.relu(self.bnorm6(h))
        h = self.conv7(h)
        h = self.voxres8(h)
        h = self.voxres9(h)
        c4 = self.clipped_relu(self.c4deconv(h))
        c4 = self.c4conv(c4)

        # crop/pad each side-branch output to a common spatial size before summing
        # (the asymmetric deconv kernels above can produce off-by-a-few-voxel shapes;
        # the original Chainer DeconvolutionND output_shape args pin this to match x
        # exactly, which torch's ConvTranspose3d does not take as a direct argument)
        target_shape = c1.shape[-3:]
        c2 = _match_shape(c2, target_shape)
        c3 = _match_shape(c3, target_shape)
        c4 = _match_shape(c4, target_shape)

        c = c1 + c2 + c3 + c4

        if train:
            return [c1, c2, c3, c4, c]
        else:
            return torch.softmax(c, dim=1)


def _match_shape(t, target_shape):
    """Center-crop or zero-pad the trailing 3 spatial dims of t to target_shape."""
    diffs = [t.shape[-3 + i] - target_shape[i] for i in range(3)]
    if any(d < 0 for d in diffs):
        pad = []
        for d in reversed(diffs):
            lo = max(-d, 0) // 2
            hi = max(-d, 0) - lo
            pad.extend([lo, hi])
        t = F.pad(t, pad)
        diffs = [t.shape[-3 + i] - target_shape[i] for i in range(3)]
    if any(d > 0 for d in diffs):
        starts = [d // 2 for d in diffs]
        t = t[
            ...,
            starts[0] : starts[0] + target_shape[0],
            starts[1] : starts[1] + target_shape[1],
            starts[2] : starts[2] + target_shape[2],
        ]
    return t


def build_voxresnet():
    return VoxResNet(in_channels=1, n_classes=4)


def example_input_voxresnet():
    return torch.randn(1, 1, 32, 32, 32)


MENAGERIE_ENTRIES = [
    ("VoxResNet", "build_voxresnet", "example_input_voxresnet", 2018, "ported"),
]

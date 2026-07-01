# FAITHFUL PORT of bm2-lab/DeepCRISPR @ master (original framework: TensorFlow 1.x + DeepMind Sonnet + tf.contrib.slim)
#
# Source file transcribed: deepcrispr/deepcrispr_src.py (functions
# `build_ontar_model` / `DCModelOntar`), the on-target sgRNA-efficiency
# classifier from DeepCRISPR (Chuai et al., "DeepCRISPR: optimized CRISPR
# guide RNA design by deep learning", Genome Biology 2018).
#
# The original graph is built with `tensorflow.contrib.slim` and DeepMind's
# `sonnet` library, both long removed from any TF version installable
# alongside a modern torch stack (tf.contrib was deleted in TF2, sonnet 1.x
# targeted TF1 graph-mode) -- neither is in the base-lib allowlist for this
# repo, so this is a faithful architectural transcription into self-contained
# torch of `build_ontar_model`, preserving every layer/mechanism of the
# original graph:
#
#   A 5-layer convolutional "denoising-autoencoder-style" encoder (channel
#   sizes [32, 64, 64, 256, 256], all 1x3 kernels operating over the sequence
#   axis of a (1, 23, C) "image" -- one-hot + epigenetic-feature channels),
#   with stride-2 downsampling on layers e2 and e4, BatchNorm on every layer,
#   and a per-channel learned additive bias ("beta") applied *before* the ReLU
#   on layers e1..e5 (the "denoising feature learning" module DeepCRISPR
#   pretrains as an autoencoder and then reuses as an on-target/off-target
#   feature extractor) -- followed by a 4-layer classifier head
#   (channel sizes [512, 512, 1024, 2], with BatchNorm + ReLU on the first
#   three and stride-2 downsampling on e6, VALID padding + kernel (1,3) on e8
#   collapsing the sequence axis, then a final 1x1 conv to 2 logits) and a
#   softmax, taking the positive-class probability as the on-target efficacy
#   score (`sig_l = softmax(l_last)[..., 1]`).
#
# Input: (batch, 1, 23, 8) -- DeepCRISPR's on-target "sg" tensor is a 23bp
# sgRNA+PAM window one-hot encoded over 4 bases x 2 strands (8 channels),
# matching `DCModelOntar.__init__` (`inputs_sg` placeholder shape
# `[None, 1, 23, 8]` for the full `seq_feature_only=False` configuration).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderLayer(nn.Module):
    """One `snt.Conv2D` + `snt.BatchNorm` + additive-beta + ReLU stage.

    Faithful port of the per-layer body of the loop in
    `build_ontar_model` / `build_ontar_reg_model`:
        pre_u = encoder[i](hu_pre)
        u = encoder_bn_u[i](pre_u, ...)
        hu = relu(u + betas[i])
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        # kernel_shape=[1, 3] in the original operates over a (1, 23, C) map;
        # we keep the sequence axis as the Conv2d "width" dim with a (1, 3)
        # kernel and 'same'-equivalent padding (1,) on the width axis only,
        # matching Sonnet's default SAME padding for these layers.
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1))
        self.bn = nn.BatchNorm2d(out_ch)
        self.beta = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.bn(self.conv(x))
        return F.relu(u + self.beta.view(1, -1, 1, 1))


class _ClassifierLayer(nn.Module):
    """One classifier-head stage: `snt.Conv2D` + `snt.BatchNorm` + ReLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        padding: tuple[int, int] = (0, 1),
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=(1, 3), stride=(1, stride), padding=padding
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class DeepCRISPROnTarget(nn.Module):
    """DeepCRISPR on-target sgRNA-efficacy classifier.

    Faithful port of `build_ontar_model` (bm2-lab/DeepCRISPR).
    """

    def __init__(self, in_channels: int = 8):
        super().__init__()
        channel_size = [8, 32, 64, 64, 256, 256]  # channel_size in the original
        channel_size[0] = in_channels

        self.e1 = _EncoderLayer(channel_size[0], channel_size[1], stride=1)
        self.e2 = _EncoderLayer(channel_size[1], channel_size[2], stride=2)
        self.e3 = _EncoderLayer(channel_size[2], channel_size[3], stride=1)
        self.e4 = _EncoderLayer(channel_size[3], channel_size[4], stride=2)
        self.e5 = _EncoderLayer(channel_size[4], channel_size[5], stride=1)

        cls_channel_size = [512, 512, 1024, 2]
        self.e6 = _ClassifierLayer(channel_size[5], cls_channel_size[0], stride=2, padding=(0, 1))
        self.e7 = _ClassifierLayer(
            cls_channel_size[0], cls_channel_size[1], stride=1, padding=(0, 1)
        )
        # e8 uses VALID padding + kernel (1,3): collapses the sequence axis.
        self.e8 = _ClassifierLayer(
            cls_channel_size[1], cls_channel_size[2], stride=1, padding=(0, 0)
        )
        self.bn8 = self.e8.bn  # exposed for clarity; unused directly
        # e9: final 1x1 conv to 2 logits (no BatchNorm/ReLU in the original).
        self.e9 = nn.Conv2d(cls_channel_size[2], cls_channel_size[3], kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, 1, 23) -- channels-first NCHW, sequence axis
        # is the width (last) dimension, matching the original's (1, 23, C)
        # NHWC "image" transposed to torch's NCHW convention.
        hu = self.e1(x)
        hu = self.e2(hu)
        hu = self.e3(hu)
        hu = self.e4(hu)
        hu_last = self.e5(hu)

        hl = self.e6(hu_last)
        hl = self.e7(hl)
        hl = self.e8(hl)
        l_last = self.e9(hl)  # (batch, 2, 1, W')

        hl_last = F.softmax(l_last, dim=1)
        sig_l = hl_last[:, 1].squeeze(-1).squeeze(-1)  # positive-class probability
        return sig_l


def build_deepcrispr() -> DeepCRISPROnTarget:
    return DeepCRISPROnTarget(in_channels=8)


def example_input_deepcrispr() -> torch.Tensor:
    torch.manual_seed(0)
    batch = 2
    # (batch, channels=8, height=1, width=23) matching DCModelOntar's
    # inputs_sg placeholder [None, 1, 23, 8] after the NHWC->NCHW transpose
    # `ontar_predict` performs before feeding the graph.
    return torch.rand(batch, 8, 1, 23)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepCRISPR-OnTarget", "build_deepcrispr", "example_input_deepcrispr", 2018, "ported-pytorch"),
]

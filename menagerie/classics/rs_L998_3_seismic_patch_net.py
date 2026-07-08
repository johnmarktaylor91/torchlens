# FAITHFUL PORT of gzoutlook/SeismicPatchNet_v1 @ master (original framework: TensorFlow
# v2.3.x / Keras, from SeismicPatchNet_TF2.py::Build_SPN_TF2)
#
# SeismicPatchNet: a multi-scale-filter CNN for seismic-patch classification, whose topology
# was discovered by a random/NAS architecture search over "fusion" (perception) blocks. The
# original TF1-era `libs/CNN_Models/SeismicPatchNet.py` builds the searched topology with
# stochastic per-run channel splits (`np.random.dirichlet`/`np.random.randint` sampling
# inside `perception_layer_topologyV1`), so it has no single deterministic architecture on
# its own. `SeismicPatchNet_TF2.py::Build_SPN_TF2`, however, hard-codes the exact channel
# counts and kernel sizes the NAS search converged on as its `archi_dict` default (the
# concrete, fixed architecture the paper reports and ships) and rebuilds the identical
# topology with plain `tf.keras.layers.Conv2D`/`MaxPool2D`/`Dense` -- deterministic, no
# runtime randomness. This port transcribes that exact fixed-topology Keras graph into torch:
# 3 stem conv+pool layers, 3 "fusion" (perception) blocks (each: a MxM conv branch, two
# 1x1->CxC->1xC "asymmetric" conv branches, and max/min-pool branches concatenated along the
# channel axis) with a max-pool between each pair, a global average pool, and a final dense
# classifier head. Fixed strides/paddings/kernel-sizes/channel-splits are copied 1:1 from the
# real `archi_dict` defaults; only the activation semantics that TF2 spells with a Lambda-style
# negation (`-MaxPool2D(-x)`, i.e. a "min-pool" implemented as negated max-pool) are kept
# literally rather than re-derived. Keras `padding='same'` at stride=1 keeps the spatial size
# fixed for ANY kernel size (odd or even) by padding `k-1` total, split as `(k-1)//2` before /
# `k - 1 - (k-1)//2` after -- symmetric for odd k, asymmetric (short on the left/top) for even
# k. `same_pad_stride1` below reproduces that exactly via explicit `F.pad` before each
# `padding=0` conv/pool, so the even `convMM_size=2` branches used throughout `archi_dict`
# match Keras's SAME output size bit-for-bit rather than relying on torch's symmetric
# `padding=k//2` (which is only equivalent to SAME for odd kernels).
#
# Geng, Z., Wang, Y. "Automated design of a convolutional neural network with multi-scale
# filters for cost-efficient seismic data classification." Nat Commun 11, 3311 (2020).
# https://doi.org/10.1038/s41467-020-17123-6

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


def _same_pad_1d(in_size, kernel, stride):
    out_size = -(-in_size // stride)  # ceil division, Keras SAME output size
    total = max((out_size - 1) * stride + kernel - in_size, 0)
    before = total // 2
    after = total - before
    return before, after


def same_pad(x, kh, kw, stride=1):
    """Reproduce Keras `padding='same'` spatial padding for any kernel size/stride."""
    h, w = x.shape[-2], x.shape[-1]
    top, bottom = _same_pad_1d(h, kh, stride)
    left, right = _same_pad_1d(w, kw, stride)
    return F.pad(x, [left, right, top, bottom])


def same_pad_stride1(x, kh, kw):
    """Reproduce Keras `padding='same'`, `strides=1` spatial padding for any kernel size."""
    return same_pad(x, kh, kw, stride=1)


class FusionLayer(nn.Module):
    """Port of `fusion_layer_template_tf2` (perception/"fusion" block).

    Branches, concatenated along the channel axis (TF `axis=-1` -> torch `dim=1`):
      - conv_MxM: a single convMM_size x convMM_size conv ("MxM" branch)
      - conv_C1:  1x1 squeeze -> Cx1 -> 1xC ("asymmetric" branch 1)
      - conv_C2:  1x1 squeeze -> Cx1 -> 1xC ("asymmetric" branch 2)
      - pool1: max-pool branch (1x1 conv on max-pool) + min-pool branch (1x1 conv on
        negated max-pool of negated input, i.e. an actual min-pool)
      - pool2: same pattern with a second pool kernel size
    """

    def __init__(
        self,
        in_channels,
        convMM_size,
        convMM_num,
        num_of_c11_1,
        conv1_size,
        conv1_num,
        num_of_c11_2,
        conv2_size,
        conv2_num,
        pool1_size,
        num_of_pool1c1_max,
        num_of_pool1c1_min,
        pool2_size,
        num_of_pool2c1_max,
        num_of_pool2c1_min,
    ):
        super().__init__()
        self.convMM_size = convMM_size
        self.conv1_size = conv1_size
        self.conv2_size = conv2_size
        self.pool1_size = pool1_size
        self.pool2_size = pool2_size

        self.convMM = nn.Conv2d(in_channels, convMM_num, convMM_size, padding=0)

        self.conv1_11 = nn.Conv2d(in_channels, num_of_c11_1, 1)
        self.conv1_a = nn.Conv2d(num_of_c11_1, conv1_num, (conv1_size, 1), padding=0)
        self.conv1_b = nn.Conv2d(conv1_num, conv1_num, (1, conv1_size), padding=0)

        self.conv2_11 = nn.Conv2d(in_channels, num_of_c11_2, 1)
        self.conv2_a = nn.Conv2d(num_of_c11_2, conv2_num, (conv2_size, 1), padding=0)
        self.conv2_b = nn.Conv2d(conv2_num, conv2_num, (1, conv2_size), padding=0)

        self.pool1_max = nn.MaxPool2d(pool1_size, stride=1, padding=0)
        self.pool1_max_c11 = nn.Conv2d(in_channels, num_of_pool1c1_max, 1)
        self.pool1_min_c11 = nn.Conv2d(in_channels, num_of_pool1c1_min, 1)

        self.pool2_max = nn.MaxPool2d(pool2_size, stride=1, padding=0)
        self.pool2_max_c11 = nn.Conv2d(in_channels, num_of_pool2c1_max, 1)
        self.pool2_min_c11 = nn.Conv2d(in_channels, num_of_pool2c1_min, 1)

    def forward(self, x):
        convMM = self.convMM(same_pad_stride1(x, self.convMM_size, self.convMM_size))

        conv1 = self.conv1_11(x)
        conv1 = self.conv1_a(same_pad_stride1(conv1, self.conv1_size, 1))
        conv1 = self.conv1_b(same_pad_stride1(conv1, 1, self.conv1_size))

        conv2 = self.conv2_11(x)
        conv2 = self.conv2_a(same_pad_stride1(conv2, self.conv2_size, 1))
        conv2 = self.conv2_b(same_pad_stride1(conv2, 1, self.conv2_size))

        pool1_max = self.pool1_max(same_pad_stride1(x, self.pool1_size, self.pool1_size))
        pool1_max = self.pool1_max_c11(pool1_max)
        pool1_min = -self.pool1_max(same_pad_stride1(-x, self.pool1_size, self.pool1_size))
        pool1_min = self.pool1_min_c11(pool1_min)

        pool2_max = self.pool2_max(same_pad_stride1(x, self.pool2_size, self.pool2_size))
        pool2_max = self.pool2_max_c11(pool2_max)
        pool2_min = -self.pool2_max(same_pad_stride1(-x, self.pool2_size, self.pool2_size))
        pool2_min = self.pool2_min_c11(pool2_min)

        return torch.cat([convMM, conv1, conv2, pool1_max, pool1_min, pool2_max, pool2_min], dim=1)

    @property
    def out_channels(self):
        return (
            self.convMM.out_channels
            + self.conv1_b.out_channels
            + self.conv2_b.out_channels
            + self.pool1_max_c11.out_channels
            + self.pool1_min_c11.out_channels
            + self.pool2_max_c11.out_channels
            + self.pool2_min_c11.out_channels
        )


# The exact channel/kernel spec the NAS search converged on, as hard-coded in
# `Build_SPN_TF2`'s `archi_dict` default (verbatim values from the real repo file).
DEFAULT_ARCHI = {
    "input shape": [1, 112, 112],
    "conv0 output size": 137,
    "conv1 output size": 137,
    "conv2 output size": 144,
    "perception_3 output size": 133,
    "perception_3/conv_MxM": [2, 83],
    "perception_3/conv_C1": [104, 3, 2],
    "perception_3/conv_C2": [112, 7, 16],
    "perception_3/pool1": [2, 12, 2, 11],
    "perception_3/pool2": [3, 4, 3, 5],
    "perception_4 output size": 151,
    "perception_4/conv_MxM": [2, 75],
    "perception_4/conv_C1": [280, 5, 5],
    "perception_4/conv_C2": [382, 7, 7],
    "perception_4/pool1": [2, 20, 2, 21],
    "perception_4/pool2": [5, 12, 5, 11],
    "perception_5 output size": 459,
    "perception_5/conv_MxM": [2, 14],
    "perception_5/conv_C1": [46, 3, 25],
    "perception_5/conv_C2": [170, 5, 116],
    "perception_5/pool1": [4, 107, 4, 107],
    "perception_5/pool2": [5, 45, 5, 45],
}


class SeismicPatchNet(nn.Module):
    """Port of `Build_SPN_TF2` (deterministic default `archi_dict` topology)."""

    def __init__(self, archi=None, num_of_class=2, dropout_rate=0.3):
        super().__init__()
        archi = archi or DEFAULT_ARCHI
        in_ch = archi["input shape"][0]

        self.conv0 = nn.Conv2d(in_ch, archi["conv0 output size"], 7, stride=2, padding=0)
        self.act0 = nn.ELU()
        self.pool0 = nn.MaxPool2d(3, stride=2, padding=0)

        self.conv1 = nn.Conv2d(
            archi["conv0 output size"], archi["conv1 output size"], 1, stride=2, padding=0
        )
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv2d(
            archi["conv1 output size"], archi["conv2 output size"], 3, stride=1, padding=0
        )
        self.act2 = nn.ELU()
        self.pooling_0 = nn.MaxPool2d(3, stride=2, padding=0)

        c3 = archi["perception_3/conv_MxM"]
        c1 = archi["perception_3/conv_C1"]
        c2 = archi["perception_3/conv_C2"]
        p1 = archi["perception_3/pool1"]
        p2 = archi["perception_3/pool2"]
        self.fusion_0 = FusionLayer(
            archi["conv2 output size"],
            c3[0],
            c3[1],
            c1[0],
            c1[1],
            c1[2],
            c2[0],
            c2[1],
            c2[2],
            p1[0],
            p1[1],
            p1[3],
            p2[0],
            p2[1],
            p2[3],
        )

        c3 = archi["perception_4/conv_MxM"]
        c1 = archi["perception_4/conv_C1"]
        c2 = archi["perception_4/conv_C2"]
        p1 = archi["perception_4/pool1"]
        p2 = archi["perception_4/pool2"]
        self.fusion_1 = FusionLayer(
            self.fusion_0.out_channels,
            c3[0],
            c3[1],
            c1[0],
            c1[1],
            c1[2],
            c2[0],
            c2[1],
            c2[2],
            p1[0],
            p1[1],
            p1[3],
            p2[0],
            p2[1],
            p2[3],
        )

        c3 = archi["perception_5/conv_MxM"]
        c1 = archi["perception_5/conv_C1"]
        c2 = archi["perception_5/conv_C2"]
        p1 = archi["perception_5/pool1"]
        p2 = archi["perception_5/pool2"]
        self.fusion_2 = FusionLayer(
            self.fusion_1.out_channels,
            c3[0],
            c3[1],
            c1[0],
            c1[1],
            c1[2],
            c2[0],
            c2[1],
            c2[2],
            p1[0],
            p1[1],
            p1[3],
            p2[0],
            p2[1],
            p2[3],
        )

        self.pooling_1 = nn.AvgPool2d(7, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.LazyLinear(num_of_class)
        self.output_act = nn.ELU()

    def forward(self, x):
        x = self.act0(self.conv0(same_pad(x, 7, 7, stride=2)))
        x = self.pool0(same_pad(x, 3, 3, stride=2))
        x = self.act1(self.conv1(same_pad(x, 1, 1, stride=2)))
        x = self.act2(self.conv2(same_pad(x, 3, 3, stride=1)))
        x = self.pooling_0(same_pad(x, 3, 3, stride=2))
        x = self.fusion_0(x)
        x = self.fusion_1(x)
        x = self.fusion_2(x)
        x = self.pooling_1(x)  # Keras padding='valid' -- no SAME padding here
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.output_act(self.output(x))
        return x


def build_seismic_patch_net():
    model = SeismicPatchNet(num_of_class=2, dropout_rate=0.3)
    model.eval()
    with torch.no_grad():
        model(example_input_seismic_patch_net())  # materialize LazyLinear
    return model


def example_input_seismic_patch_net():
    b, c, h, w = 1, 1, 112, 112
    return torch.randn(b, c, h, w)


MENAGERIE_ENTRIES = [
    (
        "SeismicPatchNet",
        build_seismic_patch_net,
        example_input_seismic_patch_net,
        2020,
        "ported-pytorch",
    ),
]

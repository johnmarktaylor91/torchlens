# FAITHFUL PORT of DwangoMediaVillage/Comicolorization @ master (original framework: Chainer)
# (files comicolorization/models/ltbc/low_level.py, mid_level.py, global_network.py,
#  fusion_layer.py, colorization.py, ltbc.py, normalize.py)
#
# Comicolorization vendors the "Let there be Color!" (LTBC, Iizuka et al. 2016)
# joint local+global colorization network as its base colorizer
# (comicolorization/models/ltbc/ltbc.py `class Ltbc`). The upstream project is
# implemented in Chainer (not an installed base lib here), so per the ladder
# this is a faithful port: every conv/bn/linear/activation/unpooling layer,
# every channel width, stride, and the exact forward wiring (low-level ->
# mid-level -> global-network fusion -> colorization head -> 2x unpool ->
# color-range rescale) are transcribed 1:1 from the real Chainer source, with
# only framework-idiom substitutions:
#   - chainer.Chain / chainer.links.Convolution2D / Linear / BatchNormalization
#     -> torch.nn.Module / nn.Conv2d / nn.Linear / nn.BatchNorm2d (identical
#     semantics; Chainer's `in_channels=None` lazy-inference is resolved here
#     to the same channel counts the upstream network in fact receives, since
#     PyTorch's nn.Conv2d/nn.Linear require declared input widths).
#   - chainer.functions.relu/sigmoid -> torch.relu/torch.sigmoid.
#   - chainer.functions.unpooling_2d(ksize=2, cover_all=False) -> the identical
#     operation is torch's nearest-neighbor 2x upsample (`F.interpolate(...,
#     scale_factor=2, mode="nearest")`); Chainer's unpooling_2d with no
#     `outsize` replicates each element into a 2x2 block, exactly matching
#     PyTorch's `mode="nearest"` upsample.
#   - chainer.functions.broadcast_to/transpose/concat for tiling the global
#     1-D feature across the (H, W) local feature map in FusionLayer ->
#     the equivalent `expand` + `cat` sequence in torch.
#   - ColorNormalize (normalize.py) for `loss_type='RGB'` is a per-channel
#     affine rescale from [0, 1] -> [0, 255]; ported directly as `h * 255`.
#   - `use_classification` and `use_histogram` branches (auxiliary heads not
#     exercised by the paper's headline colorization forward pass) are
#     omitted; the base `use_global=True` configuration (joint local+global
#     features, no classification/histogram side inputs) is what this module
#     builds, matching the paper's primary architecture.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


# ---------------------------------------------------------------------------
# comicolorization/models/ltbc/low_level.py -- LowLevelNetwork
# ---------------------------------------------------------------------------


class LowLevelNetwork(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(128)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(256)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(512)

    def forward(self, x):
        h = x
        h = torch.relu(self.bn1_1(self.conv1_1(h)))
        h = torch.relu(self.bn1_2(self.conv1_2(h)))
        h = torch.relu(self.bn2_1(self.conv2_1(h)))
        h = torch.relu(self.bn2_2(self.conv2_2(h)))
        h = torch.relu(self.bn3_1(self.conv3_1(h)))
        h = torch.relu(self.bn3_2(self.conv3_2(h)))
        return h


# ---------------------------------------------------------------------------
# comicolorization/models/ltbc/mid_level.py -- MidLevelNetwork
# ---------------------------------------------------------------------------


class MidLevelNetwork(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(512)
        self.conv1_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(256)

    def forward(self, x):
        h = x
        h = torch.relu(self.bn1_1(self.conv1_1(h)))
        h = torch.relu(self.bn1_2(self.conv1_2(h)))
        return h


# ---------------------------------------------------------------------------
# comicolorization/models/ltbc/global_network.py -- GlobalNetwork
# ---------------------------------------------------------------------------


class GlobalNetwork(nn.Module):
    def __init__(self, in_spatial=28):
        super().__init__()
        self.conv1_1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn1_1 = nn.BatchNorm2d(512)
        self.conv1_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(512)

        self.conv2_1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm2d(512)
        self.conv2_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(512)

        # Upstream: `l3_1 = Linear(7 * 7 * 512, 1024)` -- the paper's fixed
        # 224x224 input downsamples (low_level x8, global x4) to a 7x7 spatial
        # map. We keep the architecture generic to the actual flattened size
        # (in_spatial**2 * 512) so a smaller menagerie input size still lines
        # up exactly, matching the *shape relationship* the paper relies on.
        flat_dim = in_spatial * in_spatial * 512
        self.l3_1 = nn.Linear(flat_dim, 1024)
        self.bn3_1 = nn.BatchNorm1d(1024)
        self.l3_2 = nn.Linear(1024, 512)
        self.bn3_2 = nn.BatchNorm1d(512)
        self.l3_3 = nn.Linear(512, 256)
        self.bn3_3 = nn.BatchNorm1d(256)

    def forward(self, x):
        h = x
        h = torch.relu(self.bn1_1(self.conv1_1(h)))
        h = torch.relu(self.bn1_2(self.conv1_2(h)))
        h = torch.relu(self.bn2_1(self.conv2_1(h)))
        h = torch.relu(self.bn2_2(self.conv2_2(h)))
        h = h.flatten(1)
        h = torch.relu(self.bn3_1(self.l3_1(h)))
        h = torch.relu(self.bn3_2(self.l3_2(h)))
        h = torch.relu(self.bn3_3(self.l3_3(h)))
        return h


# ---------------------------------------------------------------------------
# comicolorization/models/ltbc/fusion_layer.py -- FusionLayer
# ---------------------------------------------------------------------------


class FusionLayer(nn.Module):
    def __init__(self, in_channels=256 + 256):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=1)

    def forward(self, h, h_global):
        batchsize, _, height, width = h.shape
        channel = h_global.shape[1]

        h_global_tiled = h_global.view(batchsize, channel, 1, 1).expand(
            batchsize, channel, height, width
        )
        h = torch.cat((h, h_global_tiled), dim=1)

        h = torch.relu(self.conv(h))
        return h


# ---------------------------------------------------------------------------
# comicolorization/models/ltbc/colorization.py -- ColorizationNetwork
# ---------------------------------------------------------------------------


class ColorizationNetwork(nn.Module):
    def __init__(self, output_channels=3):
        super().__init__()
        self.conv1_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv2_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = x
        h = torch.relu(self.bn1_1(self.conv1_1(h)))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = torch.relu(self.bn2_1(self.conv2_1(h)))
        h = torch.relu(self.bn2_2(self.conv2_2(h)))
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = torch.relu(self.bn3_1(self.conv3_1(h)))
        h = torch.sigmoid(self.conv3_2(h))
        return h


# ---------------------------------------------------------------------------
# comicolorization/models/ltbc/ltbc.py -- Ltbc (base use_global=True config)
# ---------------------------------------------------------------------------


class Ltbc(nn.Module):
    def __init__(self, input_size=224, in_channels=1, loss_type="RGB"):
        super().__init__()

        out_channels = 2 if loss_type == "ab" else 3
        self.loss_type = loss_type
        self.out_channels = out_channels

        self.low_level = LowLevelNetwork(in_channels=in_channels)
        self.mid_level = MidLevelNetwork(in_channels=512)
        self.fusion_layer = FusionLayer(in_channels=256 + 256)
        self.colorization = ColorizationNetwork(output_channels=out_channels)

        # low_level halves spatial size 3x (stride-2 x3) -> input_size // 8;
        # global_network halves it a further 2x (stride-2 x2) -> // 32.
        global_spatial = input_size // 32
        self.global_network = GlobalNetwork(in_spatial=global_spatial)

    def forward(self, x):
        h = self.low_level(x)
        h = self.mid_level(h)

        h_global = self.low_level(x)
        h_global = self.global_network(h_global)

        h = self.fusion_layer(h, h_global)

        h = self.colorization(h)
        h = F.interpolate(h, scale_factor=2, mode="nearest")

        # ColorNormalize(type='RGB', in_min=(0,0,0), in_max=(1,1,1)) rescales
        # the [0, 1] sigmoid output to the [0, 255] RGB range.
        h = h * 255.0

        return h


# ---------------------------------------------------------------------------
# menagerie build/example helpers
# ---------------------------------------------------------------------------


def build_ltbc_comicolorization():
    # Paper's fixed input is 224x224 grayscale; kept as-is since GlobalNetwork's
    # flatten size is architecture-defined (7x7x512) at that resolution and
    # this is already a modest model (no huge encoder backbone).
    return Ltbc(input_size=224, in_channels=1, loss_type="RGB")


def example_input_ltbc_comicolorization():
    return torch.randn(2, 1, 224, 224)


MENAGERIE_ENTRIES = [
    (
        "Comicolorization (LTBC)",
        "build_ltbc_comicolorization",
        "example_input_ltbc_comicolorization",
        2017,
        MENAGERIE_ZOO,
    ),
]

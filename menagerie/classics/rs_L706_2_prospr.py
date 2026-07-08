# SOURCE: vendored from dellacortelab/prospr @ c081249 (master)
# File: prospr/nn.py
# ProSPr / ProFOLD: open-source AlphaFold1-style dilated-residual-convolution network that
# predicts a distogram plus auxiliary secondary-structure / torsion-angle / ASA maps from
# an input feature tensor (MSA co-evolution + sequence features). queue.tsv rows for
# "ProFOLD" (dellacortelab/prospr, POTENTIAL_DEDUP) and "ProSPr" both resolve to this same
# repo/paper -- one vendored module, two catalog aliases (ProSPr is the paper's official
# name; ProFOLD is the informal AF1-clone label used in some literature). Vendored verbatim
# aside from stripping the CUDA device probe / state-dict loader (training/loading utility,
# not architecture).
import torch
import torch.nn as nn

INPUT_DIM = 547
DIST_BINS = 10
AUX_BINS = 94
SS_BINS = 9
ANGLE_BINS = 37
ASA_BINS = 11
DROPOUT_RATE = 0.15


def conv3x3(in_channels, out_channels):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


def conv3x3_dilated(in_channels, out_channels, dilation=1):
    """dilated 3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)


def conv1x1(in_channels, out_channels):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)


def conv64x1(in_channels, out_channels):
    """64x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=(64, 1))


def conv1x64(in_channels, out_channels):
    """1x64 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1, 64))


class Block(nn.Module):
    def __init__(self, in_channels, dilation=1):
        super(Block, self).__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.project_down = conv1x1(in_channels, in_channels // 2)
        self.norm2 = nn.BatchNorm2d(in_channels // 2)
        self.dilation = conv3x3_dilated(in_channels // 2, in_channels // 2, dilation=dilation)
        self.norm3 = nn.BatchNorm2d(in_channels // 2)
        self.project_up = conv1x1(in_channels // 2, in_channels)
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout2d(p=DROPOUT_RATE, inplace=True)

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.elu(out)
        out = self.project_down(out)
        out = self.norm2(out)
        out = self.elu(out)
        out = self.dilation(out)
        out = self.dropout(out)
        out = self.norm3(out)
        out = self.elu(out)
        out = self.project_up(out)
        return out + identity


class ProsprNetwork(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_blocks_wide=28, num_blocks_narrow=192):
        super(ProsprNetwork, self).__init__()
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.conv1 = conv1x1(input_dim, 256)
        self.dropout = nn.Dropout2d(p=DROPOUT_RATE, inplace=True)
        self.conv2 = conv1x1(128, DIST_BINS)
        self.conv_aux_i = conv64x1(128, AUX_BINS)
        self.conv_aux_j = conv1x64(128, AUX_BINS)
        self.blocks = self._make_layer(num_blocks_wide, num_blocks_narrow)

    def _make_layer(self, num_blocks_wide, num_blocks_narrow):
        layers = []
        dilations = [1, 2, 4, 8]
        for i in range(num_blocks_wide):
            layers.append(Block(256, dilation=dilations[i % 4]))
        layers.append(conv1x1(256, 128))
        for i in range(num_blocks_narrow):
            layers.append(Block(128, dilation=dilations[i % 4]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.blocks(x)
        distogram = self.conv2(x)
        aux_i_out = torch.squeeze(self.conv_aux_i(x), dim=2)
        aux_j_out = torch.squeeze(self.conv_aux_j(x), dim=3)
        aux_i = dict()
        aux_i["ss"] = aux_i_out[:, :SS_BINS, :]
        aux_i["phi"] = aux_i_out[:, SS_BINS : SS_BINS + ANGLE_BINS, :]
        aux_i["psi"] = aux_i_out[:, SS_BINS + ANGLE_BINS : SS_BINS + ANGLE_BINS + ANGLE_BINS, :]
        aux_i["asa"] = aux_i_out[:, SS_BINS + ANGLE_BINS + ANGLE_BINS :, :]
        aux_j = dict()
        aux_j["ss"] = aux_j_out[:, :SS_BINS, :]
        aux_j["phi"] = aux_j_out[:, SS_BINS : SS_BINS + ANGLE_BINS, :]
        aux_j["psi"] = aux_j_out[:, SS_BINS + ANGLE_BINS : SS_BINS + ANGLE_BINS + ANGLE_BINS, :]
        aux_j["asa"] = aux_j_out[:, SS_BINS + ANGLE_BINS + ANGLE_BINS :, :]

        return distogram, aux_i, aux_j


MENAGERIE_ZOO = "vendored-pytorch"


def build_prospr():
    import torch

    torch.manual_seed(0)
    # real constructor is ProsprNetwork() with INPUT_DIM=547, 28 wide blocks (256ch) + 192
    # narrow blocks (128ch); shrunk block counts + input_dim for a fast CPU trace, same
    # architecture (BatchNorm2d -> conv1x1 -> dilated residual Block stack -> distogram +
    # two auxiliary 1D-marginal heads).
    return ProsprNetwork(input_dim=24, num_blocks_wide=2, num_blocks_narrow=2)


def example_input_prospr():
    import torch

    torch.manual_seed(0)
    # real input is an [N, INPUT_DIM, CROP_SIZE, CROP_SIZE] MSA co-evolution feature crop.
    # CROP_SIZE=64 is architecturally load-bearing here: conv_aux_i/conv_aux_j use hardcoded
    # (64,1)/(1,64) kernels (see conv64x1/conv1x64 above), so the spatial dims cannot be
    # shrunk like the channel/block counts -- only input_dim and block depth are tiny.
    batch, crop = 1, 64
    return (torch.randn(batch, 24, crop, crop),)


MENAGERIE_ENTRIES = [
    ("ProSPr", "build_prospr", "example_input_prospr", 2020, "SOURCE_AVAILABLE"),
    ("ProFOLD", "build_prospr", "example_input_prospr", 2020, "SOURCE_AVAILABLE"),
]

# SOURCE: vendored from yizhou-wang/RODNet @ 44fd61fa7b43
# https://raw.githubusercontent.com/yizhou-wang/RODNet/44fd61fa7b43/rodnet/models/backbones/cdc.py
# https://raw.githubusercontent.com/yizhou-wang/RODNet/44fd61fa7b43/rodnet/models/rodnet_cdc.py
#
# RODNet (Wang et al., "RODNet: A Real-Time Radar Object Detection Network Cross-Supervised
# by Camera-Radar Fused Object 3D Localization", WACV 2021 / IEEE J-STSP 2021) predicts
# object-detection confidence maps directly from range-Doppler-angle radar heatmap
# sequences. This is the "CDC" (Convolution-Deconvolution) backbone variant --
# RadarVanilla / RODNet-CDC -- the simplest of the four official backbones (CDC, HG, HGwi,
# their -v2/-dcn siblings); it is pure base-torch (nn.Conv3d/BatchNorm3d/ConvTranspose3d/
# PReLU), unlike the HG/HGwi variants which depend on the repo's custom compiled
# Deformable-Conv (`rodnet/ops/dcn/`) extension modules. Vendored verbatim -- no
# architectural changes; only the unused `use_mse_loss` docstring/import trimming.
#
# Only base-lib deps used: torch, torch.nn.

import torch.nn as nn


class RODEncode(nn.Module):
    def __init__(self, in_channels=2):
        super(RODEncode, self).__init__()
        self.conv1a = nn.Conv3d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(9, 5, 5),
            stride=(1, 1, 1),
            padding=(4, 2, 2),
        )
        self.conv1b = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=(9, 5, 5),
            stride=(2, 2, 2),
            padding=(4, 2, 2),
        )
        self.conv2a = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=(9, 5, 5),
            stride=(1, 1, 1),
            padding=(4, 2, 2),
        )
        self.conv2b = nn.Conv3d(
            in_channels=128,
            out_channels=128,
            kernel_size=(9, 5, 5),
            stride=(2, 2, 2),
            padding=(4, 2, 2),
        )
        self.conv3a = nn.Conv3d(
            in_channels=128,
            out_channels=256,
            kernel_size=(9, 5, 5),
            stride=(1, 1, 1),
            padding=(4, 2, 2),
        )
        self.conv3b = nn.Conv3d(
            in_channels=256,
            out_channels=256,
            kernel_size=(9, 5, 5),
            stride=(1, 2, 2),
            padding=(4, 2, 2),
        )
        self.bn1a = nn.BatchNorm3d(num_features=64)
        self.bn1b = nn.BatchNorm3d(num_features=64)
        self.bn2a = nn.BatchNorm3d(num_features=128)
        self.bn2b = nn.BatchNorm3d(num_features=128)
        self.bn3a = nn.BatchNorm3d(num_features=256)
        self.bn3b = nn.BatchNorm3d(num_features=256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/4, 16, 16)
        return x


class RODDecode(nn.Module):
    def __init__(self, n_class):
        super(RODDecode, self).__init__()
        self.convt1 = nn.ConvTranspose3d(
            in_channels=256,
            out_channels=128,
            kernel_size=(4, 6, 6),
            stride=(2, 2, 2),
            padding=(1, 2, 2),
        )
        self.convt2 = nn.ConvTranspose3d(
            in_channels=128,
            out_channels=64,
            kernel_size=(4, 6, 6),
            stride=(2, 2, 2),
            padding=(1, 2, 2),
        )
        self.convt3 = nn.ConvTranspose3d(
            in_channels=64,
            out_channels=n_class,
            kernel_size=(3, 6, 6),
            stride=(1, 2, 2),
            padding=(1, 2, 2),
        )
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.prelu(self.convt1(x))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
        x = self.prelu(self.convt2(x))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
        x = self.convt3(x)  # (B, 64, W, 64, 64) -> (B, 3, W, 128, 128)
        return x


class RadarVanilla(nn.Module):
    def __init__(self, in_channels, n_class, use_mse_loss=False):
        super(RadarVanilla, self).__init__()
        self.encoder = RODEncode(in_channels=in_channels)
        self.decoder = RODDecode(n_class=n_class)
        self.sigmoid = nn.Sigmoid()
        self.use_mse_loss = use_mse_loss

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if not self.use_mse_loss:
            x = self.sigmoid(x)
        return x


class RODNetCDC(nn.Module):
    def __init__(self, in_channels, n_class):
        super(RODNetCDC, self).__init__()
        self.cdc = RadarVanilla(in_channels, n_class, use_mse_loss=False)

    def forward(self, x):
        x = self.cdc(x)
        return x


def build_rodnet_cdc():
    import torch

    torch.manual_seed(0)
    model = RODNetCDC(in_channels=2, n_class=3)
    model.eval()
    return model


def example_input_rodnet_cdc():
    import torch

    torch.manual_seed(0)
    # (B, C=2 [real/imag or I/Q RA heatmap channels], W=win_size, H, W) shrunk from the
    # repo's default 128x128 spatial size / win_size=16 for a fast trace.
    return torch.randn(1, 2, 4, 32, 32)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RODNet-CDC", "build_rodnet_cdc", "example_input_rodnet_cdc", 2021, MENAGERIE_ZOO),
]

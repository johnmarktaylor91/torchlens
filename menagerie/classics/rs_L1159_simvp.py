# SOURCE: vendored from A4Bio/SimVP @ 4dcd4451282e014573ca4346ced8223e342f7893
# https://raw.githubusercontent.com/A4Bio/SimVP/master/model.py
# https://raw.githubusercontent.com/A4Bio/SimVP/master/modules.py
#
# SimVP (Gao, Tan, Wang, Yang, Liu, Li, CVPR 2022) -- "SimVP: Simpler yet Better Video
# Prediction": a pure-CNN spatiotemporal predictor with no recurrence/attention/adversarial
# training: a strided-Conv2d spatial Encoder, an Inception-style (multi-kernel grouped-conv)
# Mid_Xnet temporal translator over the stacked-frame latent, and a transposed-Conv2d
# Decoder with a skip connection from the first encoder layer. `BasicConv2d`, `ConvSC`,
# `GroupConv2d`, `Inception` (from modules.py) and `stride_generator`, `Encoder`, `Decoder`,
# `Mid_Xnet`, `SimVP` (from model.py) are copied verbatim -- no architectural changes, pure
# torch with no external dependencies.
"""SimVP: pure-CNN encoder / Inception temporal translator / decoder video predictor.

Faithful tiny-scale assembly of the real SimVP architecture for TorchLens tracing.
Random init, no pretrained/hub downloads.
"""

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# modules.py (verbatim)
# ---------------------------------------------------------------------------
class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        transpose=False,
        act_norm=False,
    ):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if not transpose:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=stride // 2,
            )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(
            C_in,
            C_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            transpose=transpose,
            act_norm=act_norm,
        )

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False
    ):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(
                GroupConv2d(
                    C_hid,
                    C_out,
                    kernel_size=ker,
                    stride=1,
                    padding=ker // 2,
                    groups=groups,
                    act_norm=True,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y


# ---------------------------------------------------------------------------
# model.py (verbatim)
# ---------------------------------------------------------------------------
def stride_generator(N, reverse=False):
    strides = [1, 2] * 10
    if reverse:
        return list(reversed(strides[:N]))
    else:
        return strides[:N]


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        super(Encoder, self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]],
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True),
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y


class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3, 5, 7, 11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [
            Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)
        ]
        for i in range(1, N_T - 1):
            enc_layers.append(
                Inception(
                    channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups
                )
            )
        enc_layers.append(
            Inception(
                channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups
            )
        )

        dec_layers = [
            Inception(
                channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups
            )
        ]
        for i in range(1, N_T - 1):
            dec_layers.append(
                Inception(
                    2 * channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    incep_ker=incep_ker,
                    groups=groups,
                )
            )
        dec_layers.append(
            Inception(
                2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups
            )
        )

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y


class SimVP(nn.Module):
    def __init__(
        self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3, 5, 7, 11], groups=8
    ):
        super(SimVP, self).__init__()
        T, C, H, W = shape_in
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T * hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, C, N_S)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B * T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = Y.reshape(B, T, C, H, W)
        return Y


# ---------------------------------------------------------------------------
# Tiny-scale build/example helpers
# ---------------------------------------------------------------------------
def build_simvp():
    # shape_in = (T, C, H, W); small hid_S/hid_T/N_S/N_T/incep_ker/groups for a fast trace
    return SimVP(
        shape_in=(4, 1, 16, 16),
        hid_S=4,
        hid_T=8,
        N_S=2,
        N_T=2,
        incep_ker=[3, 5],
        groups=2,
    )


def example_input_simvp():
    # (B, T, C, H, W)
    return torch.randn(1, 4, 1, 16, 16)


MENAGERIE_ENTRIES = [
    ("SimVP", "build_simvp", "example_input_simvp", 2022, "vendored"),
]

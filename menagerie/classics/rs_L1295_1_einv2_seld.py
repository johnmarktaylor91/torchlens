# SOURCE: vendored from yinkalario/EIN-SELD @ master
# https://github.com/yinkalario/EIN-SELD
# Files combined: seld/methods/ein_seld/models/seld.py (EINV2) +
#                 seld/methods/utils/model_utilities.py (DoubleConv, init_layer)
# Only import paths were adjusted to be self-contained; the architecture is
# transcribed verbatim from the official repo.
import numpy as np
import torch
import torch.nn as nn


def init_layer(layer, nonlinearity="leaky_relu"):
    """
    Initialize a layer
    """
    classname = layer.__class__.__name__
    if (classname.find("Conv") != -1) or (classname.find("Linear") != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=1,
        bias=False,
    ):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.init_weights()

    def init_weights(self):
        for layer in self.double_conv:
            init_layer(layer)

    def forward(self, x):
        x = self.double_conv(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, pos_len, d_model=512, pe_type="t", dropout=0.0):
        """Positional encoding using sin and cos

        Args:
            pos_len: positional length
            d_model: number of feature maps
            pe_type: 't' | 'f' , time domain, frequency domain
            dropout: dropout probability
        """
        super().__init__()

        self.pe_type = pe_type
        pe = torch.zeros(pos_len, d_model)
        pos = torch.arange(0, pos_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = 0.1 * torch.sin(pos * div_term)
        pe[:, 1::2] = 0.1 * torch.cos(pos * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)  # (N, C, T)
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.pe_type == "t":
            x = x + self.pe[:, :, : x.shape[2]]
        elif self.pe_type == "f":
            x = x + self.pe[:, :, : x.shape[2]].unsqueeze(-1)
        return self.dropout(x)


class EINV2(nn.Module):
    """Event-Independent Network V2 (EIN-SELD).

    Joint sound-event-detection (SED) + direction-of-arrival (DOA) estimation
    network with a two-branch (SED/DOA) convolutional front end connected by
    learned "stitch" cross-channel mixing, followed by two-track transformer
    encoders and per-track linear heads. arXiv:2010.13092.
    """

    def __init__(self, in_channels=7, pe_enable=False):
        super().__init__()
        self.pe_enable = pe_enable  # True | False

        self.f_bins = 64
        self.in_channels = in_channels

        self.downsample_ratio = 2**2
        self.sed_conv_block1 = nn.Sequential(
            DoubleConv(in_channels=4, out_channels=64),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block2 = nn.Sequential(
            DoubleConv(in_channels=64, out_channels=128),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.sed_conv_block3 = nn.Sequential(
            DoubleConv(in_channels=128, out_channels=256),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )
        self.sed_conv_block4 = nn.Sequential(
            DoubleConv(in_channels=256, out_channels=512),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )

        self.doa_conv_block1 = nn.Sequential(
            DoubleConv(in_channels=self.in_channels, out_channels=64),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block2 = nn.Sequential(
            DoubleConv(in_channels=64, out_channels=128),
            nn.AvgPool2d(kernel_size=(2, 2)),
        )
        self.doa_conv_block3 = nn.Sequential(
            DoubleConv(in_channels=128, out_channels=256),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )
        self.doa_conv_block4 = nn.Sequential(
            DoubleConv(in_channels=256, out_channels=512),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )

        self.stitch = nn.ParameterList(
            [
                nn.Parameter(torch.FloatTensor(64, 2, 2).uniform_(0.1, 0.9)),
                nn.Parameter(torch.FloatTensor(128, 2, 2).uniform_(0.1, 0.9)),
                nn.Parameter(torch.FloatTensor(256, 2, 2).uniform_(0.1, 0.9)),
            ]
        )

        if self.pe_enable:
            self.pe = PositionalEncoding(pos_len=100, d_model=512, pe_type="t", dropout=0.0)
        self.sed_trans_track1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2),
            num_layers=2,
        )
        self.sed_trans_track2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2),
            num_layers=2,
        )
        self.doa_trans_track1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2),
            num_layers=2,
        )
        self.doa_trans_track2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024, dropout=0.2),
            num_layers=2,
        )

        self.fc_sed_track1 = nn.Linear(512, 14, bias=True)
        self.fc_sed_track2 = nn.Linear(512, 14, bias=True)
        self.fc_doa_track1 = nn.Linear(512, 3, bias=True)
        self.fc_doa_track2 = nn.Linear(512, 3, bias=True)
        self.final_act_sed = nn.Sequential()  # nn.Sigmoid()
        self.final_act_doa = nn.Tanh()

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc_sed_track1)
        init_layer(self.fc_sed_track2)
        init_layer(self.fc_doa_track1)
        init_layer(self.fc_doa_track2)

    def forward(self, x):
        """
        x: waveform-derived time-frequency feature, (batch_size, 7, T, F)
           channels 0:4 = SED input (log-mel + intensity subset), full 7 = DOA input
        """
        x_sed = x[:, :4]
        x_doa = x

        # cnn
        x_sed = self.sed_conv_block1(x_sed)
        x_doa = self.doa_conv_block1(x_doa)
        x_sed = torch.einsum("c, nctf -> nctf", self.stitch[0][:, 0, 0], x_sed) + torch.einsum(
            "c, nctf -> nctf", self.stitch[0][:, 0, 1], x_doa
        )
        x_doa = torch.einsum("c, nctf -> nctf", self.stitch[0][:, 1, 0], x_sed) + torch.einsum(
            "c, nctf -> nctf", self.stitch[0][:, 1, 1], x_doa
        )
        x_sed = self.sed_conv_block2(x_sed)
        x_doa = self.doa_conv_block2(x_doa)
        x_sed = torch.einsum("c, nctf -> nctf", self.stitch[1][:, 0, 0], x_sed) + torch.einsum(
            "c, nctf -> nctf", self.stitch[1][:, 0, 1], x_doa
        )
        x_doa = torch.einsum("c, nctf -> nctf", self.stitch[1][:, 1, 0], x_sed) + torch.einsum(
            "c, nctf -> nctf", self.stitch[1][:, 1, 1], x_doa
        )
        x_sed = self.sed_conv_block3(x_sed)
        x_doa = self.doa_conv_block3(x_doa)
        x_sed = torch.einsum("c, nctf -> nctf", self.stitch[2][:, 0, 0], x_sed) + torch.einsum(
            "c, nctf -> nctf", self.stitch[2][:, 0, 1], x_doa
        )
        x_doa = torch.einsum("c, nctf -> nctf", self.stitch[2][:, 1, 0], x_sed) + torch.einsum(
            "c, nctf -> nctf", self.stitch[2][:, 1, 1], x_doa
        )
        x_sed = self.sed_conv_block4(x_sed)
        x_doa = self.doa_conv_block4(x_doa)
        x_sed = x_sed.mean(dim=3)  # (N, C, T)
        x_doa = x_doa.mean(dim=3)  # (N, C, T)

        # transformer
        if self.pe_enable:
            x_sed = self.pe(x_sed)
        if self.pe_enable:
            x_doa = self.pe(x_doa)
        x_sed = x_sed.permute(2, 0, 1)  # (T, N, C)
        x_doa = x_doa.permute(2, 0, 1)  # (T, N, C)

        x_sed_1 = self.sed_trans_track1(x_sed).transpose(0, 1)  # (N, T, C)
        x_sed_2 = self.sed_trans_track2(x_sed).transpose(0, 1)  # (N, T, C)
        x_doa_1 = self.doa_trans_track1(x_doa).transpose(0, 1)  # (N, T, C)
        x_doa_2 = self.doa_trans_track2(x_doa).transpose(0, 1)  # (N, T, C)

        # fc
        x_sed_1 = self.final_act_sed(self.fc_sed_track1(x_sed_1))
        x_sed_2 = self.final_act_sed(self.fc_sed_track2(x_sed_2))
        x_sed = torch.stack((x_sed_1, x_sed_2), 2)
        x_doa_1 = self.final_act_doa(self.fc_doa_track1(x_doa_1))
        x_doa_2 = self.final_act_doa(self.fc_doa_track2(x_doa_2))
        x_doa = torch.stack((x_doa_1, x_doa_2), 2)
        output = {
            "sed": x_sed,
            "doa": x_doa,
        }

        return output


def build_einv2():
    return EINV2(in_channels=7, pe_enable=False)


def example_input_einv2():
    # (batch, 7-channel logmel+intensity feature, time frames, mel bins)
    # T and F chosen small but compatible with the four (2,2)/(1,2) AvgPool
    # stages (needs T divisible by 4, F divisible by 16).
    return torch.randn(1, 7, 8, 32)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("EINv2", "build_einv2", "example_input_einv2", 2021, MENAGERIE_ZOO),
]

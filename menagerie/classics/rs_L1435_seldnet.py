# SOURCE: vendored from sharathadavanne/seld-dcase2023 @ 24d14c9ebe6878062dc3b68eb07dd0100722f530
# (seldnet_model.py, class SeldModel — the PyTorch CRNN + self-attention SELDnet
# architecture used as the official DCASE2023 Task 3 sound-event-localization-and-detection
# baseline). This is the PyTorch-era evolution of the original sharathadavanne/seld-net
# (Keras/TF, pre-Keras-3 API, not installable here) by the same author: stacked Conv2D+BN+ReLU+
# MaxPool blocks over a (mic_channels, time, mel) spectrogram input, a bidirectional GRU with a
# GLU-style tanh-gated split, a stack of residual multi-head self-attention + LayerNorm blocks,
# and a final FNN stack regressing (ACCDOA-encoded) direction-of-arrival output. Only imports/
# docstring content changed from the source; class bodies are verbatim.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class SeldModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.nb_classes = params["unique_classes"]
        self.params = params
        self.conv_block_list = nn.ModuleList()
        if len(params["f_pool_size"]):
            for conv_cnt in range(len(params["f_pool_size"])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params["nb_cnn2d_filt"] if conv_cnt else in_feat_shape[1],
                        out_channels=params["nb_cnn2d_filt"],
                    )
                )
                self.conv_block_list.append(
                    nn.MaxPool2d((params["t_pool_size"][conv_cnt], params["f_pool_size"][conv_cnt]))
                )
                self.conv_block_list.append(nn.Dropout2d(p=params["dropout_rate"]))

        self.gru_input_dim = params["nb_cnn2d_filt"] * int(
            np.floor(in_feat_shape[-1] / np.prod(params["f_pool_size"]))
        )
        self.gru = torch.nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=params["rnn_size"],
            num_layers=params["nb_rnn_layers"],
            batch_first=True,
            dropout=params["dropout_rate"],
            bidirectional=True,
        )

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params["nb_self_attn_layers"]):
            self.mhsa_block_list.append(
                nn.MultiheadAttention(
                    embed_dim=self.params["rnn_size"],
                    num_heads=params["nb_heads"],
                    dropout=params["dropout_rate"],
                    batch_first=True,
                )
            )
            self.layer_norm_list.append(nn.LayerNorm(self.params["rnn_size"]))

        self.fnn_list = torch.nn.ModuleList()
        if params["nb_fnn_layers"]:
            for fc_cnt in range(params["nb_fnn_layers"]):
                self.fnn_list.append(
                    nn.Linear(
                        params["fnn_size"] if fc_cnt else self.params["rnn_size"],
                        params["fnn_size"],
                        bias=True,
                    )
                )
        self.fnn_list.append(
            nn.Linear(
                params["fnn_size"] if params["nb_fnn_layers"] else self.params["rnn_size"],
                out_shape[-1],
                bias=True,
            )
        )

    def forward(self, x):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2 :] * x[:, :, : x.shape[-1] // 2]

        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = torch.tanh(self.fnn_list[-1](x))
        return doa


def _tiny_params():
    return dict(
        unique_classes=3,
        f_pool_size=[2, 2],
        t_pool_size=[1, 1],
        nb_cnn2d_filt=8,
        dropout_rate=0.0,
        self_attn=True,
        nb_heads=2,
        nb_self_attn_layers=1,
        nb_rnn_layers=1,
        rnn_size=16,
        nb_fnn_layers=1,
        fnn_size=16,
    )


def build_seldnet():
    params = _tiny_params()
    in_feat_shape = (2, 4, 10, 16)  # (batch, mic_channels, time_steps, mel_bins)
    out_shape = (2, 10, 3 * 3 * params["unique_classes"])  # ACCDOA-style (track*axis*class) output
    model = SeldModel(in_feat_shape=in_feat_shape, out_shape=out_shape, params=params)
    model.eval()
    return model


def example_input_seldnet():
    return torch.randn(2, 4, 10, 16)


MENAGERIE_ENTRIES = [
    ("SELDnet", "build_seldnet", "example_input_seldnet", 2023, MENAGERIE_ZOO),
]

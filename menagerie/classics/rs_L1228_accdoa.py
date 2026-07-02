# SOURCE: vendored from sharathadavanne/seld-dcase2023 @ 24d14c9ebe6878062dc3b68eb07dd0100722f530
# https://github.com/sharathadavanne/seld-dcase2023/blob/main/seldnet_model.py
# The official DCASE2021 SELDnet baseline (queue row: sharathadavanne/seld-dcase2021) is
# a Keras/TF1 implementation (keras_model.py). The SAME author (sharathadavanne) publishes
# an official PyTorch re-release of the identical SELDnet/ACCDOA architecture for the later
# DCASE2023 SELD baseline (`SeldModel` in seldnet_model.py): CNN feature-pooling stack ->
# BiGRU (with the GRU-output tanh-gating split used in the original baseline) -> optional
# multi-head self-attention refinement blocks -> FNN stack -> tanh-squashed ACCDOA (Cartesian
# activity-coupled direction-of-arrival) output. The `ConvBlock` and `SeldModel` classes here
# are transcribed VERBATIM from seldnet_model.py (base torch/numpy only, no changes). Config
# is a tiny single-ACCDOA instantiation of parameters.py's default `params` dict
# (self_attn=True path exercised; multi_accdoa/ADPIT loss machinery in the source file is
# training-only and not needed for a forward trace, so it is omitted here).
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- seldnet_model.py (verbatim architecture) ---
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


# --- staging entry points ---
def _tiny_params():
    # tiny instantiation of parameters.py's default `params` dict (single-ACCDOA,
    # DCASE2021 class count = 12 as used by the queue-row's dataset year)
    label_hop_len_s = 0.1
    hop_len_s = 0.02
    feature_label_resolution = int(round(label_hop_len_s / hop_len_s))
    return {
        "unique_classes": 12,
        "dropout_rate": 0.05,
        "nb_cnn2d_filt": 8,
        "f_pool_size": [2, 2, 1],
        "t_pool_size": [feature_label_resolution, 1, 1],
        "self_attn": True,
        "nb_heads": 2,
        "nb_self_attn_layers": 1,
        "nb_rnn_layers": 1,
        "rnn_size": 16,
        "nb_fnn_layers": 1,
        "fnn_size": 16,
    }


def build_accdoa():
    params = _tiny_params()
    nb_mel_bins = 16
    in_feat_shape = (
        1,
        7,
        params["t_pool_size"][0] * 2,
        nb_mel_bins,
    )  # (batch, mic_channels, time, mel_bins)
    out_shape = (1, 2, params["unique_classes"] * 3)  # single-ACCDOA: 3 axes * nb_classes
    return SeldModel(in_feat_shape, out_shape, params)


def example_input_accdoa():
    params = _tiny_params()
    nb_mel_bins = 16
    time_steps = params["t_pool_size"][0] * 2
    return torch.randn(1, 7, time_steps, nb_mel_bins)


MENAGERIE_ENTRIES = [
    ("ACCDOA (SELDnet)", "build_accdoa", "example_input_accdoa", 2020, MENAGERIE_ZOO),
]

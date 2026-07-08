# FAITHFUL PORT of mdelrosa/csinet-lstm @ master (original framework: TensorFlow/Keras)
#
# Source files transcribed (raw.githubusercontent.com/mdelrosa/csinet-lstm/master/csinet-lstm/):
#   - csinet.py       (base `CsiNet` Keras functional model, used as the T=0 "high-dim"
#                       and shared T=1..T-1 "low-dim" per-timestep sub-networks)
#   - csinet_lstm.py  (`make_CsiNet`, `CsiNet_LSTM`, `stacked_LSTM` -- the recurrent
#                       feedback wiring and the stacked-LSTM temporal-correlation head)
#
# CsiNet-LSTM (Wang, Gao, Mei, Cavallaro, "CSI-LSTM: Deep Learning for
# Time-Varying CSI Feedback with Long Short-Term Memory" -- the follow-on to
# CsiNet with temporal correlation across a CSI time series). No PyTorch port
# of the LSTM-specific model exists in mdelrosa/csinet-lstm (only the base
# CsiNet, without the LSTM wiring, was ported to `torch/csinet_torch.py`;
# see the sibling `L1224_csinet.py` staging module for that separate torch
# port). This module is a from-scratch faithful transcription of the real
# `csinet.py` / `csinet_lstm.py` TensorFlow/Keras source into torch, matching
# every architectural mechanism:
#
#   - `CsiNetBase` == `csinet.py`'s `CsiNet` Keras functional model: a 3-layer
#     Conv2d+BatchNorm+LeakyReLU encoder (channels 2->8->16->2, channels_first,
#     matching `residual_block_decoded`'s conv channel path exactly) flattened
#     and linear-mapped to `encoded_dim`; a decoder that concatenates an
#     optional `aux` side-vector with the encoded vector (`concatenate([aux,
#     encoded])` in the original), linear-maps back to `img_total`, reshapes
#     to (C,H,W), then 6x Conv2d+BatchNorm+LeakyReLU at channels
#     128->64->32->32->16->16 (the original's `deconv1..deconv6`, all
#     `channels_first`) and a final Conv2d to `img_channels` with a `tanh`
#     output activation (the original's `out_activation` default, "predict"
#     layer). Keras' default `LeakyReLU()` alpha is 0.3 (not PyTorch's
#     default 0.01), used here via `nn.LeakyReLU(0.3)` to match.
#   - `CsiNetLSTM` == `CsiNet_LSTM(...)`'s per-timestep recurrent feedback
#     loop: at t=0, the high-dimensional-latent `CsiNet_hi` (encoded_dim=M_1)
#     encodes/decodes the first CSI frame; for t=1..T-1, a SHARED
#     low-dimensional-latent `CsiNet_lo` (encoded_dim=M_2, `share_bool=True`
#     path, the default in the original's training script) reconstructs each
#     subsequent frame using `EncodedLayer` -- the M_1-dim encoded vector from
#     the *first* (t=0) `CsiNet_hi` pass, held fixed and fed as the `aux` side
#     input at every later timestep, exactly as in the original's
#     `OutLayer = CsiNet_lo([EncodedLayer, CsiIn])` call (not each step's own
#     encoding -- the original always re-uses the t=0 encoding as `aux`).
#     The non-shared branch (`share_bool=False`, an independent `CsiNet_lo`
#     per timestep) is also transcribed (`csinet_lo_list`).
#   - `StackedLSTM` == `stacked_LSTM(...)`: reshapes the concatenated
#     per-timestep reconstructions (B,T,C,H,W) to (B,T,C*H*W) and applies
#     `LSTM_depth` (default 3) stacked recurrent layers of width C*H*W with
#     `return_sequences=True` at every layer (the original's `CuDNNLSTM`,
#     functionally a vanilla LSTM with cuDNN-only recurrent activations --
#     transcribed here as `nn.LSTM`, the CPU/device-agnostic equivalent),
#     then reshapes back to (B,T,C,H,W).
#
# Pretrained-weight loading (`pretrained_bool`, `.h5` weight files),
# `LSTM_only_bool` (skip CsiNets entirely) and `pass_through_bool` (bypass
# the LSTM) are training/deployment-configuration branches, not part of the
# core architecture graph, and are not exposed here; the staged
# `build_csinet_lstm()` below always exercises the full CsiNet_hi -> shared
# CsiNet_lo x(T-1) -> stacked-LSTM path (the paper's default configuration).
#
# MENAGERIE_ZOO = "ported-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn


class CsiNetBase(nn.Module):
    """Faithful port of csinet.py's `CsiNet` Keras functional model.

    Encoder: 3x (Conv2d 3x3 same-pad -> BatchNorm -> LeakyReLU(0.3)), channel
    path img_channels->8->16->2 over a (img_channels, H, W) map
    (channels_first), flattened and linear-mapped to `encoded_dim`.

    Decoder: optional `aux` (side vector, e.g. the high-dim CsiNet's encoded
    vector) concatenated with the encoded vector, linear-mapped back to
    img_total, reshaped to (img_channels, H, W), then 6x (Conv2d ->
    BatchNorm -> LeakyReLU(0.3)) at channels [128, 64, 32, 32, 16, 16]
    followed by a final Conv2d to img_channels with `out_activation` (tanh).
    """

    def __init__(
        self, img_channels, img_height, img_width, encoded_dim, aux_dim=None, out_activation="tanh"
    ):
        super().__init__()
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.img_total = img_channels * img_height * img_width
        self.encoded_dim = encoded_dim
        self.aux_dim = aux_dim

        self.enc_conv1 = nn.Conv2d(img_channels, 8, 3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(8)
        self.enc_conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(16)
        self.enc_conv3 = nn.Conv2d(16, 2, 3, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(2)
        self.enc_dense = nn.Linear(self.img_total, encoded_dim)

        dec_in_dim = encoded_dim + (aux_dim if aux_dim else 0)
        self.dec_dense = nn.Linear(dec_in_dim, self.img_total)

        self.dec_conv1 = nn.Conv2d(img_channels, 128, 1)
        self.dec_bn1 = nn.BatchNorm2d(128)
        self.dec_conv2 = nn.Conv2d(128, 64, 1)
        self.dec_bn2 = nn.BatchNorm2d(64)
        self.dec_conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(32)
        self.dec_conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.dec_bn4 = nn.BatchNorm2d(32)
        self.dec_conv5 = nn.Conv2d(32, 16, 3, padding=1)
        self.dec_bn5 = nn.BatchNorm2d(16)
        self.dec_conv6 = nn.Conv2d(16, 16, 3, padding=1)
        self.dec_bn6 = nn.BatchNorm2d(16)
        self.dec_conv7 = nn.Conv2d(16, img_channels, 3, padding=1)

        self.activ = nn.LeakyReLU(0.3)  # Keras LeakyReLU() default alpha=0.3
        if out_activation == "tanh":
            self.out_activ = nn.Tanh()
        elif out_activation == "sigmoid":
            self.out_activ = nn.Sigmoid()
        else:
            self.out_activ = nn.Identity()

    def forward(self, x, aux=None):
        y = self.activ(self.enc_bn1(self.enc_conv1(x)))
        y = self.activ(self.enc_bn2(self.enc_conv2(y)))
        y = self.activ(self.enc_bn3(self.enc_conv3(y)))
        y = torch.reshape(y, (y.size(0), -1))
        encoded = self.enc_dense(y)

        if aux is not None:
            z = torch.cat([aux, encoded], dim=1)
        else:
            z = encoded
        z = self.dec_dense(z)
        z = torch.reshape(z, (z.size(0), self.img_channels, self.img_height, self.img_width))

        z = self.activ(self.dec_bn1(self.dec_conv1(z)))
        z = self.activ(self.dec_bn2(self.dec_conv2(z)))
        z = self.activ(self.dec_bn3(self.dec_conv3(z)))
        z = self.activ(self.dec_bn4(self.dec_conv4(z)))
        z = self.activ(self.dec_bn5(self.dec_conv5(z)))
        z = self.activ(self.dec_bn6(self.dec_conv6(z)))
        out = self.out_activ(self.dec_conv7(z))
        return encoded, out


class StackedLSTM(nn.Module):
    """Faithful port of `stacked_LSTM(...)`: LSTM_depth stacked recurrent
    layers of width img_channels*H*W, each with return_sequences=True."""

    def __init__(self, img_channels, img_height, img_width, lstm_depth=3):
        super().__init__()
        self.lstm_dim = img_channels * img_height * img_width
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.layers = nn.ModuleList(
            [nn.LSTM(self.lstm_dim, self.lstm_dim, batch_first=True) for _ in range(lstm_depth)]
        )

    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, T, C*H*W)
        B, T = x.shape[0], x.shape[1]
        h = x.reshape(B, T, -1)
        for lstm in self.layers:
            h, _ = lstm(h)
        out = h.reshape(B, T, self.img_channels, self.img_height, self.img_width)
        return out


class CsiNetLSTM(nn.Module):
    """Faithful port of `CsiNet_LSTM(...)`'s default configuration
    (aux_bool=True, share_bool=True, LSTM_only_bool=False,
    pass_through_bool=False): per-timestep CsiNet_hi/CsiNet_lo reconstruction
    with recurrent aux feedback, followed by a stacked LSTM over the
    reconstructed CSI time series."""

    def __init__(
        self,
        img_channels,
        img_height,
        img_width,
        T,
        M_1,
        M_2,
        lstm_depth=3,
        aux_bool=True,
        share_bool=True,
    ):
        super().__init__()
        self.T = T
        self.aux_bool = aux_bool
        self.share_bool = share_bool

        self.csinet_hi = CsiNetBase(
            img_channels, img_height, img_width, M_1, aux_dim=(M_1 if aux_bool else None)
        )
        if share_bool:
            self.csinet_lo = CsiNetBase(
                img_channels, img_height, img_width, M_2, aux_dim=(M_1 if aux_bool else None)
            )
        else:
            self.csinet_lo_list = nn.ModuleList(
                [
                    CsiNetBase(
                        img_channels,
                        img_height,
                        img_width,
                        M_2,
                        aux_dim=(M_1 if aux_bool else None),
                    )
                    for _ in range(T - 1)
                ]
            )
        self.lstm = StackedLSTM(img_channels, img_height, img_width, lstm_depth=lstm_depth)

    def forward(self, aux, x):
        # x: (B, T, C, H, W); aux: (B, M_1) high-dim side input to CsiNet_hi
        outs = []
        encoded_hi = None
        for t in range(self.T):
            csi_in = x[:, t]
            if t == 0:
                encoded_hi, out = self.csinet_hi(csi_in, aux if self.aux_bool else None)
            else:
                lo_net = self.csinet_lo if self.share_bool else self.csinet_lo_list[t - 1]
                if self.aux_bool:
                    _, out = lo_net(csi_in, encoded_hi)
                else:
                    _, out = lo_net(csi_in, None)
            outs.append(out.unsqueeze(1))
        lstm_in = torch.cat(outs, dim=1)
        lstm_out = self.lstm(lstm_in)
        return lstm_out


# ============================================================================
# staging build/example functions
# ============================================================================


def build_csinet_lstm():
    """Tiny-config CsiNet-LSTM: T=3 timesteps, M_1=64 (high-dim latent),
    M_2=16 (low-dim latent), 2-layer stacked LSTM, shared low-rate CsiNet."""
    img_channels, H, W = 2, 32, 32
    T = 3
    M_1 = 64
    M_2 = 16
    return CsiNetLSTM(img_channels, H, W, T, M_1, M_2, lstm_depth=2, aux_bool=True, share_bool=True)


def example_input_csinet_lstm():
    batch_size = 2
    M_1 = 64
    T = 3
    img_channels, H, W = 2, 32, 32
    aux = torch.randn(batch_size, M_1)
    x = torch.randn(batch_size, T, img_channels, H, W)
    return (aux, x)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("CsiNet-LSTM", "build_csinet_lstm", "example_input_csinet_lstm", 2019, MENAGERIE_ZOO),
]

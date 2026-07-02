# SOURCE: vendored from NVIDIA/CleanUNet @ main
# https://raw.githubusercontent.com/NVIDIA/CleanUNet/main/network.py
# https://raw.githubusercontent.com/NVIDIA/CleanUNet/main/util.py (weight_scaling_init only)
#
# CleanUNet (Kong, Ping, Huang, Zhao, Catanzaro; ICASSP 2022): a 1D speech-denoising
# U-Net operating directly on the raw waveform, with a self-attention (Transformer
# encoder) block at the bottleneck. The classes below (`ScaledDotProductAttention`,
# `MultiHeadAttention`, `PositionwiseFeedForward`, `PositionalEncoding`, `EncoderLayer`,
# `TransformerEncoder`, `padding`, `CleanUNet`) are copied verbatim from the official
# repo's network.py. The only change is inlining the single helper the file imports
# from the sibling `util.py` (`weight_scaling_init`, a Conv weight-rescaling init used
# at construction time) instead of importing it via `from util import ...`, since that
# module also pulls in training-only utilities (STFT loss, checkpoint I/O) we don't
# need for tracing. No architectural code was altered.
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


def weight_scaling_init(layer):
    """
    weight rescaling initialization from https://arxiv.org/abs/1911.13254
    (verbatim from NVIDIA/CleanUNet util.py)
    """
    w = layer.weight.detach()
    alpha = 10.0 * w.std()
    layer.weight.data /= torch.sqrt(alpha)
    layer.bias.data /= torch.sqrt(alpha)


# Transformer (encoder) https://github.com/jadore801120/attention-is-all-you-need-pytorch
# Original Copyright 2017 Victor Huang
#  MIT License (https://opensource.org/licenses/MIT)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    sz_b, len_s = seq.size()
    subsequent_mask = (
        1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)
    ).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()


class EncoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class TransformerEncoder(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(
        self,
        d_word_vec=512,
        n_layers=2,
        n_head=8,
        d_k=64,
        d_v=64,
        d_model=512,
        d_inner=2048,
        dropout=0.1,
        n_position=624,
        scale_emb=False,
    ):
        super().__init__()

        if n_position > 0:
            self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        else:
            self.position_enc = lambda x: x
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []

        # -- Forward
        enc_output = src_seq
        if self.scale_emb:
            enc_output *= self.d_model**0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


# CleanUNet architecture


def padding(x, D, K, S):
    """padding zeroes to x so that denoised audio has the same length"""

    L = x.shape[-1]
    for _ in range(D):
        if L < K:
            L = 1
        else:
            L = 1 + np.ceil((L - K) / S)

    for _ in range(D):
        L = (L - 1) * S + K

    L = int(L)
    x = F.pad(x, (0, L - x.shape[-1]))
    return x


class CleanUNet(nn.Module):
    """CleanUNet architecture."""

    def __init__(
        self,
        channels_input=1,
        channels_output=1,
        channels_H=64,
        max_H=768,
        encoder_n_layers=8,
        kernel_size=4,
        stride=2,
        tsfm_n_layers=3,
        tsfm_n_head=8,
        tsfm_d_model=512,
        tsfm_d_inner=2048,
    ):
        """
        Parameters:
        channels_input (int):   input channels
        channels_output (int):  output channels
        channels_H (int):       middle channels H that controls capacity
        max_H (int):            maximum H
        encoder_n_layers (int): number of encoder/decoder layers D
        kernel_size (int):      kernel size K
        stride (int):           stride S
        tsfm_n_layers (int):    number of self attention blocks N
        tsfm_n_head (int):      number of heads in each self attention block
        tsfm_d_model (int):     d_model of self attention
        tsfm_d_inner (int):     d_inner of self attention
        """

        super(CleanUNet, self).__init__()

        self.channels_input = channels_input
        self.channels_output = channels_output
        self.channels_H = channels_H
        self.max_H = max_H
        self.encoder_n_layers = encoder_n_layers
        self.kernel_size = kernel_size
        self.stride = stride

        self.tsfm_n_layers = tsfm_n_layers
        self.tsfm_n_head = tsfm_n_head
        self.tsfm_d_model = tsfm_d_model
        self.tsfm_d_inner = tsfm_d_inner

        # encoder and decoder
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(encoder_n_layers):
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(channels_input, channels_H, kernel_size, stride),
                    nn.ReLU(),
                    nn.Conv1d(channels_H, channels_H * 2, 1),
                    nn.GLU(dim=1),
                )
            )
            channels_input = channels_H

            if i == 0:
                # no relu at end
                self.decoder.append(
                    nn.Sequential(
                        nn.Conv1d(channels_H, channels_H * 2, 1),
                        nn.GLU(dim=1),
                        nn.ConvTranspose1d(channels_H, channels_output, kernel_size, stride),
                    )
                )
            else:
                self.decoder.insert(
                    0,
                    nn.Sequential(
                        nn.Conv1d(channels_H, channels_H * 2, 1),
                        nn.GLU(dim=1),
                        nn.ConvTranspose1d(channels_H, channels_output, kernel_size, stride),
                        nn.ReLU(),
                    ),
                )
            channels_output = channels_H

            # double H but keep below max_H
            channels_H *= 2
            channels_H = min(channels_H, max_H)

        # self attention block
        self.tsfm_conv1 = nn.Conv1d(channels_output, tsfm_d_model, kernel_size=1)
        self.tsfm_encoder = TransformerEncoder(
            d_word_vec=tsfm_d_model,
            n_layers=tsfm_n_layers,
            n_head=tsfm_n_head,
            d_k=tsfm_d_model // tsfm_n_head,
            d_v=tsfm_d_model // tsfm_n_head,
            d_model=tsfm_d_model,
            d_inner=tsfm_d_inner,
            dropout=0.0,
            n_position=0,
            scale_emb=False,
        )
        self.tsfm_conv2 = nn.Conv1d(tsfm_d_model, channels_output, kernel_size=1)

        # weight scaling initialization
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                weight_scaling_init(layer)

    def forward(self, noisy_audio):
        # (B, L) -> (B, C, L)
        if len(noisy_audio.shape) == 2:
            noisy_audio = noisy_audio.unsqueeze(1)
        B, C, L = noisy_audio.shape
        assert C == 1

        # normalization and padding
        std = noisy_audio.std(dim=2, keepdim=True) + 1e-3
        noisy_audio = noisy_audio / std
        x = padding(noisy_audio, self.encoder_n_layers, self.kernel_size, self.stride)

        # encoder
        skip_connections = []
        for downsampling_block in self.encoder:
            x = downsampling_block(x)
            skip_connections.append(x)
        skip_connections = skip_connections[::-1]

        # attention mask for causal inference; for non-causal, set attn_mask to None
        len_s = x.shape[-1]  # length at bottleneck
        attn_mask = (
            1 - torch.triu(torch.ones((1, len_s, len_s), device=x.device), diagonal=1)
        ).bool()

        x = self.tsfm_conv1(x)  # C 1024 -> 512
        x = x.permute(0, 2, 1)
        x = self.tsfm_encoder(x, src_mask=attn_mask)
        x = x.permute(0, 2, 1)
        x = self.tsfm_conv2(x)  # C 512 -> 1024

        # decoder
        for i, upsampling_block in enumerate(self.decoder):
            skip_i = skip_connections[i]
            x = x + skip_i[:, :, : x.shape[-1]]
            x = upsampling_block(x)

        x = x[:, :, :L] * std
        return x


def build_cleanunet():
    # Tiny menagerie-scale config: source defaults (channels_H=64, max_H=768,
    # encoder_n_layers=8, tsfm_d_model=512, tsfm_d_inner=2048) shrunk for fast tracing
    # while keeping the real architecture (U-Net encoder/decoder + Transformer
    # bottleneck) intact.
    return CleanUNet(
        channels_input=1,
        channels_output=1,
        channels_H=4,
        max_H=16,
        encoder_n_layers=3,
        kernel_size=4,
        stride=2,
        tsfm_n_layers=1,
        tsfm_n_head=2,
        tsfm_d_model=8,
        tsfm_d_inner=16,
    )


def example_input_cleanunet():
    # Raw single-channel waveform, matching the source's `(B, L)` / `(B, 1, L)`
    # noisy_audio convention.
    torch.manual_seed(0)
    return (torch.randn(1, 256),)


MENAGERIE_ENTRIES = [
    (
        "CleanUNet",
        "build_cleanunet",
        "example_input_cleanunet",
        2022,
        "vendored-pytorch",
    ),
]

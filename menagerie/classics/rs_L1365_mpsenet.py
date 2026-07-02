# SOURCE: vendored from https://github.com/yxlu-0102/MP-SENet @ 89932cfe90d1dacb8e170e4a331d762462c21792
#   Vendored files:
#     - models/model.py       -> `MPNet` and its DenseEncoder/DenseBlock/SPConvTranspose2d/
#       MaskDecoder/PhaseDecoder/TSTransformerBlock building blocks (the explicit-parallel
#       magnitude + phase spectra denoising network, "MP-SENet: A Speech Enhancement Model
#       with Parallel Denoising of Magnitude and Phase Spectra", Lu et al. 2023, MIT license).
#     - models/transformer.py -> `TransformerBlock`/`FFN` (the time- and frequency-axis
#       transformer used inside each TSTransformerBlock).
#     - utils.py               -> `LearnableSigmoid2d` (the learnable-slope sigmoid mask gate;
#       only this one class is vendored, the plotting/checkpoint helpers are dropped).
#
# `env.py`'s pesq/joblib evaluation helpers and `train.py`'s discriminator/loss machinery are
# training-only and dropped; the `MPNet` generator (the actual network architecture) is
# reproduced verbatim, with a small AttrDict-like config object standing in for the real
# `h = AttrDict(json.loads(open(config.json).read()))` config loader.

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- utils.py (verbatim, trimmed to LearnableSigmoid2d) ----


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


# ---- models/transformer.py (verbatim) ----


class FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model * 2, 1, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(d_model * 2 * 2, d_model)
        else:
            self.linear = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = torch.nn.functional.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, bidirectional=True, dropout=0):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, bidirectional=bidirectional)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        xt = self.norm1(x)
        xt, _ = self.attention(xt, xt, xt, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

        x = self.norm3(x)

        return x


# ---- models/model.py (verbatim) ----


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, h, kernel_size=(2, 3), depth=4):
        super().__init__()
        self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2**i
            pad_length = dilation
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
                nn.Conv2d(
                    h.dense_channel * (i + 1), h.dense_channel, kernel_size, dilation=(dilation, 1)
                ),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel),
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, h, in_channel):
        super().__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

        self.dense_block = DenseBlock(h, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(h.dense_channel, h.dense_channel, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)  # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class MaskDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, (1, 2)),
        )
        self.lsigmoid = LearnableSigmoid2d(h.n_fft // 2 + 1, beta=h.beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)  # [B, F, T]
        x = self.lsigmoid(x)
        return x


class PhaseDecoder(nn.Module):
    def __init__(self, h, out_channel=1):
        super().__init__()
        self.dense_block = DenseBlock(h, depth=4)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(h.dense_channel, h.dense_channel, (1, 3), 2),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
        )
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        x = x.permute(0, 3, 2, 1).squeeze(-1)  # [B, F, T]
        return x


class TSTransformerBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.time_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=4)
        self.freq_transformer = TransformerBlock(d_model=h.dense_channel, n_heads=4)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time_transformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_transformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x


class MPNet(nn.Module):
    def __init__(self, h, num_tsblocks=4):
        super().__init__()
        self.h = h
        self.num_tscblocks = num_tsblocks
        self.dense_encoder = DenseEncoder(h, in_channel=2)

        self.TSTransformer = nn.ModuleList([])
        for _i in range(num_tsblocks):
            self.TSTransformer.append(TSTransformerBlock(h))

        self.mask_decoder = MaskDecoder(h, out_channel=1)
        self.phase_decoder = PhaseDecoder(h, out_channel=1)

    def forward(self, noisy_amp, noisy_pha):  # [B, F, T]
        x = torch.stack((noisy_amp, noisy_pha), dim=-1).permute(0, 3, 2, 1)  # [B, 2, T, F]
        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSTransformer[i](x)

        denoised_amp = noisy_amp * self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)
        denoised_com = torch.stack(
            (denoised_amp * torch.cos(denoised_pha), denoised_amp * torch.sin(denoised_pha)), dim=-1
        )

        return denoised_amp, denoised_pha, denoised_com


# ---- tiny build/example (architecture unmodified from the real repo) ----


class _AttrDict(dict):
    """Matches env.py's `AttrDict`: dict with attribute access, used for `h`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


# Real config.json values from the repo (dense_channel/compress_factor/beta/n_fft/hop_size/
# win_size/sampling_rate), scaled down only in n_fft/hop_size/win_size/dense_channel for a
# fast trace.
_TINY_CONFIG = _AttrDict(
    dense_channel=8,
    compress_factor=0.3,
    num_tsconformers=2,
    beta=2.0,
    sampling_rate=16000,
    n_fft=16,
    hop_size=4,
    win_size=16,
)


def build_mpsenet():
    """Tiny MPNet (dense_channel=8, n_fft=16, 2 TS-transformer blocks) for tracing."""
    torch.manual_seed(0)
    model = MPNet(_TINY_CONFIG, num_tsblocks=2)
    model.eval()
    return model


def example_input_mpsenet():
    """Matches MPNet.forward: noisy magnitude and phase spectra [B, F, T] with
    F = n_fft // 2 + 1."""
    torch.manual_seed(0)
    freq_bins = _TINY_CONFIG.n_fft // 2 + 1
    noisy_amp = torch.rand(1, freq_bins, 6, dtype=torch.float32)
    noisy_pha = (torch.rand(1, freq_bins, 6, dtype=torch.float32) - 0.5) * 2 * np.pi
    return noisy_amp, noisy_pha


MENAGERIE_ENTRIES = [
    ("MP-SENet", "build_mpsenet", "example_input_mpsenet", 2023, MENAGERIE_ZOO),
]

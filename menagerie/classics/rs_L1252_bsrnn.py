# SOURCE: vendored from amanteur/BandSplitRNN-Pytorch @ f3375dc7c99633fd4756f2376a130e80c7b805f8
# https://raw.githubusercontent.com/amanteur/BandSplitRNN-Pytorch/f3375dc7c99633fd4756f2376a130e80c7b805f8/src/model/bandsplitrnn.py
# https://raw.githubusercontent.com/amanteur/BandSplitRNN-Pytorch/f3375dc7c99633fd4756f2376a130e80c7b805f8/src/model/modules/bandsplit.py
# https://raw.githubusercontent.com/amanteur/BandSplitRNN-Pytorch/f3375dc7c99633fd4756f2376a130e80c7b805f8/src/model/modules/bandsequence.py
# https://raw.githubusercontent.com/amanteur/BandSplitRNN-Pytorch/f3375dc7c99633fd4756f2376a130e80c7b805f8/src/model/modules/maskestimation.py
# https://raw.githubusercontent.com/amanteur/BandSplitRNN-Pytorch/f3375dc7c99633fd4756f2376a130e80c7b805f8/src/model/modules/utils.py
#
# Luo & Yu, "Music Source Separation with Band-split RNN" (arXiv:2209.15174) --
# this unofficial-but-widely-used PyTorch implementation (amanteur/BandSplitRNN-Pytorch)
# implements BSRNN exactly: a `BandSplitModule` (1st stage) that partitions the
# complex STFT spectrogram into non-uniform frequency subbands (`freq2bands`,
# computed here via `torch.fft.fftfreq` -- the source's own torch workaround
# for `librosa.fft_frequencies`, so no extra deps needed) and runs each
# subband through per-band LayerNorm+Linear; a `BandSequenceModelModule`
# (2nd/bottleneck stage) of stacked dual-path BiLSTMs alternating across the
# time axis and the subband axis (`RNNModule`, GroupNorm + LSTM + residual
# Linear); and a `MaskEstimationModule` (3rd stage) that reconstructs each
# subband's complex T-F mask via per-band LayerNorm+MLP+GLU. `BandSplitRNN`
# wires the three stages together plus per-sample mean/std normalization
# and multiplies the mask back onto the (optionally complex-as-channel)
# input spectrogram. All four vendored files are base-lib torch only
# (`torch`, `torch.nn`) -- the only mechanical edit is flattening the
# original package's `from model.modules import ...` relative imports into
# this single file (no architecture line altered). `BandSplitRNN.wiener()`
# (real repo: `# TODO: add Wiener Filtering`, an identity no-op stub) and the
# `BandTransformerModelModule` attention-bottleneck alternative
# (`bottleneck_layer='att'`) are included verbatim but the RNN bottleneck
# (`bottleneck_layer='rnn'`) is the one exercised by the tiny build below,
# matching the repo's shipped configs (`bandsplitrnnbass.yaml` /
# `bandsplitrnndrums.yaml` / `bandsplitrnnV7.yaml` all use `bottleneck_layer: rnn`).

from __future__ import annotations

import typing as tp

import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention


# ---------------------------------------------------------------------------
# modules/utils.py -- ported verbatim
# ---------------------------------------------------------------------------
def get_fftfreq(sr: int = 44100, n_fft: int = 2048) -> torch.Tensor:
    """
    Torch workaround of librosa.fft_frequencies
    """
    out = sr * torch.fft.fftfreq(n_fft)[: n_fft // 2 + 1]
    out[-1] = sr // 2
    return out


def get_subband_indices(
    freqs: torch.Tensor,
    splits: tp.List[tp.Tuple[int, int]],
) -> tp.List[tp.Tuple[int, int]]:
    """
    Computes subband frequency indices with given bandsplits
    """
    indices = []
    start_freq, start_index = 0, 0
    for end_freq, step in splits:
        bands = torch.arange(start_freq + step, end_freq + step, step)
        start_freq = end_freq
        for band in bands:
            end_index = freqs[freqs < band].shape[0]
            indices.append((start_index, end_index))
            start_index = end_index
    indices.append((start_index, freqs.shape[0]))
    return indices


def freq2bands(
    bandsplits: tp.List[tp.Tuple[int, int]], sr: int = 44100, n_fft: int = 2048
) -> tp.List[tp.Tuple[int, int]]:
    """
    Returns start and end FFT indices of given bandsplits
    """
    freqs = get_fftfreq(sr=sr, n_fft=n_fft)
    band_indices = get_subband_indices(freqs, bandsplits)
    return band_indices


# ---------------------------------------------------------------------------
# modules/bandsplit.py -- ported verbatim
# ---------------------------------------------------------------------------
class BandSplitModule(nn.Module):
    """
    BandSplit (1st) Module of BandSplitRNN.
    Separates input in k subbands and runs through LayerNorm+FC layers.
    """

    def __init__(
        self,
        sr: int,
        n_fft: int,
        bandsplits: tp.List[tp.Tuple[int, int]],
        t_timesteps: int = 517,
        fc_dim: int = 128,
        complex_as_channel: bool = True,
        is_mono: bool = False,
    ):
        super(BandSplitModule, self).__init__()

        frequency_mul = 1
        if complex_as_channel:
            frequency_mul *= 2
        if not is_mono:
            frequency_mul *= 2

        self.cac = complex_as_channel
        self.is_mono = is_mono
        self.bandwidth_indices = freq2bands(bandsplits, sr, n_fft)
        self.layernorms = nn.ModuleList(
            [
                nn.LayerNorm([(e - s) * frequency_mul, t_timesteps])
                for s, e in self.bandwidth_indices
            ]
        )
        self.fcs = nn.ModuleList(
            [nn.Linear((e - s) * frequency_mul, fc_dim) for s, e in self.bandwidth_indices]
        )

    def generate_subband(self, x: torch.Tensor) -> tp.Iterator[torch.Tensor]:
        for start_index, end_index in self.bandwidth_indices:
            yield x[:, :, start_index:end_index]

    def forward(self, x: torch.Tensor):
        """
        Input: [batch_size, n_channels, freq, time]
        Output: [batch_size, k_subbands, time, fc_output_shape]
        """
        xs = []
        for i, x in enumerate(self.generate_subband(x)):
            B, C, F, T = x.shape
            # view complex as channels
            if x.dtype == torch.cfloat:
                x = torch.view_as_real(x).permute(0, 1, 4, 2, 3)
            # from channels to frequency
            x = x.reshape(B, -1, T)
            # run through model
            x = self.layernorms[i](x)
            x = x.transpose(-1, -2)
            x = self.fcs[i](x)
            xs.append(x)
        return torch.stack(xs, dim=1)


# ---------------------------------------------------------------------------
# modules/bandsequence.py -- ported verbatim
# ---------------------------------------------------------------------------
class RNNModule(nn.Module):
    """
    RNN submodule of BandSequence module
    """

    def __init__(
        self,
        input_dim_size: int,
        hidden_dim_size: int,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
    ):
        super(RNNModule, self).__init__()
        self.groupnorm = nn.GroupNorm(input_dim_size, input_dim_size)
        self.rnn = getattr(nn, rnn_type)(
            input_dim_size, hidden_dim_size, batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(
            hidden_dim_size * 2 if bidirectional else hidden_dim_size, input_dim_size
        )

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            across T - [batch_size, k_subbands, time, n_features]
            OR
            across K - [batch_size, time, k_subbands, n_features]
        """
        B, K, T, N = x.shape  # across T      across K (keep in mind T->K, K->T)

        out = x.view(B * K, T, N)  # [BK, T, N]    [BT, K, N]

        out = self.groupnorm(out.transpose(-1, -2)).transpose(-1, -2)  # [BK, T, N]    [BT, K, N]
        out = self.rnn(out)[0]  # [BK, T, H]    [BT, K, H]
        out = self.fc(out)  # [BK, T, N]    [BT, K, N]

        x = out.view(B, K, T, N) + x  # [B, K, T, N]  [B, T, K, N]

        x = x.permute(0, 2, 1, 3).contiguous()  # [B, T, K, N]  [B, K, T, N]
        return x


class BandSequenceModelModule(nn.Module):
    """
    BandSequence (2nd) Module of BandSplitRNN.
    Runs input through n BiLSTMs in two dimensions - time and subbands.
    """

    def __init__(
        self,
        input_dim_size: int,
        hidden_dim_size: int,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        num_layers: int = 12,
    ):
        super(BandSequenceModelModule, self).__init__()

        self.bsrnn = nn.ModuleList([])

        for _ in range(num_layers):
            rnn_across_t = RNNModule(input_dim_size, hidden_dim_size, rnn_type, bidirectional)
            rnn_across_k = RNNModule(input_dim_size, hidden_dim_size, rnn_type, bidirectional)
            self.bsrnn.append(nn.Sequential(rnn_across_t, rnn_across_k))

    def forward(self, x: torch.Tensor):
        """
        Input shape: [batch_size, k_subbands, time, n_features]
        Output shape: [batch_size, k_subbands, time, n_features]
        """
        for i in range(len(self.bsrnn)):
            x = self.bsrnn[i](x)
        return x


# ---------------------------------------------------------------------------
# modules/bandtransformer.py -- ported verbatim (alternative bottleneck; not
# exercised by the build below, kept for architectural completeness since
# BandSplitRNN.__init__ can construct it for bottleneck_layer='att')
# ---------------------------------------------------------------------------
class TransformerModule(nn.Module):
    """
    Transformer module based on Dual-Path Transformer paper [1].
    Almost the same as in https://github.com/asteroid-team/asteroid/blob/master/asteroid/masknn/attention.py

    References
        [1] Chen, Jingjing, Qirong Mao, and Dong Liu. "Dual-Path Transformer
        Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation."
        arXiv (2020).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        dim_ff: int = 512,
        n_heads: int = 4,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super(TransformerModule, self).__init__()

        self.groupnorm = nn.GroupNorm(embed_dim, embed_dim)
        self.mha = MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.recurrent = nn.LSTM(embed_dim, dim_ff, bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(2 * dim_ff if bidirectional else dim_ff, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            across T - [batch_size, k_subbands, time, n_features]
            OR
            across K - [batch_size, time, k_subbands, n_features]
        """
        B, K, T, N = x.shape  # across T, across K - keep in mind T->K, K->T

        x = x.view(B * K, T, N)  # [BK, T, N] across T,      [BT, K, N] across K

        # groupnorm (result unused -- verbatim from upstream source, a
        # pre-existing dead-store in the real repo's TransformerModule.forward;
        # not corrected here since this is a vendored faithful copy)
        out = self.groupnorm(  # noqa: F841
            x.transpose(-1, -2)
        ).transpose(-1, -2)  # [BK, T, N]    [BT, K, N]

        # Attention
        mha_in = x.transpose(0, 1)
        mha_out, _ = self.mha(mha_in, mha_in, mha_in)
        x = mha_out.transpose(0, 1) + x

        # RNN
        rnn_out, _ = self.recurrent(x)
        x = self.linear(rnn_out) + x

        # returning to the initial shape
        x = x.view(B, K, T, N)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x


class BandTransformerModelModule(nn.Module):
    """
    Modified BandSequence (2nd) Module of BandSplitRNN.
    Runs input through n Transformers in two dimensions - time and subbands.
    """

    def __init__(
        self,
        input_dim_size: int,
        hidden_dim_size: int,
        num_layers: int = 6,
    ):
        super(BandTransformerModelModule, self).__init__()

        self.dptransformers = nn.ModuleList([])

        for _ in range(num_layers):
            transformer_across_t = TransformerModule(input_dim_size, hidden_dim_size)
            transformer_across_k = TransformerModule(input_dim_size, hidden_dim_size)
            self.dptransformers.append(nn.Sequential(transformer_across_t, transformer_across_k))

    def forward(self, x: torch.Tensor):
        """
        Input shape: [batch_size, k_subbands, time, n_features]
        Output shape: [batch_size, k_subbands, time, n_features]
        """
        for i in range(len(self.dptransformers)):
            x = self.dptransformers[i](x)
        return x


# ---------------------------------------------------------------------------
# modules/maskestimation.py -- ported verbatim
# ---------------------------------------------------------------------------
class GLU(nn.Module):
    """
    GLU Activation Module.
    """

    def __init__(self, input_dim: int):
        super(GLU, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, input_dim * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = x[..., : self.input_dim] * self.sigmoid(x[..., self.input_dim :])
        return x


class MLP(nn.Module):
    """
    Just a simple MLP with tanh activation (by default).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation_type: str = "tanh",
    ):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.select_activation(activation_type)(),
            nn.Linear(hidden_dim, output_dim),
            GLU(output_dim),
        )

    @staticmethod
    def select_activation(activation_type: str) -> nn.modules.activation:
        if activation_type == "tanh":
            return nn.Tanh
        elif activation_type == "relu":
            return nn.ReLU
        elif activation_type == "gelu":
            return nn.GELU
        else:
            raise ValueError("wrong activation function was selected")

    def forward(self, x: torch.Tensor):
        x = self.mlp(x)
        return x


class MaskEstimationModule(nn.Module):
    """
    MaskEstimation (3rd) Module of BandSplitRNN.
    Recreates from input initial subband dimensionality via running through LayerNorms+MLPs and forms the T-F mask.
    """

    def __init__(
        self,
        sr: int,
        n_fft: int,
        bandsplits: tp.List[tp.Tuple[int, int]],
        t_timesteps: int = 517,
        fc_dim: int = 128,
        mlp_dim: int = 512,
        complex_as_channel: bool = True,
        is_mono: bool = False,
    ):
        super(MaskEstimationModule, self).__init__()

        frequency_mul = 1
        if complex_as_channel:
            frequency_mul *= 2
        if not is_mono:
            frequency_mul *= 2

        self.cac = complex_as_channel
        self.is_mono = is_mono
        self.frequency_mul = frequency_mul

        self.bandwidths = [(e - s) for s, e in freq2bands(bandsplits, sr, n_fft)]
        self.layernorms = nn.ModuleList(
            [nn.LayerNorm([t_timesteps, fc_dim]) for _ in range(len(self.bandwidths))]
        )
        self.mlp = nn.ModuleList(
            [
                MLP(fc_dim, mlp_dim, bw * frequency_mul, activation_type="tanh")
                for bw in self.bandwidths
            ]
        )

    def forward(self, x: torch.Tensor):
        """
        Input: [batch_size, k_subbands, time, fc_shape]
        Output: [batch_size, freq, time]
        """
        outs = []
        for i in range(x.shape[1]):
            # run through model
            out = self.layernorms[i](x[:, i])
            out = self.mlp[i](out)
            B, T, F = out.shape
            # return to complex
            if self.cac:
                out = out.permute(0, 2, 1).contiguous()
                out = out.view(B, -1, 2, F // self.frequency_mul, T).permute(0, 1, 3, 4, 2)
                out = torch.view_as_complex(out.contiguous())
            else:
                out = out.view(B, -1, F // self.frequency_mul, T).contiguous()
            outs.append(out)

        # concat all subbands
        outs = torch.cat(outs, dim=-2)
        return outs


# ---------------------------------------------------------------------------
# model/bandsplitrnn.py -- ported verbatim
# ---------------------------------------------------------------------------
class BandSplitRNN(nn.Module):
    """
    BandSplitRNN as described in paper.
    """

    def __init__(
        self,
        sr: int,
        n_fft: int,
        bandsplits: tp.List[tp.Tuple[int, int]],
        complex_as_channel: bool,
        is_mono: bool,
        bottleneck_layer: str,
        t_timesteps: int,
        fc_dim: int,
        rnn_dim: int,
        rnn_type: str,
        bidirectional: bool,
        num_layers: int,
        mlp_dim: int,
        return_mask: bool = False,
    ):
        super(BandSplitRNN, self).__init__()

        # encoder layer
        self.bandsplit = BandSplitModule(
            sr=sr,
            n_fft=n_fft,
            bandsplits=bandsplits,
            t_timesteps=t_timesteps,
            fc_dim=fc_dim,
            complex_as_channel=complex_as_channel,
            is_mono=is_mono,
        )

        # bottleneck layer
        if bottleneck_layer == "rnn":
            self.bandsequence = BandSequenceModelModule(
                input_dim_size=fc_dim,
                hidden_dim_size=rnn_dim,
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                num_layers=num_layers,
            )
        elif bottleneck_layer == "att":
            self.bandsequence = BandTransformerModelModule(
                input_dim_size=fc_dim,
                hidden_dim_size=rnn_dim,
                num_layers=num_layers,
            )
        else:
            raise NotImplementedError

        # decoder layer
        self.maskest = MaskEstimationModule(
            sr=sr,
            n_fft=n_fft,
            bandsplits=bandsplits,
            t_timesteps=t_timesteps,
            fc_dim=fc_dim,
            mlp_dim=mlp_dim,
            complex_as_channel=complex_as_channel,
            is_mono=is_mono,
        )
        self.cac = complex_as_channel
        self.return_mask = return_mask

    def wiener(self, x_hat: torch.Tensor, x_complex: torch.Tensor) -> torch.Tensor:
        """
        Wiener filtering of the input signal
        """
        # TODO: add Wiener Filtering
        return x_hat

    def compute_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes complex-valued T-F mask.
        """
        x = self.bandsplit(x)  # [batch_size, k_subbands, time, fc_dim]
        x = self.bandsequence(x)  # [batch_size, k_subbands, time, fc_dim]
        x = self.maskest(x)  # [batch_size, freq, time]

        return x

    def forward(self, x: torch.Tensor):
        """
        Input and output are T-F complex-valued features.
        Input shape: batch_size, n_channels, freq, time]
        Output shape: batch_size, n_channels, freq, time]
        """
        # use only magnitude if not using complex input
        x_complex = None
        if not self.cac:
            x_complex = x
            x = x.abs()
        # normalize
        # TODO: Try to normalize in bandsplit and denormalize in maskest
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (std + 1e-5)

        # compute T-F mask
        mask = self.compute_mask(x)

        # multiply with original tensor
        x = mask if self.return_mask else mask * x

        # denormalize
        x = x * std + mean

        if not self.cac:
            x = self.wiener(x, x_complex)

        return x


# ---------------------------------------------------------------------------
# staging build (tiny sizes; real repo default cfg shown in bandsplitrnn.py's
# __main__ block: sr=44100, n_fft=2048, 5 bandsplits, fc_dim=128, rnn_dim=256,
# num_layers=1..12). `sr`/`n_fft` are kept at the real repo's values (needed
# for `freq2bands`'s FFT-bin math to produce sane non-degenerate subbands);
# `bandsplits` is coarsened to 2 splits (5 resulting subbands, matching real
# freq2bands geometry) and t_timesteps/fc_dim/rnn_dim/mlp_dim/num_layers are
# shrunk for a fast trace-verification build. bottleneck_layer='rnn' (the
# config the repo's shipped YAMLs all use) and complex_as_channel=True
# (cfloat input) kept exactly as the repo's own __main__ smoke test.
# ---------------------------------------------------------------------------
def build_bsrnn() -> BandSplitRNN:
    cfg = {
        "sr": 44100,
        "n_fft": 2048,
        "bandsplits": [
            (2000, 1000),
            (8000, 4000),
        ],
        "complex_as_channel": True,
        "is_mono": False,
        "bottleneck_layer": "rnn",
        "t_timesteps": 4,
        "fc_dim": 8,
        "rnn_dim": 8,
        "rnn_type": "LSTM",
        "bidirectional": True,
        "num_layers": 1,
        "mlp_dim": 16,
        "return_mask": False,
    }
    return BandSplitRNN(**cfg).eval()


def example_input_bsrnn():
    batch_size, n_channels, freq, time = 1, 2, 1025, 4
    return torch.rand(batch_size, n_channels, freq, time, dtype=torch.cfloat)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("BSRNN", "build_bsrnn", "example_input_bsrnn", 2022, "vendored-pytorch"),
]

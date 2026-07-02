# SOURCE: vendored from https://github.com/plumprc/MTS-Mixers @ main
# (models/MTSMixer.py::MLPBlock/FactorizedTemporalMixing/FactorizedChannelMixing/
#  MixerBlock/Model + layers/Invertible.py::RevIN + layers/Projection.py::
#  ChannelProjection)
#
# MTS-Mixers (Li et al., 2023, "MTS-Mixers: Multivariate Time Series Forecasting via
# Factorized Temporal and Channel Mixing"): an all-MLP time-series forecaster in the
# MLP-Mixer family. Alternates factorized token-mixing (per-subsampled-phase MLPs
# applied along the time axis, "FactorizedTemporalMixing") with factorized
# channel-mixing (a low-rank MLP bottleneck along the channel axis,
# "FactorizedChannelMixing") across `e_layers` MixerBlocks, wrapped in optional
# Reversible Instance Normalization (RevIN) and a final per-channel/shared linear
# projection from `seq_len` to `pred_len`.
#
# Vendored real repo code verbatim: MLPBlock, FactorizedTemporalMixing,
# FactorizedChannelMixing, MixerBlock, Model (models/MTSMixer.py); RevIN (layers/
# Invertible.py); ChannelProjection (layers/Projection.py). Every Linear/GELU/
# LayerNorm layer, factorization/sampling scheme, and the norm -> mix -> project ->
# denorm forward path is unchanged from the original. Only non-architectural
# scaffolding was dropped: the unused `svd_denoise`/`NMF` decomposition-utility import
# (utils/decomposition.py provides optional preprocessing helpers the real `Model`
# imports but never calls in forward()) and the commented-out `refine`/plain-Linear
# `projection` alternates already dead in the source. `configs` here is a plain
# `types.SimpleNamespace` standing in for the real repo's argparse Namespace (same
# field names: seq_len, pred_len, enc_in, d_model, d_ff, fac_T, fac_C, sampling,
# norm, individual, e_layers, rev) -- no architectural difference, just how the
# hyperparameters are passed in.

from types import SimpleNamespace

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --------------------------------------------------------------------------------
# layers/Invertible.py
# --------------------------------------------------------------------------------


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean

        return x


# --------------------------------------------------------------------------------
# layers/Projection.py
# --------------------------------------------------------------------------------


class ChannelProjection(nn.Module):
    def __init__(self, seq_len, pred_len, num_channel, individual):
        super().__init__()

        self.linears = (
            nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(num_channel)])
            if individual
            else nn.Linear(seq_len, pred_len)
        )
        self.individual = individual

    def forward(self, x):
        # x: [B, L, D]
        x_out = []
        if self.individual:
            for idx in range(x.shape[-1]):
                x_out.append(self.linears[idx](x[:, :, idx]))

            x = torch.stack(x_out, dim=-1)
        else:
            x = self.linears(x.transpose(1, 2)).transpose(1, 2)

        return x


# --------------------------------------------------------------------------------
# models/MTSMixer.py
# --------------------------------------------------------------------------------


class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling):
        super().__init__()

        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList(
            [MLPBlock(input_dim // sampling, mlp_dim) for _ in range(sampling)]
        )

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx :: self.sampling] = x_pad

        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx :: self.sampling]))

        x = self.merge(x.shape, x_samp)

        return x


class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim):
        super().__init__()

        assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):
        return self.channel_mixing(x)


class MixerBlock(nn.Module):
    def __init__(
        self,
        tokens_dim,
        channels_dim,
        tokens_hidden_dim,
        channels_hidden_dim,
        fac_T,
        fac_C,
        sampling,
        norm_flag,
    ):
        super().__init__()
        self.tokens_mixing = (
            FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling)
            if fac_T
            else MLPBlock(tokens_dim, tokens_hidden_dim)
        )
        self.channels_mixing = (
            FactorizedChannelMixing(channels_dim, channels_hidden_dim) if fac_C else None
        )
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self, x):
        # token-mixing [B, D, #tokens]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        # channel-mixing [B, #tokens, D]
        if self.channels_mixing:
            y = y + x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y)

        return y


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.mlp_blocks = nn.ModuleList(
            [
                MixerBlock(
                    configs.seq_len,
                    configs.enc_in,
                    configs.d_model,
                    configs.d_ff,
                    configs.fac_T,
                    configs.fac_C,
                    configs.sampling,
                    configs.norm,
                )
                for _ in range(configs.e_layers)
            ]
        )
        self.norm = nn.LayerNorm(configs.enc_in) if configs.norm else None
        self.projection = ChannelProjection(
            configs.seq_len, configs.pred_len, configs.enc_in, configs.individual
        )
        self.rev = RevIN(configs.enc_in) if configs.rev else None

    def forward(self, x):
        x = self.rev(x, "norm") if self.rev else x

        for block in self.mlp_blocks:
            x = block(x)

        x = self.norm(x) if self.norm else x
        x = self.projection(x)
        x = self.rev(x, "denorm") if self.rev else x

        return x


def build_mts_mixers():
    # Tiny config matching the real repo's default field names (scripts/*.sh use
    # e.g. seq_len=336, pred_len=96, enc_in=7/21, d_model=..., sampling in
    # {1,2,3,4,6,8,12}); shrunk here for a fast, small trace. sampling=6 must
    # evenly divide seq_len=12 per FactorizedTemporalMixing's assert; enc_in
    # (channels_dim) must exceed d_ff (channels_hidden_dim, the factorized-mixing
    # bottleneck width) per FactorizedChannelMixing's assert.
    configs = SimpleNamespace(
        seq_len=12,
        pred_len=6,
        enc_in=8,
        d_model=8,
        d_ff=4,
        fac_T=True,
        fac_C=True,
        sampling=6,
        norm=True,
        individual=False,
        e_layers=2,
        rev=True,
    )
    return Model(configs)


def example_input_mts_mixers():
    # x: [B, seq_len, enc_in]
    return torch.randn(2, 12, 8)


MENAGERIE_ENTRIES = [
    ("MTS-Mixers", build_mts_mixers, example_input_mts_mixers, 2023, MENAGERIE_ZOO),
]

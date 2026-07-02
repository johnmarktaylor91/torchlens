# FAITHFUL PORT of dianixn/Channelformer @ main (original framework: MATLAB / Deep
# Learning Toolbox custom dlarray code)
#
# Source model files transcribed (paths in the original repo):
#   +transformer/model.m                       -- top-level Channelformer forward
#   +transformer/+HA03/Encoder_block.m         -- self-attention encoder block
#   +transformer/+HA03/Decoder_block.m         -- residual-conv decoder block
#   +transformer/+layer/attention.m            -- multi-head QKV projection + split
#   +transformer/+layer/multiheadAttention.m   -- scaled dot-product attention core
#   +transformer/+layer/FeedforwardNN.m        -- encoder position-wise conv FFN
#   +transformer/+layer/normalization.m        -- per-position (feature-axis) LayerNorm
#   +transformer/+layer/gelu.m                 -- tanh-approximate GELU
#   +transformer/+layer/FC1.m                  -- dlmtimes(W,X)+b "1x1" projection
#   +Parameter/parameters_hybrid.m             -- weight shapes / hyperparameters
#   +Training/Training_hybrid_offline.m        -- NumHeads=6, Encoder_num_layers=1,
#                                                  Decoder_num_layers=3
#   +Parameter/parameters.m                    -- Num_of_FFT=72 => Feature_size=72
#
# This repo (https://github.com/dianixn/Channelformer, paper: Luan & Thompson, IEEE
# TWC 2023) ships ONLY MATLAB (.m) source using MATLAB's Deep Learning Toolbox
# `dlarray`/`dlconv`/`dlmtimes` custom-training-loop primitives; there is no PyTorch
# (or any Python deep-learning-framework) implementation anywhere in the repo or its
# dependents, so the MATLAB->base-torch environment cannot be installed (rung 2 is
# unavailable) and this is transcribed as a faithful port (rung 3) instead.
#
# MATLAB data layout is "SSCB" = [Spatial1, Spatial2, Channel, Batch], which for
# this model is [Feature_size(=Num_of_FFT=72), Time/OFDM-symbol-position, 1, Batch].
# `dlconv(..., 'DataFormat','SSCB')` maps directly onto PyTorch NCHW Conv2d with
# N=Batch, C=Channel, H=Feature_size, W=Time. `FC1`/`dlmtimes(W,X)` is a
# position-wise linear projection applied along the Feature_size axis, independent
# per (Time, Channel, Batch) position -- ported here as a 1x1 Conv2d over the H axis
# (equivalently a Linear applied per spatial position), which is exactly what the
# MATLAB per-position matmul computes.
#
# The self-attention block folds the Time axis together with the split-head axis
# per iSplitHeads/iMergeHeads' MATLAB *column-major* reshape (verified empirically
# against the actual reshape/permute index arithmetic); attention is computed over
# the folded (Time, Batch) span per head, which the original code's batched
# `dlmtimes` performs by iterating the two trailing (Channel, Batch) dims (Channel
# is 1 here) -- i.e. per Batch element, attention is computed over Time positions,
# matching a standard per-example self-attention over the Time/OFDM-symbol axis.
# That is the faithful behavior this port reproduces (per-batch-element attention
# over the Time axis, independently for each example in the batch).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    # +transformer/+layer/gelu.m : 0.5*X.*(1+tanh(sqrt(2/pi)*(X+0.044715*X.^3)))
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class PositionwiseProjection(nn.Module):
    """+transformer/+layer/FC1.m : Z = dlmtimes(W, X) + b

    A position-wise linear projection applied along the Feature_size (H) axis,
    independently at every (Time, Batch) position -- implemented as a 1x1 Conv2d
    over NCHW-format [B, C=1, H=Feature_size, W=Time] tensors, exactly matching
    the MATLAB per-position matmul.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # FC1 in Channelformer applies a [out_features, in_features] weight matrix
        # along the Feature_size (H) axis at every (Time, Batch) position, exactly
        # matching dlmtimes(W, X) + b -- represented faithfully as a Linear over H.
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H_in, W] -> apply Linear over H -> [B, C, H_out, W]
        x = x.permute(0, 1, 3, 2)  # [B, C, W, H_in]
        x = self.linear(x)  # [B, C, W, H_out]
        x = x.permute(0, 1, 3, 2)  # [B, C, H_out, W]
        return x


class FeaturePositionLayerNorm(nn.Module):
    """+transformer/+layer/normalization.m

    U = mean(X, dim=1); S = mean((X-U)^2, dim=1); X = (X-U)/sqrt(S+eps); Z = g.*X+b
    Normalizes over the Feature_size (H) axis, with a per-position (per-H-index)
    affine g, b of shape [Feature_size, 1] (broadcast over Time/Batch/Channel).
    """

    def __init__(self, feature_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, 1, feature_size, 1))
        self.b = nn.Parameter(torch.zeros(1, 1, feature_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]; normalize across H (dim=2)
        u = x.mean(dim=2, keepdim=True)
        s = ((x - u) ** 2).mean(dim=2, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.g * x + self.b


class MultiHeadSelfAttention(nn.Module):
    """+transformer/+layer/attention.m + multiheadAttention.m

    QKV projected along the Feature_size (H) axis via FC1, split into NumHeads
    along H, attention computed over the Time (W) axis per batch element per head
    (see module docstring for the MATLAB fold-semantics this reproduces), then
    heads merged and re-projected along H via FC1.
    """

    def __init__(self, feature_size: int, num_heads: int):
        super().__init__()
        assert feature_size % num_heads == 0
        self.feature_size = feature_size
        self.num_heads = num_heads
        self.head_dim = feature_size // num_heads

        self.c_attn = PositionwiseProjection(feature_size, 3 * feature_size)
        self.c_proj = PositionwiseProjection(feature_size, feature_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C=1, H=Feature_size, W=Time]
        b, c, h, w = x.shape

        qkv = self.c_attn(x)  # [B, C, 3*Feature_size, W]
        q, k, v = qkv.split(self.feature_size, dim=2)  # each [B, C, Feature_size, W]

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # [B, C, Feature_size, W] -> [B, C, num_heads, head_dim, W]
            t = t.view(b, c, self.num_heads, self.head_dim, w)
            # -> [B, C, num_heads, W, head_dim] for standard attention matmul
            return t.permute(0, 1, 2, 4, 3)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # scaled dot-product attention over the Time (W) axis, per (B, C, head)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # [B, C, num_heads, W, head_dim]

        # merge heads back to Feature_size
        out = out.permute(0, 1, 4, 2, 3)  # [B, C, head_dim, num_heads, W]
        out = out.reshape(b, c, self.feature_size, w)

        return self.c_proj(out)


class ChannelformerEncoderBlock(nn.Module):
    """+transformer/+HA03/Encoder_block.m"""

    def __init__(self, feature_size: int, num_heads: int, num_filters: int = 5):
        super().__init__()
        self.attn = MultiHeadSelfAttention(feature_size, num_heads)
        self.ln_1 = FeaturePositionLayerNorm(feature_size)
        # +transformer/+layer/FeedforwardNN.m : two "same"-padded 3x3 conv layers
        # over the [1 -> num_filters -> 1] channel axis, GELU between them.
        self.mlp_fc = nn.Conv2d(1, num_filters, kernel_size=3, padding="same")
        self.mlp_proj = nn.Conv2d(num_filters, 1, kernel_size=3, padding="same")
        self.ln_2 = FeaturePositionLayerNorm(feature_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attn(x)
        a = a + x
        a = self.ln_1(a)

        z = self.mlp_fc(a)
        z = gelu_tanh(z)
        z = self.mlp_proj(z)

        z = z + a
        z = self.ln_2(z)
        return z


class ChannelformerDecoderBlock(nn.Module):
    """+transformer/+HA03/Decoder_block.m"""

    def __init__(self, feature_size: int, num_filters: int = 12):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=5, padding="same")
        self.ln = FeaturePositionLayerNorm(feature_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = F.relu(y)
        y = self.conv2(y)
        y = y + x
        y = self.ln(y)
        return y


class Channelformer(nn.Module):
    """+transformer/model.m -- full Channelformer: HA03 self-attention encoder
    (Encoder_num_layers blocks) followed by a residual-convolutional decoder
    (Decoder_num_layers residual blocks + input/output projection convs), as used
    for OFDM downlink channel estimation.
    """

    def __init__(
        self,
        feature_size: int = 72,
        num_heads: int = 6,
        encoder_num_layers: int = 1,
        decoder_num_layers: int = 3,
        decoder_num_filters: int = 12,
        output_feature_size: int | None = None,
    ):
        super().__init__()
        self.feature_size = feature_size
        output_feature_size = output_feature_size or feature_size

        self.encoder_layers = nn.ModuleList(
            [ChannelformerEncoderBlock(feature_size, num_heads) for _ in range(encoder_num_layers)]
        )

        # decoder input projection: 1 -> num_filters, 5x5 "same" conv
        self.decoder_in_conv = nn.Conv2d(1, decoder_num_filters, kernel_size=5, padding="same")

        self.decoder_layers = nn.ModuleList(
            [
                ChannelformerDecoderBlock(feature_size, decoder_num_filters)
                for _ in range(decoder_num_layers)
            ]
        )

        # final FC1 projection along the Feature_size axis (Feature_size -> output_feature_size)
        self.decoder_out_fc = PositionwiseProjection(feature_size, output_feature_size)

        # final 5x5 "same" conv: num_filters -> 1
        self.decoder_out_conv = nn.Conv2d(decoder_num_filters, 1, kernel_size=5, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, Feature_size, Time]
        z = x
        for layer in self.encoder_layers:
            z = layer(z)

        z = self.decoder_in_conv(z)

        for layer in self.decoder_layers:
            z = layer(z)

        z = self.decoder_out_fc(z)
        z = self.decoder_out_conv(z)
        return z


# --- menagerie staging entrypoints -----------------------------------------------

MENAGERIE_ZOO = "ported-pytorch"


def build_channelformer():
    # Feature_size=72 (Num_of_FFT), NumHeads=6, Encoder_num_layers=1,
    # Decoder_num_layers=3, Number_of_filters(decoder)=12 -- all straight from
    # +Training/Training_hybrid_offline.m and +Parameter/parameters_hybrid.m.
    return Channelformer(
        feature_size=72,
        num_heads=6,
        encoder_num_layers=1,
        decoder_num_layers=3,
        decoder_num_filters=12,
    )


def example_input_channelformer():
    # [Batch, Channel=1, Feature_size=72, Time=Frame_size(=Num_of_symbols+Num_of_pilot=14)]
    return torch.rand(2, 1, 72, 14)


MENAGERIE_ENTRIES = [
    ("ChannelFormer", build_channelformer, example_input_channelformer, 2023, MENAGERIE_ZOO),
]

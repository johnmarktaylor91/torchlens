# FAITHFUL PORT of dianixn/Channelformer @ main (original framework: MATLAB Deep Learning
# Toolbox, custom dlarray/dlnetwork training loop -- no Python/PyTorch model code shipped)
# https://github.com/dianixn/Channelformer  (Luan & Thompson, "Channelformer: Attention
# based Neural Solution for Wireless Channel Estimation and Effective Online Training",
# IEEE TWC 2023)
# The repo's model definition is entirely MATLAB (+transformer/model.m, +transformer/+HA03/
# {Encoder_block,Decoder_block}.m, +transformer/+layer/{attention,multiheadAttention,
# FeedforwardNN,normalization,gelu,FC1}.m). MATLAB's Deep Learning Toolbox with a custom
# `dlfeval`/`dlarray` training loop cannot be installed alongside torch in this environment,
# so the architecture is transcribed faithfully into torch (rung 3) directly from the
# released .m files (fetched from the `main` branch), preserving every mechanism and the
# exact hyperparameters used by `+Training/Training_hybrid_offline.m` (offline Channelformer
# / HA03 variant): NumHeads=6, Encoder_num_layers=1, Decoder_num_layers=3.
#
# MATLAB tensors use the 'SSCB' (Spatial1, Spatial2, Channel, Batch) `dlarray` format
# throughout; this port keeps the same 4 logical axes as torch dims
# [Spatial1=Feature, Spatial2=IQ(2), Channel, Batch] and transcribes each `.m` file 1:1:
#
#   +transformer/+layer/FC1.m: Z = dlmtimes(W, X) + b -- a matmul contracting the
#     Spatial1 (Feature) axis only, broadcast over Spatial2/Channel/Batch. Ported as
#     `torch.einsum("oi,i...->o...", W, X) + b` (b broadcast over Spatial1).
#
#   +transformer/+layer/attention.m: C = FC1(X, attn_c_attn_w, attn_c_attn_b) splits into
#     Q/K/V along the (post-FC1) Feature axis (size 3F -> 3 x F), each reshaped by
#     iSplitHeads (splits F into [F/numHeads, numHeads] and moves numHeads next to the
#     Spatial2=IQ axis, i.e. attention keys/queries attend over the IQ(2) sequence
#     position, with F/numHeads as the per-head embedding dim), scaled dot-product via
#     multiheadAttention.m (QK^T / sqrt(d), softmax over the key axis, then V), heads
#     merged back (iMergeHeads), then FC1(attn_c_proj_w/b) projects back to F.
#
#   +transformer/+HA03/Encoder_block.m: A = attention(X); A = A + permute(X); A =
#     normalization(A, ln_1_g_0, ln_1_b_0); Z = FeedforwardNN(A); Z = Z + A; Z =
#     normalization(Z, ln_2_g_0, ln_2_b_0).  Post-norm residual transformer block
#     (norm AFTER the residual add, not pre-norm).
#
#   +transformer/+layer/FeedforwardNN.m: two `dlconv` 2D convolutions (3x3, 'same'
#     padding, 1 input channel -> 5 filters -> 1 filter) applied directly to the
#     [Feature, IQ, Channel=1, Batch] tensor (i.e. a small 2D CNN over the
#     Feature x IQ grid, not a per-token MLP), with `gelu` (tanh approximation, exactly
#     as coded in +transformer/+layer/gelu.m) between the two convs.
#
#   +transformer/+layer/normalization.m: per-example LayerNorm over the Spatial1
#     (Feature) axis only (`normalizationDimension = 1`), with per-Feature affine
#     gain/bias g, b (shape [Feature, 1], broadcasting over IQ/Channel/Batch).
#
#   +transformer/model.m (HA03 decoder): after the encoder stack, a 5x5 'same' conv
#     (1 -> 12 filters, `ln_de_w`/`ln_de_b`) projects Channel 1->12, then
#     `Decoder_num_layers` (=3) HA03/Decoder_block.m residual conv blocks run:
#     conv5x5(12->12) -> ReLU -> conv5x5(12->12) -> += input -> normalization
#     (over Feature axis, gain/bias `ln_de_w3`/`ln_de_b3`). A final regression head
#     (+transformer/model.m tail) applies FC1 (`ln_de_w1`/`ln_de_b1`, Feature ->
#     Training_Y's Feature dim) then a 5x5 'same' conv projecting Channel 12->1
#     (`ln_de_w0`/`ln_de_b0`), producing the estimated (real, imag) channel response.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _fc1(x, w, b):
    """+transformer/+layer/FC1.m: Z = dlmtimes(W, X) + b.

    x: [..., F_in, S2, C, B] (Spatial1=F_in leading logical axis).  Here we keep x as
    [B, C, S2, F_in] (torch-native layout) and contract the last dim (F_in) with W
    ([F_out, F_in]), matching `dlmtimes(W, X)` which matrix-multiplies over the
    Spatial1 axis and broadcasts over Spatial2/Channel/Batch.
    """
    return torch.einsum("oi,bcsi->bcso", w, x) + b.view(1, 1, 1, -1)


def _gelu_matlab(x):
    """+transformer/+layer/gelu.m: tanh-approximation GELU, transcribed exactly."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _layernorm_feature_axis(x, g, b, eps=1e-5):
    """+transformer/+layer/normalization.m: mean/var over the Feature (Spatial1) axis
    only, per (Batch, Channel, Spatial2) example, then affine gain/bias per-Feature.

    x: [B, C, S2, F]; g, b: [F].
    """
    u = x.mean(dim=-1, keepdim=True)
    s = ((x - u) ** 2).mean(dim=-1, keepdim=True)
    x = (x - u) / torch.sqrt(s + eps)
    return g.view(1, 1, 1, -1) * x + b.view(1, 1, 1, -1)


class MultiHeadAttention(nn.Module):
    """+transformer/+layer/attention.m + multiheadAttention.m, HA03 encoder variant."""

    def __init__(self, feature_size, num_heads):
        super().__init__()
        assert feature_size % num_heads == 0
        self.feature_size = feature_size
        self.num_heads = num_heads
        self.head_dim = feature_size // num_heads

        self.attn_c_attn_w = nn.Parameter(torch.empty(3 * feature_size, feature_size))
        self.attn_c_attn_b = nn.Parameter(torch.zeros(3 * feature_size))
        self.attn_c_proj_w = nn.Parameter(torch.empty(feature_size, feature_size))
        self.attn_c_proj_b = nn.Parameter(torch.zeros(feature_size))
        nn.init.xavier_uniform_(self.attn_c_attn_w)
        nn.init.xavier_uniform_(self.attn_c_proj_w)

    def _split_heads(self, x):
        # x: [B, C, S2, F] -> split F into (head_dim, num_heads), move num_heads next
        # to the S2 (IQ) axis, matching iSplitHeads' reshape+permute in MATLAB.
        b, c, s2, f = x.shape
        x = x.reshape(b, c, s2, self.head_dim, self.num_heads)
        return x.permute(0, 1, 4, 2, 3)  # [B, C, numHeads, S2, head_dim]

    def _merge_heads(self, x):
        # inverse of _split_heads: [B, C, numHeads, S2, head_dim] -> [B, C, S2, F]
        b, c, nh, s2, hd = x.shape
        x = x.permute(0, 1, 3, 2, 4)  # [B, C, S2, numHeads, head_dim]
        return x.reshape(b, c, s2, nh * hd)

    def forward(self, x):
        # x: [B, C, S2, F]
        c = _fc1(x, self.attn_c_attn_w, self.attn_c_attn_b)  # [B, C, S2, 3F]
        q, k, v = c.split(self.feature_size, dim=-1)

        q = self._split_heads(q)  # [B, C, nh, S2, hd]
        k = self._split_heads(k)
        v = self._split_heads(v)

        # multiheadAttention.m: W = K^T Q / sqrt(d); softmax over key axis; A = V W
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)  # [B, C, nh, S2, hd]

        merged = self._merge_heads(attn)  # [B, C, S2, F]
        return _fc1(merged, self.attn_c_proj_w, self.attn_c_proj_b)


class FeedforwardNN(nn.Module):
    """+transformer/+layer/FeedforwardNN.m: two 3x3 'same' 2D convs over the
    (Feature, IQ) grid (1 -> 5 -> 1 channel), with MATLAB-gelu between them."""

    def __init__(self):
        super().__init__()
        self.mlp_c_fc = nn.Conv2d(1, 5, kernel_size=3, padding="same")
        self.mlp_c_proj = nn.Conv2d(5, 1, kernel_size=3, padding="same")

    def forward(self, x):
        # x: [B, C=1, S2, F] treated directly as a [B, 1, S2, F] image for dlconv.
        z = self.mlp_c_fc(x)
        z = _gelu_matlab(z)
        z = self.mlp_c_proj(z)
        return z


class EncoderBlockHA03(nn.Module):
    """+transformer/+HA03/Encoder_block.m: post-norm residual transformer block."""

    def __init__(self, feature_size, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(feature_size, num_heads)
        self.ln1_g = nn.Parameter(torch.zeros(feature_size))
        self.ln1_b = nn.Parameter(torch.zeros(feature_size))
        self.ffn = FeedforwardNN()
        self.ln2_g = nn.Parameter(torch.zeros(feature_size))
        self.ln2_b = nn.Parameter(torch.zeros(feature_size))
        nn.init.xavier_uniform_(self.ln1_g.view(1, -1))
        nn.init.xavier_uniform_(self.ln2_g.view(1, -1))

    def forward(self, x):
        a = self.attn(x)
        a = a + x
        a = _layernorm_feature_axis(a, self.ln1_g, self.ln1_b)

        z = self.ffn(a)
        z = z + a
        z = _layernorm_feature_axis(z, self.ln2_g, self.ln2_b)
        return z


class DecoderBlockHA03(nn.Module):
    """+transformer/+HA03/Decoder_block.m: 5x5 conv residual block."""

    def __init__(self, num_filters, feature_size):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=5, padding="same")
        self.ln_g = nn.Parameter(torch.zeros(feature_size))
        self.ln_b = nn.Parameter(torch.zeros(feature_size))
        nn.init.xavier_uniform_(self.ln_g.view(1, -1))

    def forward(self, x):
        # x: [B, num_filters, S2, F]
        y = self.conv1(x)
        y = torch.relu(y)
        y = self.conv2(y)
        y = y + x
        # normalization.m operates on [B, C, S2, F]-with-Spatial1=F axis; ln_de_w3/b3
        # are per-Feature (shape [Feature_size,1]) exactly as the encoder LN.
        y = _layernorm_feature_axis(y, self.ln_g, self.ln_b)
        return y


class Channelformer(nn.Module):
    """Full offline Channelformer (+transformer/model.m, HA03 encoder/decoder)."""

    def __init__(
        self,
        feature_size=72,
        num_heads=6,
        encoder_layers=1,
        decoder_layers=3,
        decoder_filters=12,
        output_feature_size=72,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.encoder_layers = nn.ModuleList(
            [EncoderBlockHA03(feature_size, num_heads) for _ in range(encoder_layers)]
        )

        # Regression head, first stage: dlconv Channel 1 -> decoder_filters, 5x5 'same'
        self.ln_de_w = nn.Conv2d(1, decoder_filters, kernel_size=5, padding="same")

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlockHA03(decoder_filters, feature_size) for _ in range(decoder_layers)]
        )

        # Final FC1 projection (Feature -> output_feature_size) then Channel
        # decoder_filters -> 1 via a last 5x5 'same' conv (ln_de_w0/b0).
        self.ln_de_w1 = nn.Parameter(torch.empty(output_feature_size, feature_size))
        self.ln_de_b1 = nn.Parameter(torch.zeros(output_feature_size))
        nn.init.xavier_uniform_(self.ln_de_w1)
        self.ln_de_w0 = nn.Conv2d(decoder_filters, 1, kernel_size=5, padding="same")

    def forward(self, x):
        # x: [B, C=1, S2=2 (real/imag), F=feature_size]
        z = x
        for layer in self.encoder_layers:
            z = layer(z)

        z = self.ln_de_w(z)  # Channel 1 -> decoder_filters

        for block in self.decoder_blocks:
            z = block(z)

        z = _fc1(z, self.ln_de_w1, self.ln_de_b1)  # project Feature axis
        z = self.ln_de_w0(z)  # Channel decoder_filters -> 1
        return z


MENAGERIE_ZOO = "ported-pytorch"


def build_channelformer():
    # Tiny menagerie-scale config: real feature_size=72 (36 pilot rows x 2 pilot
    # symbols, from +Parameter/parameters.m) shrunk to 12 (divisible by num_heads=6,
    # matching the released offline-Channelformer hyperparameters NumHeads=6,
    # Encoder_num_layers=1, Decoder_num_layers=3 from
    # +Training/Training_hybrid_offline.m) and decoder_filters shrunk from 12 to 4.
    return Channelformer(
        feature_size=12,
        num_heads=6,
        encoder_layers=1,
        decoder_layers=3,
        decoder_filters=4,
        output_feature_size=12,
    )


def example_input_channelformer():
    torch.manual_seed(0)
    # [Batch, Channel=1, IQ=2, Feature] -- least-squares channel estimate at pilot
    # positions, split into real/imag "spatial2" planes as in Data_Generation_Transformer.m.
    return (torch.randn(2, 1, 2, 12),)


MENAGERIE_ENTRIES = [
    (
        "Channelformer (attention channel estimation)",
        "build_channelformer",
        "example_input_channelformer",
        2023,
        "ported-pytorch",
    ),
]

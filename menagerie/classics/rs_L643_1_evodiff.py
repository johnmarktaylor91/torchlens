# SOURCE: vendored from microsoft/evodiff @ main
# Vendored files: evodiff/model.py::ByteNetTime, ByteNetLMTime (the real CARP/D3PM/OADM
# sequence-diffusion backbone used by evodiff.pretrained.D3PM_BLOSUM_640M and friends),
# plus the small real dependency pieces they import from the `sequence_models` package
# (microsoft/protein-sequence-models, pip name `sequence-models`) that are NOT installed
# in the base env: sequence_models.layers::PositionFeedForward, DoubleEmbedding and
# sequence_models.convolutional::MaskedConv1d, ByteNetBlock. Only imports/relative paths
# were adjusted to route to the vendored copies below; the architecture code is untouched.
#
# EvoDiff (Alamdari et al., "Protein generation with evolutionary diffusion: sequence is
# all you need", 2023) generates protein sequences via order-agnostic autoregressive /
# discrete (D3PM) diffusion over a ByteNet convolutional backbone with per-timestep
# conditioning (`ByteNetTime`/`ByteNetLMTime`), exactly the class instantiated by every
# non-MSA pretrained EvoDiff checkpoint (carp-38M/640M, oa_dm-38M/640M, d3pm-*-38M/640M).
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Vendored from sequence_models/layers.py (microsoft/protein-sequence-models)
# ---------------------------------------------------------------------------
class DoubleEmbedding(nn.Module):
    """Embedding layer that allows some frozen and some trainable embeddings.

    An embedding layer where the first n_trainable embeddings are trainable and the
    remaining n_frozen embeddings are frozen.
    """

    def __init__(self, n_trainable, n_frozen, embedding_dim, padding_idx=None):
        super().__init__()
        if padding_idx is None:
            train_padding_idx = None
            freeze_padding_idx = None
        elif padding_idx < n_trainable:
            train_padding_idx = padding_idx
            freeze_padding_idx = None
        else:
            train_padding_idx = None
            freeze_padding_idx = padding_idx - n_trainable
        self.n_trainable = n_trainable
        self.embedding_dim = embedding_dim
        self.trainable = nn.Embedding(n_trainable, embedding_dim, padding_idx=train_padding_idx)
        self.frozen = nn.Embedding(n_frozen, embedding_dim, padding_idx=freeze_padding_idx)
        self.frozen.weight.requires_grad = False

    def forward(self, idx):
        i = torch.where(idx < self.n_trainable)
        j = torch.where(idx >= self.n_trainable)
        b, ell = idx.shape
        e = torch.empty(
            b, ell, self.embedding_dim, device=idx.device, dtype=self.trainable.weight.dtype
        )
        e[i] = self.trainable(idx[i])
        e[j] = self.frozen(idx[j] - self.n_trainable)
        return e


class PositionFeedForward(nn.Module):
    def __init__(self, d_in, d_out, rank=None):
        super().__init__()
        if rank is None:
            self.conv = nn.Conv1d(d_in, d_out, 1)
            self.factorized = False
        else:
            layer = nn.Linear(d_in, d_out)
            w = layer.weight.data
            self.bias = layer.bias
            u, s, v = torch.svd(w)
            s = torch.diag(s[:rank].sqrt())
            u = u[:, :rank]
            v = v.t()[:rank]
            self.u = nn.Parameter(u @ s)
            self.v = nn.Parameter(s @ v)
            self.factorized = True

    def forward(self, x):
        if self.factorized:
            w = self.u @ self.v
            return x @ w.t() + self.bias
        else:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)


# ---------------------------------------------------------------------------
# Vendored from sequence_models/convolutional.py (microsoft/protein-sequence-models)
# ---------------------------------------------------------------------------
class MaskedConv1d(nn.Conv1d):
    """A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set
    automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class ByteNetBlock(nn.Module):
    """Residual block from ByteNet paper (https://arxiv.org/abs/1610.10099).

    Shape:
       Input: (N, L, d_in)
       input_mask: (N, L, 1), optional
       Output: (N, L, d_out)
    """

    def __init__(
        self,
        d_in,
        d_h,
        d_out,
        kernel_size,
        dilation=1,
        groups=1,
        causal=False,
        activation="relu",
        rank=None,
    ):
        super().__init__()
        # NOTE: only the non-causal path is vendored (EvoDiff's pretrained non-MSA
        # checkpoints all use causal=False); MaskedCausalConv1d is unused here.
        self.conv = MaskedConv1d(
            d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups
        )
        if activation == "relu":
            act = nn.ReLU
        elif activation == "gelu":
            act = nn.GELU
        layers1 = [
            nn.LayerNorm(d_in),
            act(),
            PositionFeedForward(d_in, d_h, rank=rank),
            nn.LayerNorm(d_h),
            act(),
        ]
        layers2 = [
            nn.LayerNorm(d_h),
            act(),
            PositionFeedForward(d_h, d_out, rank=rank),
        ]
        self.sequence1 = nn.Sequential(*layers1)
        self.sequence2 = nn.Sequential(*layers2)

    def forward(self, x, input_mask=None):
        return x + self.sequence2(self.conv(self.sequence1(x), input_mask=input_mask))


# ---------------------------------------------------------------------------
# Vendored from evodiff/model.py (microsoft/evodiff)
# ---------------------------------------------------------------------------
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model=8, length=500):
        super().__init__()
        self.d_model = d_model
        self.length = length

    def forward(self, x):
        """
        Used for encoding timestep in diffusion models

        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if self.d_model % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(
                    self.d_model
                )
            )
        pe = torch.zeros(self.length, self.d_model)
        position = torch.arange(0, self.length).unsqueeze(1)
        div_term = torch.exp(
            (
                torch.arange(0, self.d_model, 2, dtype=torch.float)
                * -(np.log(10000.0) / self.d_model)
            )
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        device = x.device
        pe = pe.to(device)
        return pe[x]


class ByteNetTime(nn.Module):
    """Stacked residual blocks from ByteNet paper defined by n_layers

    Shape:
       Input: (N, L,)
       input_mask: (N, L, 1), optional
       Output: (N, L, d)
    """

    def __init__(
        self,
        n_tokens,
        d_embedding,
        d_model,
        n_layers,
        kernel_size,
        r,
        rank=None,
        n_frozen_embs=None,
        padding_idx=None,
        causal=False,
        dropout=0.0,
        slim=True,
        activation="relu",
        down_embed=True,
        timesteps=None,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.time_encoding = PositionalEncoding1D(d_embedding, timesteps)  # Timestep encoding
        if n_tokens is not None:
            if n_frozen_embs is None:
                self.embedder = nn.Embedding(n_tokens, d_embedding, padding_idx=padding_idx)
            else:
                self.embedder = DoubleEmbedding(
                    n_tokens - n_frozen_embs, n_frozen_embs, d_embedding, padding_idx=padding_idx
                )
        else:
            self.embedder = nn.Identity()
        if down_embed:
            self.up_embedder = PositionFeedForward(d_embedding, d_model)
        else:
            self.up_embedder = nn.Identity()
            assert n_tokens == d_embedding
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        d_h = d_model
        if slim:
            d_h = d_h // 2
        layers = [
            ByteNetBlock(
                d_model,
                d_h,
                d_model,
                kernel_size,
                dilation=d,
                causal=causal,
                rank=rank,
                activation=activation,
            )
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.dropout = dropout

    def forward(self, x, y, input_mask=None):
        """
        :param x: (batch, length)
        :param y: (batch)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        e = self._embed(x, y, timesteps=self.timesteps)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x, y, timesteps=None):
        e = self.embedder(x)
        if timesteps is not None:
            e2 = self.time_encoding(y)
            # expand dim of e2 to match e1
            e2 = e2.expand(e.shape[1], e2.shape[0], e2.shape[1])
            e2 = e2.reshape(e.shape[0], e.shape[1], e.shape[2])
            e = torch.add(e2, e)
        e = self.up_embedder(e)
        return e

    def _convolve(self, e, input_mask=None):
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout)
        return e


class ByteNetLMTime(nn.Module):
    """The real EvoDiff sequence-diffusion backbone (evodiff/model.py::ByteNetLMTime),
    instantiated by evodiff.pretrained.load_sequence_checkpoint for every non-MSA
    pretrained model (carp-*, oa_dm-*, d3pm-*)."""

    def __init__(
        self,
        n_tokens,
        d_embedding,
        d_model,
        n_layers,
        kernel_size,
        r,
        rank=None,
        n_frozen_embs=None,
        padding_idx=None,
        causal=False,
        dropout=0.0,
        final_ln=False,
        slim=True,
        activation="relu",
        tie_weights=False,
        down_embed=True,
        timesteps=None,
    ):
        super().__init__()
        self.embedder = ByteNetTime(
            n_tokens,
            d_embedding,
            d_model,
            n_layers,
            kernel_size,
            r,
            padding_idx=padding_idx,
            causal=causal,
            dropout=dropout,
            down_embed=down_embed,
            slim=slim,
            activation=activation,
            rank=rank,
            n_frozen_embs=n_frozen_embs,
            timesteps=timesteps,
        )
        if tie_weights:
            self.decoder = nn.Linear(d_model, n_tokens, bias=False)
            self.decoder.weight = self.embedder.embedder.weight
        else:
            self.decoder = PositionFeedForward(d_model, n_tokens)
        if final_ln:
            self.last_norm = nn.LayerNorm(d_model)
        else:
            self.last_norm = nn.Identity()

    def forward(self, x, y, input_mask=None):
        e = self.embedder(x, y, input_mask=input_mask)
        e = self.last_norm(e)
        return self.decoder(e)


# ---------------------------------------------------------------------------
# Menagerie staging entry point
# ---------------------------------------------------------------------------
def build_evodiff():
    # Tiny stand-in for the real "carp-38M"-style config (config/config38M.json uses
    # n_tokens=31 (MSA_ALPHABET), d_embed=8, d_model=1024, n_layers=16, kernel_size=5,
    # r=128, final_ln=True); shrunk here to a random-init smoke-sized model with the
    # real architecture and real default hyperparameter roles preserved.
    return ByteNetLMTime(
        n_tokens=31,
        d_embedding=8,
        d_model=32,
        n_layers=2,
        kernel_size=5,
        r=8,
        rank=None,
        n_frozen_embs=None,
        padding_idx=28,
        causal=False,
        dropout=0.0,
        final_ln=True,
        slim=True,
        activation="relu",
        tie_weights=False,
        down_embed=True,
        timesteps=500,
    )


def example_input_evodiff():
    torch.manual_seed(0)
    batch, length = 2, 16
    x = torch.randint(0, 31, (batch, length))
    y = torch.randint(0, 500, (batch,))
    return (x, y)


MENAGERIE_ENTRIES = [
    ("EvoDiff-ByteNetLMTime", "build_evodiff", "example_input_evodiff", 2023, MENAGERIE_ZOO),
]

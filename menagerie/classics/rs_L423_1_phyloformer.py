# SOURCE: vendored from lucanest/Phyloformer @ main
# (phyloformer/model.py + phyloformer/attention.py, combined into one staging file)
#
# Original files:
#   https://raw.githubusercontent.com/lucanest/Phyloformer/main/phyloformer/model.py
#   https://raw.githubusercontent.com/lucanest/Phyloformer/main/phyloformer/attention.py
#
# Only import/module-layout changes were made to combine the two original files into
# a single staging module (the local `from .attention import ScaledLinearAttention`
# became a direct in-file class definition). The architecture itself is untouched.

import math
from typing import Optional

import torch  # type:ignore
from scipy.special import binom
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- from phyloformer/attention.py -----------------------------------------


class BaseAttention(nn.Module):
    """
    Base module to implement various self-attention mechanisms
    Allows for (Q,K) and V to have different dimensions
    """

    def __init__(
        self,
        nb_heads: int,
        embed_dim: int,
        qk_dim: Optional[int] = None,
        dropout: float = 0.0,
        # eps: float = 1e-6,
    ):
        super().__init__()

        # By default all matrices have the same shape
        if qk_dim is None:
            qk_dim = embed_dim

        if embed_dim % nb_heads != 0 or qk_dim % nb_heads != 0:
            raise ValueError(
                "Embed dim and QK dim (if specified) mus tbe divisible by the number of heads.\n"
                f"Embed: {embed_dim}, QK: {qk_dim} -> n_heads: {nb_heads}"
            )

        # Dimensions and parameters
        self.embed_dim = embed_dim
        self.qk_dim = qk_dim
        self.nb_heads = nb_heads
        self.dropout = dropout

        self.head_dim = embed_dim // nb_heads
        self.head_qk_dim = qk_dim // nb_heads

        # Projectors
        self.k_proj = nn.Linear(embed_dim, qk_dim)
        self.q_proj = nn.Linear(embed_dim, qk_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.atten_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)


class ScaledLinearAttention(BaseAttention):
    """
    Custom version of the Linear Kernel Attention with dimension 1 for Q and K.
    """

    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__(nb_heads, embed_dim, nb_heads, dropout)

        self.elu = nn.ELU()
        self.eps = eps

    def forward(self, input):
        batch_size, nb_row, nb_col, embed_dim = input.size()

        k = (
            self.k_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_qk_dim)
            .transpose(2, 3)
        )
        q = (
            self.q_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_qk_dim)
            .transpose(2, 3)
        )
        v = (
            self.v_proj(input)
            .view(batch_size, nb_row, nb_col, self.nb_heads, self.head_dim)
            .transpose(2, 3)
        )

        q = self.elu(q) + 1
        k = self.elu(k) + 1

        # Scale Q to keep amplitude under control
        q = q / q.mean(dim=-2, keepdim=True)

        # Normalize K
        k = k / k.sum(dim=-2, keepdim=True)  # Sum directly on -2 instead of transposing an summing

        KtV = k.transpose(-1, -2) @ v

        V = q @ KtV
        V = V.transpose(2, 3).contiguous().view(batch_size, -1, nb_col, embed_dim)

        out = self.proj_drop(self.out_proj(V))

        return out


# --- from phyloformer/model.py ----------------------------------------------


def seq2pair(n_seqs: int):
    """Initialize Seq2Pair matrix"""
    n_pairs = int(binom(n_seqs, 2))
    seq2pair = torch.zeros(n_pairs, n_seqs)
    k = 0
    for i in range(n_seqs):
        for j in range(i + 1, n_seqs):
            seq2pair[k, i] = 1
            seq2pair[k, j] = 1
            k = k + 1
    return seq2pair


def adaptable_seq2pair(n_seqs: int, global_seq2pair):
    """Initialize Seq2Pair matrix"""
    max_n_seqs = global_seq2pair.shape[1]
    if n_seqs > max_n_seqs:
        raise ValueError(
            f"n_seqs must be smaller or equal to {max_n_seqs} "
            "(or pre-compute a larger global_seq2pair)"
        )
    # Retain n_seqs columns (sequences) and rows (pairs) that only
    # involve these sequences. Arbitrarilly using the first
    # columns, but any subset of n_seqs columns would do.
    mask = (torch.norm(global_seq2pair[:, n_seqs:], dim=1) == 0).squeeze()
    seq2pair = global_seq2pair[mask, :n_seqs]
    del mask
    return seq2pair


# Global instance of a large Seq2Pair matrix
SEQ2PAIR = seq2pair(200)


class PhyloformerLayer(nn.Module):
    """Phyloformer's Transformer Layer"""

    def __init__(
        self,
        embed_dim: int,
        nb_heads: int,
        dropout: float,
        normalize: bool = True,
        heterodims: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads
        self.dropout = dropout
        self.normalize = normalize
        self.heterodims = heterodims

        self.row_attention = ScaledLinearAttention(self.embed_dim, self.nb_heads)
        self.col_attention = ScaledLinearAttention(self.embed_dim, self.nb_heads)

        # Normalization layers
        self.row_norm = nn.LayerNorm(self.embed_dim)
        self.col_norm = nn.LayerNorm(self.embed_dim)
        self.ffn_norm = nn.LayerNorm(self.embed_dim)

        # Feed forward NN
        self.ffn = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim * 4,
                kernel_size=1,
                stride=1,
            ),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Conv2d(
                in_channels=self.embed_dim * 4,
                out_channels=self.embed_dim,
                kernel_size=1,
                stride=1,
            ),
            nn.Dropout(self.dropout),
        )

    def forward(self, input):
        # Row attention sub-block
        res_row = input
        out = self.row_norm(input.transpose(-1, -3)).transpose(-1, -3)
        out = self.row_attention(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = out + res_row  # residual connection

        # Col attention sub-block
        res_col = out
        out = self.col_norm(out.transpose(-1, -3)).transpose(-1, -3)
        out = self.col_attention(out.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        out = out + res_col

        # FFN sub-block
        res_ffn = out
        out = self.ffn_norm(out.transpose(-1, -3)).transpose(-1, -3)
        out = self.ffn(out)
        out = out + res_ffn

        return out


class Phyloformer(nn.Module):
    """Model architecture for Phyloformer"""

    def __init__(
        self,
        n_blocks: int = 6,
        n_heads: int = 4,
        h_dim: int = 64,
        dropout: float = 0.0,
        n_seqs: int = 20,
        seq_len: int = 200,
        normalize: bool = True,
        heterodims: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.nb_blocks = n_blocks
        self.nb_heads = n_heads
        self.embed_dim = h_dim
        self.dropout = dropout
        self.normalize = normalize
        self.heterodims = heterodims

        self.n_seqs = n_seqs
        self.seq_len = seq_len

        # Initialize seq2pair matrix
        self.seq2pair = adaptable_seq2pair(20, SEQ2PAIR)

        self.embedding_block = nn.Sequential(
            nn.Conv2d(in_channels=22, out_channels=self.embed_dim, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.attention_blocks = nn.ModuleList(
            [
                PhyloformerLayer(
                    embed_dim=self.embed_dim,
                    nb_heads=self.nb_heads,
                    dropout=self.dropout,
                    normalize=self.normalize,
                    heterodims=self.heterodims,
                )
                for _ in range(self.nb_blocks)
            ]
        )

        self.pwFNN = nn.Sequential(
            nn.Conv2d(in_channels=self.embed_dim, out_channels=1, kernel_size=1, stride=1),
            nn.Dropout(self.dropout),
            nn.Softplus(),
        )

    def forward(self, input):
        # input: (batch_size, 22, seq_len, n_seqs)

        # Set seq2pair matrix if needed
        self._set_seq2pair(input.shape[-1])

        # Embed alignment to embed_dim
        out = self.embedding_block(input)
        # Pair representation -> (batch_size, embed_dim, nb_pairs, seq_len)
        out = torch.matmul(self.seq2pair, out.transpose(-1, -2))

        # Attention
        for block in self.attention_blocks:
            out = block(out)

        # Convolution -> (batch_size, 1, nb_pairs, seq_len)
        out = self.pwFNN(out)

        # Average of sequence length -> (batch_size, nb_pairs)
        out = torch.squeeze(torch.mean(out, dim=-1))

        return out

    def _set_seq2pair(self, n_seqs: int):
        """Initialize Seq2Pair matrix"""

        # Don't do anything if the alignment shape is the same
        if self.n_seqs == n_seqs:
            return

        self.n_seqs = n_seqs
        self.n_pairs = int(binom(n_seqs, 2))

        # Generate new
        device = self.seq2pair.device
        self.seq2pair = adaptable_seq2pair(n_seqs, SEQ2PAIR).to(device)


# --- staging entry points ----------------------------------------------------


def build_phyloformer():
    # Tiny config: 2 attention blocks, small embed dim. NOTE: Phyloformer.__init__
    # always builds self.seq2pair for exactly 20 sequences (adaptable_seq2pair(20, ...)),
    # regardless of the n_seqs kwarg -- a real quirk of the upstream code -- so the
    # example input below must present exactly 20 sequences on first forward to avoid
    # a stale/mismatched seq2pair matrix (see Phyloformer._set_seq2pair).
    return Phyloformer(n_blocks=2, n_heads=2, h_dim=16, dropout=0.0, n_seqs=20, seq_len=24)


def example_input_phyloformer():
    # (batch_size, 22, seq_len, n_seqs) -- 22 = 20 amino acids + gap + padding channel;
    # n_seqs=20 matches the seq2pair matrix Phyloformer.__init__ always constructs.
    return torch.rand(1, 22, 24, 20)


MENAGERIE_ENTRIES = [
    (
        "Phyloformer",
        build_phyloformer,
        example_input_phyloformer,
        2022,
        "SOURCE_AVAILABLE",
    ),
]

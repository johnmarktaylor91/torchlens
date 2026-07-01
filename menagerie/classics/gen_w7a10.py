"""Wave 7 batch 10 menagerie classics: RNA language/structure models + all-atom
biomolecular structure prediction.

Sources checked (repo_url / desc_source columns of the build queue, web research
2026-07-01; no cloning, no pip installs beyond the base env):
  - RiNALMo: https://github.com/lbcb-sci/RiNALMo ; Penic et al. 2024,
    bioRxiv 2024.03.17.585376 / Nature Communications 2025, "RiNALMo:
    general-purpose RNA language models can generalize well on structure
    prediction tasks". A 650M-parameter, 33-block BERT-style encoder for
    non-coding RNA pretrained with masked-token modeling, distinguished from a
    plain BERT by three modern upgrades applied together: rotary positional
    embedding (RoPE) instead of learned absolute position embeddings, a SwiGLU
    gated feed-forward block instead of a plain ReLU/GELU MLP, and pre-norm
    residual blocks (FlashAttention-2 is a training-only kernel optimization,
    not an architectural feature, so it is omitted here).
  - RNA-MSM: https://github.com/yikunpku/RNA-MSM ; Zhang et al. 2024, Nucleic
    Acids Research 52(1):e3, "Multiple sequence alignment-based RNA language
    model and its application to structural inference". An MSA-Transformer-
    style model operating on a 2D (num_sequences x length) alignment tensor of
    homologous RNA sequences: alternating row-wise (across positions, within
    one sequence) and column-wise (across sequences, at one position)
    axial attention, with separate learnable row/column positional
    embeddings, so it explicitly models coevolutionary covariation across the
    MSA depth axis (unlike a single-sequence BERT such as RiNALMo above).
  - RNADiffFold: https://github.com/HIM-AIM/RNADiffFold ; Wang et al. 2025,
    Briefings in Bioinformatics 26(1):bbae618, "RNADiffFold: generative RNA
    secondary structure prediction using discrete diffusion models". Treats
    the (length x length) binary base-pair contact map as a discrete
    diffusion target (categorical/absorbing-style corruption over {pair,
    no-pair}) and trains a conditional denoiser that, at each reverse step,
    refines the contact map conditioned on sequence features (one-hot
    sequence + an outer-product pairwise prior) and a sinusoidal timestep
    embedding -- i.e. secondary-structure prediction reframed as conditional
    image-segmentation-style denoising rather than one-shot classification.
  - RNAformer: https://github.com/automl/RNAformer ; Franke et al. 2024,
    bioRxiv 2023.01.25.525393 / ICML workshop, "RNAformer: A Simple yet
    Effective Model for Homology-Aware RNA Secondary Structure Prediction".
    Lifts the 1D sequence into a 2D (length x length) latent via an
    outer-product/broadcast expansion, then refines it with blocks of
    row-wise + column-wise axial self-attention followed by a convolutional
    "transition" layer (captures local stem-loop structure that pure
    attention misses), with the whole 2D latent recycled through the block
    stack for several iterations before a final pairing-probability readout
    -- the axial-attention + conv-transition + recycling combination is the
    distinctive mechanism (vs. plain 2D self-attention or a 1D-only model).
  - RNAsnap2: https://github.com/jaswindersingh2/RNAsnap2 ; Hanumanthappa
    et al. 2020, Bioinformatics 36(21):5169-5176, "Single-sequence and
    profile-based prediction of RNA solvent accessibility using dilated
    convolutional neural network". A 1D dilated-CNN stack (exponentially
    increasing dilation per layer) over per-nucleotide input features,
    giving a large effective receptive field without pooling so long-range
    sequence context informs the per-nucleotide solvent-accessibility
    regression head -- the dilation ladder is the distinctive mechanism.
  - RoseTTAFold All-Atom: https://github.com/baker-laboratory/RoseTTAFold-All-Atom
    ; Krishna et al. 2024, Science 384(6693):eadl2528, "Generalized
    biomolecular modeling and design with RoseTTAFold All-Atom". Represents
    a biological assembly with a hybrid node set: protein/DNA residues get
    one node each (residue-frame track, as in RoseTTAFold2), while every atom
    of a small molecule/ligand/modified residue gets its own node (atom
    track), with all nodes sharing one pair track and one coordinate-frame
    update; this residue-vs-atom hybrid tokenization (not diffusion -- that
    is the separate downstream RFdiffusion All-Atom / RFAA design model
    already in the menagerie) is what distinguishes it from the
    protein-only RoseTTAFold2 and the nucleic-acid-only RF2NA.

All six are faithful compact reimplementations: random init, small dims, few
blocks/iterations, forward-only, kept just large enough to exercise each
architecture's distinctive mechanism so the traced/unrolled atlas graph
renders quickly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. RiNALMo: BERT-style RNA encoder upgraded with rotary position embeddings
#    (RoPE) and a SwiGLU gated feed-forward block, pre-norm residual blocks.
# ---------------------------------------------------------------------------


def _rope_angles(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """Compute rotary-embedding angles for a sequence of length ``seq_len``.

    Parameters
    ----------
    seq_len:
        Number of positions.
    dim:
        Per-head channel width (must be even).
    device:
        Device to place the angle tensor on.

    Returns
    -------
    torch.Tensor
        Angle tensor of shape ``(seq_len, dim // 2)``.
    """

    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    pos = torch.arange(seq_len, device=device).float()
    return torch.outer(pos, inv_freq)


def _apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate query/key vectors by the RoPE angles.

    Parameters
    ----------
    x:
        Tensor of shape ``(seq_len, n_head, head_dim)``.
    angles:
        Angle tensor of shape ``(seq_len, head_dim // 2)``.

    Returns
    -------
    torch.Tensor
        Rotated tensor of the same shape as ``x``.
    """

    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = torch.cos(angles)[:, None, :]
    sin = torch.sin(angles)[:, None, :]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = torch.stack([rot1, rot2], dim=-1).flatten(-2)
    return out


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention with rotary positional embeddings."""

    def __init__(self, dim: int, n_head: int = 4) -> None:
        """Build a RoPE self-attention block.

        Parameters
        ----------
        dim:
            Model channel width.
        n_head:
            Number of attention heads.
        """

        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE self-attention.

        Parameters
        ----------
        x:
            Input of shape ``(seq_len, dim)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(seq_len, dim)``.
        """

        seq_len = x.shape[0]
        qkv = self.qkv(x).reshape(seq_len, 3, self.n_head, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        angles = _rope_angles(seq_len, self.head_dim, x.device)
        q = _apply_rope(q, angles).permute(1, 0, 2)
        k = _apply_rope(k, angles).permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5), dim=-1)
        out = torch.matmul(att, v).permute(1, 0, 2).reshape(seq_len, -1)
        return self.out(out)


class SwiGLU(nn.Module):
    """Gated SwiGLU feed-forward block."""

    def __init__(self, dim: int, hidden: int) -> None:
        """Build a SwiGLU feed-forward block.

        Parameters
        ----------
        dim:
            Model channel width.
        hidden:
            Hidden gate width.
        """

        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU transform.

        Parameters
        ----------
        x:
            Input of shape ``(..., dim)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(..., dim)``.
        """

        return self.proj(F.silu(self.gate(x)) * self.value(x))


class RiNALMoBlock(nn.Module):
    """One pre-norm RiNALMo transformer block (RoPE attention + SwiGLU FFN)."""

    def __init__(self, dim: int, n_head: int = 4, ffn_hidden: int = 64) -> None:
        """Build a RiNALMo transformer block.

        Parameters
        ----------
        dim:
            Model channel width.
        n_head:
            Number of attention heads.
        ffn_hidden:
            SwiGLU hidden width.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPESelfAttention(dim, n_head)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the pre-norm attention + SwiGLU residual block.

        Parameters
        ----------
        x:
            Input of shape ``(seq_len, dim)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(seq_len, dim)``.
        """

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RiNALMo(nn.Module):
    """Compact RiNALMo: RoPE + SwiGLU BERT-style RNA encoder with an MLM head."""

    def __init__(self, dim: int = 32, n_block: int = 3, n_token: int = 6) -> None:
        """Build a compact RiNALMo encoder.

        Parameters
        ----------
        dim:
            Model channel width.
        n_block:
            Number of transformer blocks.
        n_token:
            Vocabulary size (4 nucleotides + mask + pad).
        """

        super().__init__()
        self.embed = nn.Embedding(n_token, dim)
        self.blocks = nn.ModuleList([RiNALMoBlock(dim) for _ in range(n_block)])
        self.norm_out = nn.LayerNorm(dim)
        self.mlm_head = nn.Linear(dim, n_token)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict per-position masked-token logits.

        Parameters
        ----------
        tokens:
            Integer token sequence of shape ``(seq_len,)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(seq_len, n_token)``.
        """

        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.mlm_head(self.norm_out(x))


def build_rinalmo() -> nn.Module:
    """Build a compact RiNALMo RNA language model.

    Returns
    -------
    nn.Module
        Random-initialized RiNALMo in eval mode.
    """

    return RiNALMo().eval()


def example_input_rinalmo() -> torch.Tensor:
    """Create a small masked RNA token sequence.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape ``(24,)`` with tokens in ``[0, 6)``.
    """

    return torch.randint(0, 6, (24,))


# ---------------------------------------------------------------------------
# 2. RNA-MSM: MSA-transformer-style model with alternating row-wise and
#    column-wise axial attention over a (num_sequences x length) alignment.
# ---------------------------------------------------------------------------


class AxialMSABlock(nn.Module):
    """One row-wise + column-wise axial self-attention block over an MSA."""

    def __init__(self, dim: int, n_head: int = 4) -> None:
        """Build an axial MSA attention block.

        Parameters
        ----------
        dim:
            Per-token channel width.
        n_head:
            Number of attention heads (shared by both axes).
        """

        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.norm_row = nn.LayerNorm(dim)
        self.row_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.row_out = nn.Linear(dim, dim)
        self.norm_col = nn.LayerNorm(dim)
        self.col_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.col_out = nn.Linear(dim, dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def _axial_attend(self, x: torch.Tensor, qkv: nn.Linear, out: nn.Linear) -> torch.Tensor:
        """Run self-attention along the last (sequence) axis of ``x``.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch_axis, seq_axis, dim)`` where attention
            mixes over ``seq_axis`` independently for each ``batch_axis``
            slice.
        qkv:
            Linear projection producing packed query/key/value.
        out:
            Output projection.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``.
        """

        b, n, dim = x.shape
        proj = qkv(x).reshape(b, n, 3, self.n_head, self.head_dim)
        q, k, v = proj[:, :, 0], proj[:, :, 1], proj[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5), dim=-1)
        merged = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b, n, dim)
        return out(merged)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply row-wise then column-wise axial attention plus an FFN.

        Parameters
        ----------
        x:
            MSA representation of shape ``(n_seq, length, dim)``.

        Returns
        -------
        torch.Tensor
            Updated MSA representation of the same shape.
        """

        h = self.norm_row(x)
        x = x + self._axial_attend(h, self.row_qkv, self.row_out)
        h = self.norm_col(x).transpose(0, 1)
        col_out = self._axial_attend(h, self.col_qkv, self.col_out).transpose(0, 1)
        x = x + col_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class RNAMSM(nn.Module):
    """Compact RNA-MSM: axial-attention MSA transformer with row/col position embeds."""

    def __init__(
        self,
        dim: int = 24,
        n_block: int = 2,
        n_token: int = 6,
        max_seqs: int = 8,
        max_len: int = 20,
    ) -> None:
        """Build a compact RNA-MSM encoder.

        Parameters
        ----------
        dim:
            Per-token channel width.
        n_block:
            Number of axial MSA blocks.
        n_token:
            Vocabulary size (4 nucleotides + gap + mask).
        max_seqs:
            Maximum MSA depth for the row (sequence-index) position embedding.
        max_len:
            Maximum alignment length for the column position embedding.
        """

        super().__init__()
        self.embed = nn.Embedding(n_token, dim)
        self.row_pos = nn.Parameter(torch.zeros(max_seqs, 1, dim))
        self.col_pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList([AxialMSABlock(dim) for _ in range(n_block)])
        self.contact_left = nn.Linear(dim, dim)
        self.contact_right = nn.Linear(dim, dim)
        self.contact_head = nn.Linear(dim, 1)

    def forward(self, msa_tokens: torch.Tensor) -> torch.Tensor:
        """Predict base-pairing contact logits from an RNA MSA.

        Parameters
        ----------
        msa_tokens:
            Integer alignment tensor of shape ``(n_seq, length)``.

        Returns
        -------
        torch.Tensor
            Pairwise contact logits of shape ``(length, length)``, derived
            from the query (first) sequence's row after MSA refinement.
        """

        n_seq, length = msa_tokens.shape
        x = self.embed(msa_tokens)
        x = x + self.row_pos[:n_seq] + self.col_pos[:, :length]
        for block in self.blocks:
            x = block(x)
        query = x[0]
        left = self.contact_left(query)
        right = self.contact_right(query)
        pair = left[:, None, :] + right[None, :, :]
        return self.contact_head(pair).squeeze(-1)


def build_rna_msm() -> nn.Module:
    """Build a compact RNA-MSM model.

    Returns
    -------
    nn.Module
        Random-initialized RNA-MSM in eval mode.
    """

    return RNAMSM().eval()


def example_input_rna_msm() -> torch.Tensor:
    """Create a small toy RNA multiple sequence alignment.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape ``(6, 16)`` (6 homologous sequences, length 16).
    """

    return torch.randint(0, 6, (6, 16))


# ---------------------------------------------------------------------------
# 3. RNADiffFold: discrete diffusion denoiser over an RNA base-pair contact
#    map, conditioned on sequence features and a timestep embedding.
# ---------------------------------------------------------------------------


class DiffFoldDenoiseBlock(nn.Module):
    """One conditional contact-map denoising block."""

    def __init__(self, channels: int) -> None:
        """Build a denoising block.

        Parameters
        ----------
        channels:
            Number of feature-map channels.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, channels)
        self.time_proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """Apply one residual conv block modulated by the timestep embedding.

        Parameters
        ----------
        x:
            Feature map of shape ``(channels, L, L)``.
        t_embed:
            Timestep embedding of shape ``(channels,)``.

        Returns
        -------
        torch.Tensor
            Updated feature map of shape ``(channels, L, L)``.
        """

        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_embed)[:, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return x + h


def _sinusoidal_timestep_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute a sinusoidal embedding for a scalar diffusion timestep.

    Parameters
    ----------
    t:
        Scalar timestep tensor.
    dim:
        Embedding width (must be even).

    Returns
    -------
    torch.Tensor
        Embedding of shape ``(dim,)``.
    """

    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class RNADiffFold(nn.Module):
    """Compact RNADiffFold: conditional discrete-diffusion contact-map denoiser."""

    def __init__(self, channels: int = 16, n_block: int = 2, n_base: int = 4) -> None:
        """Build a compact RNADiffFold denoiser.

        Parameters
        ----------
        channels:
            Feature-map channel width.
        n_block:
            Number of denoising blocks.
        n_base:
            Number of one-hot base channels (A/C/G/U).
        """

        super().__init__()
        self.cond_proj = nn.Conv2d(2 * n_base + 1, channels, kernel_size=1)
        self.time_mlp = nn.Sequential(nn.Linear(channels, channels), nn.SiLU())
        self.blocks = nn.ModuleList([DiffFoldDenoiseBlock(channels) for _ in range(n_block)])
        self.out_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.channels = channels

    def forward(
        self, seq_onehot: torch.Tensor, noisy_contacts: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Denoise one reverse-diffusion step of the base-pair contact map.

        Parameters
        ----------
        seq_onehot:
            One-hot RNA sequence of shape ``(length, n_base)``.
        noisy_contacts:
            Corrupted contact-map probabilities of shape ``(length, length)``.
        t:
            Scalar diffusion timestep.

        Returns
        -------
        torch.Tensor
            Denoised contact-map logits of shape ``(length, length)``.
        """

        length = seq_onehot.shape[0]
        left = seq_onehot.transpose(0, 1)[:, :, None].expand(-1, -1, length)
        right = seq_onehot.transpose(0, 1)[:, None, :].expand(-1, length, -1)
        cond = torch.cat([left, right, noisy_contacts[None]], dim=0)
        x = self.cond_proj(cond[None])[0]
        t_embed = self.time_mlp(_sinusoidal_timestep_embed(t, self.channels))
        for block in self.blocks:
            x = block(x, t_embed)
        return self.out_head(x[None])[0, 0]


def build_rnadifffold() -> nn.Module:
    """Build a compact RNADiffFold denoiser.

    Returns
    -------
    nn.Module
        Random-initialized RNADiffFold in eval mode.
    """

    return RNADiffFold().eval()


def example_input_rnadifffold() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create one reverse-diffusion step of RNADiffFold's inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(seq_onehot, noisy_contacts, t)`` for a length-16 RNA.
    """

    length = 16
    seq_onehot = F.one_hot(torch.randint(0, 4, (length,)), num_classes=4).float()
    noisy_contacts = torch.rand(length, length)
    t = torch.tensor(500.0)
    return seq_onehot, noisy_contacts, t


# ---------------------------------------------------------------------------
# 4. RNAformer: 1D-to-2D outer-product lift, axial row/column attention +
#    convolutional transition, with the 2D latent recycled across iterations.
# ---------------------------------------------------------------------------


class RNAformerBlock(nn.Module):
    """One row/column axial-attention + convolutional-transition block."""

    def __init__(self, dim: int, n_head: int = 4) -> None:
        """Build an RNAformer block.

        Parameters
        ----------
        dim:
            2D latent channel width.
        n_head:
            Number of attention heads (shared by both axes).
        """

        super().__init__()
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.norm_row = nn.LayerNorm(dim)
        self.row_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.row_out = nn.Linear(dim, dim)
        self.norm_col = nn.LayerNorm(dim)
        self.col_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.col_out = nn.Linear(dim, dim)
        self.norm_conv = nn.LayerNorm(dim)
        self.transition = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def _axial(self, x: torch.Tensor, qkv: nn.Linear, out: nn.Linear) -> torch.Tensor:
        """Run self-attention along axis 1 of a ``(axis0, axis1, dim)`` tensor.

        Parameters
        ----------
        x:
            Tensor of shape ``(a0, a1, dim)``.
        qkv:
            Linear projection producing packed query/key/value.
        out:
            Output projection.

        Returns
        -------
        torch.Tensor
            Tensor of the same shape as ``x``.
        """

        a0, a1, dim = x.shape
        proj = qkv(x).reshape(a0, a1, 3, self.n_head, self.head_dim)
        q, k, v = proj[:, :, 0], proj[:, :, 1], proj[:, :, 2]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5), dim=-1)
        merged = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(a0, a1, dim)
        return out(merged)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply row attention, column attention, and a conv transition.

        Parameters
        ----------
        z:
            2D latent of shape ``(length, length, dim)``.

        Returns
        -------
        torch.Tensor
            Updated 2D latent of the same shape.
        """

        h = self.norm_row(z)
        z = z + self._axial(h, self.row_qkv, self.row_out)
        h = self.norm_col(z).transpose(0, 1)
        z = z + self._axial(h, self.col_qkv, self.col_out).transpose(0, 1)
        h = self.norm_conv(z).permute(2, 0, 1)[None]
        z = z + self.transition(h)[0].permute(1, 2, 0)
        return z


class RNAformer(nn.Module):
    """Compact RNAformer: 2D-latent axial-attention model with recycling."""

    def __init__(
        self, dim: int = 16, n_block: int = 2, n_recycle: int = 2, n_token: int = 5
    ) -> None:
        """Build a compact RNAformer.

        Parameters
        ----------
        dim:
            2D latent channel width.
        n_block:
            Number of RNAformer blocks per recycling pass.
        n_recycle:
            Number of times the block stack is recycled over the 2D latent.
        n_token:
            Vocabulary size (4 nucleotides + pad).
        """

        super().__init__()
        self.embed = nn.Embedding(n_token, dim)
        self.lift_left = nn.Linear(dim, dim)
        self.lift_right = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([RNAformerBlock(dim) for _ in range(n_block)])
        self.n_recycle = n_recycle
        self.pair_head = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict a base-pairing probability matrix from an RNA sequence.

        Parameters
        ----------
        tokens:
            Integer sequence tensor of shape ``(length,)``.

        Returns
        -------
        torch.Tensor
            Pairing logits of shape ``(length, length)``.
        """

        s = self.embed(tokens)
        z = self.lift_left(s)[:, None, :] + self.lift_right(s)[None, :, :]
        for _ in range(self.n_recycle):
            for block in self.blocks:
                z = block(z)
        return self.pair_head(z).squeeze(-1)


def build_rnaformer() -> nn.Module:
    """Build a compact RNAformer.

    Returns
    -------
    nn.Module
        Random-initialized RNAformer in eval mode.
    """

    return RNAformer().eval()


def example_input_rnaformer() -> torch.Tensor:
    """Create a small RNA sequence for RNAformer.

    Returns
    -------
    torch.Tensor
        Integer tensor of shape ``(18,)`` with tokens in ``[0, 5)``.
    """

    return torch.randint(0, 5, (18,))


# ---------------------------------------------------------------------------
# 5. RNAsnap2: 1D dilated-convolution stack (exponentially increasing
#    dilation) for per-nucleotide solvent-accessibility regression.
# ---------------------------------------------------------------------------


class DilatedConvBlock(nn.Module):
    """One dilated 1D conv block with a residual connection."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Build a dilated conv block.

        Parameters
        ----------
        channels:
            Number of feature channels.
        dilation:
            Dilation factor for the conv kernel.
        """

        super().__init__()
        padding = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the dilated conv block with a residual add.

        Parameters
        ----------
        x:
            Feature tensor of shape ``(channels, length)``.

        Returns
        -------
        torch.Tensor
            Updated feature tensor of the same shape.
        """

        return x + F.relu(self.norm(self.conv(x)))


class RNAsnap2(nn.Module):
    """Compact RNAsnap2: exponential-dilation 1D CNN for solvent accessibility."""

    def __init__(self, channels: int = 24, dilations: tuple[int, ...] = (1, 2, 4, 8, 16)) -> None:
        """Build a compact RNAsnap2 predictor.

        Parameters
        ----------
        channels:
            Feature channel width.
        dilations:
            Per-layer dilation ladder (exponentially increasing).
        """

        super().__init__()
        self.in_proj = nn.Conv1d(9, channels, kernel_size=1)
        self.blocks = nn.ModuleList([DilatedConvBlock(channels, d) for d in dilations])
        self.out_head = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict per-nucleotide relative solvent accessibility.

        Parameters
        ----------
        features:
            Per-nucleotide input features of shape ``(length, 9)`` (one-hot
            base + profile + predicted base-pairing-probability summary).

        Returns
        -------
        torch.Tensor
            Predicted accessibility of shape ``(length,)`` in ``[0, 1]``.
        """

        x = features.transpose(0, 1)[None]
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return torch.sigmoid(self.out_head(x))[0, 0]


def build_rnasnap2() -> nn.Module:
    """Build a compact RNAsnap2 predictor.

    Returns
    -------
    nn.Module
        Random-initialized RNAsnap2 in eval mode.
    """

    return RNAsnap2().eval()


def example_input_rnasnap2() -> torch.Tensor:
    """Create small per-nucleotide input features for RNAsnap2.

    Returns
    -------
    torch.Tensor
        Feature tensor of shape ``(30, 9)``.
    """

    return torch.randn(30, 9)


# ---------------------------------------------------------------------------
# 6. RoseTTAFold All-Atom: hybrid residue-frame + atom-graph node set sharing
#    one pair track and one coordinate update (structure prediction, not the
#    downstream diffusion design model RFAA already in the menagerie).
# ---------------------------------------------------------------------------


class HybridResidueAtomBlock(nn.Module):
    """One residue+atom hybrid-node pair/coordinate update block."""

    def __init__(self, c_1d: int, c_2d: int, n_head: int = 4) -> None:
        """Build a hybrid residue/atom structure-update block.

        Parameters
        ----------
        c_1d:
            Per-node (residue-frame or atom) channel width.
        c_2d:
            Pair channel width.
        n_head:
            Attention head count.
        """

        super().__init__()
        self.h = n_head
        self.c = c_1d // n_head
        self.norm = nn.LayerNorm(c_1d)
        self.q = nn.Linear(c_1d, c_1d, bias=False)
        self.k = nn.Linear(c_1d, c_1d, bias=False)
        self.v = nn.Linear(c_1d, c_1d, bias=False)
        self.pair_bias = nn.Linear(c_2d, n_head, bias=False)
        self.attn_out = nn.Linear(c_1d, c_1d)
        # bonded-graph message passing restricted to the atom-track nodes
        self.bond_msg = nn.Linear(c_1d, c_1d)
        self.outer_left = nn.Linear(c_1d, c_2d)
        self.outer_right = nn.Linear(c_1d, c_2d)
        self.pair_update = nn.Sequential(nn.LayerNorm(c_2d), nn.Linear(c_2d, c_2d), nn.ReLU())
        self.coord_from_pair = nn.Linear(c_2d, 3)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        coords: torch.Tensor,
        bond_adj: torch.Tensor,
        is_atom: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update node/pair features and node coordinates for one block.

        Parameters
        ----------
        s:
            Per-node representation of shape ``(N, c_1d)``, mixing
            residue-frame nodes and small-molecule atom nodes.
        z:
            Pair representation of shape ``(N, N, c_2d)``.
        coords:
            Node coordinates (residue Ca / atom position) of shape ``(N, 3)``.
        bond_adj:
            Bonded-graph adjacency of shape ``(N, N)`` (nonzero only between
            atom-track nodes that share a covalent bond).
        is_atom:
            Boolean mask of shape ``(N, 1)``; ``True`` marks atom-track nodes.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated ``(s, z, coords)``.
        """

        h = self.norm(s)
        q = _heads(self.q(h), self.h).permute(1, 0, 2)
        k = _heads(self.k(h), self.h).permute(1, 0, 2)
        v = _heads(self.v(h), self.h).permute(1, 0, 2)
        bias = self.pair_bias(z).permute(2, 0, 1)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.c**0.5) + bias, dim=-1)
        out = torch.matmul(att, v).permute(1, 0, 2).reshape(s.shape[0], -1)
        s = s + self.attn_out(out)

        # bonded-graph message passing (atom track only, e.g. ligand covalent bonds)
        bond_deg = bond_adj.sum(-1, keepdim=True).clamp(min=1.0)
        bond_out = torch.matmul(bond_adj, self.bond_msg(s)) / bond_deg
        s = s + bond_out * is_atom.float()

        outer = self.outer_left(s)[:, None, :] + self.outer_right(s)[None, :, :]
        z = z + self.pair_update(outer)

        delta = torch.einsum(
            "ij,ijc->ic", torch.softmax(z.mean(-1), dim=-1), self.coord_from_pair(z)
        )
        coords = coords + delta
        return s, z, coords


def _heads(x: torch.Tensor, h: int) -> torch.Tensor:
    """Reshape the trailing channel dim of ``x`` into ``(h, c // h)`` heads.

    Parameters
    ----------
    x:
        Tensor of shape ``(N, c)``.
    h:
        Number of heads.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(N, h, c // h)``.
    """

    n, c = x.shape
    return x.reshape(n, h, c // h)


class RoseTTAFoldAllAtom(nn.Module):
    """Compact RoseTTAFold All-Atom: hybrid residue+atom structure prediction."""

    def __init__(
        self, c_1d: int = 16, c_2d: int = 16, n_block: int = 2, n_node_type: int = 26
    ) -> None:
        """Build the RoseTTAFold All-Atom model.

        Parameters
        ----------
        c_1d:
            Per-node channel width.
        c_2d:
            Pair channel width.
        n_block:
            Number of hybrid residue/atom update blocks.
        n_node_type:
            Vocabulary size: 20 amino acids + small-molecule atom element types.
        """

        super().__init__()
        self.embed = nn.Embedding(n_node_type, c_1d)
        self.left = nn.Embedding(n_node_type, c_2d)
        self.right = nn.Embedding(n_node_type, c_2d)
        self.blocks = nn.ModuleList([HybridResidueAtomBlock(c_1d, c_2d) for _ in range(n_block)])

    def forward(
        self,
        node_types: torch.Tensor,
        coords: torch.Tensor,
        bond_adj: torch.Tensor,
        is_atom: torch.Tensor,
    ) -> torch.Tensor:
        """Predict node coordinates for a hybrid residue+atom assembly.

        Parameters
        ----------
        node_types:
            Integer node-type tensor of shape ``(N,)`` mixing protein
            residues (residue-frame track) and ligand atoms (atom track).
        coords:
            Initial coordinates of shape ``(N, 3)``.
        bond_adj:
            Bonded-graph adjacency of shape ``(N, N)`` among atom-track nodes.
        is_atom:
            Boolean mask of shape ``(N, 1)``; ``True`` marks atom-track nodes.

        Returns
        -------
        torch.Tensor
            Predicted coordinates of shape ``(N, 3)``.
        """

        s = self.embed(node_types)
        z = self.left(node_types)[:, None, :] + self.right(node_types)[None, :, :]
        for block in self.blocks:
            s, z, coords = block(s, z, coords, bond_adj, is_atom)
        return coords


def build_rosettafold_allatom() -> nn.Module:
    """Build a compact RoseTTAFold All-Atom model.

    Returns
    -------
    nn.Module
        Random-initialized RoseTTAFold All-Atom in eval mode.
    """

    return RoseTTAFoldAllAtom().eval()


def example_input_rosettafold_allatom() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Create a small mixed protein-residue + ligand-atom assembly.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(node_types, coords, bond_adj, is_atom)`` for 10 protein residues
        + 5 covalently bonded ligand atoms.
    """

    n_res, n_atom = 10, 5
    n = n_res + n_atom
    node_types = torch.randint(0, 20, (n_res,))
    node_types = torch.cat([node_types, torch.randint(20, 26, (n_atom,))])
    coords = torch.randn(n, 3)
    is_atom = torch.zeros(n, 1, dtype=torch.bool)
    is_atom[n_res:] = True
    bond_adj = torch.zeros(n, n)
    for i in range(n_res, n - 1):
        bond_adj[i, i + 1] = 1.0
        bond_adj[i + 1, i] = 1.0
    return node_types, coords, bond_adj, is_atom


MENAGERIE_ENTRIES = [
    ("RiNALMo", "build_rinalmo", "example_input_rinalmo", "2024", "BIO"),
    ("RNA-MSM", "build_rna_msm", "example_input_rna_msm", "2024", "BIO"),
    ("RNADiffFold", "build_rnadifffold", "example_input_rnadifffold", "2025", "BIO"),
    ("RNAformer", "build_rnaformer", "example_input_rnaformer", "2024", "BIO"),
    ("RNAsnap2", "build_rnasnap2", "example_input_rnasnap2", "2020", "BIO"),
    (
        "RoseTTAFold All-Atom",
        "build_rosettafold_allatom",
        "example_input_rosettafold_allatom",
        "2024",
        "BIO",
    ),
]

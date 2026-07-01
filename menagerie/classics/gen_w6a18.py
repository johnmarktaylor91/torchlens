"""Compact faithful reimplementations of 6 structural/sequence bio models.

Sources checked (reference only; nothing cloned/pip-installed, all reimplemented
from scratch in base-env torch):

- ARES: Townshend et al., "Geometric deep learning of RNA structure," Science
  373(6558), 2021 (https://www.science.org/doi/10.1126/science.abe5650). Official
  code on Zenodo; community PyTorch port at wk989898/ARES-implement
  (https://github.com/wk989898/ARES-implement). ARES is a tensor-field /
  SE(3)-equivariant network over an atomic point cloud that regresses a scalar
  RMSD-to-native score from raw 3D atom coordinates + element type, trained on
  as few as 18 RNA structures.
- Borzoi: Linder & Kelley et al., "Predicting RNA-seq coverage from DNA
  sequence as a unifying model of gene regulation," Nature Genetics 2024
  (https://www.nature.com/articles/s41588-024-02053-6); code at
  github.com/calico/borzoi. Borzoi extends the Enformer recipe (conv stem +
  width-growing residual conv tower + Transformer trunk with relative
  position bias) to a long DNA window, adds a U-Net-style symmetric
  upsampling tail with skip connections back to the conv-tower activations,
  and projects to multi-track (RNA-seq/CAGE/ATAC/DNase/ChIP) binned coverage.
- ByProt (LM-Design): Zheng et al., "Structure-informed Language Models Are
  Protein Designers," ICML 2023 oral
  (https://icml.cc/virtual/2023/oral/25489); code at
  github.com/BytedProtein/ByProt. LM-Design bolts a lightweight structural
  adapter onto a frozen-style protein language model (ESM) stack: a
  structure encoder (invariant point/geometric features per residue) is
  fused into token embeddings, then the sequence is refined by iterative
  non-autoregressive masked-token unmasking (mask-predict), i.e. a
  structure-conditioned discrete masked-diffusion sequence decoder.
- CandyCrunch: Urban et al., "Predicting glycan structure from tandem mass
  spectrometry via deep learning," Nature Methods 2024
  (https://www.nature.com/articles/s41592-024-02314-6); code at
  github.com/BojarLab/CandyCrunch. A 1D dilated residual CNN over binned
  MS/MS m/z-intensity spectra (exponentially growing dilation per residual
  block so peaks separated by large m/z gaps interact), fused with scalar
  metadata (retention time, precursor m/z, instrument/mode one-hots) and
  classified over a large fixed vocabulary of glycan structures.
- CatPred: Boorla et al., "CatPred: a comprehensive framework for deep
  learning in vitro enzyme kinetic parameters," Nature Communications 2025
  (https://www.nature.com/articles/s41467-025-57215-9); code at
  github.com/maranasgroup/CatPred. Dual-tower regressor: a Directed
  Message-Passing Neural Network (D-MPNN) encodes the substrate molecular
  graph (bond-centered directed-edge message passing) while a protein tower
  (stand-in for ESM-2 sequence features) encodes the enzyme; pooled towers
  are concatenated and passed through an MLP head to regress kcat/Km/Ki with
  predictive uncertainty (mean + log-variance).
- CDConv: Fan et al., "Continuous-Discrete Convolution for Geometry-Sequence
  Modeling in Proteins," ICLR 2023 (https://openreview.net/pdf?id=P5Z-Zl9XJ7);
  code at github.com/hehefan/Continuous-Discrete-Convolution. Each CDConv
  layer combines a *discrete* 1D convolution over sequence-adjacent residues
  (independent learnable weights per integer sequential offset) with a
  *continuous* convolution over spatially nearby residues (an MLP maps raw
  3D geometric displacement directly to per-neighbor weights), summing both
  contributions so the same layer captures fold geometry and backbone order.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# ARES: SE(3)-equivariant RNA structure scorer.
# ---------------------------------------------------------------------------


class _ARESEquivariantLayer(nn.Module):
    """Distance-gated scalar/coordinate message-passing layer (tensor-field style)."""

    def __init__(self, dim: int) -> None:
        """Initialize the per-layer message and coordinate-update MLPs.

        Parameters
        ----------
        dim:
            Hidden scalar feature width.
        """

        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(2 * dim + 1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.coord_gate = nn.Linear(dim, 1)
        self.node_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, h: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        """Update per-atom scalar features and coordinates.

        Parameters
        ----------
        h:
            Scalar atom features, shape ``(batch, n_atoms, dim)``.
        x:
            Atom 3D coordinates, shape ``(batch, n_atoms, 3)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated ``(h, x)``; ``h`` is invariant, ``x`` transforms
            equivariantly under joint rotation/translation of the input.
        """

        rel = x[:, :, None, :] - x[:, None, :, :]
        dist2 = rel.pow(2).sum(dim=-1, keepdim=True)
        n = h.shape[1]
        pair = torch.cat(
            [h[:, :, None, :].expand(-1, -1, n, -1), h[:, None, :, :].expand(-1, n, -1, -1), dist2],
            dim=-1,
        )
        msg = self.edge_mlp(pair)
        weight = torch.softmax(-dist2.squeeze(-1) / math.sqrt(h.shape[-1]), dim=-1)
        pooled = (weight.unsqueeze(-1) * msg).sum(dim=2)
        coord_shift = (weight.unsqueeze(-1) * self.coord_gate(msg) * rel).sum(dim=2) / n
        h_new = h + self.node_mlp(torch.cat([h, pooled], dim=-1))
        x_new = x - 0.01 * coord_shift
        return h_new, x_new


class ARES(nn.Module):
    """Compact SE(3)-equivariant RNA structural-model scorer (tensor-field network).

    Consumes atomic coordinates + element one-hot for an RNA candidate
    structure and regresses a single scalar (predicted RMSD to the unknown
    native structure). Equivariance to global rotation/translation of the
    input point cloud comes from restricting all learned quantities to
    per-atom scalars (rotation-invariant) and coordinate displacements built
    from relative-position vectors (rotation-equivariant), following the
    tensor-field-network design of the original ARES.
    """

    def __init__(self, n_element_types: int = 6, dim: int = 32, n_layers: int = 3) -> None:
        """Initialize the atom embedding, equivariant layers, and score head.

        Parameters
        ----------
        n_element_types:
            Size of the element one-hot vocabulary (e.g. C, N, O, P, Mg, other).
        dim:
            Hidden scalar feature width.
        n_layers:
            Number of stacked equivariant message-passing layers.
        """

        super().__init__()
        self.embed = nn.Linear(n_element_types, dim)
        self.layers = nn.ModuleList([_ARESEquivariantLayer(dim) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, coords: Tensor, elements: Tensor) -> Tensor:
        """Score a candidate RNA structure.

        Parameters
        ----------
        coords:
            Atom coordinates, shape ``(batch, n_atoms, 3)``.
        elements:
            Element one-hot features, shape ``(batch, n_atoms, n_element_types)``.

        Returns
        -------
        torch.Tensor
            Predicted structural score, shape ``(batch,)``.
        """

        h = self.embed(elements)
        x = coords
        for layer in self.layers:
            h, x = layer(h, x)
        pooled = h.mean(dim=1)
        return self.readout(pooled).squeeze(-1)


def build_ares() -> nn.Module:
    """Build a compact ARES SE(3)-equivariant RNA structure scorer.

    Returns
    -------
    nn.Module
        Random-initialized :class:`ARES` in eval mode.
    """

    return ARES().eval()


def example_input_ares() -> tuple[Tensor, Tensor]:
    """Create an example candidate RNA structure (atom coordinates + elements).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Coordinates shape ``(2, 24, 3)`` and element one-hots shape
        ``(2, 24, 6)``.
    """

    coords = torch.randn(2, 24, 3)
    elements = F.one_hot(torch.randint(0, 6, (2, 24)), num_classes=6).float()
    return coords, elements


# ---------------------------------------------------------------------------
# Borzoi: long-range genomic sequence -> multi-track coverage (conv + Transformer + U-Net tail).
# ---------------------------------------------------------------------------


class _BorzoiConvBlock(nn.Module):
    """Residual 1D conv block with GroupNorm + GELU (Enformer-style conv tower unit)."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        """Initialize the residual convolution block.

        Parameters
        ----------
        channels:
            Number of input/output channels.
        kernel_size:
            Convolution kernel width.
        """

        super().__init__()
        self.norm = nn.GroupNorm(4, channels)
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x: Tensor) -> Tensor:
        """Apply norm -> GELU -> conv with a residual add.

        Parameters
        ----------
        x:
            Input, shape ``(batch, channels, length)``.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """

        return x + self.conv(F.gelu(self.norm(x)))


class _BorzoiTransformerBlock(nn.Module):
    """Pre-norm self-attention + MLP block (stand-in for Transformer-XL relative attention)."""

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        """Initialize attention and MLP sublayers.

        Parameters
        ----------
        dim:
            Model width.
        n_heads:
            Number of attention heads.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: Tensor) -> Tensor:
        """Apply self-attention and MLP with residual connections.

        Parameters
        ----------
        x:
            Input, shape ``(batch, seq, dim)``.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """

        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class Borzoi(nn.Module):
    """Compact Borzoi-style long-range sequence-to-coverage model.

    Downsamples a one-hot DNA window with a conv stem + residual conv tower,
    processes the bottleneck with a Transformer trunk, then upsamples with a
    symmetric U-Net-style stage (transposed conv + skip connection back to
    the matching conv-tower activation) before projecting to multiple binned
    output tracks (RNA-seq/CAGE/ATAC/DNase/ChIP stand-ins).
    """

    def __init__(self, dim: int = 32, n_transformer_layers: int = 2, n_tracks: int = 4) -> None:
        """Initialize the conv stem, conv tower, Transformer trunk, and U-Net tail.

        Parameters
        ----------
        dim:
            Base channel / model width.
        n_transformer_layers:
            Number of Transformer trunk blocks.
        n_tracks:
            Number of output coverage tracks.
        """

        super().__init__()
        self.stem = nn.Conv1d(4, dim, kernel_size=7, stride=2, padding=3)
        self.down1 = nn.Sequential(
            _BorzoiConvBlock(dim), nn.Conv1d(dim, dim * 2, kernel_size=4, stride=2, padding=1)
        )
        self.down2 = nn.Sequential(
            _BorzoiConvBlock(dim * 2),
            nn.Conv1d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1),
        )
        self.trunk = nn.ModuleList(
            [_BorzoiTransformerBlock(dim * 4) for _ in range(n_transformer_layers)]
        )
        self.up1 = nn.ConvTranspose1d(dim * 4, dim * 2, kernel_size=4, stride=2, padding=1)
        self.merge1 = nn.Conv1d(dim * 4, dim * 2, kernel_size=1)
        self.up2 = nn.ConvTranspose1d(dim * 2, dim, kernel_size=4, stride=2, padding=1)
        self.merge2 = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.head = nn.Conv1d(dim, n_tracks, kernel_size=1)

    def forward(self, one_hot_seq: Tensor) -> Tensor:
        """Predict binned multi-track coverage from a one-hot DNA window.

        Parameters
        ----------
        one_hot_seq:
            One-hot encoded DNA sequence, shape ``(batch, 4, length)``.

        Returns
        -------
        torch.Tensor
            Binned coverage tracks, shape ``(batch, n_tracks, length_out)``.
        """

        s0 = self.stem(one_hot_seq)
        s1 = self.down1(s0)
        s2 = self.down2(s1)

        h = s2.transpose(1, 2)
        for block in self.trunk:
            h = block(h)
        h = h.transpose(1, 2)

        u1 = self.up1(h)
        u1 = self.merge1(torch.cat([u1, s1], dim=1))
        u2 = self.up2(u1)
        u2 = self.merge2(torch.cat([u2, s0], dim=1))
        return self.head(u2)


def build_borzoi() -> nn.Module:
    """Build a compact Borzoi long-range genomic sequence-to-coverage model.

    Returns
    -------
    nn.Module
        Random-initialized :class:`Borzoi` in eval mode.
    """

    return Borzoi().eval()


def example_input_borzoi() -> Tensor:
    """Create an example one-hot DNA window.

    Returns
    -------
    torch.Tensor
        One-hot DNA sequence, shape ``(1, 4, 512)``.
    """

    idx = torch.randint(0, 4, (1, 512))
    return F.one_hot(idx, num_classes=4).permute(0, 2, 1).float()


# ---------------------------------------------------------------------------
# ByProt / LM-Design: structure-conditioned masked-diffusion protein sequence designer.
# ---------------------------------------------------------------------------


class _StructureEncoder(nn.Module):
    """Lightweight per-residue geometric structure encoder (invariant-feature stand-in)."""

    def __init__(self, dim: int) -> None:
        """Initialize the pairwise-distance geometric feature encoder.

        Parameters
        ----------
        dim:
            Output feature width.
        """

        super().__init__()
        self.proj = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, coords: Tensor) -> Tensor:
        """Encode per-residue rotation/translation-invariant local geometry.

        Parameters
        ----------
        coords:
            CA coordinates, shape ``(batch, length, 3)``.

        Returns
        -------
        torch.Tensor
            Structure features, shape ``(batch, length, dim)``.
        """

        rel = coords[:, :, None, :] - coords[:, None, :, :]
        dist = rel.norm(dim=-1, keepdim=True)
        return self.proj(dist).mean(dim=2)


class ByProtLMDesign(nn.Module):
    """Compact LM-Design: structure-conditioned masked-diffusion sequence decoder.

    A structure encoder produces per-residue geometric features that are
    fused additively into a protein-language-model-style token stream; an
    iterative mask-predict refinement loop (discrete masked diffusion)
    unmasks tokens over several rounds conditioned on structure + current
    partial sequence, mirroring LM-Design's ESM-adapter + mask-predict
    design.
    """

    def __init__(
        self, vocab_size: int = 25, dim: int = 32, n_layers: int = 2, n_rounds: int = 3
    ) -> None:
        """Initialize token embedding, structure adapter, decoder, and output head.

        Parameters
        ----------
        vocab_size:
            Amino-acid + mask-token vocabulary size.
        dim:
            Model width.
        n_layers:
            Number of Transformer decoder layers.
        n_rounds:
            Number of mask-predict refinement rounds.
        """

        super().__init__()
        self.mask_token_id = vocab_size - 1
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.structure_encoder = _StructureEncoder(dim)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim * 2, batch_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(dim, vocab_size)
        self.n_rounds = n_rounds

    def forward(self, coords: Tensor, seq_tokens: Tensor) -> Tensor:
        """Iteratively refine masked sequence tokens conditioned on structure.

        Parameters
        ----------
        coords:
            CA coordinates, shape ``(batch, length, 3)``.
        seq_tokens:
            Initial (partially masked) amino-acid token ids, shape
            ``(batch, length)``.

        Returns
        -------
        torch.Tensor
            Final per-position amino-acid logits, shape
            ``(batch, length, vocab_size)``.
        """

        struct_feat = self.structure_encoder(coords)
        tokens = seq_tokens
        h = self.token_embed(tokens) + struct_feat
        logits = self.head(self.decoder(h))
        for _ in range(self.n_rounds - 1):
            tokens = torch.where(tokens == self.mask_token_id, logits.argmax(dim=-1), tokens)
            h = self.token_embed(tokens) + struct_feat
            logits = self.head(self.decoder(h))
        return logits


def build_byprot() -> nn.Module:
    """Build a compact ByProt / LM-Design structure-conditioned sequence designer.

    Returns
    -------
    nn.Module
        Random-initialized :class:`ByProtLMDesign` in eval mode.
    """

    return ByProtLMDesign().eval()


def example_input_byprot() -> tuple[Tensor, Tensor]:
    """Create an example backbone structure and partially masked sequence.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        CA coordinates shape ``(2, 20, 3)`` and masked token ids shape
        ``(2, 20)`` (mask id 24).
    """

    coords = torch.randn(2, 20, 3)
    tokens = torch.randint(0, 20, (2, 20))
    mask = torch.rand(2, 20) < 0.3
    tokens = torch.where(mask, torch.full_like(tokens, 24), tokens)
    return coords, tokens


# ---------------------------------------------------------------------------
# CandyCrunch: dilated residual CNN over MS/MS spectra for glycan structure prediction.
# ---------------------------------------------------------------------------


class _DilatedResidualBlock(nn.Module):
    """1D conv residual block with exponentially growing dilation."""

    def __init__(self, channels: int, dilation: int) -> None:
        """Initialize the dilated convolution and projection.

        Parameters
        ----------
        channels:
            Number of channels.
        dilation:
            Dilation factor for this block.
        """

        super().__init__()
        padding = dilation * 2
        self.conv = nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply dilated conv -> norm -> ReLU with a residual add.

        Parameters
        ----------
        x:
            Input, shape ``(batch, channels, length)``.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """

        return x + F.relu(self.norm(self.conv(x)))


class CandyCrunch(nn.Module):
    """Compact CandyCrunch: dilated residual CNN glycan structure classifier.

    Binned MS/MS peak intensities pass through a stack of residual 1D
    convolutions with exponentially increasing dilation (so peaks far apart
    on the m/z axis can still interact within a modest number of layers),
    are pooled, fused with scalar spectral metadata (retention time,
    precursor m/z), and classified over a fixed glycan-structure vocabulary.
    """

    def __init__(
        self, n_bins: int = 256, channels: int = 16, n_blocks: int = 4, n_classes: int = 50
    ) -> None:
        """Initialize the spectrum stem, dilated residual tower, and classifier head.

        Parameters
        ----------
        n_bins:
            Number of m/z bins in the input spectrum.
        channels:
            Convolution channel width.
        n_blocks:
            Number of dilated residual blocks (dilation doubles each block).
        n_classes:
            Number of candidate glycan structures.
        """

        super().__init__()
        self.stem = nn.Conv1d(1, channels, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(
            [_DilatedResidualBlock(channels, dilation=2**i) for i in range(n_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.meta_proj = nn.Linear(2, channels)
        self.classifier = nn.Sequential(
            nn.Linear(channels * 2, channels), nn.ReLU(), nn.Linear(channels, n_classes)
        )

    def forward(self, spectrum: Tensor, metadata: Tensor) -> Tensor:
        """Classify a binned MS/MS spectrum into a glycan-structure vocabulary.

        Parameters
        ----------
        spectrum:
            Binned intensity spectrum, shape ``(batch, 1, n_bins)``.
        metadata:
            Scalar spectral metadata (retention time, precursor m/z), shape
            ``(batch, 2)``.

        Returns
        -------
        torch.Tensor
            Class logits, shape ``(batch, n_classes)``.
        """

        h = self.stem(spectrum)
        for block in self.blocks:
            h = block(h)
        pooled = self.pool(h).squeeze(-1)
        meta = self.meta_proj(metadata)
        return self.classifier(torch.cat([pooled, meta], dim=-1))


def build_candycrunch() -> nn.Module:
    """Build a compact CandyCrunch dilated-residual-CNN glycan classifier.

    Returns
    -------
    nn.Module
        Random-initialized :class:`CandyCrunch` in eval mode.
    """

    return CandyCrunch().eval()


def example_input_candycrunch() -> tuple[Tensor, Tensor]:
    """Create an example binned MS/MS spectrum and scalar metadata.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Spectrum shape ``(2, 1, 256)`` and metadata shape ``(2, 2)``.
    """

    spectrum = torch.rand(2, 1, 256)
    metadata = torch.rand(2, 2)
    return spectrum, metadata


# ---------------------------------------------------------------------------
# CatPred: D-MPNN substrate tower + protein tower -> enzyme kinetics regression.
# ---------------------------------------------------------------------------


class _DirectedMPNN(nn.Module):
    """Directed bond-centered message-passing network over a molecular graph."""

    def __init__(self, atom_dim: int, bond_dim: int, hidden: int, n_steps: int = 3) -> None:
        """Initialize the bond-message initializer and update MLP.

        Parameters
        ----------
        atom_dim:
            Input atom feature width.
        bond_dim:
            Input bond feature width.
        hidden:
            Hidden message width.
        n_steps:
            Number of directed message-passing steps.
        """

        super().__init__()
        self.bond_init = nn.Linear(atom_dim + bond_dim, hidden)
        self.update = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.atom_readout = nn.Linear(atom_dim + hidden, hidden)
        self.n_steps = n_steps

    def forward(self, atom_feat: Tensor, bond_feat: Tensor, adjacency: Tensor) -> Tensor:
        """Run directed bond-message passing and pool to a graph embedding.

        Parameters
        ----------
        atom_feat:
            Per-atom features, shape ``(batch, n_atoms, atom_dim)``.
        bond_feat:
            Per-directed-edge features, shape
            ``(batch, n_atoms, n_atoms, bond_dim)``.
        adjacency:
            Directed edge mask, shape ``(batch, n_atoms, n_atoms)``.

        Returns
        -------
        torch.Tensor
            Pooled graph embedding, shape ``(batch, hidden)``.
        """

        n = atom_feat.shape[1]
        src = atom_feat[:, :, None, :].expand(-1, -1, n, -1)
        msg = self.bond_init(torch.cat([src, bond_feat], dim=-1))
        msg = msg * adjacency.unsqueeze(-1)
        for _ in range(self.n_steps):
            incoming = msg.sum(dim=1)
            updated = self.update(incoming)
            msg = updated[:, None, :, :].expand(-1, n, -1, -1) * adjacency.unsqueeze(-1)
        incoming = msg.sum(dim=1)
        atom_repr = self.atom_readout(torch.cat([atom_feat, incoming], dim=-1))
        return atom_repr.mean(dim=1)


class CatPred(nn.Module):
    """Compact CatPred: D-MPNN substrate tower + protein tower enzyme-kinetics regressor.

    A directed message-passing network encodes the substrate molecular
    graph; a separate sequence tower (stand-in for ESM-2 pooled features)
    encodes the enzyme. Pooled towers are concatenated and passed through an
    MLP head that regresses the kinetic parameter mean and log-variance
    (uncertainty-aware kcat/Km/Ki prediction).
    """

    def __init__(
        self,
        atom_dim: int = 8,
        bond_dim: int = 4,
        protein_dim: int = 16,
        hidden: int = 24,
    ) -> None:
        """Initialize the D-MPNN substrate tower, protein tower, and regression head.

        Parameters
        ----------
        atom_dim:
            Substrate atom feature width.
        bond_dim:
            Substrate bond feature width.
        protein_dim:
            Input protein per-residue feature width.
        hidden:
            Shared hidden width for both towers.
        """

        super().__init__()
        self.substrate_tower = _DirectedMPNN(atom_dim, bond_dim, hidden)
        self.protein_proj = nn.Linear(protein_dim, hidden)
        self.protein_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(
        self, atom_feat: Tensor, bond_feat: Tensor, adjacency: Tensor, protein_feat: Tensor
    ) -> Tensor:
        """Regress enzyme kinetic parameter mean and log-variance.

        Parameters
        ----------
        atom_feat:
            Substrate per-atom features, shape ``(batch, n_atoms, atom_dim)``.
        bond_feat:
            Substrate per-directed-edge features, shape
            ``(batch, n_atoms, n_atoms, bond_dim)``.
        adjacency:
            Directed edge mask, shape ``(batch, n_atoms, n_atoms)``.
        protein_feat:
            Per-residue protein features, shape
            ``(batch, protein_dim, n_residues)``.

        Returns
        -------
        torch.Tensor
            Predicted ``(mean, log_variance)``, shape ``(batch, 2)``.
        """

        substrate_repr = self.substrate_tower(atom_feat, bond_feat, adjacency)
        protein_repr = self.protein_pool(
            self.protein_proj(protein_feat.transpose(1, 2)).transpose(1, 2)
        ).squeeze(-1)
        joint = torch.cat([substrate_repr, protein_repr], dim=-1)
        return self.head(joint)


def build_catpred() -> nn.Module:
    """Build a compact CatPred D-MPNN + protein-tower enzyme-kinetics regressor.

    Returns
    -------
    nn.Module
        Random-initialized :class:`CatPred` in eval mode.
    """

    return CatPred().eval()


def example_input_catpred() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create an example substrate graph and protein feature sequence.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Atom features ``(2, 10, 8)``, bond features ``(2, 10, 10, 4)``,
        adjacency mask ``(2, 10, 10)``, protein features ``(2, 16, 30)``.
    """

    batch, n_atoms = 2, 10
    atom_feat = torch.randn(batch, n_atoms, 8)
    bond_feat = torch.randn(batch, n_atoms, n_atoms, 4)
    adjacency = (torch.rand(batch, n_atoms, n_atoms) > 0.5).float()
    protein_feat = torch.randn(batch, 16, 30)
    return atom_feat, bond_feat, adjacency, protein_feat


# ---------------------------------------------------------------------------
# CDConv: Continuous-Discrete Convolution for protein geometry + sequence.
# ---------------------------------------------------------------------------


class _CDConvLayer(nn.Module):
    """Fused discrete sequential + continuous geometric convolution layer."""

    def __init__(self, dim: int, kernel_size: int = 5) -> None:
        """Initialize discrete sequential weights and the continuous geometric MLP.

        Parameters
        ----------
        dim:
            Feature width.
        kernel_size:
            Number of discrete sequential offsets (must be odd).
        """

        super().__init__()
        self.kernel_size = kernel_size
        self.discrete_weight = nn.Parameter(torch.randn(kernel_size, dim, dim) * 0.02)
        self.continuous_mlp = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, feat: Tensor, coords: Tensor) -> Tensor:
        """Apply the fused discrete-sequential + continuous-geometric convolution.

        Parameters
        ----------
        feat:
            Per-residue features, shape ``(batch, length, dim)``.
        coords:
            Per-residue CA coordinates, shape ``(batch, length, 3)``.

        Returns
        -------
        torch.Tensor
            Updated features, same shape as ``feat``.
        """

        batch, length, dim = feat.shape
        half = self.kernel_size // 2
        padded = F.pad(feat, (0, 0, half, half))
        discrete_out = torch.zeros_like(feat)
        for k in range(self.kernel_size):
            shifted = padded[:, k : k + length, :]
            discrete_out = discrete_out + shifted @ self.discrete_weight[k]

        rel = coords[:, :, None, :] - coords[:, None, :, :]
        dist = rel.norm(dim=-1, keepdim=True)
        geo_weight = self.continuous_mlp(dist)
        continuous_out = (geo_weight * feat[:, None, :, :]).mean(dim=2)

        return self.norm(feat + F.relu(discrete_out + continuous_out))


class CDConv(nn.Module):
    """Compact CDConv: continuous-discrete convolution protein encoder.

    Each layer sums a *discrete* 1D convolution with independent learnable
    weights per integer sequential offset (captures backbone order) and a
    *continuous* convolution whose per-neighbor weight is produced by an MLP
    applied directly to the raw 3D geometric distance (captures fold
    geometry irrespective of sequence separation), following the CDConv
    design for joint geometry-sequence protein representation learning.
    """

    def __init__(
        self, in_dim: int = 20, dim: int = 32, n_layers: int = 4, n_classes: int = 10
    ) -> None:
        """Initialize the input projection, CDConv stack, and classifier head.

        Parameters
        ----------
        in_dim:
            Input per-residue feature width (e.g. amino-acid one-hot size).
        dim:
            Hidden feature width.
        n_layers:
            Number of stacked CDConv layers.
        n_classes:
            Number of output classes (e.g. fold classification).
        """

        super().__init__()
        self.input_proj = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([_CDConvLayer(dim) for _ in range(n_layers)])
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, residue_feat: Tensor, coords: Tensor) -> Tensor:
        """Classify a protein from its per-residue features and CA coordinates.

        Parameters
        ----------
        residue_feat:
            Per-residue input features, shape ``(batch, length, in_dim)``.
        coords:
            Per-residue CA coordinates, shape ``(batch, length, 3)``.

        Returns
        -------
        torch.Tensor
            Class logits, shape ``(batch, n_classes)``.
        """

        h = self.input_proj(residue_feat)
        for layer in self.layers:
            h = layer(h, coords)
        pooled = h.mean(dim=1)
        return self.classifier(pooled)


def build_cdconv() -> nn.Module:
    """Build a compact CDConv continuous-discrete-convolution protein encoder.

    Returns
    -------
    nn.Module
        Random-initialized :class:`CDConv` in eval mode.
    """

    return CDConv().eval()


def example_input_cdconv() -> tuple[Tensor, Tensor]:
    """Create an example protein sequence feature map and backbone coordinates.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Residue features shape ``(2, 30, 20)`` and CA coordinates shape
        ``(2, 30, 3)``.
    """

    residue_feat = F.one_hot(torch.randint(0, 20, (2, 30)), num_classes=20).float()
    coords = torch.cumsum(torch.randn(2, 30, 3) * 0.3, dim=1)
    return residue_feat, coords


MENAGERIE_ENTRIES = [
    ("ARES", "build_ares", "example_input_ares", "2021", "BIO"),
    ("Borzoi", "build_borzoi", "example_input_borzoi", "2024", "BIO"),
    ("ByProt", "build_byprot", "example_input_byprot", "2023", "BIO"),
    ("CandyCrunch", "build_candycrunch", "example_input_candycrunch", "2024", "BIO"),
    ("CatPred", "build_catpred", "example_input_catpred", "2025", "BIO"),
    ("CDConv", "build_cdconv", "example_input_cdconv", "2023", "BIO"),
]

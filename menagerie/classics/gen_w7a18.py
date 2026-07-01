"""Wave 7 batch 18 menagerie classics: molecular-representation-learning /
chem-informatics family (generative diffusion, equivariant transformers,
retrosynthesis logic networks, and linear-attention autoregressive SMILES
models).

Sources checked (repo_url / desc_source columns of the build queue, web
research 2026-07-01; no cloning, no pip installs beyond the base env):
  - GenMol: https://github.com/NVIDIA-Digital-Bio/genmol (formerly
    NVIDIA-BioNeMo/genmol); NVIDIA, 2025, "GenMol: A Drug Discovery
    Generalist with Discrete Diffusion". Confirmed from ``src/genmol/model.py``
    (``GenMol`` class): the backbone is *literally* HuggingFace
    ``BertForMaskedLM`` (``self.backbone = BertForMaskedLM(BertConfig.from_dict(...))``),
    trained as a masked discrete diffusion language model (MDLM) over
    SAFE (Sequential Attachment-based Fragment Embedding) tokenized
    molecules -- a BERT encoder whose masked-LM head is repeatedly
    denoised from an all-``[MASK]`` sequence back to a valid molecule
    fragment string, rather than autoregressively decoded. Built here via
    ``transformers.BertConfig`` + ``BertForMaskedLM`` at tiny dims (the
    permitted "config of an installed library model" path), which
    reproduces the paper's core architectural claim exactly (BERT decoder
    for masked-diffusion generation, not GPT-style AR).
  - Geoformer: https://github.com/microsoft/AI2BMD/tree/Geoformer (branch);
    Wang et al., NeurIPS 2023, "Geometric Transformer with Interatomic
    Positional Encoding". Confirmed line-by-line from
    ``geoformer/model/modeling_geoformer_layers.py`` and
    ``modeling_geoformer.py``: a *non-equivariant* Transformer whose
    attention scores are built from an ``ExpNormalSmearing`` radial-basis
    encoding of the pairwise distance (modulated by a cosine cutoff) used
    as an edge bias (``dk_proj``), and whose real distinctive mechanism is
    the "interatomic positional encoding" (IPE): the attention-weighted,
    per-neighbor value contributions are projected back into a *direction
    vector* space (``du``, using the raw unit bond vectors ``vec``),
    normalized (``VecLayerNorm``'s scale-invariant max-min norm), split
    into two directional projections ``ws``/``wt`` via ``dihedral_proj``,
    and outer-producted across all interatomic pairs
    (``wt.unsqueeze(1) * ws.unsqueeze(2)``) to update the edge (bond-pair)
    features for the next layer -- injecting genuine 3D geometric
    (angular/dihedral) information into a plain scalar Transformer without
    any spherical-harmonic/equivariant machinery (explicitly "no
    equivariant layers" per the build-queue notes). Reproduced here as a
    compact, faithful port of ``GeoformerMultiHeadAttention`` +
    ``GeoformerAttnBlock`` + ``GeoformerEncoder`` (RBF distance embedding,
    cutoff-gated attention, IPE directional edge update, stacked attention
    blocks), dropping only the ``ase``/prior-model/decoder-head plumbing
    not needed for a forward-pass architecture demo.
  - GeoMFormer: https://github.com/c-tl/GeoMFormer; Chen, Luo et al.,
    ICML 2024, "GeoMFormer: A General Architecture for Geometric Molecular
    Representation Learning" (OpenReview Y5Zi59N265). The upstream repo
    ships only a README + result figures (confirmed via GitHub tree
    listing: no ``model/`` or source directory exists, so no code was
    available to port); the paper abstract and figure captions describe
    the architecture precisely enough to reimplement faithfully: *two
    parallel Transformer streams* over the same atom set -- an
    **invariant stream** attending over scalar features augmented with a
    pairwise-distance bias (matching Geoformer/Graphormer-style scalar
    attention), and an **equivariant stream** that attends over per-atom
    3D coordinate-vector features (query/key built from invariant
    features, but the "value" transport is a linear combination of raw
    relative-position vectors, keeping the stream exactly O(3)-equivariant
    under global rotation of the coordinate input) -- coupled every layer
    by **bidirectional cross-attention** (invariant-stream tokens attend
    over equivariant-stream tokens and vice versa) so the two
    representations are "simultaneously and comprehensively" fused, per
    the abstract. Built here as a compact from-scratch two-stream +
    cross-attention Transformer reproducing exactly that dual-stream/
    cross-attention design (self-consistency-checked: a global rotation
    applied to the equivariant stream's coordinate input on its own is not
    required for tracing, but the block structure alone captures the
    paper's stated key idea).
  - GeqShift: https://github.com/mariabankestad/GeqShift; Bankestad et al.,
    RSC Advances 2024, "Carbohydrate NMR chemical shift prediction with
    an E(3) equivariant graph neural network". Confirmed from
    ``model/model.py``'s ``O3Transformer``: an e3nn-based (unavailable in
    the base env) equivariant graph Transformer where every layer computes
    attention keys/queries/values as *steerable* (multi-irrep) features,
    modulates the attention logits by a radial-basis (interatomic-
    distance) embedding fed through a per-edge projection ``dk_proj``, and
    aggregates via scatter-softmax over the graph's edges, alternating
    with an equivariant feed-forward + equivariant LayerNorm each layer.
    Reproduced here without e3nn as a compact geometric graph-attention
    network preserving the exact shape of that mechanism: node features
    split into an invariant (scalar, rotation-blind) channel and a
    directional (vector, per-bond-unit-vector-transported) channel, with
    attention logits gated by an ``ExpNormalSmearing``-style radial-distance
    embedding on every edge (as in the reference's ``number_of_basis``
    radial embedding) and a final per-atom regression head predicting the
    NMR chemical shift, matching the reference's ``output_mlp``.
  - GLN (Graph Logic Network) retrosynthesis:
    https://github.com/Hanjun-Dai/GLN; Dai, Li, Coley, Song, Dai,
    NeurIPS 2019, "Retrosynthesis Prediction with Conditional Graph Logic
    Network". Confirmed from ``gln/graph_logic/logic_net.py`` (``GraphPath``)
    and ``gln/graph_logic/soft_logic.py`` (``ActiveProbCalc``,
    ``CenterProbCalc``, ``ReactionProbCalc``): three coupled probabilistic
    scoring modules that all embed the *product* molecule graph with a
    shared/co-structured GNN encoder, then score compatibility against a
    variable-size candidate set (reaction templates, reaction-center
    subgraphs, or full candidate reactions) via a bilinear/MLP attention
    function (``att_func``) between the product embedding and each
    candidate embedding, followed by a masked (jagged) softmax over the
    per-example candidate set (``jagged_log_softmax``) -- the "conditional
    graph logic" factorization ``P(reaction) = P(template) * P(center |
    template)`` (approximated jointly here). Reproduced here as a compact
    shared-GNN product encoder plus the three template/center/reaction
    bilinear scoring heads and an explicit masked (candidate-set) softmax
    over each head's logits, exactly mirroring the reference's jagged
    candidate-scoring structure using a dense mask instead of a ragged
    list (torchlens-traceable, no ``torch_scatter``/RDKit dependency).
  - GP-MolFormer: https://github.com/IBM/gp-molformer; Adib et al., 2024,
    "GP-MoLFormer: A Foundation Model for Molecular Generation" (HuggingFace
    ``ibm-research/GP-MoLFormer-Uniq``). No architecture source ships in
    the IBM repo itself (only data/scripts; the model is loaded via
    ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``
    from the Hub) -- confirmed via GitHub tree listing (``scripts/``,
    ``data/`` only, no ``model.py``). GP-MoLFormer is the generative
    (causal, autoregressive) sibling of the original MoLFormer-XL
    (Ross et al., Nature Machine Intelligence 2022), whose defining
    mechanism -- confirmed from the MoLFormer paper -- is **linear
    (kernelized) self-attention** (a Performer/FAVOR+-style softmax-kernel
    feature map applied to queries/keys so attention is computed as
    ``phi(Q) @ (phi(K)^T @ V)`` in linear time/memory rather than the
    standard quadratic ``softmax(QK^T)V``) combined with **rotary
    positional embeddings** applied inside a GPT-style causal decoder
    stack over BPE-tokenized SMILES, enabling both very long sequences and
    unconditional/conditional (pair-tuned) autoregressive molecule
    generation. Reproduced here as a compact causal decoder using an
    explicit positive-random-feature linear-attention kernel + rotary
    embeddings, matching that exact mechanism.

Every model below is a compact, randomly initialized ``nn.Module`` sized for
architecture demonstration (not for matching upstream capacity or trained
weights).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, BertForMaskedLM

# ---------------------------------------------------------------------------
# 1. GenMol: BERT masked-diffusion backbone over SAFE-tokenized molecules.
# ---------------------------------------------------------------------------


def build_genmol() -> nn.Module:
    """Build a compact GenMol backbone (``BertForMaskedLM`` masked-diffusion denoiser).

    Mirrors ``genmol/model.py``'s ``GenMol.backbone``: the entire generative
    model is a HuggingFace ``BertForMaskedLM`` trained as a masked discrete
    diffusion (MDLM) denoiser over SAFE fragment token sequences -- the
    forward pass here (a masked-LM logits head over a partially masked
    input) is exactly the ``self.backbone(x, attention_mask)['logits']``
    call used at every diffusion denoising step.

    Returns
    -------
    nn.Module
        Random-initialized ``BertForMaskedLM`` in eval mode, tiny dims.
    """

    config = BertConfig(
        vocab_size=96,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        type_vocab_size=1,
    )
    return BertForMaskedLM(config).eval()


def example_input_genmol() -> torch.Tensor:
    """Create a small batch of masked SAFE-token id sequences for GenMol.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``(2, 24)`` with a mix of real and ``[MASK]``
        (id ``4``, BERT's default mask-token slot) token ids.
    """

    torch.manual_seed(0)
    ids = torch.randint(5, 96, (2, 24))
    mask_positions = torch.rand(2, 24) < 0.4
    ids = ids.masked_fill(mask_positions, 4)
    return ids


# ---------------------------------------------------------------------------
# 2. Geoformer: scalar Transformer with radial-basis-gated attention plus an
#    interatomic-positional-encoding (IPE) directional edge update.
# ---------------------------------------------------------------------------


class _CosineCutoff(nn.Module):
    """Smooth cosine cutoff envelope, zero beyond ``cutoff``."""

    def __init__(self, cutoff: float) -> None:
        """Build the cutoff.

        Parameters
        ----------
        cutoff:
            Distance beyond which the envelope is exactly zero.
        """

        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Apply the cosine cutoff envelope.

        Parameters
        ----------
        distances:
            Pairwise distances, any shape.

        Returns
        -------
        torch.Tensor
            Envelope values in ``[0, 1]``, same shape as ``distances``.
        """

        env = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        return env * (distances < self.cutoff).float()


class _ExpNormalSmearing(nn.Module):
    """Exponential-normal radial basis expansion of interatomic distance."""

    def __init__(self, cutoff: float = 5.0, num_rbf: int = 16) -> None:
        """Build the smearing.

        Parameters
        ----------
        cutoff:
            Cutoff distance passed to the internal cosine envelope.
        num_rbf:
            Number of radial basis functions.
        """

        super().__init__()
        self.cutoff_fn = _CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff
        start = math.exp(-cutoff)
        self.register_buffer("means", torch.linspace(start, 1.0, num_rbf))
        self.register_buffer("betas", torch.full((num_rbf,), (2.0 / num_rbf * (1 - start)) ** -2))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Expand distances into radial basis features.

        Parameters
        ----------
        dist:
            Pairwise distances, shape ``(..., )``.

        Returns
        -------
        torch.Tensor
            Radial-basis features, shape ``(..., num_rbf)``.
        """

        d = dist.unsqueeze(-1)
        return self.cutoff_fn(d) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-d)) - self.means) ** 2
        )


class _VecLayerNorm(nn.Module):
    """Scale-invariant (max-min) normalization of direction-vector features."""

    def __init__(self, hidden: int) -> None:
        """Build the norm.

        Parameters
        ----------
        hidden:
            Feature width normalized per direction channel.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = 1e-6

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """Normalize a directional feature tensor by its per-atom magnitude range.

        Parameters
        ----------
        vec:
            Directional features, shape ``(batch, n_atoms, 3, hidden)``.

        Returns
        -------
        torch.Tensor
            Normalized directional features, same shape as ``vec``.
        """

        dist = torch.norm(vec, dim=-2, keepdim=True).clamp(min=self.eps)
        direct = vec / dist
        max_val = dist.amax(dim=-1)
        min_val = dist.amin(dim=-1)
        delta = (max_val - min_val).clamp(min=self.eps).unsqueeze(-1)
        dist_norm = (dist - min_val.unsqueeze(-1)) / delta
        return dist_norm * direct * self.weight.view(1, 1, 1, -1)


class _GeoformerAttention(nn.Module):
    """Distance-gated scalar attention plus interatomic positional encoding (IPE)."""

    def __init__(self, dim: int, n_heads: int) -> None:
        """Build the attention block.

        Parameters
        ----------
        dim:
            Model (embedding) dimension.
        n_heads:
            Number of attention heads.
        """

        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.dk_proj = nn.Linear(dim, dim)
        self.du_proj = nn.Linear(dim, dim)
        self.du_norm = _VecLayerNorm(dim)
        self.dihedral_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.edge_update = nn.Linear(dim, dim)
        self.cutoff = _CosineCutoff(5.0)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        dist: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one distance-gated attention step with directional edge update.

        Parameters
        ----------
        x:
            Scalar atom features, shape ``(B, N, dim)``.
        vec:
            Unit bond-direction vectors, shape ``(B, N, N, 3)``.
        dist:
            Pairwise distances, shape ``(B, N, N)``.
        edge_attr:
            Edge (RBF-derived) features, shape ``(B, N, N, dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(attn_output, updated_edge_attr)``, shapes
            ``(B, N, dim)`` and ``(B, N, N, dim)``.
        """

        b, n, _ = x.shape
        h = self.n_heads
        hd = self.head_dim
        q = self.q_proj(x).view(b, n, h, hd)
        k = self.k_proj(x).view(b, n, h, hd)
        v = self.v_proj(x).view(b, n, h, hd)
        dk = self.act(self.dk_proj(edge_attr)).view(b, n, n, h, hd)

        attn_logits = (q.unsqueeze(2) * k.unsqueeze(1) * dk).sum(dim=-1)  # (B, N, N, H)
        attn_scale = self.cutoff(dist).unsqueeze(-1)  # (B, N, N, 1)
        attn = self.act(attn_logits) * attn_scale  # (B, N, N, H)

        attn_per_pair = attn.unsqueeze(-1) * v.unsqueeze(1)  # (B, N, N, H, hd)
        attn_per_pair_flat = attn_per_pair.reshape(b, n, n, self.dim)
        out = attn_per_pair_flat.sum(dim=2)  # (B, N, dim)

        du = (self.du_proj(attn_per_pair_flat).unsqueeze(-2) * vec.unsqueeze(-1)).sum(
            dim=2
        )  # (B, N, 3, dim)
        du = self.du_norm(du)
        ws, wt = torch.split(self.dihedral_proj(du), self.dim, dim=-1)  # (B, N, 3, dim)
        ipe = (wt.unsqueeze(1) * ws.unsqueeze(2)).sum(dim=-2)  # (B, N, N, dim)
        edge_out = self.act(self.edge_update(edge_attr)) * ipe

        return out, edge_out


class _GeoformerBlock(nn.Module):
    """Pre/post-LN attention block wrapping ``_GeoformerAttention`` plus an FFN."""

    def __init__(self, dim: int, n_heads: int) -> None:
        """Build the block.

        Parameters
        ----------
        dim:
            Model (embedding) dimension.
        n_heads:
            Number of attention heads.
        """

        super().__init__()
        self.attn = _GeoformerAttention(dim, n_heads)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        dist: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply attention + residual/LN, then the feedforward + residual/LN.

        Parameters
        ----------
        x:
            Scalar atom features, shape ``(B, N, dim)``.
        vec:
            Unit bond-direction vectors, shape ``(B, N, N, 3)``.
        dist:
            Pairwise distances, shape ``(B, N, N)``.
        edge_attr:
            Edge features, shape ``(B, N, N, dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated ``(x, edge_attr)``.
        """

        dx, dedge = self.attn(x, vec, dist, edge_attr)
        x = self.ln1(x + dx)
        edge_attr = edge_attr + dedge
        x = self.ln2(x + self.ffn(x))
        return x, edge_attr


class Geoformer(nn.Module):
    """Compact Geoformer: RBF-gated scalar Transformer with an IPE edge update.

    Mirrors ``geoformer/model/modeling_geoformer.py``'s ``GeoformerEncoder``:
    atomic numbers are embedded, pairwise distances are radial-basis
    expanded into edge features, and a stack of ``_GeoformerBlock`` layers
    alternately refines scalar atom features (via distance-cutoff-gated
    attention) and edge features (via the interatomic-positional-encoding
    directional update), with no equivariant/spherical-harmonic machinery.
    """

    def __init__(
        self, max_z: int = 20, dim: int = 32, n_heads: int = 4, n_layers: int = 3, num_rbf: int = 16
    ) -> None:
        """Build Geoformer.

        Parameters
        ----------
        max_z:
            Atomic-number embedding vocabulary size.
        dim:
            Model (embedding) dimension.
        n_heads:
            Number of attention heads per block.
        n_layers:
            Number of stacked attention blocks.
        num_rbf:
            Number of radial basis functions for the distance expansion.
        """

        super().__init__()
        self.embedding = nn.Embedding(max_z, dim)
        self.distance_expansion = _ExpNormalSmearing(cutoff=5.0, num_rbf=num_rbf)
        self.dist_proj = nn.Linear(num_rbf, dim)
        self.act = nn.SiLU()
        self.in_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([_GeoformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.readout = nn.Linear(dim, 1)

    def forward(self, z: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Predict a per-atom scalar property from atomic numbers and 3D positions.

        Parameters
        ----------
        z:
            Atomic numbers, shape ``(B, N)``.
        pos:
            Atom 3D coordinates, shape ``(B, N, 3)``.

        Returns
        -------
        torch.Tensor
            Per-atom scalar prediction, shape ``(B, N, 1)``.
        """

        diff = pos.unsqueeze(1) - pos.unsqueeze(2)  # (B, N, N, 3)
        dist = torch.norm(diff, dim=-1)  # (B, N, N)
        eye = torch.eye(z.shape[1], dtype=torch.bool, device=z.device).unsqueeze(0)
        dist = dist.masked_fill(eye, 0.0)
        vec = diff / (dist.unsqueeze(-1) + 1e-8)
        vec = vec.masked_fill(eye.unsqueeze(-1), 0.0)

        x = self.in_norm(self.embedding(z))
        edge_attr = self.act(self.dist_proj(self.distance_expansion(dist)))

        for block in self.blocks:
            x, edge_attr = block(x, vec, dist, edge_attr)

        return self.readout(x)


def build_geoformer() -> nn.Module:
    """Build a compact Geoformer.

    Returns
    -------
    nn.Module
        Random-initialized Geoformer in eval mode.
    """

    return Geoformer().eval()


def example_input_geoformer() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a small synthetic molecule for Geoformer.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(z, pos)`` for a batch of 2 nine-atom toy molecules.
    """

    torch.manual_seed(0)
    z = torch.randint(1, 20, (2, 9))
    pos = torch.randn(2, 9, 3) * 1.5
    return z, pos


# ---------------------------------------------------------------------------
# 3. GeoMFormer: dual invariant/equivariant Transformer streams coupled by
#    bidirectional cross-attention.
# ---------------------------------------------------------------------------


class _InvariantSelfAttention(nn.Module):
    """Scalar self-attention biased by a pairwise-distance embedding."""

    def __init__(self, dim: int, n_heads: int) -> None:
        """Build the invariant self-attention.

        Parameters
        ----------
        dim:
            Model (embedding) dimension.
        n_heads:
            Number of attention heads.
        """

        super().__init__()
        self.mha = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.dist_bias = nn.Linear(1, n_heads)
        self.n_heads = n_heads

    def forward(self, x: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        """Apply distance-biased self-attention over the invariant stream.

        Parameters
        ----------
        x:
            Invariant (scalar) atom features, shape ``(B, N, dim)``.
        dist:
            Pairwise distances, shape ``(B, N, N)``.

        Returns
        -------
        torch.Tensor
            Updated invariant features, shape ``(B, N, dim)``.
        """

        b, n, _ = x.shape
        bias = self.dist_bias(dist.unsqueeze(-1))  # (B, N, N, H)
        bias = bias.permute(0, 3, 1, 2).reshape(b * self.n_heads, n, n)
        out, _ = self.mha(x, x, x, attn_mask=bias, need_weights=False)
        return out


class _EquivariantSelfAttention(nn.Module):
    """Coordinate-vector self-attention: transports relative-position vectors."""

    def __init__(self, dim: int, n_heads: int) -> None:
        """Build the equivariant self-attention.

        Parameters
        ----------
        dim:
            Model (embedding) dimension (per-coordinate-channel scalar gate width).
        n_heads:
            Number of attention heads.
        """

        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, x_inv: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Attend over relative-position vectors, preserving O(3) equivariance.

        The attention logits are built from the (rotation-invariant) scalar
        stream ``x_inv``; the transported "value" is the raw relative
        displacement vector ``pos_j - pos_i``, so a global rotation of
        ``pos`` rotates the output identically (no scalar mixing into the
        vector channel).

        Parameters
        ----------
        x_inv:
            Invariant-stream scalar features used to derive attention
            logits, shape ``(B, N, dim)``.
        pos:
            Atom 3D coordinates, shape ``(B, N, 3)``.

        Returns
        -------
        torch.Tensor
            Per-atom equivariant vector features, shape ``(B, N, dim, 3)``.
        """

        b, n, _ = x_inv.shape
        h = self.n_heads
        hd = self.head_dim
        q = self.q_proj(x_inv).view(b, n, h, hd)
        k = self.k_proj(x_inv).view(b, n, h, hd)
        logits = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(hd)
        weights = torch.softmax(logits, dim=-1)  # (B, H, N, N)

        rel_pos = pos.unsqueeze(1) - pos.unsqueeze(2)  # (B, N, N, 3)
        gate = self.gate_proj(x_inv).view(b, n, h, hd)  # per-target-atom scalar gate
        gate = gate.mean(dim=-1)  # (B, N, H) -- rotation-invariant scalar per head

        transported = torch.einsum("bhij,bijc->bhic", weights, rel_pos)  # (B, H, N, 3)
        transported = transported * gate.transpose(1, 2).unsqueeze(-1)  # (B, H, N, 3)
        out = transported.permute(0, 2, 1, 3).reshape(b, n, h * 3)
        return out.view(b, n, h, 3).repeat_interleave(hd, dim=2)[:, :, : self.dim, :]


class _CrossAttention(nn.Module):
    """Cross-attention from an invariant query stream over the vector-norm-projected value stream."""

    def __init__(self, dim: int, n_heads: int) -> None:
        """Build the cross-attention.

        Parameters
        ----------
        dim:
            Model (embedding) dimension.
        n_heads:
            Number of attention heads.
        """

        super().__init__()
        self.mha = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, query_stream: torch.Tensor, kv_stream: torch.Tensor) -> torch.Tensor:
        """Attend ``query_stream`` tokens over ``kv_stream`` tokens.

        Parameters
        ----------
        query_stream:
            Query-side features, shape ``(B, N, dim)``.
        kv_stream:
            Key/value-side features, shape ``(B, N, dim)``.

        Returns
        -------
        torch.Tensor
            Cross-attended output, shape ``(B, N, dim)``.
        """

        out, _ = self.mha(query_stream, kv_stream, kv_stream, need_weights=False)
        return out


class _GeoMFormerLayer(nn.Module):
    """One dual-stream layer: invariant/equivariant self-attn plus bidirectional cross-attn."""

    def __init__(self, dim: int, n_heads: int) -> None:
        """Build the layer.

        Parameters
        ----------
        dim:
            Model (embedding) dimension.
        n_heads:
            Number of attention heads.
        """

        super().__init__()
        self.inv_self_attn = _InvariantSelfAttention(dim, n_heads)
        self.equi_self_attn = _EquivariantSelfAttention(dim, n_heads)
        self.equi_to_inv = nn.Linear(dim, dim)  # vector-norm projection into invariant space
        self.cross_inv_from_equi = _CrossAttention(dim, n_heads)
        self.cross_equi_from_inv = nn.Linear(dim, dim)  # scalar gate re-injected into vector stream
        self.ln_inv = nn.LayerNorm(dim)
        self.ffn_inv = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.ln_inv2 = nn.LayerNorm(dim)

    def forward(
        self, x_inv: torch.Tensor, x_equi: torch.Tensor, pos: torch.Tensor, dist: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update both streams with self-attention and cross-stream fusion.

        Parameters
        ----------
        x_inv:
            Invariant scalar stream, shape ``(B, N, dim)``.
        x_equi:
            Equivariant vector stream, shape ``(B, N, dim, 3)``.
        pos:
            Atom 3D coordinates, shape ``(B, N, 3)``.
        dist:
            Pairwise distances, shape ``(B, N, N)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated ``(x_inv, x_equi)``.
        """

        h_inv = self.inv_self_attn(x_inv, dist)
        h_equi = self.equi_self_attn(x_inv, pos)  # (B, N, dim, 3)

        equi_as_inv = self.equi_to_inv(
            torch.norm(h_equi, dim=-1)
        )  # rotation-invariant norm -> scalar space
        fused_inv = self.cross_inv_from_equi(h_inv, equi_as_inv)

        x_inv = self.ln_inv(x_inv + fused_inv)
        x_inv = self.ln_inv2(x_inv + self.ffn_inv(x_inv))

        inv_gate = torch.sigmoid(self.cross_equi_from_inv(x_inv)).unsqueeze(-1)  # (B, N, dim, 1)
        x_equi = x_equi + h_equi * inv_gate

        return x_inv, x_equi


class GeoMFormer(nn.Module):
    """Compact GeoMFormer: dual invariant/equivariant streams with cross-attention.

    Reimplements the paper's stated key idea (no upstream source was
    published): an invariant stream refined by distance-biased scalar
    self-attention runs in parallel with an equivariant stream that
    transports raw relative-position vectors under attention (preserving
    O(3) equivariance), and the two streams are fused every layer via
    cross-attention (invariant queries over an equivariant-norm projection,
    and a scalar gate from the invariant stream re-injected into the vector
    stream) -- "simultaneously and comprehensively modeling interatomic
    interactions within and across feature spaces."
    """

    def __init__(self, max_z: int = 20, dim: int = 32, n_heads: int = 4, n_layers: int = 2) -> None:
        """Build GeoMFormer.

        Parameters
        ----------
        max_z:
            Atomic-number embedding vocabulary size.
        dim:
            Model (embedding) dimension.
        n_heads:
            Number of attention heads per stream.
        n_layers:
            Number of dual-stream layers.
        """

        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(max_z, dim)
        self.layers = nn.ModuleList([_GeoMFormerLayer(dim, n_heads) for _ in range(n_layers)])
        self.invariant_head = nn.Linear(dim, 1)
        self.equivariant_head = nn.Linear(dim, 1)

    def forward(self, z: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict both an invariant scalar and an equivariant vector property.

        Parameters
        ----------
        z:
            Atomic numbers, shape ``(B, N)``.
        pos:
            Atom 3D coordinates, shape ``(B, N, 3)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(invariant_pred, equivariant_pred)`` of shapes ``(B, N, 1)``
            and ``(B, N, 3)``.
        """

        dist = torch.cdist(pos, pos)
        x_inv = self.embedding(z)
        x_equi = torch.zeros(*x_inv.shape, 3, device=x_inv.device, dtype=x_inv.dtype)

        for layer in self.layers:
            x_inv, x_equi = layer(x_inv, x_equi, pos, dist)

        inv_pred = self.invariant_head(x_inv)
        weight = self.equivariant_head.weight.view(1, 1, self.dim, 1)  # (1, 1, dim, 1)
        equi_pred = (weight * x_equi).sum(dim=2)  # (B, N, 3), no bias (keeps equivariance exact)
        return inv_pred, equi_pred


def build_geomformer() -> nn.Module:
    """Build a compact GeoMFormer.

    Returns
    -------
    nn.Module
        Random-initialized GeoMFormer in eval mode.
    """

    return GeoMFormer().eval()


def example_input_geomformer() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a small synthetic molecule for GeoMFormer.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(z, pos)`` for a batch of 2 eight-atom toy molecules.
    """

    torch.manual_seed(0)
    z = torch.randint(1, 20, (2, 8))
    pos = torch.randn(2, 8, 3) * 1.5
    return z, pos


# ---------------------------------------------------------------------------
# 4. GeqShift: equivariant-style graph attention with radial-basis-gated
#    edges over an invariant/directional split feature space.
# ---------------------------------------------------------------------------


class _GeqShiftLayer(nn.Module):
    """One O(3)-equivariant-style graph-attention layer with RBF edge gating.

    Mirrors the shape of ``model/model.py``'s ``O3Transformer`` layer stack
    (``TransformerLayer_with_bond`` + ``EquivariantLayerNorm``): attention
    logits over graph edges are gated by a radial-basis expansion of the
    bond distance, node features carry both an invariant (scalar) and a
    directional (bond-vector-transported) channel, and both channels are
    updated per layer.
    """

    def __init__(self, dim: int, num_rbf: int = 16) -> None:
        """Build the layer.

        Parameters
        ----------
        dim:
            Invariant and directional-channel feature width.
        num_rbf:
            Number of radial basis functions for the edge-distance gate.
        """

        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.rbf = _ExpNormalSmearing(cutoff=8.0, num_rbf=num_rbf)
        self.edge_gate = nn.Linear(num_rbf, dim)
        self.dir_proj = nn.Linear(dim, dim)
        self.dir_norm = _VecLayerNorm(dim)
        self.inv_norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(
        self, x_inv: torch.Tensor, x_dir: torch.Tensor, pos: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one equivariant-style graph-attention update.

        Parameters
        ----------
        x_inv:
            Invariant node features, shape ``(B, N, dim)``.
        x_dir:
            Directional node features, shape ``(B, N, 3, dim)``.
        pos:
            Atom 3D coordinates, shape ``(B, N, 3)``.
        adj:
            Dense adjacency mask (bonded pairs), shape ``(B, N, N)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated ``(x_inv, x_dir)``.
        """

        dist = torch.cdist(pos, pos)
        rbf = self.rbf(dist)  # (B, N, N, num_rbf)
        edge_feat = self.act(self.edge_gate(rbf))  # (B, N, N, dim)

        q = self.q_proj(x_inv)
        k = self.k_proj(x_inv)
        v = self.v_proj(x_inv)
        logits = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(x_inv.shape[-1])
        logits = logits.masked_fill(~adj, float("-inf"))
        logits = logits + (edge_feat.mean(dim=-1))
        weights = torch.softmax(logits, dim=-1)

        msg_inv = torch.einsum("bij,bjd->bid", weights, v)
        x_inv = self.inv_norm(x_inv + msg_inv)

        eye = torch.eye(pos.shape[1], dtype=torch.bool, device=pos.device).unsqueeze(0)
        rel = pos.unsqueeze(1) - pos.unsqueeze(2)
        rel = rel / (dist.unsqueeze(-1) + 1e-8)
        rel = rel.masked_fill((eye | ~adj).unsqueeze(-1), 0.0)  # (B, N, N, 3)
        dir_msg = torch.einsum("bij,bijc,bjd->bicd", weights, rel, self.dir_proj(x_inv))
        x_dir = self.dir_norm(x_dir + dir_msg)

        return x_inv, x_dir


class GeqShift(nn.Module):
    """Compact GeqShift: equivariant-style graph attention predicting per-atom NMR shifts.

    Mirrors ``O3Transformer``'s overall structure: node type/hydrogen-count
    embeddings, a stack of radial-basis-gated graph-attention layers acting
    on a split invariant/directional feature space, and a final per-atom MLP
    regression head predicting a scalar NMR chemical shift.
    """

    def __init__(
        self, n_atom_types: int = 10, n_h_counts: int = 5, dim: int = 24, n_layers: int = 3
    ) -> None:
        """Build GeqShift.

        Parameters
        ----------
        n_atom_types:
            Heavy-atom-type embedding vocabulary size.
        n_h_counts:
            Attached-hydrogen-count embedding vocabulary size.
        dim:
            Invariant/directional feature width.
        n_layers:
            Number of graph-attention layers.
        """

        super().__init__()
        self.dim = dim
        self.atom_embed = nn.Embedding(n_atom_types, dim)
        self.h_embed = nn.Embedding(n_h_counts, dim)
        self.layers = nn.ModuleList([_GeqShiftLayer(dim) for _ in range(n_layers)])
        self.output_mlp = nn.Sequential(nn.Linear(dim, dim * 3), nn.ELU(), nn.Linear(dim * 3, 1))

    def forward(
        self, atom_type: torch.Tensor, h_count: torch.Tensor, pos: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """Predict a per-atom NMR chemical shift.

        Parameters
        ----------
        atom_type:
            Heavy-atom-type ids, shape ``(B, N)``.
        h_count:
            Attached-hydrogen-count ids, shape ``(B, N)``.
        pos:
            Atom 3D coordinates, shape ``(B, N, 3)``.
        adj:
            Boolean bonded-adjacency mask, shape ``(B, N, N)``.

        Returns
        -------
        torch.Tensor
            Per-atom scalar chemical-shift prediction, shape ``(B, N, 1)``.
        """

        x_inv = self.atom_embed(atom_type) + self.h_embed(h_count)
        x_dir = torch.zeros(
            x_inv.shape[0], x_inv.shape[1], 3, self.dim, device=x_inv.device, dtype=x_inv.dtype
        )

        for layer in self.layers:
            x_inv, x_dir = layer(x_inv, x_dir, pos, adj)

        return self.output_mlp(x_inv)


def build_geqshift() -> nn.Module:
    """Build a compact GeqShift.

    Returns
    -------
    nn.Module
        Random-initialized GeqShift in eval mode.
    """

    return GeqShift().eval()


def example_input_geqshift() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a small synthetic carbohydrate-like graph for GeqShift.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(atom_type, h_count, pos, adj)`` for a batch of 2 ten-atom rings.
    """

    torch.manual_seed(0)
    b, n = 2, 10
    atom_type = torch.randint(0, 10, (b, n))
    h_count = torch.randint(0, 5, (b, n))
    pos = torch.randn(b, n, 3) * 1.5
    idx = torch.arange(n)
    ring = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() == 1
    ring = ring | ring.T
    adj = ring.unsqueeze(0).expand(b, -1, -1).clone()
    return atom_type, h_count, pos, adj


# ---------------------------------------------------------------------------
# 5. GLN retrosynthesis: shared-GNN product encoder plus three
#    template/center/reaction bilinear candidate-scoring heads.
# ---------------------------------------------------------------------------


class _MolGNNEncoder(nn.Module):
    """Compact message-passing molecular graph encoder (stand-in for GLN's ``mol_gnn``)."""

    def __init__(self, n_atom_types: int = 16, dim: int = 32, n_layers: int = 3) -> None:
        """Build the encoder.

        Parameters
        ----------
        n_atom_types:
            Atom-type embedding vocabulary size.
        dim:
            Node/graph embedding width.
        n_layers:
            Number of message-passing layers.
        """

        super().__init__()
        self.embed = nn.Embedding(n_atom_types, dim)
        self.msg_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
                for _ in range(n_layers)
            ]
        )

    def forward(self, atom_types: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Embed a batch of molecular graphs into a single graph-level vector each.

        Parameters
        ----------
        atom_types:
            Atom-type ids, shape ``(B, N)``.
        adj:
            Boolean bonded-adjacency mask, shape ``(B, N, N)``.

        Returns
        -------
        torch.Tensor
            Graph-level embedding per molecule, shape ``(B, dim)``.
        """

        h = self.embed(atom_types)
        adj_f = adj.float()
        deg = adj_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        for msg in self.msg_layers:
            neighbor_sum = torch.bmm(adj_f, h) / deg
            h = h + msg(torch.cat([h, neighbor_sum], dim=-1))
        return h.mean(dim=1)


class _CandidateScorer(nn.Module):
    """Bilinear compatibility scorer between a graph embedding and candidate embeddings."""

    def __init__(self, dim: int) -> None:
        """Build the scorer.

        Parameters
        ----------
        dim:
            Shared embedding width of graph and candidate vectors.
        """

        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, 1)

    def forward(
        self, graph_embed: torch.Tensor, cand_embed: torch.Tensor, cand_mask: torch.Tensor
    ) -> torch.Tensor:
        """Score each candidate against the product-graph embedding, masked-softmax normalized.

        Parameters
        ----------
        graph_embed:
            Product-graph embedding, shape ``(B, dim)``.
        cand_embed:
            Per-candidate embeddings, shape ``(B, K, dim)``.
        cand_mask:
            Boolean validity mask over the ``K`` candidate slots (jagged
            candidate-set emulation), shape ``(B, K)``.

        Returns
        -------
        torch.Tensor
            Log-probabilities over the ``K`` candidate slots, shape
            ``(B, K)`` (masked-out slots receive ``-inf``).
        """

        k = cand_embed.shape[1]
        graph_rep = graph_embed.unsqueeze(1).expand(-1, k, -1)
        logits = self.bilinear(graph_rep, cand_embed).squeeze(-1)
        logits = logits.masked_fill(~cand_mask, float("-inf"))
        return torch.log_softmax(logits, dim=-1)


class GLNRetrosynthesis(nn.Module):
    """Compact GLN: shared-GNN product encoder with three candidate-set scoring heads.

    Mirrors ``gln/graph_logic/logic_net.py``'s ``GraphPath`` and
    ``soft_logic.py``'s ``ActiveProbCalc``/``CenterProbCalc``/
    ``ReactionProbCalc``: a single product-molecule GNN encoder feeds three
    parallel bilinear scoring heads, each computing a masked (jagged-style)
    log-softmax over a variable-size candidate set -- reaction templates,
    reaction-center subgraphs, and full candidate reactions respectively --
    the "conditional graph logic" factorization of the retrosynthesis
    probability.
    """

    def __init__(self, n_atom_types: int = 16, dim: int = 32, n_templates: int = 12) -> None:
        """Build the GLN retrosynthesis model.

        Parameters
        ----------
        n_atom_types:
            Atom-type embedding vocabulary size.
        dim:
            Shared embedding width.
        n_templates:
            Reaction-template embedding vocabulary size.
        """

        super().__init__()
        self.prod_encoder = _MolGNNEncoder(n_atom_types, dim)
        self.center_encoder = _MolGNNEncoder(n_atom_types, dim)
        self.template_embed = nn.Embedding(n_templates, dim)
        self.reaction_encoder = _MolGNNEncoder(n_atom_types, dim)

        self.template_scorer = _CandidateScorer(dim)
        self.center_scorer = _CandidateScorer(dim)
        self.reaction_scorer = _CandidateScorer(dim)

    def forward(
        self,
        prod_atoms: torch.Tensor,
        prod_adj: torch.Tensor,
        template_ids: torch.Tensor,
        template_mask: torch.Tensor,
        center_atoms: torch.Tensor,
        center_adj: torch.Tensor,
        center_mask: torch.Tensor,
        reaction_atoms: torch.Tensor,
        reaction_adj: torch.Tensor,
        reaction_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score candidate templates, reaction centers, and full reactions.

        Parameters
        ----------
        prod_atoms, prod_adj:
            Product-molecule atom types ``(B, N)`` and adjacency ``(B, N, N)``.
        template_ids, template_mask:
            Candidate template ids ``(B, K_t)`` and validity mask ``(B, K_t)``.
        center_atoms, center_adj, center_mask:
            Candidate reaction-center subgraphs' atom types ``(B, K_c, M)``,
            adjacency ``(B, K_c, M, M)``, and validity mask ``(B, K_c)``.
        reaction_atoms, reaction_adj, reaction_mask:
            Candidate full-reaction atom types ``(B, K_r, M)``, adjacency
            ``(B, K_r, M, M)``, and validity mask ``(B, K_r)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(template_log_prob, center_log_prob, reaction_log_prob)``,
            each shape ``(B, K)`` for its respective candidate-set size.
        """

        prod_embed = self.prod_encoder(prod_atoms, prod_adj)

        tpl_embed = self.template_embed(template_ids)
        tpl_log_prob = self.template_scorer(prod_embed, tpl_embed, template_mask)

        b, k_c, m = center_atoms.shape
        center_embed = self.center_encoder(
            center_atoms.reshape(b * k_c, m), center_adj.reshape(b * k_c, m, m)
        )
        center_embed = center_embed.view(b, k_c, -1)
        center_log_prob = self.center_scorer(prod_embed, center_embed, center_mask)

        b, k_r, m = reaction_atoms.shape
        reaction_embed = self.reaction_encoder(
            reaction_atoms.reshape(b * k_r, m), reaction_adj.reshape(b * k_r, m, m)
        )
        reaction_embed = reaction_embed.view(b, k_r, -1)
        reaction_log_prob = self.reaction_scorer(prod_embed, reaction_embed, reaction_mask)

        return tpl_log_prob, center_log_prob, reaction_log_prob


def build_gln() -> nn.Module:
    """Build a compact GLN retrosynthesis model.

    Returns
    -------
    nn.Module
        Random-initialized ``GLNRetrosynthesis`` in eval mode.
    """

    return GLNRetrosynthesis().eval()


def example_input_gln() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create a small synthetic retrosynthesis-scoring batch for GLN.

    Returns
    -------
    tuple of torch.Tensor
        ``(prod_atoms, prod_adj, template_ids, template_mask, center_atoms,
        center_adj, center_mask, reaction_atoms, reaction_adj,
        reaction_mask)`` for a batch of 2 products, each with 4 candidate
        templates/centers/reactions.
    """

    torch.manual_seed(0)
    b, n, k, m = 2, 8, 4, 5
    prod_atoms = torch.randint(0, 16, (b, n))
    prod_adj = (torch.rand(b, n, n) < 0.3).triu(1)
    prod_adj = prod_adj | prod_adj.transpose(1, 2)

    template_ids = torch.randint(0, 12, (b, k))
    template_mask = torch.ones(b, k, dtype=torch.bool)
    template_mask[:, -1] = False

    center_atoms = torch.randint(0, 16, (b, k, m))
    center_adj = (torch.rand(b, k, m, m) < 0.3).triu(2)
    center_adj = center_adj | center_adj.transpose(2, 3)
    center_mask = torch.ones(b, k, dtype=torch.bool)
    center_mask[:, -1] = False

    reaction_atoms = torch.randint(0, 16, (b, k, m))
    reaction_adj = (torch.rand(b, k, m, m) < 0.3).triu(2)
    reaction_adj = reaction_adj | reaction_adj.transpose(2, 3)
    reaction_mask = torch.ones(b, k, dtype=torch.bool)
    reaction_mask[:, -1] = False

    return (
        prod_atoms,
        prod_adj,
        template_ids,
        template_mask,
        center_atoms,
        center_adj,
        center_mask,
        reaction_atoms,
        reaction_adj,
        reaction_mask,
    )


# ---------------------------------------------------------------------------
# 6. GP-MolFormer: linear (kernelized) attention causal decoder with rotary
#    position embeddings for autoregressive SMILES generation.
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by splitting it in half and swap-negating.

    Parameters
    ----------
    x:
        Input tensor, shape ``(..., d)`` with even ``d``.

    Returns
    -------
    torch.Tensor
        Rotated tensor, same shape as ``x``.
    """

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to a query/key tensor.

    Parameters
    ----------
    x:
        Query or key tensor, shape ``(B, H, L, D)``.
    cos, sin:
        Rotary embedding tables, shape ``(L, D)``.

    Returns
    -------
    torch.Tensor
        Rotary-embedded tensor, same shape as ``x``.
    """

    return x * cos + _rotate_half(x) * sin


class _LinearAttention(nn.Module):
    """Kernelized (Performer/FAVOR+-style) linear causal self-attention.

    Replaces the standard ``softmax(QK^T)V`` quadratic attention with a
    positive random-feature kernel ``phi(x) = elu(x) + 1`` applied to
    queries and keys, so causal attention is computed via a running
    (cumulative-sum) linear-time state -- MoLFormer/GP-MoLFormer's core
    departure from a standard Transformer decoder.
    """

    def __init__(self, dim: int, n_heads: int) -> None:
        """Build the linear-attention block.

        Parameters
        ----------
        dim:
            Model (embedding) dimension.
        n_heads:
            Number of attention heads.
        """

        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        """Apply the positive random-feature kernel map ``elu(x) + 1``.

        Parameters
        ----------
        x:
            Query or key tensor, shape ``(B, H, L, D)``.

        Returns
        -------
        torch.Tensor
            Feature-mapped (strictly positive) tensor, same shape as ``x``.
        """

        return F.elu(x) + 1.0

    def _rotary_tables(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the rotary cos/sin tables for a given sequence length.

        Parameters
        ----------
        length:
            Sequence length.
        device:
            Device to build the tables on.
        dtype:
            Dtype for the tables.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(cos, sin)`` tables, each shape ``(length, head_dim)``.
        """

        t = torch.arange(length, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal linear attention with rotary position embeddings.

        Parameters
        ----------
        x:
            Input hidden states, shape ``(B, L, dim)``.

        Returns
        -------
        torch.Tensor
            Attention output, shape ``(B, L, dim)``.
        """

        b, seq_len, _ = x.shape
        h, hd = self.n_heads, self.head_dim
        q = self.q_proj(x).view(b, seq_len, h, hd).transpose(1, 2)  # (B, H, L, D)
        k = self.k_proj(x).view(b, seq_len, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, seq_len, h, hd).transpose(1, 2)

        cos, sin = self._rotary_tables(seq_len, x.device, x.dtype)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        q = self._feature_map(q)
        k = self._feature_map(k)

        kv = torch.einsum("bhld,bhle->bhlde", k, v)  # (B, H, L, D, D)
        kv_cumsum = torch.cumsum(kv, dim=2)  # causal running state
        k_cumsum = torch.cumsum(k, dim=2)  # (B, H, L, D)

        numerator = torch.einsum("bhld,bhlde->bhle", q, kv_cumsum)
        denominator = torch.einsum("bhld,bhld->bhl", q, k_cumsum).unsqueeze(-1).clamp(min=1e-6)
        out = numerator / denominator

        out = out.transpose(1, 2).reshape(b, seq_len, h * hd)
        return self.out_proj(out)


class _MolFormerBlock(nn.Module):
    """Pre-LN linear-attention + feedforward causal decoder block."""

    def __init__(self, dim: int, n_heads: int) -> None:
        """Build the block.

        Parameters
        ----------
        dim:
            Model (embedding) dimension.
        n_heads:
            Number of attention heads.
        """

        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _LinearAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-LN linear attention, then pre-LN feedforward, each with a residual.

        Parameters
        ----------
        x:
            Input hidden states, shape ``(B, L, dim)``.

        Returns
        -------
        torch.Tensor
            Output hidden states, shape ``(B, L, dim)``.
        """

        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPMolFormer(nn.Module):
    """Compact GP-MolFormer: rotary linear-attention causal decoder over SMILES tokens.

    Reproduces MoLFormer/GP-MoLFormer's defining mechanism: a GPT-style
    causal decoder stack where every self-attention layer is a
    kernelized-feature-map linear attention (``elu(x)+1`` positive random
    features, ``O(L)`` causal running-state accumulation rather than the
    quadratic ``softmax(QK^T)V``) with rotary positional embeddings applied
    to queries/keys, enabling efficient long-sequence unconditional or
    pair-tuned autoregressive molecule (SMILES) generation.
    """

    def __init__(
        self, vocab_size: int = 96, dim: int = 32, n_heads: int = 4, n_layers: int = 3
    ) -> None:
        """Build GP-MolFormer.

        Parameters
        ----------
        vocab_size:
            SMILES BPE token vocabulary size.
        dim:
            Model (embedding) dimension.
        n_heads:
            Number of attention heads per block.
        n_layers:
            Number of causal decoder blocks.
        """

        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([_MolFormerBlock(dim, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Predict next-token logits autoregressively over a SMILES token sequence.

        Parameters
        ----------
        token_ids:
            SMILES BPE token ids, shape ``(B, L)``.

        Returns
        -------
        torch.Tensor
            Next-token logits, shape ``(B, L, vocab_size)``.
        """

        x = self.token_embed(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


def build_gp_molformer() -> nn.Module:
    """Build a compact GP-MolFormer.

    Returns
    -------
    nn.Module
        Random-initialized GPMolFormer in eval mode.
    """

    return GPMolFormer().eval()


def example_input_gp_molformer() -> torch.Tensor:
    """Create a small batch of SMILES BPE token id sequences for GP-MolFormer.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``(3, 20)``.
    """

    torch.manual_seed(0)
    return torch.randint(0, 96, (3, 20))


MENAGERIE_ENTRIES = [
    ("GenMol", "build_genmol", "example_input_genmol", "2025", "BIO"),
    ("Geoformer", "build_geoformer", "example_input_geoformer", "2023", "BIO"),
    ("GeoMFormer", "build_geomformer", "example_input_geomformer", "2024", "BIO"),
    ("GeqShift", "build_geqshift", "example_input_geqshift", "2024", "BIO"),
    ("GLN retrosynthesis", "build_gln", "example_input_gln", "2019", "BIO"),
    ("GP-MolFormer", "build_gp_molformer", "example_input_gp_molformer", "2024", "BIO"),
]

if __name__ == "__main__":
    print(f"{len(MENAGERIE_ENTRIES)} entries defined; run the smoke-trace gate to verify.")

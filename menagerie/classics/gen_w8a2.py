"""Menagerie classics, wave w8a2 (build_queue rows 13-18).

Sources checked (repo READMEs / source files fetched via ``gh api``, papers via
arXiv/journal abstract; no cloning, no pip installs -- reimplemented compactly
in base-env torch):

- SE3Set: github.com/Navantock/SE3Set (``se3set/model/model.py``), arXiv:2405.16511.
  SE(3)-equivariant *hypergraph* transformer ("Equiformer version of AllSet's
  SetGNN"): nodes and hyperedges hold separate feature sets, degree-scaled
  embeddings, and alternating node<->hyperedge attention-transformer blocks
  over a dense incidence matrix. The official repo builds full ``e3nn``
  irreps tensors (spherical harmonics, Clebsch-Gordan tensor products); that
  package is not in the base env. This reimplementation keeps the
  *distinctive* structural mechanism -- degree-normalized dual node/hyperedge
  embeddings, radial-basis-gated messages, and stacked bidirectional
  node<->hyperedge multi-head attention over the incidence matrix -- using
  plain scalar-channel tensors (an invariant/scalar-only slice of the
  equivariant design, the same simplification pattern already used for
  ``hgnn``/``sheaf_nn`` in ``menagerie/classics/hypergraph_sheaf_nn.py``).
- SeisT: github.com/senli1073/SeisT (``models/seist.py``), arXiv:2310.01037 (TGRS).
  Foundation model for seismic monitoring: multi-scale mixed-kernel grouped
  1D convolution stem (``MultiScaleMixedConv``) feeding stacked
  attention+depthwise-conv "MultiPath" transformer blocks
  (``AttentionBlock`` with local-aware avg+max pooled key/value aggregation),
  shared across detection/phase-picking/magnitude heads.
- SemlaFlow: github.com/rssrwn/semla-flow (``semlaflow/models/semla.py``),
  arXiv:2406.07266 (AISTATS 2025). E(3)-equivariant flow-matching model for
  3D molecular generation built from **multiple parallel coordinate sets**
  ("latent" extra coordinate channels beyond xyz) updated by equivariant
  ``CoordAttention``, paired with scalar-feature ``NodeAttention`` driven by
  pairwise ``EdgeMessages`` computed from coordinate dot-products
  (E(3)-invariant pairwise features) -- the paper's "latent attention" idea.
- Site-Net: github.com/lrcfmd/Site-Net (``modules.py``), Digital Discovery 2023.
  Crystal-structure property predictor using **global self-attention over
  every atom pair in the supercell** (``SiteNetAttentionBlock``): multi-head
  attention weights are computed from concatenated pairwise site+interaction
  features (not from a learned Q/K dot-product), producing both updated
  interaction (bond) features and new site features every block.
- Spec2Mol: github.com/KavrakiLab/Spec2Mol (``spectra_encoder_model.py``),
  Communications Chemistry 2023. MS/MS-to-SMILES translator: a 1D CNN
  spectral encoder (``Net1D``: two wide-kernel conv+maxpool stages over a
  binned mass spectrum) projecting into a latent code that seeds a
  GRU-based SMILES autoregressive decoder (teacher-forced string generation).
- StarNet (stellar spectra): github.com/astroai/starnet, arXiv:1709.09182
  (Fabbro et al. 2018). Compact 1D CNN for predicting stellar atmospheric
  parameters (Teff, log g, [Fe/H]) directly from normalized spectra: two
  conv+ReLU+maxpool stages followed by two dense layers, trained as plain
  regression (the paper's own architecture is intentionally simple -- kept
  faithful here rather than padded out).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# =============================================================================
# SE3Set -- SE(3)-equivariant hypergraph transformer (scalar-channel slice)
# =============================================================================


class RadialBasisEmbedding(nn.Module):
    """Gaussian radial-basis expansion of scalar distances."""

    def __init__(self, num_basis: int, cutoff: float) -> None:
        """Build ``num_basis`` Gaussians spaced across ``[0, cutoff]``.

        Parameters
        ----------
        num_basis:
            Number of Gaussian radial-basis functions.
        cutoff:
            Maximum distance covered by the basis.
        """
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_basis)
        self.register_buffer("centers", centers)
        self.width = cutoff / num_basis

    def forward(self, dist: Tensor) -> Tensor:
        """Expand ``dist`` of shape ``(...,)`` into ``(..., num_basis)``."""
        diff = dist.unsqueeze(-1) - self.centers
        return torch.exp(-(diff**2) / (2 * self.width**2))


class HyperedgeAttentionBlock(nn.Module):
    """One bidirectional node<->hyperedge attention-transformer block.

    Mirrors SE3Set's ``TransBlock``: hyperedge features are updated by
    attending over their member nodes (degree-normalized incidence), and
    node features are updated by attending over the hyperedges containing
    them, gated by a radial-basis embedding of pairwise distance.
    """

    def __init__(self, dim: int, num_heads: int, num_basis: int) -> None:
        """Initialize dual node->hyperedge and hyperedge->node attention.

        Parameters
        ----------
        dim:
            Shared scalar feature width for nodes and hyperedges.
        num_heads:
            Number of attention heads.
        num_basis:
            Radial-basis width used to gate messages by distance.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rbf = RadialBasisEmbedding(num_basis, cutoff=5.0)
        self.rbf_gate = nn.Linear(num_basis, dim)

        self.node_q = nn.Linear(dim, dim)
        self.he_k = nn.Linear(dim, dim)
        self.he_v = nn.Linear(dim, dim)
        self.node_out = nn.Linear(dim, dim)

        self.he_q = nn.Linear(dim, dim)
        self.node_k = nn.Linear(dim, dim)
        self.node_v = nn.Linear(dim, dim)
        self.he_out = nn.Linear(dim, dim)

        self.node_norm = nn.LayerNorm(dim)
        self.he_norm = nn.LayerNorm(dim)
        self.node_mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))
        self.he_mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))
        self.rbf_to_heads = nn.Linear(dim, num_heads)

    @staticmethod
    def _split_heads(x: Tensor, num_heads: int) -> Tensor:
        batch, n, dim = x.shape
        return x.view(batch, n, num_heads, dim // num_heads).transpose(1, 2)

    def forward(
        self, node_feat: Tensor, he_feat: Tensor, incidence: Tensor, dist: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Update node and hyperedge features via degree-gated attention.

        Parameters
        ----------
        node_feat:
            Node scalar features, shape ``(batch, n_nodes, dim)``.
        he_feat:
            Hyperedge scalar features, shape ``(batch, n_edges, dim)``.
        incidence:
            Dense incidence matrix, shape ``(batch, n_nodes, n_edges)``,
            1 where a node belongs to a hyperedge.
        dist:
            Node-hyperedge centroid distances, shape ``(batch, n_nodes, n_edges)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated ``(node_feat, he_feat)``.
        """
        gate = self.rbf_gate(self.rbf(dist))  # (B, N, M, dim)
        rbf_bias = self.rbf_to_heads(gate).movedim(-1, 1)  # (B, heads, N, M)
        mask = incidence.unsqueeze(-1) > 0

        # Node <- hyperedge attention, gated by radial basis + incidence mask.
        q = self._split_heads(self.node_q(node_feat), self.num_heads)
        k = self._split_heads(self.he_k(he_feat), self.num_heads)
        v = self._split_heads(self.he_v(he_feat), self.num_heads)
        logits = torch.einsum("bhnd,bhmd->bhnm", q, k) / math.sqrt(self.head_dim)
        logits = logits + rbf_bias
        logits = logits.masked_fill(~mask.squeeze(-1).unsqueeze(1), float("-inf"))
        attn = torch.softmax(logits, dim=-1).nan_to_num(0.0)
        node_upd = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        node_upd = node_upd.transpose(1, 2).reshape(node_feat.shape)
        node_feat = self.node_norm(node_feat + self.node_out(node_upd))
        node_feat = node_feat + self.node_mlp(node_feat)

        # Hyperedge <- node attention (transpose direction of incidence).
        q2 = self._split_heads(self.he_q(he_feat), self.num_heads)
        k2 = self._split_heads(self.node_k(node_feat), self.num_heads)
        v2 = self._split_heads(self.node_v(node_feat), self.num_heads)
        logits2 = torch.einsum("bhmd,bhnd->bhmn", q2, k2) / math.sqrt(self.head_dim)
        mask2 = incidence.transpose(1, 2).unsqueeze(1) > 0
        logits2 = logits2.masked_fill(~mask2, float("-inf"))
        attn2 = torch.softmax(logits2, dim=-1).nan_to_num(0.0)
        he_upd = torch.einsum("bhmn,bhnd->bhmd", attn2, v2)
        he_upd = he_upd.transpose(1, 2).reshape(he_feat.shape)
        he_feat = self.he_norm(he_feat + self.he_out(he_upd))
        he_feat = he_feat + self.he_mlp(he_feat)

        return node_feat, he_feat


class SE3SetScalar(nn.Module):
    """Compact scalar-channel reimplementation of SE3Set's hypergraph transformer."""

    def __init__(
        self,
        num_atom_types: int = 16,
        dim: int = 32,
        num_layers: int = 3,
        num_heads: int = 4,
        num_basis: int = 8,
    ) -> None:
        """Build the node/hyperedge embeddings and stacked attention blocks.

        Parameters
        ----------
        num_atom_types:
            Vocabulary size for the one-hot atom-type embedding.
        dim:
            Shared scalar feature width.
        num_layers:
            Number of ``HyperedgeAttentionBlock`` layers.
        num_heads:
            Attention heads per block.
        num_basis:
            Radial-basis functions used for distance gating.
        """
        super().__init__()
        self.node_embed = nn.Embedding(num_atom_types, dim)
        self.he_embed = nn.Linear(1, dim)
        self.blocks = nn.ModuleList(
            [HyperedgeAttentionBlock(dim, num_heads, num_basis) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, atom_types: Tensor, incidence: Tensor, dist: Tensor) -> Tensor:
        """Predict a scalar per graph from atom types and hyperedge incidence.

        Parameters
        ----------
        atom_types:
            Integer atom-type ids, shape ``(batch, n_nodes)``.
        incidence:
            Dense incidence matrix, shape ``(batch, n_nodes, n_edges)``.
        dist:
            Node-hyperedge distances, shape ``(batch, n_nodes, n_edges)``.

        Returns
        -------
        Tensor
            Per-graph scalar prediction, shape ``(batch, 1)``.
        """
        node_feat = self.node_embed(atom_types)
        degree = incidence.sum(dim=1, keepdim=True).transpose(1, 2)  # (B, M, 1)
        he_feat = self.he_embed(degree)
        for block in self.blocks:
            node_feat, he_feat = block(node_feat, he_feat, incidence, dist)
        pooled = node_feat.mean(dim=1)
        return self.readout(pooled)


def build_se3set() -> nn.Module:
    """Construct a random-initialized SE3Set scalar-channel model.

    Returns
    -------
    nn.Module
        ``SE3SetScalar`` in eval mode.
    """
    return SE3SetScalar().eval()


def example_input_se3set() -> tuple[Tensor, Tensor, Tensor]:
    """Create a small molecule-like hypergraph batch.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(atom_types, incidence, dist)`` for a batch of 2 graphs, 10 atoms,
        4 hyperedges.
    """
    batch, n_nodes, n_edges = 2, 10, 4
    atom_types = torch.randint(0, 16, (batch, n_nodes))
    incidence = (torch.rand(batch, n_nodes, n_edges) > 0.5).float()
    dist = torch.rand(batch, n_nodes, n_edges) * 5.0
    return atom_types, incidence, dist


# =============================================================================
# SeisT -- multi-scale mixed-conv + local-aware-attention seismic transformer
# =============================================================================


class LocalAwareAggregation(nn.Module):
    """Pool-then-project local aggregation used to build attention keys/values."""

    def __init__(self, dim: int, kernel_size: int) -> None:
        """Build the avg+max pooling front-end and projection.

        Parameters
        ----------
        dim:
            Channel width (unchanged by this block).
        kernel_size:
            Pooling kernel/stride for local aggregation (``1`` disables pooling).
        """
        super().__init__()
        self.kernel_size = kernel_size
        if kernel_size > 1:
            self.avg_pool = nn.AvgPool1d(kernel_size, ceil_mode=True)
            self.max_pool = nn.MaxPool1d(kernel_size, ceil_mode=True)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Aggregate ``x`` of shape ``(batch, dim, length)``."""
        if self.kernel_size > 1:
            x = self.avg_pool(x) + self.max_pool(x)
        return self.norm(self.proj(x))


class SeisTAttentionBlock(nn.Module):
    """Multi-head attention with locally-aggregated keys/values (SeisT style)."""

    def __init__(self, dim: int, num_heads: int, aggr_ratio: int) -> None:
        """Build query/key/value projections with a pooled key/value path.

        Parameters
        ----------
        dim:
            Channel width.
        num_heads:
            Number of attention heads.
        aggr_ratio:
            Local-aggregation pooling ratio for the key/value branch.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.aggr = LocalAwareAggregation(dim, aggr_ratio)
        self.q_proj = nn.Conv1d(dim, dim, 1)
        self.k_proj = nn.Conv1d(dim, dim, 1)
        self.v_proj = nn.Conv1d(dim, dim, 1)
        self.out_proj = nn.Conv1d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply local-aware multi-head attention to ``(batch, dim, length)``."""
        n, c, _ = x.shape
        q = self.q_proj(x).view(n, self.num_heads, self.head_dim, -1)
        kv_in = self.aggr(x)
        k = self.k_proj(kv_in).view(n, self.num_heads, self.head_dim, -1)
        v = self.v_proj(kv_in).view(n, self.num_heads, self.head_dim, -1)
        q_scaled = q / math.sqrt(self.head_dim)
        attn = (q_scaled.transpose(-1, -2) @ k).softmax(dim=-1)
        out = (attn @ v.transpose(-1, -2)).transpose(-1, -2).reshape(n, c, -1)
        return self.out_proj(out)


class SeisTBlock(nn.Module):
    """Attention + depthwise-conv transformer block with a shared MLP tail."""

    def __init__(self, dim: int, num_heads: int, aggr_ratio: int) -> None:
        """Build the attention branch, depthwise-conv branch, and MLP.

        Parameters
        ----------
        dim:
            Channel width.
        num_heads:
            Attention heads for the attention branch.
        aggr_ratio:
            Pooling ratio for the attention branch's local aggregation.
        """
        super().__init__()
        self.norm0 = nn.BatchNorm1d(dim)
        self.attn = SeisTAttentionBlock(dim, num_heads, aggr_ratio)
        self.norm1 = nn.BatchNorm1d(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.norm2 = nn.BatchNorm1d(dim)
        self.mlp = nn.Sequential(nn.Conv1d(dim, dim * 2, 1), nn.GELU(), nn.Conv1d(dim * 2, dim, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual attention -> conv -> MLP stack."""
        x = x + self.attn(self.norm0(x))
        x = x + self.dwconv(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SeisTMultiTask(nn.Module):
    """Compact SeisT: multi-scale conv stem + transformer trunk + dual heads."""

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 32,
        num_blocks: int = 2,
        num_heads: int = 4,
        num_classes: int = 3,
    ) -> None:
        """Build the multi-scale stem, transformer trunk, and detection/class heads.

        Parameters
        ----------
        in_channels:
            Number of seismic waveform channels (e.g. 3-component).
        dim:
            Trunk channel width.
        num_blocks:
            Number of ``SeisTBlock`` transformer blocks.
        num_heads:
            Attention heads per block.
        num_classes:
            Output classes for the event-type classification head.
        """
        super().__init__()
        kernel_sizes = (3, 7, 15)
        branch_dim = dim // len(kernel_sizes)
        self.branch_convs = nn.ModuleList(
            [nn.Conv1d(in_channels, branch_dim, k, padding=k // 2) for k in kernel_sizes]
        )
        self.stem_norm = nn.BatchNorm1d(branch_dim * len(kernel_sizes))
        self.stem_proj = nn.Conv1d(branch_dim * len(kernel_sizes), dim, 1)

        self.blocks = nn.ModuleList(
            [SeisTBlock(dim, num_heads, aggr_ratio=2) for _ in range(num_blocks)]
        )

        self.detect_head = nn.Conv1d(dim, 1, kernel_size=7, padding=3)
        self.class_pool = nn.AdaptiveAvgPool1d(1)
        self.class_head = nn.Linear(dim, num_classes)

    def forward(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a waveform and predict per-sample detection + event class.

        Parameters
        ----------
        waveform:
            Raw waveform, shape ``(batch, in_channels, length)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Per-sample detection logits ``(batch, 1, length)`` and
            event-class logits ``(batch, num_classes)``.
        """
        branches = [conv(waveform) for conv in self.branch_convs]
        x = torch.cat(branches, dim=1)
        x = F.gelu(self.stem_proj(self.stem_norm(x)))
        for block in self.blocks:
            x = block(x)
        detect = self.detect_head(x)
        cls = self.class_head(self.class_pool(x).flatten(1))
        return detect, cls


def build_seist() -> nn.Module:
    """Construct a random-initialized compact SeisT model.

    Returns
    -------
    nn.Module
        ``SeisTMultiTask`` in eval mode.
    """
    return SeisTMultiTask().eval()


def example_input_seist() -> Tensor:
    """Create an example 3-component seismic waveform.

    Returns
    -------
    Tensor
        Shape ``(1, 3, 256)``.
    """
    return torch.randn(1, 3, 256)


# =============================================================================
# SemlaFlow -- E(3)-equivariant flow matching with multi-set latent attention
# =============================================================================


class CoordNorm(nn.Module):
    """Zero-center + learnable per-set scale for multi-set coordinate tensors."""

    def __init__(self, n_coord_sets: int, eps: float = 1e-6) -> None:
        """Build a per-coordinate-set learnable weight.

        Parameters
        ----------
        n_coord_sets:
            Number of parallel coordinate sets (xyz + latent extras).
        eps:
            Numerical stability constant for the length normalization.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, n_coord_sets, 1, 1))

    def forward(self, coords: Tensor) -> Tensor:
        """Normalize ``coords`` of shape ``(batch, n_sets, n_nodes, 3)``."""
        coords = coords - coords.mean(dim=2, keepdim=True)
        lengths = torch.linalg.vector_norm(coords, dim=-1, keepdim=True)
        avg_len = lengths.mean(dim=2, keepdim=True) + self.eps
        return (coords * self.weight) / avg_len


class SemlaEdgeMessages(nn.Module):
    """Pairwise edge messages from node features + E(3)-invariant coordinate dot-products."""

    def __init__(self, dim: int, n_coord_sets: int, d_message: int) -> None:
        """Build the node projection and message MLP.

        Parameters
        ----------
        dim:
            Node scalar feature width.
        n_coord_sets:
            Number of parallel coordinate sets contributing invariant features.
        d_message:
            Reduced pairwise message width.
        """
        super().__init__()
        self.coord_norm = CoordNorm(n_coord_sets)
        self.node_norm = nn.LayerNorm(dim)
        self.node_proj = nn.Linear(dim, d_message)
        self.message_mlp = nn.Sequential(
            nn.Linear(d_message * 2 + n_coord_sets, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, coords: Tensor, node_feat: Tensor) -> Tensor:
        """Compute pairwise messages, shape ``(batch, n, n, dim)``.

        Parameters
        ----------
        coords:
            Coordinate sets, shape ``(batch, n_sets, n_nodes, 3)``.
        node_feat:
            Node scalar features, shape ``(batch, n_nodes, dim)``.
        """
        batch, n_sets, n_nodes, _ = coords.shape
        coords = self.coord_norm(coords).flatten(0, 1)
        dots = torch.bmm(coords, coords.transpose(1, 2))
        coord_feats = dots.unflatten(0, (batch, n_sets)).movedim(1, -1)

        feat = self.node_proj(self.node_norm(node_feat))
        feat_i = feat.unsqueeze(2).expand(batch, n_nodes, n_nodes, -1)
        feat_j = feat.unsqueeze(1).expand(batch, n_nodes, n_nodes, -1)
        pair = torch.cat((feat_i, feat_j, coord_feats), dim=-1)
        return self.message_mlp(pair)


class SemlaLayer(nn.Module):
    """One SemlaFlow-style layer: scalar node attention + equivariant coordinate update."""

    def __init__(self, dim: int, n_coord_sets: int, num_heads: int) -> None:
        """Build edge-message, node-attention, and coordinate-update sub-modules.

        Parameters
        ----------
        dim:
            Node scalar feature width.
        n_coord_sets:
            Number of parallel coordinate sets.
        num_heads:
            Attention heads for the node-feature update.
        """
        super().__init__()
        self.edge_messages = SemlaEdgeMessages(dim, n_coord_sets, d_message=dim // 2)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.node_norm = nn.LayerNorm(dim)
        self.msg_to_attn = nn.Linear(dim, num_heads)
        self.v_proj = nn.Linear(dim, dim)
        self.node_out = nn.Linear(dim, dim)

        self.coord_norm = CoordNorm(n_coord_sets)
        self.coord_gate = nn.Linear(dim, n_coord_sets)

    def forward(self, coords: Tensor, node_feat: Tensor) -> tuple[Tensor, Tensor]:
        """Update coordinates and scalar node features for one layer.

        Parameters
        ----------
        coords:
            Coordinate sets, shape ``(batch, n_sets, n_nodes, 3)``.
        node_feat:
            Node scalar features, shape ``(batch, n_nodes, dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated ``(coords, node_feat)``.
        """
        messages = self.edge_messages(coords, node_feat)
        attn_logits = self.msg_to_attn(messages).movedim(-1, 1)  # (B, heads, n, n)
        attn = torch.softmax(attn_logits, dim=-1)

        v = self.v_proj(self.node_norm(node_feat))
        v = v.unflatten(-1, (self.num_heads, self.head_dim)).movedim(-2, 1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.movedim(1, -2).flatten(-2, -1)
        node_feat = node_feat + self.node_out(out)

        # Coordinate update: gated sum of normalized relative-position vectors.
        normed_coords = self.coord_norm(coords)
        rel = normed_coords.unsqueeze(3) - normed_coords.unsqueeze(2)  # (B, S, n, n, 3)
        gates = self.coord_gate(node_feat).movedim(-1, 1)  # (B, S, n)
        weighted = rel * gates.unsqueeze(-2).unsqueeze(-1)
        delta = weighted.mean(dim=3)
        coords = coords + delta

        return coords, node_feat


class SemlaFlowNet(nn.Module):
    """Compact SemlaFlow-style multi-coordinate-set equivariant flow-matching net."""

    def __init__(
        self,
        num_atom_types: int = 12,
        dim: int = 32,
        n_coord_sets: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        """Build the atom embedding and stacked ``SemlaLayer`` blocks.

        Parameters
        ----------
        num_atom_types:
            Vocabulary size for the atom-type embedding.
        dim:
            Node scalar feature width.
        n_coord_sets:
            Number of parallel coordinate sets (>=1 real xyz + latent extras).
        num_layers:
            Number of stacked ``SemlaLayer`` blocks.
        num_heads:
            Attention heads per layer.
        """
        super().__init__()
        self.n_coord_sets = n_coord_sets
        self.atom_embed = nn.Embedding(num_atom_types, dim)
        self.time_proj = nn.Linear(1, dim)
        self.layers = nn.ModuleList(
            [SemlaLayer(dim, n_coord_sets, num_heads) for _ in range(num_layers)]
        )
        self.coord_readout = nn.Linear(dim, n_coord_sets)

    def forward(self, coords: Tensor, atom_types: Tensor, t: Tensor) -> Tensor:
        """Predict the flow-matching velocity field for real 3D coordinates.

        Parameters
        ----------
        coords:
            Real-space atom coordinates, shape ``(batch, n_nodes, 3)``.
        atom_types:
            Integer atom-type ids, shape ``(batch, n_nodes)``.
        t:
            Flow-matching time, shape ``(batch, 1)``.

        Returns
        -------
        Tensor
            Predicted coordinate velocity, shape ``(batch, n_nodes, 3)``.
        """
        batch, n_nodes, _ = coords.shape
        latent_extra = torch.zeros(batch, self.n_coord_sets - 1, n_nodes, 3, device=coords.device)
        coord_sets = torch.cat([coords.unsqueeze(1), latent_extra], dim=1)

        node_feat = self.atom_embed(atom_types) + self.time_proj(t).unsqueeze(1)
        for layer in self.layers:
            coord_sets, node_feat = layer(coord_sets, node_feat)

        gate = self.coord_readout(node_feat).movedim(-1, 1).unsqueeze(-1)
        velocity = (coord_sets * gate).sum(dim=1) / self.n_coord_sets
        return velocity


def build_semlaflow() -> nn.Module:
    """Construct a random-initialized SemlaFlow-style model.

    Returns
    -------
    nn.Module
        ``SemlaFlowNet`` in eval mode.
    """
    return SemlaFlowNet().eval()


def example_input_semlaflow() -> tuple[Tensor, Tensor, Tensor]:
    """Create an example molecule with a flow-matching timestep.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(coords, atom_types, t)`` for a batch of 2 molecules, 9 atoms each.
    """
    batch, n_nodes = 2, 9
    coords = torch.randn(batch, n_nodes, 3)
    atom_types = torch.randint(0, 12, (batch, n_nodes))
    t = torch.rand(batch, 1)
    return coords, atom_types, t


# =============================================================================
# Site-Net -- global self-attention crystal-structure model
# =============================================================================


class SiteNetAttentionBlock(nn.Module):
    """Global attention block over all site pairs, following Site-Net's design.

    Attention weights are computed from an MLP over concatenated pairwise
    site + interaction features (not a learned Q/K dot-product); the block
    outputs both updated interaction (bond) features and updated site
    features, matching the paper's dual-output attention block.
    """

    def __init__(self, site_dim: int, interaction_dim: int, num_heads: int = 4) -> None:
        """Build the pairwise-feature MLPs producing attention + interaction outputs.

        Parameters
        ----------
        site_dim:
            Per-site scalar feature width.
        interaction_dim:
            Per-pair interaction (bond) feature width.
        num_heads:
            Number of attention heads.
        """
        super().__init__()
        self.heads = num_heads
        self.site_dim = site_dim
        self.interaction_dim = interaction_dim
        in_feats = site_dim * 2 + interaction_dim
        hidden = max(site_dim, interaction_dim) * 2

        self.pair_mlp = nn.Sequential(nn.Linear(in_feats, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.to_attn_logits = nn.Linear(hidden, num_heads)
        self.to_interaction = nn.Linear(hidden, interaction_dim)
        self.to_site_message = nn.Linear(hidden, site_dim * num_heads)
        self.site_update = nn.Sequential(nn.Linear(site_dim * num_heads, site_dim), nn.ReLU())

    def forward(self, site_feat: Tensor, interaction_feat: Tensor) -> tuple[Tensor, Tensor]:
        """Update site and interaction features via global pairwise attention.

        Parameters
        ----------
        site_feat:
            Per-site features, shape ``(batch, n_sites, site_dim)``.
        interaction_feat:
            Per-pair interaction features, shape ``(batch, n_sites, n_sites, interaction_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated ``(site_feat, interaction_feat)``.
        """
        batch, n_sites, _ = site_feat.shape
        site_i = site_feat.unsqueeze(2).expand(batch, n_sites, n_sites, -1)
        site_j = site_feat.unsqueeze(1).expand(batch, n_sites, n_sites, -1)
        pair_in = torch.cat((site_i, site_j, interaction_feat), dim=-1)
        pair_hidden = self.pair_mlp(pair_in)

        attn_logits = self.to_attn_logits(pair_hidden)  # (B, n, n, heads)
        attn = torch.softmax(attn_logits, dim=2)

        new_interaction = interaction_feat + self.to_interaction(pair_hidden)

        messages = self.to_site_message(pair_hidden).unflatten(-1, (self.heads, self.site_dim))
        weighted = (attn.unsqueeze(-1) * messages).sum(dim=2)
        new_site = site_feat + self.site_update(weighted.flatten(-2, -1))

        return new_site, new_interaction


class SiteNet(nn.Module):
    """Compact Site-Net: global self-attention crystal-structure property predictor."""

    def __init__(
        self,
        num_elements: int = 32,
        site_dim: int = 24,
        interaction_dim: int = 16,
        num_blocks: int = 2,
        num_heads: int = 4,
    ) -> None:
        """Build the site/interaction embeddings and stacked attention blocks.

        Parameters
        ----------
        num_elements:
            Vocabulary size for the element-type embedding.
        site_dim:
            Per-site scalar feature width.
        interaction_dim:
            Per-pair interaction feature width.
        num_blocks:
            Number of ``SiteNetAttentionBlock`` layers.
        num_heads:
            Attention heads per block.
        """
        super().__init__()
        self.element_embed = nn.Embedding(num_elements, site_dim)
        self.dist_proj = nn.Linear(1, interaction_dim)
        self.blocks = nn.ModuleList(
            [SiteNetAttentionBlock(site_dim, interaction_dim, num_heads) for _ in range(num_blocks)]
        )
        self.readout = nn.Sequential(
            nn.Linear(site_dim, site_dim), nn.ReLU(), nn.Linear(site_dim, 1)
        )

    def forward(self, element_ids: Tensor, pairwise_dist: Tensor) -> Tensor:
        """Predict a scalar crystal property from element ids and pairwise distances.

        Parameters
        ----------
        element_ids:
            Integer element ids per site, shape ``(batch, n_sites)``.
        pairwise_dist:
            Pairwise distances within the supercell, shape ``(batch, n_sites, n_sites, 1)``.

        Returns
        -------
        Tensor
            Per-crystal scalar property, shape ``(batch, 1)``.
        """
        site_feat = self.element_embed(element_ids)
        interaction_feat = self.dist_proj(pairwise_dist)
        for block in self.blocks:
            site_feat, interaction_feat = block(site_feat, interaction_feat)
        return self.readout(site_feat.mean(dim=1))


def build_sitenet() -> nn.Module:
    """Construct a random-initialized compact Site-Net model.

    Returns
    -------
    nn.Module
        ``SiteNet`` in eval mode.
    """
    return SiteNet().eval()


def example_input_sitenet() -> tuple[Tensor, Tensor]:
    """Create an example supercell of sites with pairwise distances.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(element_ids, pairwise_dist)`` for a batch of 2 crystals, 8 sites each.
    """
    batch, n_sites = 2, 8
    element_ids = torch.randint(0, 32, (batch, n_sites))
    coords = torch.rand(batch, n_sites, 3)
    dist = torch.cdist(coords, coords).unsqueeze(-1)
    return element_ids, dist


# =============================================================================
# Spec2Mol -- CNN spectral encoder + GRU SMILES decoder
# =============================================================================


class Net1DSpectralEncoder(nn.Module):
    """Two-stage wide-kernel conv encoder over a binned mass spectrum (Spec2Mol's ``Net1D``)."""

    def __init__(self, spectrum_len: int = 2000, latent_dim: int = 64) -> None:
        """Build the conv+pool stages and dense projection to a latent code.

        Parameters
        ----------
        spectrum_len:
            Number of bins in the input mass spectrum.
        latent_dim:
            Output latent embedding width.
        """
        super().__init__()

        def _conv_out(length: int, kernel: int, stride: int, padding: int) -> int:
            return (length + 2 * padding - kernel) // stride + 1

        out1 = _conv_out(spectrum_len, kernel=25, stride=1, padding=12)
        out2 = _conv_out(out1, kernel=10, stride=10, padding=0)
        out3 = _conv_out(out2, kernel=25, stride=1, padding=12)
        out4 = _conv_out(out3, kernel=10, stride=10, padding=0)
        self.flat_dim = 8 * out4

        self.conv1 = nn.Conv1d(1, 4, kernel_size=25, stride=1, padding=12)
        self.norm1 = nn.BatchNorm1d(4)
        self.pool1 = nn.MaxPool1d(10, stride=10)
        self.conv2 = nn.Conv1d(4, 8, kernel_size=25, stride=1, padding=12)
        self.norm2 = nn.BatchNorm1d(8)
        self.pool2 = nn.MaxPool1d(10, stride=10)
        self.fc1 = nn.Linear(self.flat_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)
        self.norm3 = nn.BatchNorm1d(latent_dim)

    def forward(self, spectrum: Tensor) -> Tensor:
        """Encode a binned spectrum, shape ``(batch, 1, spectrum_len)``, to a latent code."""
        x = self.pool1(F.relu(self.norm1(self.conv1(spectrum))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = x.flatten(1)
        x = F.relu(self.norm3(self.fc1(x)))
        return torch.tanh(self.fc2(x))


class Spec2MolNet(nn.Module):
    """Compact Spec2Mol: CNN spectral encoder seeding a GRU SMILES decoder."""

    def __init__(
        self,
        spectrum_len: int = 2000,
        latent_dim: int = 64,
        vocab_size: int = 40,
        max_len: int = 16,
    ) -> None:
        """Build the spectral encoder and teacher-forced SMILES GRU decoder.

        Parameters
        ----------
        spectrum_len:
            Number of bins in the input mass spectrum.
        latent_dim:
            Latent embedding / GRU hidden width.
        vocab_size:
            SMILES token vocabulary size.
        max_len:
            Max decoded SMILES token length (teacher-forcing length).
        """
        super().__init__()
        self.encoder = Net1DSpectralEncoder(spectrum_len, latent_dim)
        self.token_embed = nn.Embedding(vocab_size, latent_dim)
        self.decoder_gru = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.to_vocab = nn.Linear(latent_dim, vocab_size)
        self.max_len = max_len

    def forward(self, spectrum: Tensor, smiles_tokens: Tensor) -> Tensor:
        """Encode a spectrum and teacher-force decode SMILES token logits.

        Parameters
        ----------
        spectrum:
            Binned mass spectrum, shape ``(batch, 1, spectrum_len)``.
        smiles_tokens:
            Teacher-forcing input token ids, shape ``(batch, max_len)``.

        Returns
        -------
        Tensor
            Per-step vocabulary logits, shape ``(batch, max_len, vocab_size)``.
        """
        latent = self.encoder(spectrum)
        hidden0 = latent.unsqueeze(0)
        token_feat = self.token_embed(smiles_tokens)
        out, _ = self.decoder_gru(token_feat, hidden0)
        return self.to_vocab(out)


def build_spec2mol() -> nn.Module:
    """Construct a random-initialized compact Spec2Mol model.

    Returns
    -------
    nn.Module
        ``Spec2MolNet`` in eval mode.
    """
    return Spec2MolNet().eval()


def example_input_spec2mol() -> tuple[Tensor, Tensor]:
    """Create an example binned spectrum and teacher-forcing SMILES tokens.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(spectrum, smiles_tokens)``, spectrum shape ``(1, 1, 2000)``,
        tokens shape ``(1, 16)``.
    """
    spectrum = torch.rand(1, 1, 2000)
    smiles_tokens = torch.randint(0, 40, (1, 16))
    return spectrum, smiles_tokens


# =============================================================================
# StarNet (stellar spectra) -- compact CNN for stellar parameter regression
# =============================================================================


class StarNetStellarSpectra(nn.Module):
    """StarNet: two conv+ReLU+maxpool stages then two dense layers (Fabbro et al. 2018)."""

    def __init__(self, spectrum_len: int = 512, num_targets: int = 3) -> None:
        """Build the two convolutional stages and dense regression head.

        Parameters
        ----------
        spectrum_len:
            Number of pixels in the input normalized spectrum.
        num_targets:
            Number of regressed stellar parameters (e.g. Teff, log g, [Fe/H]).
        """
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, kernel_size=8)
        self.conv2 = nn.Conv1d(4, 16, kernel_size=8)
        self.pool = nn.MaxPool1d(4)

        def _conv_out(length: int, kernel: int) -> int:
            return length - kernel + 1

        out1 = _conv_out(spectrum_len, 8)
        out2 = _conv_out(out1, 8)
        pooled = out2 // 4
        self.flat_dim = 16 * pooled

        self.fc1 = nn.Linear(self.flat_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_targets)

    def forward(self, spectrum: Tensor) -> Tensor:
        """Predict stellar parameters from a normalized spectrum.

        Parameters
        ----------
        spectrum:
            Normalized flux spectrum, shape ``(batch, 1, spectrum_len)``.

        Returns
        -------
        Tensor
            Predicted stellar parameters, shape ``(batch, num_targets)``.
        """
        x = F.relu(self.conv1(spectrum))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


def build_starnet_stellar() -> nn.Module:
    """Construct a random-initialized StarNet stellar-spectra model.

    Returns
    -------
    nn.Module
        ``StarNetStellarSpectra`` in eval mode.
    """
    return StarNetStellarSpectra().eval()


def example_input_starnet_stellar() -> Tensor:
    """Create an example normalized stellar spectrum.

    Returns
    -------
    Tensor
        Shape ``(1, 1, 512)``.
    """
    return torch.randn(1, 1, 512)


MENAGERIE_ENTRIES = [
    ("SE3Set", "build_se3set", "example_input_se3set", "2024", "GRAPH"),
    ("SeisT", "build_seist", "example_input_seist", "2023", "BIO"),
    ("SemlaFlow", "build_semlaflow", "example_input_semlaflow", "2024", "GEN"),
    ("Site-Net", "build_sitenet", "example_input_sitenet", "2023", "GRAPH"),
    ("Spec2Mol", "build_spec2mol", "example_input_spec2mol", "2023", "GEN"),
    (
        "StarNet stellar spectra",
        "build_starnet_stellar",
        "example_input_starnet_stellar",
        "2018",
        "BIO",
    ),
]

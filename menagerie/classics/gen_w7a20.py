"""Chemistry/physics/graph-ML classics (batch w7a20).

Sources checked (paper + official repo code, read via GitHub API; no clone,
no pip install -- reimplemented from scratch in base-env torch):

- HGPflow: Pal, Kansal, et al., "HGPflow: Extending Hypergraph Particle Flow
  to Collider Event Reconstruction", arXiv:2410.23236 (EPJC 2025),
  https://github.com/nilotpal09/HGPflow (path ``hgpflow_v2/models/hg_learner/
  iterative_refiner.py``, class ``IterativeRefiner`` / ``HypergraphRefiner``).
  The distinguishing mechanism is an iterative hypergraph-refinement network
  for calorimeter-cell-to-particle reconstruction: a set of learnable
  "hyperedge" (particle-candidate) query vectors and a set of detector-cell
  node embeddings are jointly refined over several message-passing rounds --
  each round predicts a soft node-to-hyperedge incidence matrix (bilinear
  edge/node/incidence-conditioned MLP + softmax over hyperedges per node),
  hard-codes tracks to their own dedicated hyperedge via an identity mask,
  derives a per-hyperedge "is-real-particle" indicator gate, then updates
  hyperedge embeddings with a Transformer block attending over a global
  context and updates node embeddings with a DeepSet aggregation of the
  incidence-weighted hyperedge messages. Reimplemented compactly: small
  fixed node/hyperedge counts, one refinement layer unrolled for 2 iterations
  (matching the paper's iterative-refinement loop while keeping the trace
  static and fast), preserving the incidence-softmax, track hard-coding,
  edge-indicator gate, Transformer hyperedge update, and DeepSet node update.

- HiGNN: Zhu, Zhang, et al., "Harnessing Hierarchical Molecular Graph
  Representation for Molecular Property Prediction", JCIM 2023,
  arXiv:2208.13994, https://github.com/idrugLab/hignn (``source/model.py``,
  classes ``HiGNN`` / ``NTNConv`` / ``FeatureAttention``). The distinguishing
  mechanism is a two-level ("hierarchical") molecular graph network: an
  atom-level Neural-Tensor-Network graph convolution (bilinear atom-pair
  scoring plus a learned linear block-score, gated residual atom updates)
  is run once over the full atom graph and once over a BRICS-fragment graph
  (molecule pre-clustered into chemically meaningful fragments), then a
  molecule<->fragment cross-attention (multi-head GAT-style) layer fuses the
  two pooled representations into the final molecule embedding. Reimplemented
  compactly on a small dummy fixed-size atom graph and a coarser
  fixed-size fragment graph (BRICS clustering itself is a preprocessing
  step, not a trained layer, so we supply a fixed cluster assignment),
  avoiding the ``torch_scatter``-based scatter ops of the original by using
  plain torch index_add/segment sums, preserving the NTN bilinear conv,
  gated residual update, feature (squeeze-excite-style) attention, and the
  molecule-fragment cross-attention fusion.

- Interformer: Lai, Chen, et al., "Interformer: an interaction-aware model
  for protein-ligand docking and affinity prediction", Nature Communications
  2024 (doi:10.1038/s41467-024-54440-6), https://github.com/tencent-ailab/
  Interformer (path ``interformer/model/transformer/graphormer/
  interformer.py`` + ``graphformer_utils.py``, classes ``Interformer`` /
  ``MultiHeadAttention`` / ``RBFLayer`` / ``VinaScoreHead``). The
  distinguishing mechanism is a Graphormer-style edge-conditioned Transformer
  run in two stages -- an "intra" block that attends only within each of the
  ligand/pocket sub-graphs (attention bias masks out cross terms) and an
  "inter" block that attends across the full ligand+pocket complex -- where
  every attention layer folds in a learned RBF (radial-basis-function)
  expansion of interatomic distances as a multiplicative edge-embedding term
  inside the QK-dot-product (not just an additive bias), and a final
  physically-interpretable Gaussian mixture "Vina-score" head decomposes the
  predicted pairwise energy into van-der-Waals, hydrophobic, and hydrogen-bond
  Gaussian components gated by a learned mixture-weight softmax.
  Reimplemented compactly: small dummy ligand+pocket complex graph, one intra
  layer + one inter layer of the RBF-gated multi-head attention, and the
  Gaussian VinaScore head producing a per-pair energy decomposition alongside
  the pooled affinity prediction.

- iShiftML: Li, Liang, et al., "Highly Accurate Prediction of NMR Chemical
  Shifts from Low-Level Quantum Mechanics Calculations Using Machine
  Learning", arXiv:2306.08269 (JCTC 2024), https://github.com/THGLab/
  iShiftML (path ``nmrpred/models/metamodels.py``, class ``Attention_TEV``).
  The distinguishing mechanism is a "delta-learning" correction network for
  NMR chemical shieldings: a per-atom structure encoder (message-passing
  over interatomic distances, RBF-expanded) produces a latent environment
  vector that is concatenated with a low-level-QM Tensorial Environment
  Vector (TEV -- the diamagnetic/paramagnetic shielding-tensor eigenvalue
  decomposition from a cheap ab-initio calculation) and fed through an
  attention-mask MLP; the mask's first two channels linearly re-weight the
  two dominant TEV components (not a softmax-normalized attention -- a
  learned affine correction) and the third channel is an additive bias,
  giving the corrected high-level chemical shift. Reimplemented compactly:
  small dummy molecule graph, one message-passing encoder round with
  Gaussian RBF distance features, and the TEV-attention correction head.

- Junction Tree VAE (JT-VAE): Jin, Barzilay, Jaakkola, "Junction Tree
  Variational Autoencoder for Molecular Graph Generation", ICML 2018,
  arXiv:1802.04364, https://github.com/wengong-jin/icml18-jtnn (``jtnn/
  jtnn_enc.py``, ``jtnn/mpn.py``, ``jtnn/jtnn_vae.py``, classes
  ``JTNNEncoder`` / ``MPN`` / ``JTNNVAE``). The distinguishing mechanism is
  a dual-level graph VAE: an atom-level message-passing network (MPN) encodes
  the raw molecular graph, and a separate tree-structured GRU message-passing
  network (JTNNEncoder) encodes the "junction tree" -- a coarser tree of
  chemically valid ring/bond clusters covering the molecule -- with the two
  pooled representations mapped to two independent halves of the VAE latent
  (tree_mean/logvar, mol_mean/logvar); a GRU-based tree decoder then
  reconstructs the junction-tree topology and per-node cluster labels
  autoregressively from the latent. The reference encoder does a *dynamic*
  Python BFS over each molecule's specific tree (variable depth, ragged
  neighbor lists) which TorchLens/eager tracing cannot capture as a static
  graph, so we reimplement the same dual-encoder / split-latent architecture
  on a small **fixed-topology** junction tree (fixed node count, fixed
  adjacency) unrolled for a fixed number of tree-GRU message rounds --
  preserving the tree-GRU message update, the atom-level MPN, the split
  tree/molecule latent heads, and a GRU-based decoder that reconstructs
  per-node topology and label logits from the sampled latent.

Already in catalog (skipped, not rebuilt here):

- iComFormer (cand_01008): a faithful compact iComFormer (SE(3)-invariant
  crystal graph transformer with RBF-expanded distance/angle edge features)
  is already shipped as ``ComFormer`` / ``build_comformer`` in
  ``menagerie/classics/gen_w7a14.py`` (class ``IComformer``), reimplemented
  from the same ``divelab/AIRS`` ``comformer.py`` source. No duplicate built.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# HGPflow
# ---------------------------------------------------------------------------


class DeepSetLayer(nn.Module):
    """Permutation-equivariant DeepSet layer (per-element + mean-subtracted)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the DeepSet update.

        Parameters
        ----------
        x:
            Node features, shape ``(batch, n_nodes, dim)``.

        Returns
        -------
        torch.Tensor
            Updated node features, same shape.
        """

        return self.layer1(x) + self.layer2(x - x.mean(dim=1, keepdim=True))


class HypergraphRefinerLayer(nn.Module):
    """One round of the HGPflow iterative hypergraph incidence refinement."""

    def __init__(self, dim: int, n_edges: int, n_nodes: int) -> None:
        super().__init__()
        self.n_edges = n_edges
        self.n_nodes = n_nodes

        self.proj_e = nn.Linear(dim, dim)
        self.proj_n = nn.Linear(dim, dim)
        self.proj_i = nn.Linear(1, dim)
        self.incidence_out = nn.Linear(dim, 1)

        self.edge_indicator = nn.Linear(dim + 1, 1)

        self.attn = nn.MultiheadAttention(dim, num_heads=2, batch_first=True)
        self.norm_e = nn.LayerNorm(dim)

        self.deepset = DeepSetLayer(dim)
        self.norm_pre_n = nn.LayerNorm(dim)
        self.norm_n = nn.LayerNorm(dim)

    def forward(
        self,
        node_feat: Tensor,
        edge_feat: Tensor,
        incidence: Tensor,
        track_eye: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Refine hyperedge embeddings, node embeddings, and the incidence matrix.

        Parameters
        ----------
        node_feat:
            Cell-node embeddings, shape ``(batch, n_nodes, dim)``.
        edge_feat:
            Hyperedge (particle-candidate) embeddings, shape
            ``(batch, n_edges, dim)``.
        incidence:
            Current soft incidence matrix, shape ``(batch, n_edges, n_nodes)``.
        track_eye:
            Fixed track-to-hyperedge identity mask, shape
            ``(batch, n_edges, n_nodes)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Updated ``(edge_feat, node_feat, incidence)``.
        """

        b, n_e, n_n = incidence.shape
        e_proj = self.proj_e(edge_feat).unsqueeze(2)  # (b, e, 1, d)
        n_proj = self.proj_n(node_feat).unsqueeze(1)  # (b, 1, n, d)
        i_proj = self.proj_i(incidence.unsqueeze(-1))  # (b, e, n, d)
        incidence_logit = self.incidence_out(e_proj + n_proj + i_proj).squeeze(-1)

        # softmax over hyperedges per node (dim=1) so each node's mass sums to 1
        incidence = F.softmax(incidence_logit, dim=1)
        # hard-code the track part: identity-mapped nodes bypass the softmax
        incidence = incidence * (1.0 - track_eye) + track_eye

        i_sum = incidence.sum(dim=2, keepdim=True)
        edge_ind_logit = self.edge_indicator(torch.cat([edge_feat, i_sum], dim=-1))
        edge_ind = torch.sigmoid(edge_ind_logit)
        gated_incidence = incidence * edge_ind

        edge_updates = torch.einsum("ben,bnd->bed", gated_incidence, node_feat)
        edge_attn_in = edge_feat + edge_updates
        attn_out, _ = self.attn(edge_attn_in, edge_attn_in, edge_attn_in)
        edge_feat = self.norm_e(edge_feat + attn_out)

        node_updates = torch.einsum("ben,bed->bnd", gated_incidence, edge_feat)
        node_feat = self.norm_n(node_feat + self.deepset(self.norm_pre_n(node_feat + node_updates)))

        return edge_feat, node_feat, incidence


class HGPflow(nn.Module):
    """Compact HGPflow: iterative hypergraph particle-flow reconstruction."""

    def __init__(
        self,
        n_features: int = 6,
        dim: int = 24,
        n_nodes: int = 10,
        n_edges: int = 6,
        n_tracks: int = 3,
        n_iters: int = 2,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.n_tracks = n_tracks

        self.proj_in = nn.Linear(n_features, dim)
        self.edge_init = nn.Parameter(torch.randn(1, n_edges, dim) * 0.1)
        self.layers = nn.ModuleList(
            [HypergraphRefinerLayer(dim, n_edges, n_nodes) for _ in range(n_iters)]
        )
        self.kin_head = nn.Linear(dim, 3)

    def forward(self, cell_features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict per-hyperedge incidence, kinematics, and existence.

        Parameters
        ----------
        cell_features:
            Calorimeter/track cell features, shape
            ``(batch, n_nodes, n_features)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``incidence`` shape ``(batch, n_edges, n_nodes)``, ``kinematics``
            shape ``(batch, n_edges, 3)``, and ``edge_embed`` shape
            ``(batch, n_edges, dim)``.
        """

        b, n_n, _ = cell_features.shape
        node_feat = self.proj_in(cell_features)
        edge_feat = self.edge_init.expand(b, -1, -1).contiguous()

        track_eye = torch.zeros(b, self.n_edges, n_n, device=cell_features.device)
        idx = torch.arange(self.n_tracks, device=cell_features.device)
        track_eye[:, idx, idx] = 1.0

        incidence = torch.full(
            (b, self.n_edges, n_n), 1.0 / self.n_edges, device=cell_features.device
        )
        incidence = incidence * (1.0 - track_eye) + track_eye

        for layer in self.layers:
            edge_feat, node_feat, incidence = layer(node_feat, edge_feat, incidence, track_eye)

        kinematics = self.kin_head(edge_feat)
        return incidence, kinematics, edge_feat


def build_hgpflow() -> nn.Module:
    """Build the compact HGPflow hypergraph-particle-flow model.

    Returns
    -------
    nn.Module
        ``HGPflow`` in eval mode.
    """

    model = HGPflow()
    model.eval()
    return model


def example_input_hgpflow() -> Tensor:
    """Example input for :func:`build_hgpflow`.

    Returns
    -------
    torch.Tensor
        Cell features, shape ``(2, 10, 6)``.
    """

    return torch.randn(2, 10, 6)


# ---------------------------------------------------------------------------
# HiGNN
# ---------------------------------------------------------------------------


class FeatureAttention(nn.Module):
    """Squeeze-excite-style channel attention over max+sum pooled graph stats."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Gate node features by a graph-level max+sum excitation signal.

        Parameters
        ----------
        x:
            Node features, shape ``(batch, n_nodes, channels)``.

        Returns
        -------
        torch.Tensor
            Gated node features, same shape.
        """

        max_result = x.amax(dim=1, keepdim=True)
        sum_result = x.sum(dim=1, keepdim=True)
        y = torch.sigmoid(self.mlp(max_result) + self.mlp(sum_result))
        return x * y


class NTNConv(nn.Module):
    """Dense neural-tensor-network graph convolution (bilinear atom-pair score)."""

    def __init__(self, dim: int, slices: int = 4) -> None:
        super().__init__()
        self.slices = slices
        self.dim = dim
        self.bilinear = nn.Bilinear(dim, dim, slices, bias=False)
        self.linear = nn.Linear(3 * dim, slices)

    def forward(self, x: Tensor, adj: Tensor, edge_attr: Tensor) -> Tensor:
        """Apply one dense NTN message-passing round.

        Parameters
        ----------
        x:
            Node features, shape ``(batch, n_nodes, dim)``.
        adj:
            Dense adjacency mask, shape ``(batch, n_nodes, n_nodes)``.
        edge_attr:
            Dense edge features, shape ``(batch, n_nodes, n_nodes, dim)``.

        Returns
        -------
        torch.Tensor
            Updated node features, shape ``(batch, n_nodes, dim)``.
        """

        b, n, d = x.shape
        x_i = x.unsqueeze(2).expand(b, n, n, d)
        x_j = x.unsqueeze(1).expand(b, n, n, d)

        score = self.bilinear(x_i.reshape(-1, d), x_j.reshape(-1, d)).view(b, n, n, self.slices)
        vec = torch.cat([x_i, edge_attr, x_j], dim=-1)
        block_score = self.linear(vec)
        alpha = torch.tanh(score + block_score)

        dim_split = self.dim // self.slices
        out = torch.maximum(x_j, edge_attr).view(b, n, n, self.slices, dim_split)
        out = out * alpha.unsqueeze(-1)
        out = out.view(b, n, n, self.dim)

        out = out * adj.unsqueeze(-1)
        return out.sum(dim=2)


class HiGNN(nn.Module):
    """Compact HiGNN: hierarchical atom-graph + BRICS-fragment-graph fusion."""

    def __init__(
        self,
        in_channels: int = 12,
        edge_dim: int = 6,
        hidden: int = 24,
        num_layers: int = 2,
        slices: int = 4,
    ) -> None:
        super().__init__()
        self.lin_a = nn.Linear(in_channels, hidden)
        self.lin_b = nn.Linear(edge_dim, hidden)

        self.atom_convs = nn.ModuleList([NTNConv(hidden, slices) for _ in range(num_layers)])
        self.lin_gate = nn.Linear(3 * hidden, hidden)
        self.feature_att = FeatureAttention(hidden)

        self.cross_att = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.out = nn.Linear(2 * hidden, 1)

    def _encode(self, x: Tensor, adj: Tensor, edge_attr: Tensor) -> Tensor:
        x = F.relu(self.lin_a(x))
        edge_attr = F.relu(self.lin_b(edge_attr))
        for conv in self.atom_convs:
            h = F.relu(conv(x, adj, edge_attr))
            beta = torch.sigmoid(self.lin_gate(torch.cat([x, h, x - h], dim=-1)))
            x = beta * x + (1 - beta) * h
            x = self.feature_att(x)
        return F.relu(x.sum(dim=1))

    def forward(
        self,
        atom_x: Tensor,
        atom_adj: Tensor,
        atom_edge_attr: Tensor,
        frag_x: Tensor,
        frag_adj: Tensor,
        frag_edge_attr: Tensor,
    ) -> Tensor:
        """Predict a molecular property from a hierarchical atom+fragment graph.

        Parameters
        ----------
        atom_x:
            Atom node features, shape ``(batch, n_atoms, in_channels)``.
        atom_adj:
            Dense atom adjacency, shape ``(batch, n_atoms, n_atoms)``.
        atom_edge_attr:
            Dense atom-bond edge features, shape
            ``(batch, n_atoms, n_atoms, edge_dim)``.
        frag_x:
            BRICS-fragment node features (pre-clustered), shape
            ``(batch, n_frags, in_channels)``.
        frag_adj:
            Dense fragment adjacency, shape ``(batch, n_frags, n_frags)``.
        frag_edge_attr:
            Dense fragment-edge features, shape
            ``(batch, n_frags, n_frags, edge_dim)``.

        Returns
        -------
        torch.Tensor
            Predicted molecular property, shape ``(batch, 1)``.
        """

        mol_vec = self._encode(atom_x, atom_adj, atom_edge_attr).unsqueeze(1)
        frag_vec = self._encode(frag_x, frag_adj, frag_edge_attr).unsqueeze(1)

        fused, _ = self.cross_att(mol_vec, frag_vec, frag_vec)
        fused = F.relu(fused)

        out = torch.cat([mol_vec, fused], dim=-1).squeeze(1)
        return self.out(out)


def build_hignn() -> nn.Module:
    """Build the compact hierarchical HiGNN molecular-property model.

    Returns
    -------
    nn.Module
        ``HiGNN`` in eval mode.
    """

    model = HiGNN()
    model.eval()
    return model


def example_input_hignn() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Example input for :func:`build_hignn`.

    Returns
    -------
    tuple[torch.Tensor, ...]
        ``(atom_x, atom_adj, atom_edge_attr, frag_x, frag_adj, frag_edge_attr)``.
    """

    n_atoms, n_frags = 8, 3
    atom_x = torch.randn(1, n_atoms, 12)
    atom_adj = (torch.rand(1, n_atoms, n_atoms) > 0.5).float()
    atom_edge_attr = torch.randn(1, n_atoms, n_atoms, 6)
    frag_x = torch.randn(1, n_frags, 12)
    frag_adj = (torch.rand(1, n_frags, n_frags) > 0.3).float()
    frag_edge_attr = torch.randn(1, n_frags, n_frags, 6)
    return atom_x, atom_adj, atom_edge_attr, frag_x, frag_adj, frag_edge_attr


# ---------------------------------------------------------------------------
# Interformer
# ---------------------------------------------------------------------------


class RBFLayer(nn.Module):
    """Learned radial-basis-function expansion with a polynomial cutoff envelope."""

    def __init__(self, k_bins: int = 16, cutoff: float = 10.0) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.centers = nn.Parameter(torch.linspace(0.0, cutoff, k_bins))
        self.widths = nn.Parameter(torch.ones(k_bins))

    def forward(self, distance: Tensor) -> Tensor:
        """Expand a pairwise-distance tensor into cutoff-enveloped RBF channels.

        Parameters
        ----------
        distance:
            Shape ``(..., )``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., k_bins)``.
        """

        x = (distance / self.cutoff).clamp(max=1.0)
        envelope = torch.where(x < 1, 1 - 6 * x**5 + 15 * x**4 - 10 * x**3, torch.zeros_like(x))
        d = distance.unsqueeze(-1)
        rbf = envelope.unsqueeze(-1) * torch.exp(-self.widths.abs() * (d - self.centers) ** 2)
        return rbf


class EdgeGatedAttention(nn.Module):
    """Graphormer-style multi-head attention gated by an RBF edge embedding."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.e_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.e_out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, edge_feat: Tensor, attn_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply edge-gated multi-head attention.

        Parameters
        ----------
        x:
            Node features, shape ``(batch, n, dim)``.
        edge_feat:
            Dense edge features, shape ``(batch, n, n, dim)``.
        attn_mask:
            Additive attention bias, shape ``(batch, n, n)`` (``-inf`` to mask).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated node features ``(batch, n, dim)`` and edge features
            ``(batch, n, n, dim)``.
        """

        b, n, d = x.shape
        h, hd = self.heads, self.head_dim

        q = (self.q_proj(x) * self.scale).view(b, n, h, hd)
        k = self.k_proj(x).view(b, n, h, hd)
        v = self.v_proj(x).view(b, n, h, hd)
        e = self.e_proj(edge_feat).view(b, n, n, h, hd)

        qk_e = torch.einsum("bihd,bjhd,bijhd->bijhd", q, k, e)
        scores = qk_e.sum(dim=-1) + attn_mask.unsqueeze(-1)
        attn = F.softmax(scores, dim=2)

        out = torch.einsum("bijh,bjhd->bihd", attn, v).reshape(b, n, d)
        out = self.out_proj(out)

        edge_out = self.e_out_proj(qk_e.reshape(b, n, n, d))
        return out, edge_out


class Interformer(nn.Module):
    """Compact Interformer: RBF-edge-gated intra/inter graph-transformer + VinaScore."""

    def __init__(self, n_atom_types: int = 10, dim: int = 24, heads: int = 4) -> None:
        super().__init__()
        self.atom_embedding = nn.Embedding(n_atom_types, dim)
        self.rbf = RBFLayer(k_bins=dim)
        self.rbf_proj = nn.Linear(dim, dim)

        self.intra_layer = EdgeGatedAttention(dim, heads)
        self.inter_layer = EdgeGatedAttention(dim, heads)
        self.norm_intra = nn.LayerNorm(dim)
        self.norm_inter = nn.LayerNorm(dim)

        self.affinity_head = nn.Linear(dim, 1)

        self.mean_head = nn.Linear(dim, 4)
        self.sigma_head = nn.Linear(dim, 4)
        self.weight_head = nn.Linear(dim, 4)

    def forward(
        self, atom_types: Tensor, distance: Tensor, is_ligand: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Predict binding affinity and a Gaussian-mixture pairwise energy decomposition.

        Parameters
        ----------
        atom_types:
            Integer species indices, shape ``(batch, n_atoms)``.
        distance:
            Pairwise distances, shape ``(batch, n_atoms, n_atoms)``.
        is_ligand:
            Boolean ligand/pocket membership, shape ``(batch, n_atoms)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``affinity`` shape ``(batch, 1)`` and per-pair Gaussian-mixture
            energy terms, shape ``(batch, n_atoms, n_atoms, 4)``.
        """

        x = self.atom_embedding(atom_types)
        edge_feat = self.rbf_proj(self.rbf(distance))

        same_side = is_ligand.unsqueeze(1) == is_ligand.unsqueeze(2)
        intra_mask = torch.where(
            same_side, torch.zeros_like(distance), torch.full_like(distance, float("-inf"))
        )

        h, edge_intra = self.intra_layer(x, edge_feat, intra_mask)
        x = self.norm_intra(x + h)

        inter_mask = torch.zeros_like(distance)
        h, edge_inter = self.inter_layer(x, edge_intra, inter_mask)
        x = self.norm_inter(x + h)

        pooled = x.mean(dim=1)
        affinity = self.affinity_head(pooled)

        mean = F.elu(self.mean_head(edge_inter))
        sigma = F.elu(self.sigma_head(edge_inter)) + 1.0 + 1e-5
        weight = F.softmax(self.weight_head(edge_inter), dim=-1)

        d = distance.unsqueeze(-1)
        log_prob = -0.5 * ((d - mean) / sigma) ** 2 - torch.log(sigma * math.sqrt(2 * math.pi))
        energy_terms = weight * log_prob.exp()

        return affinity, energy_terms


def build_interformer() -> nn.Module:
    """Build the compact Interformer protein-ligand interaction model.

    Returns
    -------
    nn.Module
        ``Interformer`` in eval mode.
    """

    model = Interformer()
    model.eval()
    return model


def example_input_interformer() -> tuple[Tensor, Tensor, Tensor]:
    """Example input for :func:`build_interformer`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(atom_types, distance, is_ligand)``.
    """

    n_atoms = 10
    atom_types = torch.randint(0, 10, (1, n_atoms))
    coords = torch.randn(1, n_atoms, 3)
    distance = torch.cdist(coords, coords)
    is_ligand = torch.zeros(1, n_atoms, dtype=torch.bool)
    is_ligand[:, : n_atoms // 2] = True
    return atom_types, distance, is_ligand


# ---------------------------------------------------------------------------
# iShiftML
# ---------------------------------------------------------------------------


class GaussianSmearing(nn.Module):
    """Fixed Gaussian radial-basis-function smearing of a scalar distance."""

    def __init__(self, n_gaussians: int = 16, cutoff: float = 8.0) -> None:
        super().__init__()
        offsets = torch.linspace(0.0, cutoff, n_gaussians)
        self.register_buffer("offsets", offsets)
        self.width = (offsets[1] - offsets[0]).item()

    def forward(self, distance: Tensor) -> Tensor:
        """Expand a distance tensor into Gaussian RBF channels.

        Parameters
        ----------
        distance:
            Shape ``(..., )``.

        Returns
        -------
        torch.Tensor
            Shape ``(..., n_gaussians)``.
        """

        diff = distance.unsqueeze(-1) - self.offsets
        return torch.exp(-0.5 * (diff / self.width) ** 2)


class StructureEncoder(nn.Module):
    """Compact per-atom message-passing structure encoder."""

    def __init__(self, n_species: int = 6, dim: int = 24, n_gaussians: int = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_species, dim)
        self.smearing = GaussianSmearing(n_gaussians)
        self.filter_net = nn.Linear(n_gaussians, dim)
        self.update = nn.Linear(dim, dim)

    def forward(self, atom_types: Tensor, distance: Tensor) -> Tensor:
        """Encode a per-atom latent structural environment vector.

        Parameters
        ----------
        atom_types:
            Integer species indices, shape ``(batch, n_atoms)``.
        distance:
            Pairwise distances, shape ``(batch, n_atoms, n_atoms)``.

        Returns
        -------
        torch.Tensor
            Per-atom latent vectors, shape ``(batch, n_atoms, dim)``.
        """

        x = self.embedding(atom_types)
        rbf = self.smearing(distance)
        edge_filter = self.filter_net(rbf)
        messages = torch.einsum("bijd,bjd->bid", edge_filter, x)
        return F.silu(self.update(x + messages))


class AttentionTEV(nn.Module):
    """Compact Attention_TEV: structure-gated correction of a low-level TEV."""

    def __init__(self, n_species: int = 6, dim: int = 24, tev_dim: int = 8) -> None:
        super().__init__()
        self.encoder = StructureEncoder(n_species, dim)
        self.attention_mask_net = nn.Sequential(
            nn.Linear(dim + tev_dim - 2, 32),
            nn.SiLU(),
            nn.Linear(32, 3),
        )

    def forward(self, atom_types: Tensor, distance: Tensor, tev: Tensor) -> Tensor:
        """Predict the delta-learning-corrected NMR chemical shift.

        Parameters
        ----------
        atom_types:
            Integer species indices, shape ``(batch, n_atoms)``.
        distance:
            Pairwise distances, shape ``(batch, n_atoms, n_atoms)``.
        tev:
            Low-level Tensorial Environment Vector per atom, shape
            ``(batch, n_atoms, tev_dim)`` (first 2 channels are the dominant
            diamagnetic/paramagnetic shielding components).

        Returns
        -------
        torch.Tensor
            Corrected per-atom chemical shift, shape ``(batch, n_atoms)``.
        """

        encoded = self.encoder(atom_types, distance)
        attention_in = torch.cat([encoded, tev[..., 2:]], dim=-1)
        attention_masks = self.attention_mask_net(attention_in)

        correction = (
            tev[..., 0] * attention_masks[..., 0]
            + tev[..., 1] * attention_masks[..., 1]
            + attention_masks[..., 2]
        )
        return correction


def build_ishiftml() -> nn.Module:
    """Build the compact iShiftML TEV-attention chemical-shift correction model.

    Returns
    -------
    nn.Module
        ``AttentionTEV`` in eval mode.
    """

    model = AttentionTEV()
    model.eval()
    return model


def example_input_ishiftml() -> tuple[Tensor, Tensor, Tensor]:
    """Example input for :func:`build_ishiftml`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(atom_types, distance, tev)``.
    """

    n_atoms = 8
    atom_types = torch.randint(0, 6, (1, n_atoms))
    coords = torch.randn(1, n_atoms, 3)
    distance = torch.cdist(coords, coords)
    tev = torch.randn(1, n_atoms, 8)
    return atom_types, distance, tev


# ---------------------------------------------------------------------------
# Junction Tree VAE (JT-VAE)
# ---------------------------------------------------------------------------


class GraphMPN(nn.Module):
    """Compact atom-level message-passing network (loopy-BP-style edge GRU)."""

    def __init__(self, dim: int = 24, depth: int = 3) -> None:
        super().__init__()
        self.depth = depth
        self.w_i = nn.Linear(dim, dim, bias=False)
        self.gru_cell = nn.GRUCell(dim, dim)

    def forward(self, atom_feat: Tensor, adj: Tensor) -> Tensor:
        """Run fixed-depth loopy message passing and pool to a molecule vector.

        Parameters
        ----------
        atom_feat:
            Atom features, shape ``(batch, n_atoms, dim)``.
        adj:
            Dense bond adjacency, shape ``(batch, n_atoms, n_atoms)``.

        Returns
        -------
        torch.Tensor
            Pooled molecule vector, shape ``(batch, dim)``.
        """

        b, n, d = atom_feat.shape
        h = self.w_i(atom_feat)
        for _ in range(self.depth):
            messages = torch.einsum("bij,bjd->bid", adj, h)
            h = self.gru_cell(messages.reshape(b * n, d), h.reshape(b * n, d)).reshape(b, n, d)
        return h.sum(dim=1)


class TreeGRUEncoder(nn.Module):
    """Compact junction-tree GRU message-passing encoder (fixed topology).

    The reference ``JTNNEncoder`` performs a *dynamic* Python BFS whose
    traversal depth and per-node neighbor count vary per molecule, which is
    not traceable as a static graph. We instead unroll a fixed number of
    tree-GRU message-passing rounds over a fixed-size junction tree, which
    preserves the tree-structured GRU update rule while keeping the trace
    static.
    """

    def __init__(self, vocab_size: int = 32, dim: int = 24, depth: int = 3) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.w_z = nn.Linear(2 * dim, dim)
        self.w_r = nn.Linear(dim, dim, bias=False)
        self.u_r = nn.Linear(dim, dim)
        self.w_h = nn.Linear(2 * dim, dim)
        self.w_out = nn.Linear(2 * dim, dim)
        self.depth = depth

    def forward(self, node_wid: Tensor, tree_adj: Tensor) -> Tensor:
        """Run fixed-depth tree-GRU message passing and pool to a tree vector.

        Parameters
        ----------
        node_wid:
            Cluster-vocabulary index per tree node, shape ``(batch, n_nodes)``.
        tree_adj:
            Dense tree adjacency, shape ``(batch, n_nodes, n_nodes)``.

        Returns
        -------
        torch.Tensor
            Pooled junction-tree vector, shape ``(batch, dim)``.
        """

        b, n = node_wid.shape
        x = self.embedding(node_wid)
        h = torch.zeros_like(x)

        for _ in range(self.depth):
            h_nei_sum = torch.einsum("bij,bjd->bid", tree_adj, h)
            n_nei = tree_adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
            h_nei_mean = h_nei_sum / n_nei

            z = torch.sigmoid(self.w_z(torch.cat([x, h_nei_mean], dim=-1)))
            r = torch.sigmoid(self.w_r(x).unsqueeze(2) + self.u_r(h).unsqueeze(1))
            r_diag = torch.diagonal(r, dim1=1, dim2=2).new_zeros(b, n, h.shape[-1])
            gated_nei = torch.einsum("bij,bijd,bjd->bid", tree_adj, r, h) / n_nei
            h_tilde = torch.tanh(self.w_h(torch.cat([x, gated_nei], dim=-1)))
            h = (1 - z) * h_nei_mean + z * h_tilde
            h = h + 0.0 * r_diag  # keep r fully in the trace

        pooled_nei = h.sum(dim=1)
        node_vec = torch.cat([x.sum(dim=1), pooled_nei], dim=-1)
        return F.relu(self.w_out(node_vec))


class JunctionTreeVAE(nn.Module):
    """Compact JT-VAE: dual atom-graph / junction-tree encoder with split latents."""

    def __init__(
        self,
        vocab_size: int = 32,
        n_atom_types: int = 10,
        dim: int = 24,
        latent_size: int = 16,
    ) -> None:
        super().__init__()
        self.atom_embedding = nn.Embedding(n_atom_types, dim)
        self.mpn = GraphMPN(dim)
        self.jtnn = TreeGRUEncoder(vocab_size, dim)

        half = latent_size // 2
        self.t_mean = nn.Linear(dim, half)
        self.t_var = nn.Linear(dim, half)
        self.g_mean = nn.Linear(dim, half)
        self.g_var = nn.Linear(dim, half)

        self.decoder_gru = nn.GRUCell(latent_size, dim)
        self.topo_head = nn.Linear(dim, 1)
        self.label_head = nn.Linear(dim, vocab_size)

    def encode(
        self, atom_types: Tensor, atom_adj: Tensor, node_wid: Tensor, tree_adj: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode a molecule into split tree/molecule latent mean and log-variance.

        Parameters
        ----------
        atom_types:
            Integer atom-species indices, shape ``(batch, n_atoms)``.
        atom_adj:
            Dense atom-bond adjacency, shape ``(batch, n_atoms, n_atoms)``.
        node_wid:
            Junction-tree cluster-vocabulary indices, shape
            ``(batch, n_tree_nodes)``.
        tree_adj:
            Dense junction-tree adjacency, shape
            ``(batch, n_tree_nodes, n_tree_nodes)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``z_mean`` and ``z_log_var``, each shape ``(batch, latent_size)``.
        """

        mol_vec = self.mpn(self.atom_embedding(atom_types), atom_adj)
        tree_vec = self.jtnn(node_wid, tree_adj)

        tree_mean = self.t_mean(tree_vec)
        tree_log_var = -torch.abs(self.t_var(tree_vec))
        mol_mean = self.g_mean(mol_vec)
        mol_log_var = -torch.abs(self.g_var(mol_vec))

        z_mean = torch.cat([tree_mean, mol_mean], dim=-1)
        z_log_var = torch.cat([tree_log_var, mol_log_var], dim=-1)
        return z_mean, z_log_var

    def forward(
        self, atom_types: Tensor, atom_adj: Tensor, node_wid: Tensor, tree_adj: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Encode, reparameterize, and decode per-node topology/label logits.

        Parameters
        ----------
        atom_types:
            Integer atom-species indices, shape ``(batch, n_atoms)``.
        atom_adj:
            Dense atom-bond adjacency, shape ``(batch, n_atoms, n_atoms)``.
        node_wid:
            Junction-tree cluster-vocabulary indices, shape
            ``(batch, n_tree_nodes)``.
        tree_adj:
            Dense junction-tree adjacency, shape
            ``(batch, n_tree_nodes, n_tree_nodes)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Reconstructed per-node ``topo_logits`` shape
            ``(batch, n_tree_nodes, 1)`` and ``label_logits`` shape
            ``(batch, n_tree_nodes, vocab_size)``.
        """

        z_mean, z_log_var = self.encode(atom_types, atom_adj, node_wid, tree_adj)
        std = torch.exp(0.5 * z_log_var)
        z = z_mean + std * torch.randn_like(std)

        n_nodes = node_wid.shape[1]
        z_expanded = z.unsqueeze(1).expand(-1, n_nodes, -1)
        b, n, d = z_expanded.shape
        h0 = torch.zeros(b * n, self.decoder_gru.hidden_size, device=z.device)
        h = self.decoder_gru(z_expanded.reshape(b * n, d), h0).reshape(b, n, -1)

        topo_logits = self.topo_head(h)
        label_logits = self.label_head(h)
        return topo_logits, label_logits


def build_jtvae() -> nn.Module:
    """Build the compact Junction Tree VAE molecular graph generator.

    Returns
    -------
    nn.Module
        ``JunctionTreeVAE`` in eval mode.
    """

    model = JunctionTreeVAE()
    model.eval()
    return model


def example_input_jtvae() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Example input for :func:`build_jtvae`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(atom_types, atom_adj, node_wid, tree_adj)``.
    """

    n_atoms, n_tree_nodes = 9, 5
    atom_types = torch.randint(0, 10, (1, n_atoms))
    atom_adj = (torch.rand(1, n_atoms, n_atoms) > 0.6).float()
    atom_adj = atom_adj * (1.0 - torch.eye(n_atoms).unsqueeze(0))

    node_wid = torch.randint(0, 32, (1, n_tree_nodes))
    # a small fixed tree topology (path graph 0-1-2-3-4)
    tree_adj = torch.zeros(1, n_tree_nodes, n_tree_nodes)
    for i in range(n_tree_nodes - 1):
        tree_adj[0, i, i + 1] = 1.0
        tree_adj[0, i + 1, i] = 1.0

    return atom_types, atom_adj, node_wid, tree_adj


MENAGERIE_ENTRIES = [
    ("HGPflow", "build_hgpflow", "example_input_hgpflow", "2024", "PHYS"),
    ("HiGNN", "build_hignn", "example_input_hignn", "2023", "SCI"),
    ("Interformer", "build_interformer", "example_input_interformer", "2024", "SCI"),
    ("iShiftML", "build_ishiftml", "example_input_ishiftml", "2024", "SCI"),
    ("Junction Tree VAE", "build_jtvae", "example_input_jtvae", "2018", "GEN"),
]

"""Wave 8 batch 20 menagerie classics: graph/molecule generative and
representation-learning architectures.

Sources checked (official repos + papers, reimplemented compactly from
scratch in base-env torch; no cloning, no pip installs):

- GDSS: https://github.com/harryjo97/GDSS (Jo, Lee, Hwang, "Score-Based
  Generative Modeling of Graphs via the System of Stochastic Differential
  Equations", ICML 2022). Confirmed from ``models/ScoreNetwork_X.py`` (GCN
  stack over node features) and ``models/ScoreNetwork_A.py``
  (``AttentionLayer``/``BaselineNetworkLayer`` operating on powers of the
  adjacency matrix, ``pow_tensor``) -- a joint score network pair (X, A)
  trained on the coupled graph SDE.
- GeoDiff: https://github.com/MinkaiXu/GeoDiff (Xu, Luo, Wang, Huang, Tang,
  "GeoDiff: A Geometric Diffusion Model for Molecular Conformation
  Generation", ICLR 2022 Oral, arXiv:2203.02923). Confirmed from
  ``models/epsnet/dualenc.py`` (``DualEncoderEpsNetwork``): a dual encoder
  (global long-range + local bonded-graph) predicts SE(3)-invariant
  gradients with respect to pairwise atomic distances, which are converted
  back to equivariant per-atom coordinate updates via the chain rule
  through the distance function.
- GeoLDM: https://github.com/MinkaiXu/GeoLDM (Xu, Powers, Dror, Ermon,
  Leskovec, "Geometric Latent Diffusion Models for 3D Molecule Generation",
  ICML 2023, arXiv:2305.01140). Confirmed from
  ``equivariant_diffusion/en_diffusion.py`` (``EnHierarchicalVAE``,
  ``EnLatentDiffusion``) and ``egnn/egnn_new.py`` (``EGNN``,
  ``EquivariantBlock``): an E(n)-equivariant VAE encodes atoms into an
  invariant node-feature latent plus an equivariant (zero-CoM) coordinate
  latent, and a second EGNN denoises that latent code under a diffusion
  process.
- GeoMol: https://github.com/PattanaikL/GeoMol (Ganea*, Pattanaik*, et al.,
  "GeoMol: Torsional Geometric Generation of Molecular 3D Conformer
  Ensembles", NeurIPS 2021, arXiv:2106.07802). Confirmed from
  ``model/model.py`` (``GeoMol.embed`` / ``GeoMol.model_local_stats``): two
  parallel message-passing GNNs embed atoms with per-sample random noise
  vectors (for a non-autoregressive, diverse ensemble), then for each local
  neighborhood a permutation-invariant Transformer over the (up to 4)
  neighbor/center feature pairs predicts unit local-frame directions and
  bond-length-like distances, which are combined into local 3D coordinates
  (torsion angles / dihedrals emerge from combining adjacent local frames).
- GIMLET: https://github.com/zhao-ht/GIMLET (Zhao, Wu, Wang, Yang, et al.,
  "GIMLET: A Unified Graph-Text Model for Instruction-Based Molecule
  Zero-Shot Learning", NeurIPS 2023, arXiv:2306.13089). Confirmed from
  ``model/GIMLET/GIMLETEncoderStack.py`` and
  ``model/graphormer/modules/graphormer_graph_encoder.py``: a Graphormer
  graph encoder produces per-atom graph tokens (atom-type + degree
  embeddings) that are concatenated with instruction-text tokens; a shared
  transformer encoder attends jointly over both streams using a
  generalized/decoupled position bias where the graph-token block gets its
  own learned bias term merged into the text relative-position bias.
- GLN (Conditional Graph Logic Network):
  https://github.com/Hanjun-Dai/GLN (Dai, Li, Coley, Song, Dai, "Retrosynthesis
  Prediction with Conditional Graph Logic Network", NeurIPS 2019,
  arXiv:2001.01408). Already present in the catalog as "GLN retrosynthesis"
  (``menagerie/classics/gen_w7a18.py``, ``build_gln`` /
  ``GLNRetrosynthesis``) -- SKIPPED here as a duplicate.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# 1. GDSS: joint score-based SDE system over (node features X, adjacency A).
# ---------------------------------------------------------------------------


def _pow_tensor(adj: torch.Tensor, order: int) -> torch.Tensor:
    """Stack powers ``adj^0 .. adj^(order-1)`` along a new channel axis.

    Parameters
    ----------
    adj : torch.Tensor
        Dense adjacency, shape ``(B, N, N)``.
    order : int
        Number of powers to stack (including the identity, order 0).

    Returns
    -------
    torch.Tensor
        Shape ``(B, order, N, N)``.
    """

    out = [torch.eye(adj.shape[-1], device=adj.device).expand_as(adj)]
    cur = adj
    for _ in range(1, order):
        out.append(cur)
        cur = cur @ adj
    return torch.stack(out, dim=1)


class _DenseGCNConv(nn.Module):
    """Dense GCN layer: ``sigma(D^-1/2 A D^-1/2 X W)`` on a full adjacency."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
        norm_adj = adj / deg
        return self.lin(norm_adj @ x)


class GDSSScoreX(nn.Module):
    """GDSS node-feature score network: stacked dense-GCN residual features."""

    def __init__(self, feat_dim: int = 8, hidden: int = 16, depth: int = 3) -> None:
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList(
            [_DenseGCNConv(feat_dim if i == 0 else hidden, hidden) for i in range(depth)]
        )
        fdim = feat_dim + depth * hidden
        self.final = nn.Sequential(
            nn.Linear(fdim, 2 * fdim), nn.ELU(), nn.Linear(2 * fdim, feat_dim)
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        feats = [x]
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h, adj))
            feats.append(h)
        return self.final(torch.cat(feats, dim=-1))


class GDSSScoreA(nn.Module):
    """GDSS adjacency score network: GCN-per-power layers producing a score matrix.

    Mirrors ``BaselineNetworkLayer`` in the official repo: at each layer a
    stack of dense-GCN convolutions (one per power channel of the current
    adjacency) produces new node features, and an MLP over the outer-product
    node-feature matrix plus the current adjacency channels produces the next
    adjacency channels.
    """

    def __init__(
        self, feat_dim: int = 8, hidden: int = 8, n_powers: int = 2, depth: int = 2
    ) -> None:
        super().__init__()
        self.n_powers = n_powers
        self.depth = depth
        self.node_convs = nn.ModuleList(
            [
                nn.ModuleList(
                    [_DenseGCNConv(feat_dim if i == 0 else hidden, hidden) for _ in range(n_powers)]
                )
                for i in range(depth)
            ]
        )
        self.merge = nn.ModuleList([nn.Linear(n_powers * hidden, hidden) for _ in range(depth)])
        self.adj_mlp = nn.ModuleList(
            [nn.Linear(2 * hidden + n_powers, n_powers) for _ in range(depth)]
        )
        self.final = nn.Linear(depth * n_powers + n_powers, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adjc = _pow_tensor(adj, self.n_powers)  # (B, n_powers, N, N)
        adj_list = [adjc]
        h = x
        for layer_idx in range(self.depth):
            outs = [conv(h, adjc[:, p]) for p, conv in enumerate(self.node_convs[layer_idx])]
            h = torch.tanh(self.merge[layer_idx](torch.cat(outs, dim=-1)))
            n = h.shape[1]
            h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
            h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
            mlp_in = torch.cat([h_i, h_j, adjc.permute(0, 2, 3, 1)], dim=-1)
            adjc = self.adj_mlp[layer_idx](mlp_in).permute(0, 3, 1, 2)
            adjc = adjc + adjc.transpose(-1, -2)
            adj_list.append(adjc)
        stacked = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        score = self.final(stacked).squeeze(-1)
        return score


class GDSS(nn.Module):
    """Compact GDSS: joint score networks over node features and adjacency.

    Reference
    ---------
    Jo, Lee, Hwang. "Score-Based Generative Modeling of Graphs via the
    System of Stochastic Differential Equations." ICML 2022.
    """

    def __init__(self, feat_dim: int = 8, hidden: int = 16) -> None:
        super().__init__()
        self.score_x = GDSSScoreX(feat_dim=feat_dim, hidden=hidden)
        self.score_a = GDSSScoreA(feat_dim=feat_dim, hidden=hidden // 2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score both the node features and the adjacency jointly.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape ``(B, N, feat_dim)``.
        adj : torch.Tensor
            Dense symmetric adjacency, shape ``(B, N, N)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Score for ``x`` (``B, N, feat_dim``) and score for ``adj``
            (``B, N, N``).
        """

        score_x = self.score_x(x, adj)
        score_a = self.score_a(x, adj)
        return score_x, score_a


def build_gdss() -> nn.Module:
    """Build a compact GDSS joint graph-SDE score model.

    Returns
    -------
    nn.Module
        Random-initialized ``GDSS`` in eval mode.
    """

    return GDSS(feat_dim=8, hidden=16).eval()


def example_input_gdss() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a small synthetic dense graph batch (node feats + adjacency).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``x`` of shape ``(2, 10, 8)`` and symmetric ``adj`` of shape
        ``(2, 10, 10)`` with a zeroed diagonal.
    """

    torch.manual_seed(0)
    n = 10
    x = torch.randn(2, n, 8)
    raw = torch.rand(2, n, n)
    adj = (raw + raw.transpose(-1, -2)) / 2
    adj = adj * (1 - torch.eye(n))
    return x, adj


# ---------------------------------------------------------------------------
# 2. GeoDiff: dual-encoder equivariant diffusion for 3D conformers.
# ---------------------------------------------------------------------------


class _RBFExpansion(nn.Module):
    """Gaussian radial-basis expansion of scalar distances."""

    def __init__(self, num_basis: int = 16, cutoff: float = 10.0) -> None:
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_basis)
        self.register_buffer("centers", centers)
        self.width = cutoff / num_basis

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        diff = dist.unsqueeze(-1) - self.centers
        return torch.exp(-(diff**2) / (2 * self.width**2))


class _MessagePassingBlock(nn.Module):
    """Edge-conditioned message passing over a fully-connected atom graph."""

    def __init__(self, hidden: int, edge_dim: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, h: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # h: (B, N, hidden); edge_attr: (B, N, N, edge_dim)
        n = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        msg = self.edge_mlp(torch.cat([h_i, h_j, edge_attr], dim=-1))
        agg = msg.sum(dim=2)
        return h + self.node_mlp(torch.cat([h, agg], dim=-1))


class _GeoDiffEncoder(nn.Module):
    """One branch (global or local) of GeoDiff's dual encoder.

    Embeds atom types, expands pairwise distances via RBF, and message-passes
    to produce per-atom features plus a per-edge gradient-of-distance scalar.
    """

    def __init__(
        self, num_atom_types: int = 16, hidden: int = 24, num_basis: int = 16, depth: int = 2
    ) -> None:
        super().__init__()
        self.atom_embed = nn.Embedding(num_atom_types, hidden)
        self.rbf = _RBFExpansion(num_basis=num_basis)
        self.edge_proj = nn.Linear(num_basis, hidden)
        self.blocks = nn.ModuleList([_MessagePassingBlock(hidden, hidden) for _ in range(depth)])
        self.grad_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, atom_type: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        h = self.atom_embed(atom_type)
        edge_attr = self.edge_proj(self.rbf(dist))
        for block in self.blocks:
            h = block(h, edge_attr)
        n = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        grad = self.grad_mlp(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)
        return grad


class GeoDiff(nn.Module):
    """Compact GeoDiff: dual global/local encoders predicting SE(3)-invariant
    gradients w.r.t. pairwise atomic distance, converted back to per-atom
    coordinate updates through the distance chain rule.

    Reference
    ---------
    Xu, Luo, Wang, Huang, Tang. "GeoDiff: A Geometric Diffusion Model for
    Molecular Conformation Generation." ICLR 2022 (Oral). arXiv:2203.02923.
    """

    def __init__(self, num_atom_types: int = 16, hidden: int = 24) -> None:
        super().__init__()
        self.encoder_global = _GeoDiffEncoder(num_atom_types, hidden)
        self.encoder_local = _GeoDiffEncoder(num_atom_types, hidden)

    def forward(self, atom_type: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Predict a per-atom coordinate update (noise/score) at diffusion step.

        Parameters
        ----------
        atom_type : torch.Tensor
            Integer atom types, shape ``(B, N)``.
        pos : torch.Tensor
            Noisy 3D coordinates, shape ``(B, N, 3)``.

        Returns
        -------
        torch.Tensor
            Predicted per-atom coordinate update, shape ``(B, N, 3)``.
        """

        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, N, N, 3)
        dist = diff.norm(dim=-1).clamp(min=1e-6)
        grad_global = self.encoder_global(atom_type, dist)
        grad_local = self.encoder_local(atom_type, dist)
        grad_dist = grad_global + grad_local  # (B, N, N) scalar gradient wrt each pairwise distance
        # chain rule: d(dist_ij)/d(pos_i) = (pos_i - pos_j) / dist_ij
        unit = diff / dist.unsqueeze(-1)
        update = (grad_dist.unsqueeze(-1) * unit).sum(dim=2)
        return update


def build_geodiff() -> nn.Module:
    """Build a compact GeoDiff dual-encoder equivariant score model.

    Returns
    -------
    nn.Module
        Random-initialized ``GeoDiff`` in eval mode.
    """

    return GeoDiff(num_atom_types=16, hidden=24).eval()


def example_input_geodiff() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a small synthetic molecule (atom types + 3D coordinates).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``atom_type`` of shape ``(2, 9)`` and ``pos`` of shape ``(2, 9, 3)``.
    """

    torch.manual_seed(0)
    atom_type = torch.randint(0, 16, (2, 9))
    pos = torch.randn(2, 9, 3)
    return atom_type, pos


# ---------------------------------------------------------------------------
# 3. GeoLDM: equivariant autoencoder + latent-space equivariant diffusion.
# ---------------------------------------------------------------------------


class _EGNNLayer(nn.Module):
    """One E(n)-equivariant graph-conv layer (EGNN, fully-connected graph).

    Mirrors the GCL + coordinate-update pair from ``egnn/egnn_new.py``:
    invariant messages from (h_i, h_j, ||x_i - x_j||^2) update both the
    invariant node features ``h`` and, via a scalar-weighted sum of
    coordinate differences, the equivariant coordinates ``x``.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        coord_out = nn.Linear(hidden, 1, bias=False)
        nn.init.xavier_uniform_(coord_out.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), coord_out)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        coord_diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B, N, N, 3)
        dist2 = (coord_diff**2).sum(-1, keepdim=True)
        edge_feat = self.edge_mlp(torch.cat([h_i, h_j, dist2], dim=-1))
        agg = edge_feat.sum(dim=2)
        h_out = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        weights = self.coord_mlp(edge_feat)  # (B, N, N, 1)
        x_out = x + (weights * coord_diff).mean(dim=2)
        return h_out, x_out


class _EGNN(nn.Module):
    """Stack of EGNN layers with an input/output feature projection."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 2) -> None:
        super().__init__()
        self.embed_in = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([_EGNNLayer(hidden) for _ in range(depth)])
        self.embed_out = nn.Linear(hidden, out_dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.embed_in(h)
        for layer in self.layers:
            h, x = layer(h, x)
        return self.embed_out(h), x


class GeoLDM(nn.Module):
    """Compact GeoLDM: E(n)-equivariant VAE encoder into a joint
    invariant(h)+equivariant(x) latent code, plus a latent-space EGNN
    denoiser that predicts the diffusion score in that latent space.

    Reference
    ---------
    Xu, Powers, Dror, Ermon, Leskovec. "Geometric Latent Diffusion Models
    for 3D Molecule Generation." ICML 2023. arXiv:2305.01140.
    """

    def __init__(self, atom_feat_dim: int = 10, latent_dim: int = 8, hidden: int = 24) -> None:
        super().__init__()
        self.encoder = _EGNN(atom_feat_dim, hidden, latent_dim, depth=2)
        self.decoder = _EGNN(latent_dim, hidden, atom_feat_dim, depth=2)
        self.denoiser = _EGNN(latent_dim, hidden, latent_dim, depth=2)

    def forward(
        self, atom_feat: torch.Tensor, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode to latent (h_lat, x_lat), denoise the latent, and decode back.

        Parameters
        ----------
        atom_feat : torch.Tensor
            Invariant per-atom input features (e.g. one-hot atom type),
            shape ``(B, N, atom_feat_dim)``.
        pos : torch.Tensor
            Equivariant 3D coordinates, shape ``(B, N, 3)``.

        Returns
        -------
        tuple of torch.Tensor
            ``(h_latent, x_latent, h_recon, x_denoised)``.
        """

        x_centered = pos - pos.mean(dim=1, keepdim=True)  # zero center-of-mass
        h_latent, x_latent = self.encoder(atom_feat, x_centered)
        h_recon, _ = self.decoder(h_latent, x_latent)
        _, x_denoised = self.denoiser(h_latent, x_latent)
        return h_latent, x_latent, h_recon, x_denoised


def build_geoldm() -> nn.Module:
    """Build a compact GeoLDM equivariant-latent-diffusion model.

    Returns
    -------
    nn.Module
        Random-initialized ``GeoLDM`` in eval mode.
    """

    return GeoLDM(atom_feat_dim=10, latent_dim=8, hidden=24).eval()


def example_input_geoldm() -> tuple[torch.Tensor, torch.Tensor]:
    """Create a small synthetic molecule (one-hot atom feats + 3D coordinates).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``atom_feat`` of shape ``(2, 9, 10)`` and ``pos`` of shape
        ``(2, 9, 3)``.
    """

    torch.manual_seed(0)
    atom_feat = F.one_hot(torch.randint(0, 10, (2, 9)), num_classes=10).float()
    pos = torch.randn(2, 9, 3)
    return atom_feat, pos


# ---------------------------------------------------------------------------
# 4. GeoMol: non-autoregressive torsional conformer generation.
# ---------------------------------------------------------------------------


class _NeighborhoodEncoder(nn.Module):
    """Predicts local-frame unit directions + distances for one atom's up to
    4 neighbors, following ``GeoMol.model_local_stats``: a permutation-aware
    Transformer encoder over (neighbor, center) feature pairs produces
    per-neighbor unit vectors (local-frame directions) and bond-length-like
    distances, whose product gives local 3D offsets from the center atom.
    """

    def __init__(self, hidden: int, max_neighbors: int = 4) -> None:
        super().__init__()
        self.max_neighbors = max_neighbors
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=2 * hidden, nhead=2, dim_feedforward=3 * hidden, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.coord_pred = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3)
        )
        self.dist_pred = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, center_feat: torch.Tensor, neighbor_feat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        center_feat : torch.Tensor
            Shape ``(B, K, hidden)`` -- feature of the center atom, repeated
            per neighborhood ``K``.
        neighbor_feat : torch.Tensor
            Shape ``(B, K, max_neighbors, hidden)`` -- features of each of
            the (up to) 4 neighbors.

        Returns
        -------
        torch.Tensor
            Local 3D offsets from the center, shape
            ``(B, K, max_neighbors, 3)``.
        """

        b, k, m, hidden = neighbor_feat.shape
        center_rep = center_feat.unsqueeze(2).expand(-1, -1, m, -1)
        pair = torch.cat([neighbor_feat, center_rep], dim=-1)  # (B, K, M, 2H)
        pair = pair.reshape(b * k, m, 2 * hidden)
        enc = self.encoder(pair)
        unit = self.coord_pred(enc)
        unit = unit / (unit.norm(dim=-1, keepdim=True) + 1e-8)
        dist = F.softplus(self.dist_pred(enc))
        offsets = unit * dist
        return offsets.view(b, k, m, 3)


class GeoMol(nn.Module):
    """Compact GeoMol: dual random-noise-augmented GNN embeddings feeding a
    per-neighborhood local-frame + torsion predictor for non-autoregressive
    3D conformer assembly.

    Reference
    ---------
    Ganea*, Pattanaik*, Coley, Barzilay, Jensen, Green, Jaakkola. "GeoMol:
    Torsional Geometric Generation of Molecular 3D Conformer Ensembles."
    NeurIPS 2021. arXiv:2106.07802.
    """

    def __init__(
        self,
        num_atom_types: int = 16,
        hidden: int = 16,
        random_vec_dim: int = 4,
        max_neighbors: int = 4,
    ) -> None:
        super().__init__()
        self.random_vec_dim = random_vec_dim
        self.max_neighbors = max_neighbors
        self.atom_embed = nn.Embedding(num_atom_types, hidden - random_vec_dim)
        self.gnn1 = _MessagePassingBlock(hidden, hidden)
        self.gnn2 = _MessagePassingBlock(hidden, hidden)
        self.local_encoder = _NeighborhoodEncoder(hidden, max_neighbors)

    def forward(
        self, atom_type: torch.Tensor, neighbor_index: torch.Tensor, neighbor_mask: torch.Tensor
    ) -> torch.Tensor:
        """Predict local 3D offsets for every atom's neighborhood.

        Parameters
        ----------
        atom_type : torch.Tensor
            Integer atom types, shape ``(B, N)``.
        neighbor_index : torch.Tensor
            For each atom, indices of up to ``max_neighbors`` neighbors
            (padded with 0 and masked), shape ``(B, N, max_neighbors)``.
        neighbor_mask : torch.Tensor
            Float mask (1 = valid neighbor, 0 = padding), shape
            ``(B, N, max_neighbors)``.

        Returns
        -------
        torch.Tensor
            Local 3D offsets per neighbor, shape
            ``(B, N, max_neighbors, 3)``.
        """

        b, n = atom_type.shape
        torch.manual_seed(0)
        rand_vec = torch.randn(b, n, self.random_vec_dim, device=atom_type.device)
        h0 = torch.cat([self.atom_embed(atom_type), rand_vec], dim=-1)

        # fully-connected edge attr proxy built from concatenated node feats
        n_nodes = h0.shape[1]
        h_i = h0.unsqueeze(2).expand(-1, -1, n_nodes, -1)
        h_j = h0.unsqueeze(1).expand(-1, n_nodes, -1, -1)
        edge_attr = (h_i + h_j) * 0.5

        h1 = self.gnn1(h0, edge_attr)
        h2 = self.gnn2(h0, edge_attr)
        feat = h1 + h2  # (B, N, hidden)

        idx = neighbor_index.unsqueeze(-1).expand(-1, -1, -1, feat.shape[-1])
        neighbor_feat = torch.gather(feat.unsqueeze(1).expand(-1, n, -1, -1), 2, idx)
        neighbor_feat = neighbor_feat * neighbor_mask.unsqueeze(-1)

        offsets = self.local_encoder(feat, neighbor_feat)
        offsets = offsets * neighbor_mask.unsqueeze(-1)
        return offsets


def build_geomol() -> nn.Module:
    """Build a compact GeoMol non-autoregressive conformer generator.

    Returns
    -------
    nn.Module
        Random-initialized ``GeoMol`` in eval mode.
    """

    return GeoMol(num_atom_types=16, hidden=16, random_vec_dim=4, max_neighbors=4).eval()


def example_input_geomol() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a small synthetic molecule with a fixed local-neighborhood map.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``atom_type`` shape ``(2, 6)``, ``neighbor_index`` shape
        ``(2, 6, 4)``, ``neighbor_mask`` shape ``(2, 6, 4)``.
    """

    torch.manual_seed(0)
    atom_type = torch.randint(0, 16, (2, 6))
    # ring-like connectivity: each atom's neighbors are its ring predecessor
    # and successor (2 valid neighbors out of 4 slots, rest padded/masked).
    base = torch.arange(6)
    prev_idx = (base - 1) % 6
    next_idx = (base + 1) % 6
    pad = torch.zeros(6, dtype=torch.long)
    neighbor_index = torch.stack([prev_idx, next_idx, pad, pad], dim=-1)
    neighbor_index = neighbor_index.unsqueeze(0).expand(2, -1, -1).clone()
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0]).expand(6, 4)
    neighbor_mask = mask.unsqueeze(0).expand(2, -1, -1).clone()
    return atom_type, neighbor_index, neighbor_mask


# ---------------------------------------------------------------------------
# 5. GIMLET: unified graph-text transformer with generalized position bias.
# ---------------------------------------------------------------------------


class _GraphTokenEncoder(nn.Module):
    """Graphormer-style graph tokenizer: atom-type + degree embeddings
    produce one token per atom (mirrors ``GraphNodeFeature``).
    """

    def __init__(self, num_atom_types: int = 16, max_degree: int = 8, hidden: int = 32) -> None:
        super().__init__()
        self.atom_embed = nn.Embedding(num_atom_types, hidden)
        self.degree_embed = nn.Embedding(max_degree, hidden)

    def forward(self, atom_type: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        return self.atom_embed(atom_type) + self.degree_embed(degree)


class GIMLET(nn.Module):
    """Compact GIMLET: Graphormer graph tokens concatenated with instruction
    text tokens, attended over jointly by a shared transformer encoder with
    a generalized position bias -- a learned graph-token bias term is merged
    into the text stream's relative-position bias (mirrors
    ``GIMLETEncoderStack.forward``'s ``position_bias_merged`` construction).

    Reference
    ---------
    Zhao, Wu, Wang, Yang, Cao, Cai, Liu, Chen, Wang, Wang. "GIMLET: A
    Unified Graph-Text Model for Instruction-Based Molecule Zero-Shot
    Learning." NeurIPS 2023. arXiv:2306.13089.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        num_atom_types: int = 16,
        hidden: int = 32,
        num_heads: int = 4,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.graph_encoder = _GraphTokenEncoder(num_atom_types, hidden=hidden)
        self.text_embed = nn.Embedding(vocab_size, hidden)
        self.graph_bias = nn.Parameter(torch.zeros(1, 1, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=2 * hidden,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def forward(
        self, atom_type: torch.Tensor, degree: torch.Tensor, text_ids: torch.Tensor
    ) -> torch.Tensor:
        """Jointly encode a molecular graph and an instruction, predicting a
        per-text-token logit distribution (zero-shot instruction head).

        Parameters
        ----------
        atom_type : torch.Tensor
            Integer atom types, shape ``(B, N)``.
        degree : torch.Tensor
            Integer node degree bucket, shape ``(B, N)``.
        text_ids : torch.Tensor
            Instruction token ids, shape ``(B, T)``.

        Returns
        -------
        torch.Tensor
            Logits over the text vocabulary for every merged token position,
            shape ``(B, N + T, vocab_size)``.
        """

        graph_tokens = self.graph_encoder(atom_type, degree) + self.graph_bias
        text_tokens = self.text_embed(text_ids)
        merged = torch.cat([graph_tokens, text_tokens], dim=1)
        encoded = self.encoder(merged)
        return self.lm_head(encoded)


def build_gimlet() -> nn.Module:
    """Build a compact GIMLET unified graph-text transformer.

    Returns
    -------
    nn.Module
        Random-initialized ``GIMLET`` in eval mode.
    """

    return GIMLET(vocab_size=256, num_atom_types=16, hidden=32, num_heads=4, depth=2).eval()


def example_input_gimlet() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a small synthetic molecule graph plus instruction-text batch.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``atom_type`` shape ``(2, 7)``, ``degree`` shape ``(2, 7)``,
        ``text_ids`` shape ``(2, 12)``.
    """

    torch.manual_seed(0)
    atom_type = torch.randint(0, 16, (2, 7))
    degree = torch.randint(0, 8, (2, 7))
    text_ids = torch.randint(0, 256, (2, 12))
    return atom_type, degree, text_ids


MENAGERIE_ENTRIES = [
    ("GDSS", "build_gdss", "example_input_gdss", "2022", "GRAPH"),
    ("GeoDiff", "build_geodiff", "example_input_geodiff", "2022", "BIO"),
    ("GeoLDM", "build_geoldm", "example_input_geoldm", "2023", "BIO"),
    ("GeoMol", "build_geomol", "example_input_geomol", "2021", "BIO"),
    ("GIMLET", "build_gimlet", "example_input_gimlet", "2023", "BIO"),
]

if __name__ == "__main__":
    print(f"{len(MENAGERIE_ENTRIES)} entries defined; run the smoke-trace gate to verify.")

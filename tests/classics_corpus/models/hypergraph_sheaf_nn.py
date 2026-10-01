"""Hypergraph Neural Network (HGNN) and Cellular Sheaf Neural Network.

HGNN:
  Feng et al., "Hypergraph Neural Networks." arXiv:1809.09401 (AAAI 2019).
  Source: https://github.com/iMoonLab/HGNN

Cellular Sheaf Neural Network (SheafNN):
  Bodnar et al., "Neural Sheaf Diffusion: A Topological Perspective on
  Heterophily and Oversmoothing in GNNs." arXiv:2202.04579 (NeurIPS 2022).
  Source: https://github.com/twitter-research/neural-sheaf-diffusion

------------------------------------------------------------------------------
HGNN distinctive primitive (HGNN+ spectral hypergraph convolution):
  The hypergraph is represented by an incidence matrix H (N_nodes x N_edges)
  where H[v,e]=1 if vertex v belongs to hyperedge e.  The HGNN+ convolution is:

    Y = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Theta

  where:
    H     = incidence matrix  (N, M) binary
    W_e   = diagonal hyperedge weight matrix  (M,)
    D_v   = diagonal node degree  diag(H W_e 1)   (N,)
    D_e   = diagonal hyperedge degree  diag(H^T 1)  (M,)
    Theta = learnable weight matrix  (F_in, F_out)

  This propagates features vertex -> hyperedge -> vertex symmetrically.

Faithful-compact simplifications:
  - 6 nodes, 3 hyperedges (each containing 2-4 nodes).
  - 2 HGNN convolution layers with feature dims 8 -> 16 -> 2.
  - Unit hyperedge weights W_e = I.

------------------------------------------------------------------------------
SheafNN (Neural Sheaf Diffusion) distinctive primitive:
  A cellular sheaf assigns to each node v a stalk R^d (d-dimensional space)
  and to each directed edge e=(u,v) a RESTRICTION MAP F_{u,e}: R^d -> R^d
  (a learned d x d matrix). The sheaf Laplacian is:

    L_F = B^T diag(F_e^T F_e) B     (unnormalized form)

  where B is the incidence matrix (node x edge), and the restriction maps
  stack block-diagonally. One diffusion step is:

    X' = X - sigma * L_F X

  (or equivalently the sheaf attention version). Here the d x d restriction
  maps are LEARNED from node features via a small MLP.

Faithful-compact simplifications:
  - 5 nodes (fully connected graph), d=2 (2D stalks).
  - 2 sheaf-diffusion layers.
  - Feature dim = 8.
  - Restriction maps predicted by a small MLP from concatenated node features.
  - Random init, CPU, forward-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# HGNN
# =============================================================================


class HGNNConv(nn.Module):
    """HGNN+ spectral hypergraph convolution: D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X Theta."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.theta = nn.Linear(in_ch, out_ch, bias=False)

    def forward(
        self,
        X: torch.Tensor,  # (N, F_in)
        H: torch.Tensor,  # (N, M) incidence matrix
        W_e: torch.Tensor,  # (M,) hyperedge weights
    ) -> torch.Tensor:
        """Returns (N, F_out)."""
        # Node degrees: d_v = (H W_e 1), (N,)
        D_v_vec = (H * W_e.unsqueeze(0)).sum(dim=1)  # (N,)
        D_v_inv_half = D_v_vec.clamp(min=1e-8).rsqrt()  # (N,)

        # Hyperedge degrees: d_e = H^T 1 = number of nodes per hyperedge (M,)
        D_e_vec = H.sum(dim=0)  # (M,)
        D_e_inv = D_e_vec.clamp(min=1e-8).reciprocal()  # (M,)

        # Symmetric normalization: (D_v^-1/2 H) W_e D_e^-1 (H^T D_v^-1/2) X Theta
        # Step 1: X' = D_v^{-1/2} X
        X_ = X * D_v_inv_half.unsqueeze(-1)  # (N, F)
        # Step 2: Aggregate to hyperedges: E = D_e^{-1} W_e (H^T X')
        HtX = H.t() @ X_  # (M, F)
        HtX = HtX * (W_e * D_e_inv).unsqueeze(-1)  # (M, F)
        # Step 3: Aggregate back to nodes: D_v^{-1/2} H E
        out = H @ HtX  # (N, F)
        out = out * D_v_inv_half.unsqueeze(-1)  # (N, F)
        return F.relu(self.theta(out))  # (N, F_out)


class HGNN(nn.Module):
    """Hypergraph Neural Network (HGNN/HGNN+)."""

    def __init__(self, n_classes: int = 2, in_ch: int = 4) -> None:
        super().__init__()
        self.conv1 = HGNNConv(in_ch, 16)
        self.conv2 = HGNNConv(16, n_classes)

    def forward(
        self,
        X: torch.Tensor,  # (N, in_ch)
        H: torch.Tensor,  # (N, M) incidence
        W_e: torch.Tensor,  # (M,) edge weights
    ) -> torch.Tensor:
        X = self.conv1(X, H, W_e)
        X = self.conv2(X, H, W_e)
        return X  # (N, n_classes)


def _make_hgnn_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(4)
    N, M = 6, 3
    X = torch.randn(N, 4)
    # Build incidence matrix: hyperedge 0={0,1,2}, 1={2,3,4}, 2={0,3,5}
    H = torch.zeros(N, M)
    H[0, 0] = 1
    H[1, 0] = 1
    H[2, 0] = 1
    H[2, 1] = 1
    H[3, 1] = 1
    H[4, 1] = 1
    H[0, 2] = 1
    H[3, 2] = 1
    H[5, 2] = 1
    W_e = torch.ones(M)
    return X, H, W_e


def build_hgnn() -> nn.Module:
    return HGNN(n_classes=2, in_ch=4)


def example_input_hgnn() -> list[torch.Tensor]:
    X, H, W_e = _make_hgnn_inputs()
    return [X, H, W_e]


# =============================================================================
# Sheaf Neural Network (Neural Sheaf Diffusion)
# =============================================================================


class SheafDiffusionLayer(nn.Module):
    """One step of neural sheaf diffusion.

    Given node features X (N, F) and node positions/features for restriction
    map prediction, computes:
      X' = X - sigma * L_F X
    where L_F is the sheaf Laplacian with learned restriction maps.

    Restriction map F_{u->e}: R^d -> R^d is a (d x d) matrix predicted from
    the concatenation of node features [x_u, x_v] by a small MLP.
    """

    def __init__(self, in_ch: int, d_stalk: int = 2, sigma: float = 0.5) -> None:
        super().__init__()
        self.d = d_stalk
        self.sigma = sigma
        # Linear lift: in_ch -> d_stalk
        self.lift = nn.Linear(in_ch, d_stalk, bias=False)
        # Restriction map predictor: (2 * d_stalk) -> (d_stalk * d_stalk)
        self.map_pred = nn.Sequential(
            nn.Linear(2 * d_stalk, 16),
            nn.Tanh(),
            nn.Linear(16, d_stalk * d_stalk),
        )
        # Out projection
        self.out_proj = nn.Linear(d_stalk, in_ch, bias=False)

    def forward(
        self,
        X: torch.Tensor,  # (N, in_ch)
        edge_index: torch.Tensor,  # (2, E) row = [src, dst]
    ) -> torch.Tensor:
        N = X.size(0)
        d = self.d
        E = edge_index.size(1)

        # Lift features to stalk space
        X_lift = self.lift(X)  # (N, d)

        # Compute restriction maps for each directed edge
        src, dst = edge_index[0], edge_index[1]  # (E,)
        x_src = X_lift[src]  # (E, d)
        x_dst = X_lift[dst]  # (E, d)
        pair = torch.cat([x_src, x_dst], dim=-1)  # (E, 2d)
        F_e = self.map_pred(pair).view(E, d, d)  # (E, d, d)

        # Sheaf coboundary: for edge e=(u->v):
        #   delta_X[e] = F_{v,e} x_v - F_{u,e} x_u  (coboundary operator)
        # Sheaf Laplacian times X:
        #   (L_F X)[u] = sum_{e incident to u} F_{u,e}^T (F_{u,e} x_u - F_{v,e} x_v)
        # Here we use the simple form:
        #   (L_F X)[u] += sum_{v~u} F_{u,e}^T F_{u,e} x_u - F_{u,e}^T F_{v,e} x_v

        # F_{src,e} x_src: (E, d)
        Fx_src = torch.bmm(F_e, x_src.unsqueeze(-1)).squeeze(-1)  # (E, d)
        # F_{dst,e} x_dst: (E, d)
        Fx_dst = torch.bmm(F_e, x_dst.unsqueeze(-1)).squeeze(-1)  # (E, d)

        # For each source node: accumulate F^T (F x_src - F x_dst)
        residual_src = Fx_src - Fx_dst  # (E, d)
        Ft_res = torch.bmm(F_e.transpose(1, 2), residual_src.unsqueeze(-1)).squeeze(-1)  # (E,d)

        # Scatter to nodes (src, then dst with opposite sign for symmetry)
        LFX = X_lift.new_zeros(N, d)
        LFX.scatter_add_(0, src.unsqueeze(-1).expand_as(Ft_res), Ft_res)
        # Symmetric: dst gets -F^T (F x_src - F x_dst) = F^T (F x_dst - F x_src)
        residual_dst = Fx_dst - Fx_src
        Ft_res2 = torch.bmm(F_e.transpose(1, 2), residual_dst.unsqueeze(-1)).squeeze(-1)
        LFX.scatter_add_(0, dst.unsqueeze(-1).expand_as(Ft_res2), Ft_res2)

        # Diffusion step
        X_new_lift = X_lift - self.sigma * LFX  # (N, d)
        return F.relu(self.out_proj(X_new_lift))  # (N, in_ch)


class SheafNeuralNetwork(nn.Module):
    """Sheaf Neural Network with learned restriction maps."""

    def __init__(self, in_ch: int = 8, n_layers: int = 2, d_stalk: int = 2) -> None:
        super().__init__()
        self.embed = nn.Linear(4, in_ch)
        self.layers = nn.ModuleList([SheafDiffusionLayer(in_ch, d_stalk) for _ in range(n_layers)])
        self.cls = nn.Linear(in_ch, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """x: (N, 4),  edge_index: (2, E)  ->  (N, 2) per-node logits"""
        h = F.relu(self.embed(x))
        for layer in self.layers:
            h = h + layer(h, edge_index)
        return self.cls(h)


def _make_sheaf_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(5)
    N = 5
    x = torch.randn(N, 4)
    # All edges for fully-connected graph (both directions)
    src = []
    dst = []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return x, edge_index


def build_sheaf_nn() -> nn.Module:
    return SheafNeuralNetwork(in_ch=8, n_layers=2, d_stalk=2)


def example_input_sheaf() -> list[torch.Tensor]:
    x, edge_index = _make_sheaf_inputs()
    return [x, edge_index]


# =============================================================================
# Registry
# =============================================================================

MENAGERIE_ENTRIES = [
    (
        "HGNN (Hypergraph Neural Network)",
        "build_hgnn",
        "example_input_hgnn",
        "2019",
        "DC",
    ),
    (
        "Sheaf Neural Network",
        "build_sheaf_nn",
        "example_input_sheaf",
        "2022",
        "DC",
    ),
]

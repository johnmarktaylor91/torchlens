"""Menagerie batch w5a4: structure/sequence-based bioinformatics predictors --
protein-protein interaction site prediction, ligand-binding-site classification,
eukaryotic gene-structure prediction, and TCR-epitope binding prediction.

Sources checked (reference only; no cloning, no pip installs):

  - **GraphPPIS** ("Structure-aware protein-protein interaction site prediction using
    deep graph convolutional network", Yuan et al., Bioinformatics 2021,
    doi:10.1093/bioinformatics/btab643). Official repo github.com/yuanqm55/GraphPPIS
    (verified via `gh api` file read of `GraphPPIS_model.py`). The distinctive
    mechanism is a **deep GCNII** stack (8 layers by default) applied to a residue
    contact-map graph: each `GraphConvolution` layer combines Chebyshev/GCNII-style
    **initial residual connection** (mixing back in the layer-0 embedding `h0` with
    weight `alpha`) and **identity mapping** (`theta = log(lambda/l + 1)`, blending a
    near-identity transform with the learned weight matrix) so a very deep GCN can be
    trained without over-smoothing; a "variant" mode concatenates the propagated
    neighbor aggregate with `h0` before the linear map. Reimplemented compactly with a
    literal `GCNIILayer` (variant=True, spmm-based propagation on a normalized
    adjacency) at reduced depth/width, node-classification head predicting per-residue
    interface probability.
  - **GraphSite** / **Graphsite-classifier** ("GraphSite: Ligand Binding Site
    Classification with Deep Graph Learning", Shi et al., Biomolecules 2022,
    doi:10.3390/biom12081053). Official repo
    github.com/shiwentao00/Graphsite-classifier (verified via `gh api` file read of
    `gnn/model.py`). The default embedding net (`which_model='normal'` ->
    `EmbeddingNet`) stacks 4 **SCNWMConv** ("single-channel neural weighted message")
    layers: each inherits `GINConv`-style sum-aggregation but the message from every
    neighbor is first **scaled by a learned scalar gate produced from the edge
    attributes** (a small `edge_transformer` MLP maps edge features -> ELU-activated
    weight), i.e. edge-conditioned GIN, followed by `BatchNorm1d` + LeakyReLU. Graph
    embeddings are read out via **Set2Set** (an LSTM-based iterative attention pooling
    that produces a `2*dim` graph vector, distinct from a bare mean/sum pool) before an
    MLP classification head. Reimplemented compactly with literal edge-gated GIN layers
    (`SCNWMConv`) + a from-scratch `Set2Set` readout (LSTM processing-step attention
    over node features) + classification head.
  - **Helixer** ("Helixer: cross-species gene annotation of large eukaryotic genomes
    using deep learning", Stiehler et al., Bioinformatics 2020,
    doi:10.1093/bioinformatics/btaa1044). Official repo github.com/usadellab/Helixer
    (verified via `gh api` file read of `helixer/prediction/HybridModel.py`; shipped
    TensorFlow/Keras, env_notes flags non-PyTorch backend). The distinctive
    "HybridModel" mechanism is a **CNN -> pool-by-reshape -> bidirectional LSTM stack
    -> per-nucleotide-group softmax hat**: a 1D conv over the one-hot 4-channel DNA
    sequence, then a **pool_size-wide channel-concatenating reshape** (folds
    `pool_size` adjacent conv timesteps into the channel dimension rather than
    max/avg-pooling, preserving all activations while reducing sequence length by
    `pool_size`), a stacked `Bidirectional(LSTM)` over the pooled sequence, and a dense
    head that expands each pooled step back into `pool_size` per-base softmax
    predictions over {intergenic, UTR, CDS, intron}. Reimplemented faithfully in torch
    (`nn.LSTM(bidirectional=True)`, literal reshape-based pooling, literal
    unfold-to-per-base classification hat) at reduced depth/width/pool_size.
  - **HeteroTCR** ("HeteroTCR: a heterogeneous graph neural network-based method for
    predicting peptide-TCR interactions", Yu et al., Communications Biology 2024).
    Official repo github.com/yuzilan/HeteroTCR (verified via `gh api` file read of
    `code/HeteroModel.py`). The distinctive mechanism is a **heterogeneous bipartite
    graph** with two node types (`cdr3b`, `peptide`) and message passing via PyG's
    `HeteroConv` wrapping per-edge-type `SAGEConv` layers (default `net_type='SAGE'`,
    3 layers), stacked with LeakyReLU; a decoder MLP concatenates the two endpoint
    embeddings of a queried (cdr3b, peptide) edge and predicts a binding probability via
    sigmoid. Reimplemented compactly with a literal from-scratch heterogeneous SAGE
    layer (separate learned aggregation weights per edge-type/direction, applied over
    an explicit bipartite adjacency) + the concatenate-and-MLP decoder, avoiding a
    torch_geometric `HeteroData`/`HeteroConv` dependency for TorchLens traceability
    while preserving the two-node-type / edge-type-specific-weights mechanism.
  - **ImRex** ("Current challenges for unseen-epitope TCR interaction prediction and a
    new perspective derived from image classification", Moris et al., Briefings in
    Bioinformatics 2020, doi:10.1093/bib/bbaa318). Official repo github.com/pmoris/ImRex
    (verified via `gh api` file read of `src/models/model_padded.py`; shipped
    TensorFlow/Keras `Sequential` Conv2D stack, not PyTorch as queue notes assumed).
    The distinctive mechanism is representing a TCR-epitope pair not as two sequences
    but as a 2D **interaction map**: an outer-product-style image where pixel
    `(i, j)` encodes physicochemical-property interactions between TCR residue `i` and
    epitope residue `j` (hydrophobicity/charge/etc. as channels), then a plain 2D CNN
    (two conv-conv-pool blocks with batchnorm) classifies the whole map as
    binder/non-binder. Reimplemented faithfully: an explicit interaction-map builder
    (outer difference/product of per-residue physicochemical feature vectors -> stacked
    channels) feeding the literal two-block Conv2D-BatchNorm-Conv2D-MaxPool CNN
    classifier head.

Skipped (see MENAGERIE_ENTRIES omission + build log): **HelixRNA** -- the queue entry
cites github.com/PaddlePaddle/PaddleHelix as containing "HelixRNA RNA representation
components", but a `gh api search/code?q=%22HelixRNA%22` GitHub-wide code search returns
zero hits (including within PaddleHelix itself: `apps/`, `pahelix/model_zoo/` list no
such module), and no independently published model named "HelixRNA" exists (only
unrelated, differently-architected models under similar names: Helix-mRNA
(arXiv:2502.13785, a Mamba2/transformer hybrid), the ADAR-editing "Helix" model, and the
splicing "HELIX" model, none of which are "HelixRNA" nor RNA representation-learning
components of PaddleHelix). This is a non-existent/misattributed entity, not a case of
insufficient detail -- there is nothing concrete to faithfully reimplement under this
name.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# GraphPPIS: deep GCNII over a residue contact graph.
# ---------------------------------------------------------------------------


class GCNIILayer(nn.Module):
    """One GCNII graph-convolution layer (initial residual + identity mapping).

    Parameters
    ----------
    dim : int
        Node feature dimension (input == output for this layer).
    variant : bool
        If True, concatenate the propagated aggregate with the initial-residual
        embedding ``h0`` before the linear map (GCNII "variant" mode), doubling the
        effective input width of the learned weight matrix.
    """

    def __init__(self, dim: int, variant: bool = True) -> None:
        super().__init__()
        self.variant = variant
        in_features = 2 * dim if variant else dim
        self.weight = nn.Parameter(torch.empty(in_features, dim))
        stdv = 1.0 / math.sqrt(dim)
        nn.init.uniform_(self.weight, -stdv, stdv)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        h0: torch.Tensor,
        alpha: float,
        theta: float,
    ) -> torch.Tensor:
        """Propagate then mix with initial residual + identity mapping.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape ``(n_nodes, dim)``.
        adj : torch.Tensor
            Normalized dense adjacency, shape ``(n_nodes, n_nodes)``.
        h0 : torch.Tensor
            Initial-residual (layer-0) node embedding, shape ``(n_nodes, dim)``.
        alpha : float
            Initial-residual mixing weight.
        theta : float
            Identity-mapping mixing weight for this layer (``log(lambda/l + 1)``).

        Returns
        -------
        torch.Tensor
            Updated node features, shape ``(n_nodes, dim)``.
        """
        hi = adj @ x
        support = (1 - alpha) * hi + alpha * h0
        if self.variant:
            support = torch.cat([hi, alpha * h0], dim=-1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            r = support
        return theta * (support @ self.weight) + (1 - theta) * r


class GraphPPIS(nn.Module):
    """Compact GraphPPIS: deep-GCNII residue graph node classifier.

    Parameters
    ----------
    n_feat : int
        Input per-residue feature dimension (evolutionary + structural features).
    n_hidden : int
        Hidden node embedding dimension.
    n_layers : int
        Number of stacked GCNII layers.
    n_classes : int
        Number of output classes (2: not-binding / binding).
    lamda : float
        GCNII identity-mapping hyperparameter.
    alpha : float
        GCNII initial-residual mixing hyperparameter.
    """

    def __init__(
        self,
        n_feat: int = 20,
        n_hidden: int = 32,
        n_layers: int = 4,
        n_classes: int = 2,
        lamda: float = 1.5,
        alpha: float = 0.7,
    ) -> None:
        super().__init__()
        self.fc_in = nn.Linear(n_feat, n_hidden)
        self.fc_out = nn.Linear(n_hidden, n_classes)
        self.convs = nn.ModuleList([GCNIILayer(n_hidden, variant=True) for _ in range(n_layers)])
        self.lamda = lamda
        self.alpha = alpha
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Classify each residue node as interface / non-interface.

        Parameters
        ----------
        x : torch.Tensor
            Per-residue input features, shape ``(n_residues, n_feat)``.
        adj : torch.Tensor
            Normalized dense residue-contact adjacency, shape
            ``(n_residues, n_residues)``.

        Returns
        -------
        torch.Tensor
            Per-residue class logits, shape ``(n_residues, n_classes)``.
        """
        h = self.act(self.fc_in(x))
        h0 = h
        for i, conv in enumerate(self.convs):
            theta = min(1.0, math.log(self.lamda / (i + 1) + 1))
            h = self.act(conv(h, adj, h0, self.alpha, theta))
        return self.fc_out(h)


def build_graphppis() -> nn.Module:
    """Build a compact GraphPPIS deep-GCNII residue-interface classifier."""
    return GraphPPIS(n_feat=20, n_hidden=32, n_layers=4, n_classes=2).eval()


def example_input_graphppis() -> tuple[torch.Tensor, torch.Tensor]:
    """Random per-residue features + symmetric-normalized contact adjacency."""
    n_residues = 24
    x = torch.randn(n_residues, 20)
    coords = torch.randn(n_residues, 3).cumsum(dim=0)
    dist = torch.cdist(coords, coords)
    adj = (dist <= dist.median()).float()
    adj = adj + torch.eye(n_residues)
    deg = adj.sum(dim=1).clamp(min=1.0)
    d_inv_sqrt = deg.pow(-0.5)
    adj = d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)
    return x, adj


# ---------------------------------------------------------------------------
# GraphSite: edge-gated GIN (SCNWMConv) + Set2Set readout.
# ---------------------------------------------------------------------------


class SCNWMConv(nn.Module):
    """Single-channel neural weighted message GIN layer (edge-attribute gated).

    Each neighbor's message is scaled by a learned scalar gate produced from the
    edge attributes before sum-aggregation, then passed through a 2-layer MLP
    (GIN-style), matching GraphSite's ``SCNWMConv``.

    Parameters
    ----------
    in_dim : int
        Input node feature dimension.
    out_dim : int
        Output node feature dimension.
    edge_dim : int
        Edge attribute dimension.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.edge_transformer = nn.Sequential(
            nn.Linear(edge_dim, 8), nn.LeakyReLU(), nn.Linear(8, 1), nn.ELU()
        )
        self.nn = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LeakyReLU(), nn.Linear(out_dim, out_dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Edge-gated sum-aggregate neighbor messages, then apply GIN MLP.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape ``(n_nodes, in_dim)``.
        adj : torch.Tensor
            Dense binary adjacency, shape ``(n_nodes, n_nodes)``.
        edge_attr : torch.Tensor
            Dense per-edge-slot attributes, shape ``(n_nodes, n_nodes, edge_dim)``.

        Returns
        -------
        torch.Tensor
            Updated node features, shape ``(n_nodes, out_dim)``.
        """
        weight = self.edge_transformer(edge_attr).squeeze(-1)  # (n, n)
        gated_adj = adj * weight
        agg = gated_adj @ x
        out = agg + (1 + self.eps) * x
        return self.nn(out)


class Set2SetReadout(nn.Module):
    """LSTM-based iterative-attention graph readout (Vinyals et al. Set2Set).

    Parameters
    ----------
    dim : int
        Node feature dimension.
    processing_steps : int
        Number of attention-refinement iterations.
    """

    def __init__(self, dim: int, processing_steps: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.processing_steps = processing_steps
        self.lstm = nn.LSTMCell(2 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Iteratively attend over node features to build a graph embedding.

        Parameters
        ----------
        x : torch.Tensor
            Node features for a single graph, shape ``(n_nodes, dim)``.

        Returns
        -------
        torch.Tensor
            Graph embedding, shape ``(1, 2 * dim)``.
        """
        h = x.new_zeros(1, self.dim)
        c = x.new_zeros(1, self.dim)
        q_star = x.new_zeros(1, 2 * self.dim)
        for _ in range(self.processing_steps):
            h, c = self.lstm(q_star, (h, c))
            scores = (x * h).sum(dim=-1, keepdim=True)  # (n_nodes, 1)
            alpha = torch.softmax(scores, dim=0)
            r = (alpha * x).sum(dim=0, keepdim=True)  # (1, dim)
            q_star = torch.cat([h, r], dim=-1)
        return q_star


class GraphsiteClassifier(nn.Module):
    """Compact GraphSite: edge-gated GIN embedding net + Set2Set + MLP head.

    Parameters
    ----------
    n_feat : int
        Input per-atom/residue node feature dimension.
    edge_dim : int
        Edge attribute dimension.
    dim : int
        Hidden node/graph embedding dimension.
    n_layers : int
        Number of stacked SCNWMConv layers.
    n_classes : int
        Number of binding-site classes.
    """

    def __init__(
        self,
        n_feat: int = 16,
        edge_dim: int = 1,
        dim: int = 24,
        n_layers: int = 3,
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        dims = [n_feat] + [dim] * n_layers
        self.convs = nn.ModuleList(
            [SCNWMConv(dims[i], dims[i + 1], edge_dim) for i in range(n_layers)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for _ in range(n_layers)])
        self.set2set = Set2SetReadout(dim, processing_steps=3)
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Classify a single ligand-binding-site pocket graph.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape ``(n_nodes, n_feat)``.
        adj : torch.Tensor
            Dense binary adjacency, shape ``(n_nodes, n_nodes)``.
        edge_attr : torch.Tensor
            Dense per-edge-slot attributes, shape ``(n_nodes, n_nodes, edge_dim)``.

        Returns
        -------
        torch.Tensor
            Binding-site-class logits, shape ``(1, n_classes)``.
        """
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = F.leaky_relu(conv(h, adj, edge_attr))
            h = bn(h)
        g = self.set2set(h)
        g = F.leaky_relu(self.fc1(g))
        return self.fc2(g)


def build_graphsite() -> nn.Module:
    """Build a compact GraphSite edge-gated-GIN + Set2Set pocket classifier."""
    return GraphsiteClassifier(n_feat=16, edge_dim=1, dim=24, n_layers=3, n_classes=5).eval()


def example_input_graphsite() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random pocket-graph node features, adjacency, and scalar edge distances."""
    n_nodes = 20
    x = torch.randn(n_nodes, 16)
    coords = torch.randn(n_nodes, 3)
    dist = torch.cdist(coords, coords)
    adj = (dist <= dist.median()).float()
    adj.fill_diagonal_(0.0)
    edge_attr = dist.unsqueeze(-1)
    return x, adj, edge_attr


# ---------------------------------------------------------------------------
# Helixer: CNN -> reshape-pool -> BiLSTM stack -> per-base softmax hat.
# ---------------------------------------------------------------------------


class HelixerHybridModel(nn.Module):
    """Compact Helixer HybridModel: dilation-free CNN + reshape-pool + BiLSTM.

    Parameters
    ----------
    in_channels : int
        Number of one-hot input channels (4 nucleotides).
    filter_depth : int
        Number of Conv1d output channels.
    kernel_size : int
        Conv1d kernel width.
    pool_size : int
        Number of adjacent conv time-steps folded into the channel dim per LSTM step
        (Helixer's ``Reshape((-1, pool_size * filter_depth))`` "pooling").
    lstm_units : int
        Hidden size of each unidirectional half of the BiLSTM.
    n_classes : int
        Number of per-base classes (intergenic / UTR / CDS / intron).
    """

    def __init__(
        self,
        in_channels: int = 4,
        filter_depth: int = 16,
        kernel_size: int = 9,
        pool_size: int = 3,
        lstm_units: int = 16,
        n_classes: int = 4,
    ) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.filter_depth = filter_depth
        self.n_classes = n_classes
        self.conv = nn.Conv1d(in_channels, filter_depth, kernel_size, padding=kernel_size // 2)
        self.lstm = nn.LSTM(
            pool_size * filter_depth,
            lstm_units,
            batch_first=True,
            bidirectional=True,
        )
        self.hat = nn.Linear(2 * lstm_units, pool_size * n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a per-base 4-way softmax class distribution.

        Parameters
        ----------
        x : torch.Tensor
            One-hot DNA sequence, shape ``(batch, seq_len, 4)`` with ``seq_len``
            divisible by ``pool_size``.

        Returns
        -------
        torch.Tensor
            Per-base class probabilities, shape ``(batch, seq_len, n_classes)``.
        """
        h = x.transpose(1, 2)  # (batch, 4, seq_len)
        h = F.relu(self.conv(h))
        h = h.transpose(1, 2)  # (batch, seq_len, filter_depth)
        batch, seq_len, _ = h.shape
        pooled_len = seq_len // self.pool_size
        h = h[:, : pooled_len * self.pool_size, :]
        h = h.reshape(batch, pooled_len, self.pool_size * self.filter_depth)
        h, _ = self.lstm(h)
        h = self.hat(h)  # (batch, pooled_len, pool_size * n_classes)
        h = h.reshape(batch, pooled_len * self.pool_size, self.n_classes)
        return F.softmax(h, dim=-1)


def build_helixer() -> nn.Module:
    """Build a compact Helixer CNN-BiLSTM hybrid gene-structure predictor."""
    return HelixerHybridModel(
        in_channels=4, filter_depth=16, kernel_size=9, pool_size=3, lstm_units=16, n_classes=4
    ).eval()


def example_input_helixer() -> torch.Tensor:
    """Random one-hot DNA sequence, shape (1, 90, 4)."""
    seq_len = 90
    idx = torch.randint(0, 4, (1, seq_len))
    return F.one_hot(idx, num_classes=4).float()


# ---------------------------------------------------------------------------
# HeteroTCR: heterogeneous (cdr3b, peptide) bipartite GraphSAGE + MLP decoder.
# ---------------------------------------------------------------------------


class HeteroSAGELayer(nn.Module):
    """One heterogeneous-GraphSAGE layer over a 2-node-type bipartite graph.

    Applies edge-type-specific mean-aggregation + linear transforms in both
    directions (``cdr3b -> peptide`` and ``peptide -> cdr3b``), matching
    HeteroTCR's ``HeteroConv({edge_type: SAGEConv(...)})`` pattern without requiring
    a torch_geometric ``HeteroData`` container.

    Parameters
    ----------
    in_dim : int
        Input node feature dimension (shared across both node types).
    out_dim : int
        Output node feature dimension.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin_self_b = nn.Linear(in_dim, out_dim)
        self.lin_nbr_b = nn.Linear(in_dim, out_dim)
        self.lin_self_p = nn.Linear(in_dim, out_dim)
        self.lin_nbr_p = nn.Linear(in_dim, out_dim)

    def forward(
        self, x_b: torch.Tensor, x_p: torch.Tensor, adj_bp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Message-pass in both directions across the bipartite adjacency.

        Parameters
        ----------
        x_b : torch.Tensor
            CDR3-beta node features, shape ``(n_b, in_dim)``.
        x_p : torch.Tensor
            Peptide node features, shape ``(n_p, in_dim)``.
        adj_bp : torch.Tensor
            Dense bipartite adjacency, shape ``(n_b, n_p)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated ``(x_b, x_p)`` node features, each ``(*, out_dim)``.
        """
        deg_b = adj_bp.sum(dim=1, keepdim=True).clamp(min=1.0)
        deg_p = adj_bp.sum(dim=0, keepdim=True).clamp(min=1.0).transpose(0, 1)
        nbr_for_b = (adj_bp @ x_p) / deg_b
        nbr_for_p = (adj_bp.transpose(0, 1) @ x_b) / deg_p
        new_b = self.lin_self_b(x_b) + self.lin_nbr_b(nbr_for_b)
        new_p = self.lin_self_p(x_p) + self.lin_nbr_p(nbr_for_p)
        return new_b, new_p


class HeteroTCR(nn.Module):
    """Compact HeteroTCR: heterogeneous GraphSAGE encoder + concat-MLP decoder.

    Parameters
    ----------
    in_dim : int
        Input node feature dimension.
    hidden_dim : int
        Hidden node embedding dimension.
    n_layers : int
        Number of stacked heterogeneous SAGE layers.
    """

    def __init__(self, in_dim: int = 20, hidden_dim: int = 32, n_layers: int = 3) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList([HeteroSAGELayer(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.lin1 = nn.Linear(2 * hidden_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.lin3 = nn.Linear(hidden_dim // 4, 1)

    def forward(
        self,
        x_b: torch.Tensor,
        x_p: torch.Tensor,
        adj_bp: torch.Tensor,
        query_row: torch.Tensor,
        query_col: torch.Tensor,
    ) -> torch.Tensor:
        """Predict peptide-CDR3b binding probability for queried node pairs.

        Parameters
        ----------
        x_b : torch.Tensor
            CDR3-beta node features, shape ``(n_b, in_dim)``.
        x_p : torch.Tensor
            Peptide node features, shape ``(n_p, in_dim)``.
        adj_bp : torch.Tensor
            Dense bipartite adjacency, shape ``(n_b, n_p)``.
        query_row : torch.Tensor
            Long indices into ``x_b`` for queried pairs, shape ``(n_query,)``.
        query_col : torch.Tensor
            Long indices into ``x_p`` for queried pairs, shape ``(n_query,)``.

        Returns
        -------
        torch.Tensor
            Binding probability per queried pair, shape ``(n_query,)``.
        """
        for conv in self.convs:
            x_b, x_p = conv(x_b, x_p, adj_bp)
            x_b = F.leaky_relu(x_b)
            x_p = F.leaky_relu(x_p)
        pair = torch.cat([x_b[query_row], x_p[query_col]], dim=-1)
        h = F.relu(self.lin1(pair))
        h = F.relu(self.lin2(h))
        h = self.lin3(h)
        return torch.sigmoid(h).view(-1)


def build_heterotcr() -> nn.Module:
    """Build a compact HeteroTCR heterogeneous-GraphSAGE binding predictor."""
    return HeteroTCR(in_dim=20, hidden_dim=32, n_layers=3).eval()


def example_input_heterotcr() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Random bipartite (cdr3b, peptide) graph with a batch of query pairs."""
    n_b, n_p, n_query = 10, 6, 8
    x_b = torch.randn(n_b, 20)
    x_p = torch.randn(n_p, 20)
    adj_bp = (torch.rand(n_b, n_p) > 0.6).float()
    query_row = torch.randint(0, n_b, (n_query,))
    query_col = torch.randint(0, n_p, (n_query,))
    return x_b, x_p, adj_bp, query_row, query_col


# ---------------------------------------------------------------------------
# ImRex: TCR-epitope physicochemical interaction map + 2D CNN classifier.
# ---------------------------------------------------------------------------


class ImRexCNN(nn.Module):
    """Compact ImRex: 2D-CNN classifier over a TCR-epitope interaction map.

    Parameters
    ----------
    in_channels : int
        Number of physicochemical-property interaction channels.
    depth1 : int
        Output channels of the first conv block.
    depth2 : int
        Output channels of the second conv block.
    """

    def __init__(self, in_channels: int = 4, depth1: int = 16, depth2: int = 24) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, depth1, 3, padding=1),
            nn.BatchNorm2d(depth1),
            nn.ReLU(),
            nn.Conv2d(depth1, depth1, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(depth1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(depth1, depth2, 3, padding=1),
            nn.BatchNorm2d(depth2),
            nn.ReLU(),
            nn.Conv2d(depth2, depth2, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(depth2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(depth2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, interaction_map: torch.Tensor) -> torch.Tensor:
        """Classify a TCR-epitope interaction map as binder / non-binder.

        Parameters
        ----------
        interaction_map : torch.Tensor
            Physicochemical interaction map, shape
            ``(batch, in_channels, tcr_len, epitope_len)``.

        Returns
        -------
        torch.Tensor
            Binding probability, shape ``(batch,)``.
        """
        h = self.block1(interaction_map)
        h = F.relu(h)
        h = self.block2(h)
        h = F.relu(h)
        h = self.gap(h).flatten(1)
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        return torch.sigmoid(h).view(-1)


def build_imrex() -> nn.Module:
    """Build a compact ImRex TCR-epitope interaction-map CNN classifier."""
    return ImRexCNN(in_channels=4, depth1=16, depth2=24).eval()


def example_input_imrex() -> torch.Tensor:
    """Random TCR-epitope physicochemical interaction map, (1, 4, 20, 12)."""
    tcr_len, epitope_len, n_props = 20, 12, 4
    tcr_props = torch.randn(tcr_len, n_props)
    epitope_props = torch.randn(epitope_len, n_props)
    # outer-difference per physicochemical property -> stacked channel image
    interaction_map = tcr_props.unsqueeze(1) - epitope_props.unsqueeze(
        0
    )  # (tcr_len, epitope_len, n_props)
    interaction_map = interaction_map.permute(2, 0, 1).unsqueeze(0)
    return interaction_map


MENAGERIE_ENTRIES = [
    ("GraphPPIS", "build_graphppis", "example_input_graphppis", "2021", "BIO"),
    ("GraphSite", "build_graphsite", "example_input_graphsite", "2022", "BIO"),
    ("Helixer", "build_helixer", "example_input_helixer", "2020", "BIO"),
    ("HeteroTCR", "build_heterotcr", "example_input_heterotcr", "2024", "BIO"),
    ("ImRex", "build_imrex", "example_input_imrex", "2020", "BIO"),
]

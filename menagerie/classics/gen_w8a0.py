"""Generated menagerie classics batch w8a0.

Sources checked (web research + repo browsing via ``gh api``, no clone/pip-install):

- MultiPath: Chai, Sapp, Bansal, Anguelov, "MultiPath: Multiple Probabilistic Anchor
  Trajectory Hypotheses for Behavior Prediction", arXiv:1910.05449 (CoRL 2019). Distinct
  from the later "MultiPath++" (Varadarajan et al. 2021, already captured elsewhere in the
  menagerie): the original MultiPath encodes a rasterized top-down scene with a CNN, then
  regresses, per one of a fixed set of *static anchor trajectories*, a mixture weight plus
  per-waypoint Gaussian offset/uncertainty -- a single forward pass yields the full
  discrete-continuous (categorical-over-anchors x Gaussian-per-anchor) trajectory mixture.

- SPACE-2 (SPACE2.0): He et al., "SPACE-2: Tree-Structured Semi-Supervised Contrastive
  Pre-training for Task-Oriented Dialog Understanding", arXiv:2209.06638 (COLING 2022).
  Official code: github.com/AlibabaResearch/DAMO-ConvAI/tree/main/space-2 (browsed
  ``space/modules/embedder.py``, ``space/modules/subspace.py``, and
  ``space/models/unified_transformer.py`` directly via the GitHub contents API). The
  distinctive mechanism is the ``Subspace`` module: the pooled dialog-turn representation
  is linearly projected and reshaped into ten named "semantic tree" views
  (D=Domain, I=Intent, S=Slot, V=Value, and their unions DI/IS/SV/DIS/ISV/DISV), each of
  which gets its own supervised-contrastive feature used with a tree-similarity score
  matrix during pretraining.

- PDDFormer: "PDDFormer: Pairwise Distance Distribution Graph Transformer for Crystal
  Material Property Prediction", arXiv:2408.12984 (IJCAI 2025). The distinctive mechanism
  (read from the arXiv HTML full text, Sections 4.1/4.3/4.4/Eq. 3/5/7) is: (1) a
  Weighted/Unit-cell Pairwise Distance Distribution matrix (WPDD/UPDD) that summarizes
  each atom's k-nearest-neighbor distance profile as *global* structural information,
  (2) a multi-edge crystal graph built from a cutoff-radius neighbor search with per-edge
  squared-distance features, and (3) alternating node-wise transformer attention blocks
  (edge-feature-augmented multi-head attention) with PDD message-passing blocks that
  update the global PDD matrix via residual gated GELU blocks, so local graph attention
  and the global pairwise-distance-distribution summary co-evolve layer by layer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# 1. MultiPath: rasterized-scene CNN encoder + fixed anchor trajectory mixture
# ---------------------------------------------------------------------------


class MultiPathAnchorHead(nn.Module):
    """Per-anchor mixture-weight and Gaussian waypoint-offset regression head."""

    def __init__(self, in_dim: int, n_anchors: int, n_steps: int) -> None:
        """Initialize the shared trunk and per-anchor output projections.

        Parameters
        ----------
        in_dim:
            Dimensionality of the pooled scene-context feature.
        n_anchors:
            Number of fixed anchor trajectories (static, non-learned template paths).
        n_steps:
            Number of future waypoints predicted per anchor.
        """

        super().__init__()
        self.n_anchors = n_anchors
        self.n_steps = n_steps
        self.trunk = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU())
        self.classify = nn.Linear(in_dim, n_anchors)
        # Per waypoint: (dx, dy, log_sigma_x, log_sigma_y, rho).
        self.regress = nn.Linear(in_dim, n_anchors * n_steps * 5)

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        """Predict anchor mixture logits and per-anchor Gaussian waypoint offsets.

        Parameters
        ----------
        context:
            Pooled scene-context feature, shape ``(batch, in_dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(anchor_logits, gaussian_params)`` with shapes ``(batch, n_anchors)`` and
            ``(batch, n_anchors, n_steps, 5)``.
        """

        h = self.trunk(context)
        anchor_logits = self.classify(h)
        gaussian_params = self.regress(h).view(-1, self.n_anchors, self.n_steps, 5)
        return anchor_logits, gaussian_params


class MultiPath(nn.Module):
    """Compact MultiPath: CNN scene raster encoder + fixed-anchor trajectory mixture."""

    def __init__(
        self,
        in_channels: int = 8,
        n_anchors: int = 6,
        n_steps: int = 10,
        feat_dim: int = 32,
    ) -> None:
        """Build the raster CNN encoder, static anchor bank, and mixture head.

        Parameters
        ----------
        in_channels:
            Channels of the rasterized top-down scene (roadgraph + agent history layers).
        n_anchors:
            Number of fixed anchor trajectories.
        n_steps:
            Future trajectory horizon in waypoints.
        feat_dim:
            Pooled scene-feature width.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, feat_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = MultiPathAnchorHead(feat_dim, n_anchors, n_steps)
        # Fixed (non-trained) static anchor trajectories: buffers, not parameters, since
        # MultiPath's anchors come from offline k-means clustering of training trajectories.
        self.register_buffer(
            "anchors",
            torch.randn(n_anchors, n_steps, 2) * 0.1 + torch.linspace(0, 1, n_steps)[None, :, None],
        )

    def forward(self, raster: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode the raster and predict the anchor-mixture trajectory distribution.

        Parameters
        ----------
        raster:
            Rasterized top-down scene, shape ``(batch, in_channels, H, W)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(anchor_logits, gaussian_params, anchors)``: mixture logits over anchors,
            per-anchor per-waypoint Gaussian parameters, and the static anchor templates
            broadcast so the full predicted trajectory is ``anchors + offsets``.
        """

        feat = self.encoder(raster).flatten(1)
        anchor_logits, gaussian_params = self.head(feat)
        return anchor_logits, gaussian_params, self.anchors


def build_multipath() -> nn.Module:
    """Build a compact random-init MultiPath model.

    Returns
    -------
    nn.Module
        Random-initialized ``MultiPath`` in eval mode.
    """

    return MultiPath(in_channels=8, n_anchors=6, n_steps=10, feat_dim=32).eval()


def example_input_multipath() -> Tensor:
    """Create a small rasterized top-down scene batch.

    Returns
    -------
    Tensor
        Shape ``(2, 8, 32, 32)`` rasterized scene batch.
    """

    return torch.randn(2, 8, 32, 32)


# ---------------------------------------------------------------------------
# 2. SPACE-2: BERT-style dialog encoder + tree-structured multi-view subspaces
# ---------------------------------------------------------------------------


class Space2Embedder(nn.Module):
    """Composite token/position/turn/type embedding for dialog input."""

    def __init__(self, vocab_size: int, hidden_dim: int, max_pos: int, max_turn: int) -> None:
        """Initialize the four additive embedding tables.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        hidden_dim:
            Model hidden width.
        max_pos:
            Maximum token position.
        max_turn:
            Maximum dialog turn index.
        """

        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_dim)
        self.pos = nn.Embedding(max_pos, hidden_dim)
        self.turn = nn.Embedding(max_turn, hidden_dim)
        self.speaker_type = nn.Embedding(2, hidden_dim)  # user vs. system turn.

    def forward(self, token_ids: Tensor, turn_ids: Tensor, type_ids: Tensor) -> Tensor:
        """Sum token/position/turn/type embeddings for a dialog token sequence.

        Parameters
        ----------
        token_ids:
            Token ids, shape ``(batch, seq_len)``.
        turn_ids:
            Per-token dialog-turn index, shape ``(batch, seq_len)``.
        type_ids:
            Per-token speaker-type index (0/1), shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Composite embeddings, shape ``(batch, seq_len, hidden_dim)``.
        """

        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        return (
            self.token(token_ids)
            + self.pos(positions)
            + self.turn(turn_ids)
            + self.speaker_type(type_ids)
        )


class Space2Subspace(nn.Module):
    """Project the pooled dialog feature into semantic-tree-structure subspace views.

    Mirrors DAMO-ConvAI's ``space/modules/subspace.py``: the ten named views are the
    semantic-tree-structure (STS) levels Domain / Intent / Slot / Value and their unions
    (D, I, S, V, DI, IS, SV, DIS, ISV, DISV). Each view later receives its own
    tree-similarity-gated supervised-contrastive loss during pretraining.
    """

    subspace_names: tuple[str, ...] = ("D", "I", "S", "V", "DI", "IS", "SV", "DIS", "ISV", "DISV")

    def __init__(self, hidden_dim: int, subspace_dim: int) -> None:
        """Initialize the joint linear projection into all named subspaces.

        Parameters
        ----------
        hidden_dim:
            Pooled encoder feature width.
        subspace_dim:
            Width of each individual subspace view.
        """

        super().__init__()
        self.subspace_dim = subspace_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, subspace_dim * len(self.subspace_names)), nn.Tanh()
        )

    def forward(self, pooled: Tensor) -> Tensor:
        """Reshape the joint projection into per-view subspace features.

        Parameters
        ----------
        pooled:
            Pooled dialog-turn representation, shape ``(batch, hidden_dim)``.

        Returns
        -------
        Tensor
            Stacked subspace views, shape ``(batch, n_views, subspace_dim)``.
        """

        out = self.projection(pooled)
        return out.view(pooled.shape[0], len(self.subspace_names), self.subspace_dim)


class Space2Model(nn.Module):
    """Compact SPACE-2: BERT-style dialog encoder + tree-structured multi-view head."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        subspace_dim: int = 8,
        max_pos: int = 32,
        max_turn: int = 8,
    ) -> None:
        """Build the embedder, transformer encoder, and multi-view subspace head.

        Parameters
        ----------
        vocab_size:
            Token vocabulary size.
        hidden_dim:
            Model hidden width.
        n_layers:
            Number of transformer encoder layers.
        n_heads:
            Number of self-attention heads.
        subspace_dim:
            Width of each semantic-tree subspace view.
        max_pos:
            Maximum token position.
        max_turn:
            Maximum dialog turn index.
        """

        super().__init__()
        self.embedder = Space2Embedder(vocab_size, hidden_dim, max_pos, max_turn)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = nn.Linear(hidden_dim, hidden_dim)
        self.subspace = Space2Subspace(hidden_dim, subspace_dim)

    def forward(self, token_ids: Tensor, turn_ids: Tensor, type_ids: Tensor) -> Tensor:
        """Encode a dialog and project it into semantic-tree subspace views.

        Parameters
        ----------
        token_ids:
            Token ids, shape ``(batch, seq_len)``.
        turn_ids:
            Per-token dialog-turn index, shape ``(batch, seq_len)``.
        type_ids:
            Per-token speaker-type index, shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Stacked multi-view subspace features, shape ``(batch, 10, subspace_dim)``.
        """

        embedded = self.embedder(token_ids, turn_ids, type_ids)
        encoded = self.encoder(embedded)
        pooled = torch.tanh(self.pool(encoded[:, 0]))
        return self.subspace(pooled)


def build_space2() -> nn.Module:
    """Build a compact random-init SPACE-2 model.

    Returns
    -------
    nn.Module
        Random-initialized ``Space2Model`` in eval mode.
    """

    return Space2Model().eval()


def example_input_space2() -> tuple[Tensor, Tensor, Tensor]:
    """Create a small tokenized dialog batch.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(token_ids, turn_ids, type_ids)``, each shape ``(2, 16)``.
    """

    torch.manual_seed(0)
    token_ids = torch.randint(0, 256, (2, 16))
    turn_ids = torch.randint(0, 8, (2, 16))
    type_ids = torch.randint(0, 2, (2, 16))
    return token_ids, turn_ids, type_ids


# ---------------------------------------------------------------------------
# 3. PDDFormer: pairwise-distance-distribution graph transformer for crystals
# ---------------------------------------------------------------------------


class PDDNodeTransformerBlock(nn.Module):
    """Edge-feature-augmented multi-head attention over the multi-edge crystal graph."""

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        """Initialize query/key/value/edge projections for one attention block.

        Parameters
        ----------
        dim:
            Node feature width.
        n_heads:
            Number of attention heads.
        """

        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(1, dim)
        self.gate = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, nodes: Tensor, edge_dist2: Tensor) -> Tensor:
        """Attend each atom to its multi-edge neighbors using distance-augmented keys.

        Parameters
        ----------
        nodes:
            Atom feature matrix, shape ``(batch, n_atoms, dim)``.
        edge_dist2:
            Squared pairwise Euclidean distances (the multi-edge feature ``|e_ij|^2``),
            shape ``(batch, n_atoms, n_atoms)``.

        Returns
        -------
        Tensor
            Updated node features, shape ``(batch, n_atoms, dim)``.
        """

        b, n, d = nodes.shape
        q = self.q(nodes)
        k = self.k(nodes)
        v = self.v(nodes)
        edge_bias = self.edge_proj(edge_dist2.unsqueeze(-1)).mean(dim=-1)
        att = torch.matmul(q, k.transpose(-1, -2)) / (d**0.5) + edge_bias
        weights = torch.softmax(att, dim=-1)
        message = torch.matmul(weights, v)
        gated = torch.sigmoid(self.gate(nodes)) * message
        return self.norm(nodes + self.out(gated))


class PDDMessagePassingBlock(nn.Module):
    """Residual gated-GELU update of the global pairwise-distance-distribution matrix."""

    def __init__(self, dim: int) -> None:
        """Initialize the PDD-matrix gated update projections.

        Parameters
        ----------
        dim:
            PDD feature width (matches the node feature width after alignment).
        """

        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.branch_a = nn.Linear(dim, dim)
        self.branch_b = nn.Linear(dim, dim)
        self.drop = nn.Dropout(0.0)

    def forward(self, pdd: Tensor, nodes: Tensor) -> Tensor:
        """Fold updated node information back into the global PDD summary matrix.

        Parameters
        ----------
        pdd:
            Global pairwise-distance-distribution matrix, shape ``(batch, n_atoms, dim)``.
        nodes:
            Latest node features from the transformer block, shape ``(batch, n_atoms, dim)``.

        Returns
        -------
        Tensor
            Updated PDD matrix, shape ``(batch, n_atoms, dim)``.
        """

        pdd = pdd + nodes
        normed = self.norm_in(pdd)
        gated = self.branch_a(normed) * self.drop(F.gelu(self.branch_b(normed)))
        return pdd + gated


class PDDFormer(nn.Module):
    """Compact PDDFormer: WPDD-derived node features + alternating attention/PDD blocks."""

    def __init__(
        self, n_atom_types: int = 20, dim: int = 32, n_layers: int = 3, k_neighbors: int = 8
    ) -> None:
        """Build the atom embedding, WPDD projection, and alternating block stack.

        Parameters
        ----------
        n_atom_types:
            Number of distinct atomic species embedded.
        dim:
            Node / PDD feature width.
        n_layers:
            Number of alternating (node-transformer, PDD-message-passing) block pairs.
        k_neighbors:
            Number of nearest-neighbor distances retained per atom in the WPDD row.
        """

        super().__init__()
        self.k_neighbors = k_neighbors
        self.atom_embed = nn.Embedding(n_atom_types, dim)
        # WPDD(S;k) in R^{n x (k+1)}: atomic-mass weight column concatenated with the
        # k sorted nearest-neighbor distances; project that compact row into node space.
        self.wpdd_proj = nn.Linear(k_neighbors + 1, dim)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": PDDNodeTransformerBlock(dim),
                        "pdd": PDDMessagePassingBlock(dim),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.readout = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, atom_types: Tensor, wpdd_rows: Tensor, coords: Tensor) -> Tensor:
        """Predict a crystal-level scalar property from atom types and WPDD rows.

        Parameters
        ----------
        atom_types:
            Atomic species index per atom, shape ``(batch, n_atoms)``.
        wpdd_rows:
            Per-atom weighted pairwise-distance-distribution rows
            ``WPDD(S;k) in R^{n x (k+1)}``, shape ``(batch, n_atoms, k_neighbors + 1)``.
        coords:
            Cartesian atom coordinates used to build the multi-edge squared-distance
            feature, shape ``(batch, n_atoms, 3)``.

        Returns
        -------
        Tensor
            Predicted crystal property, shape ``(batch, 1)``.
        """

        nodes = self.atom_embed(atom_types) + self.wpdd_proj(wpdd_rows)
        pdd = nodes.clone()
        edge_dist2 = torch.cdist(coords, coords, p=2) ** 2
        for block in self.blocks:
            nodes = block["attn"](nodes, edge_dist2)
            pdd = block["pdd"](pdd, nodes)
        pooled = pdd.mean(dim=1)
        return self.readout(pooled)


def build_pddformer() -> nn.Module:
    """Build a compact random-init PDDFormer model.

    Returns
    -------
    nn.Module
        Random-initialized ``PDDFormer`` in eval mode.
    """

    return PDDFormer(n_atom_types=20, dim=32, n_layers=3, k_neighbors=8).eval()


def example_input_pddformer() -> tuple[Tensor, Tensor, Tensor]:
    """Create a small synthetic crystal-structure batch.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        ``(atom_types, wpdd_rows, coords)`` for a batch of 2 crystals with 12 atoms each.
    """

    torch.manual_seed(0)
    batch, n_atoms, k = 2, 12, 8
    atom_types = torch.randint(0, 20, (batch, n_atoms))
    wpdd_rows = torch.rand(batch, n_atoms, k + 1)
    coords = torch.randn(batch, n_atoms, 3)
    return atom_types, wpdd_rows, coords


MENAGERIE_ENTRIES = [
    ("MultiPath", "build_multipath", "example_input_multipath", "2019", "SEQ"),
    ("SPACE (deep learning Hi-C)", "build_space2", "example_input_space2", "2022", "NLP"),
    ("PDDFormer", "build_pddformer", "example_input_pddformer", "2024", "GRAPH"),
]

if __name__ == "__main__":
    print(f"{len(MENAGERIE_ENTRIES)} entries defined; run the smoke-trace gate to verify.")

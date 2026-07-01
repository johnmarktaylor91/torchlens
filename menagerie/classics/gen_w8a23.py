"""Wave 8 batch 23 menagerie classics: molecular representation-learning / generative models.

Sources checked (reference only -- no cloning, no pip installs; reimplemented
compactly from scratch in base-env torch):

  * PretrainGNN (ContextPred): Hu, Liu, Gomes, Zitnik, Liang, Pande, Leskovec,
    "Strategies for Pre-training Graph Neural Networks", ICLR 2020,
    arXiv:1905.12265; https://github.com/snap-stanford/pretrain-gnns. A
    GIN-style molecular-graph encoder (edge-embedding-augmented message
    passing over atom/bond-type embeddings, ``chem/model.py::GINConv``) is
    pretrained with the ContextPred objective (``pretrain_contextpred.py``):
    a *substructure* encoder produces a center-atom embedding from a
    K-hop neighborhood, and a separate *context* encoder produces an
    embedding of the surrounding r1..r2-hop annular context, and the two are
    trained (skip-gram/CBOW style) so a substructure's embedding predicts its
    own surrounding context's embedding (and not other molecules' contexts).
    Reimplemented here as the two tied-architecture GIN encoders plus the
    context-pooling + substructure-context dot-product scoring head.
  * HierVAE: Jin, Barzilay, Jaakkola, "Hierarchical Generation of Molecular
    Graphs using Structural Motifs", ICML 2020, arXiv:2002.03230;
    https://github.com/wengong-jin/hgraph2graph. A hierarchical VAE
    (``hgraph/hgnn.py::HierVAE``) whose encoder (``hgraph/encoder.py::
    HierMPNEncoder``) runs *two* levels of directed message passing -- a
    fine atom-level graph encoder and a coarse motif-level tree encoder
    (each a GRU-style bond-message MPN, ``MPNEncoder``) -- and fuses the two
    into a root graph vector via a tanh-gated linear combiner
    (``W_root``), from which a diagonal-Gaussian latent is sampled
    (reparameterization trick) exactly as in ``HierVAE.rsample``. The
    autoregressive tree/graph *decoder* (variable-length motif attachment
    order) is not control-flow-traceable by TorchLens with a fixed shape, so
    this reimplementation captures the encoder + VAE bottleneck faithfully
    (the paper's central "read structure at two granularities, latent is a
    fusion of both" mechanism) on small fixed-size molecular graphs, using
    dense scatter-add neighbor aggregation in place of the original's ragged
    ``index_select_ND`` gather (equivalent aggregation, traceable shapes).
  * IPDiff: Huang, Peng, Ma, Zhou, Xu, Chen, Wang, Wang, "Interaction-Prior
    Guided Diffusion Model for Target-Aware Ligand Generation", ICLR 2024
    (as "IPDiff" in the SOURCE_AVAILABLE row; matches the official
    YangLing0818/IPDiff repo), https://github.com/YangLing0818/IPDiff. Builds
    on the TargetDiff/ScorePosNet3D backbone (``models/molopt_score_model.py
    ::ScorePosNet3D.forward``): protein-pocket and ligand atoms are embedded,
    concatenated ("composed") into one point cloud, and refined jointly by an
    E(3)-equivariant transformer/EGNN (``UniTransformerO2TwoUpdateGeneral`` /
    ``EGNN``). IPDiff's distinctive addition is the *interaction prior*: a
    history-derived binding-affinity-prior (HBAP) embedding for protein and
    ligand atoms (``hbap_protein``, ``hbap_ligand``) is fused into the atom
    features via a dedicated ``emb_mlp`` *before* the refine-net runs
    (``self.emb_mlp(torch.cat([h_protein, hbap_protein], dim=1))``),
    conditioning the denoiser on a cross-attention-derived protein-ligand
    interaction summary rather than raw diffusion noise alone. Reimplemented
    as a single denoising step: prior-fusion MLP + EGNN-style equivariant
    message passing over a composed protein+ligand point cloud, with a final
    atom-type/coordinate prediction head, on small fixed atom counts.
  * JANUS: Nigam, Pollice, Krenn, Gomes, Aspuru-Guzik, "Beyond Generative
    Models: Superfast Traversal, Optimization, Modification and Exploration
    (STONED) Algorithm for Molecules using SELFIES", 2021/2022 (JANUS =
    parallel-tempered genetic algorithm using SELFIES string mutation +
    crossover); https://github.com/aspuru-guzik-group/JANUS (PyPI
    ``janus-ga``). The genetic-algorithm loop over SELFIES strings itself is
    not a neural network; JANUS's one trained component is a small
    all-sigmoid MLP classifier (``src/janus/network.py::MLP``) used as a
    fitness-shortlisting discriminator over RDKit molecular descriptors,
    trained with ``BCELoss`` to separate "high value" vs "low value"
    molecules and used to pre-filter GA-generated candidates before scoring.
    Reimplemented faithfully as that sigmoid-activated MLP (arbitrary hidden
    sizes, sigmoid output).
  * KANO: Fang, Zhang, Zhang, Chen, Fan, Chen, "Knowledge Graph-enhanced
    Molecular Contrastive Learning with Functional Prompt", Nature Machine
    Intelligence 2023; https://github.com/HICAI-ZJU/KANO. A CMPN
    (directed bond-message D-MPNN, ``chemprop/models/cmpn.py::CMPNEncoder``,
    sum-then-max neighbor aggregation each depth) whose atom-input embedding
    is fused with an ElementKG-pretrained functional-group knowledge
    embedding via the ``Prompt_generator`` (``chemprop/models/model.py``):
    two stacked ``AttentionLayer`` self-attention blocks over per-molecule
    functional-group states produce a CLS-style summary that is
    alpha-gated-residual-added onto the atom hidden states
    (``atom_hiddens + self.alpha * fg_out``) before message passing. This is
    the paper's "knowledge graph as prompt" mechanism. Reimplemented on
    small fixed-size molecular graphs with dense per-molecule tensors
    (batch, atoms, feat) in place of the original's flattened ragged
    scatter/gather indexing.
  * KV-PLM: Zeng, Yao, Liu, Sun, "A deep-learning system bridging molecule
    structure and biomedical text with comprehension comparable to human
    professionals", Nature Communications 2022;
    https://github.com/thunlp/KV-PLM. A single shared BERT/SciBERT tower
    (``modeling.py``) encodes *both* SMILES strings (treated as ordinary
    subword-tokenized text, KV-PLM's key idea of using one PLM's vocabulary
    for both chemistry and biomedical natural language, optionally with
    byte-pair encoding of SMILES) and biomedical text; ``demo_matching.py``
    shows the downstream use -- pool both encodings (``BigModel``, pooled
    CLS + dropout) and score their cosine similarity for
    structure-text matching/retrieval. Reimplemented as a compact shared
    Transformer-encoder tower (token + position embeddings, a handful of
    pre-norm self-attention blocks) run once on a SMILES-token sequence and
    once on a text-token sequence, with a CLS-pooling head and cosine-
    similarity matching score, faithfully capturing the "one shared PLM
    tower for both modalities" mechanism.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# PretrainGNN (ContextPred) -- tied GIN substructure/context encoders with a
# skip-gram-style substructure<->context dot-product scoring head.
# ---------------------------------------------------------------------------


class GINMolConv(nn.Module):
    """One GIN message-passing layer with additive edge-type embeddings.

    Mirrors ``chem/model.py::GINConv`` in ``pretrain-gnns``: edge features
    (bond type, bond direction) are embedded and added to source-node
    features before an MLP update, aggregated over a dense adjacency matrix
    (an equivalent, TorchLens-traceable substitute for the original's
    sparse ``torch_geometric`` scatter aggregation).
    """

    def __init__(self, emb_dim: int, n_bond_type: int = 6) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding = nn.Embedding(n_bond_type, emb_dim)

    def forward(self, x: Tensor, adj: Tensor, edge_type: Tensor) -> Tensor:
        """Aggregate neighbor features (plus edge embeddings) and update.

        Parameters
        ----------
        x : Tensor
            Node features, shape ``(batch, n_atoms, emb_dim)``.
        adj : Tensor
            Dense adjacency (with self-loops), shape ``(batch, n_atoms, n_atoms)``.
        edge_type : Tensor
            Bond-type index per (target, source) pair, shape ``(batch, n_atoms, n_atoms)``.

        Returns
        -------
        Tensor
            Updated node features, same shape as ``x``.
        """
        edge_emb = self.edge_embedding(edge_type)  # (batch, n, n, emb_dim)
        messages = x.unsqueeze(1) + edge_emb  # broadcast source features + edge embedding
        agg = (messages * adj.unsqueeze(-1)).sum(dim=2)
        return self.mlp(agg)


class GINMolEncoder(nn.Module):
    """Stack of :class:`GINMolConv` layers with atom-type input embedding."""

    def __init__(self, emb_dim: int = 32, n_layers: int = 3, n_atom_type: int = 20) -> None:
        super().__init__()
        self.atom_embedding = nn.Embedding(n_atom_type, emb_dim)
        self.layers = nn.ModuleList([GINMolConv(emb_dim) for _ in range(n_layers)])

    def forward(self, atom_type: Tensor, adj: Tensor, edge_type: Tensor) -> Tensor:
        """Encode a batch of small molecular graphs into per-atom embeddings."""
        h = self.atom_embedding(atom_type)
        for layer in self.layers:
            h = F.relu(layer(h, adj, edge_type))
        return h


class ContextPredGNN(nn.Module):
    """PretrainGNN's ContextPred pretraining head: substructure vs context GIN encoders.

    Two independently-parameterized GIN encoders (mirroring
    ``model_substruct`` / ``model_context`` in ``pretrain_contextpred.py``)
    each encode the *same* small molecular graph; the substructure encoder's
    center-atom embedding and the mean-pooled context encoder embedding are
    combined via a dot product, exactly as the skip-gram/CBOW ContextPred
    scoring in the original training loop.
    """

    def __init__(self, emb_dim: int = 32, n_layers: int = 3, n_atom_type: int = 20) -> None:
        super().__init__()
        self.substruct_encoder = GINMolEncoder(emb_dim, n_layers, n_atom_type)
        self.context_encoder = GINMolEncoder(emb_dim, n_layers, n_atom_type)

    def forward(
        self, atom_type: Tensor, adj: Tensor, edge_type: Tensor, center_idx: Tensor
    ) -> Tensor:
        """Return the substructure<->context matching score per graph.

        Parameters
        ----------
        atom_type, adj, edge_type
            As in :class:`GINMolEncoder`.
        center_idx : Tensor
            Index of the "center" atom per graph, shape ``(batch,)``.

        Returns
        -------
        Tensor
            Per-graph dot-product score, shape ``(batch,)``.
        """
        substruct_h = self.substruct_encoder(atom_type, adj, edge_type)
        context_h = self.context_encoder(atom_type, adj, edge_type)
        batch_idx = torch.arange(atom_type.shape[0], device=atom_type.device)
        center_rep = substruct_h[batch_idx, center_idx]  # (batch, emb_dim)
        context_rep = context_h.mean(dim=1)  # pooled annular-context embedding
        return (center_rep * context_rep).sum(dim=-1)


def build_pretraingnn_contextpred() -> nn.Module:
    """Build a compact PretrainGNN ContextPred model.

    Returns
    -------
    nn.Module
        ``ContextPredGNN`` in eval mode.
    """
    return ContextPredGNN(emb_dim=32, n_layers=3, n_atom_type=20).eval()


def example_input_pretraingnn_contextpred() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (atom types, dense adjacency, bond types, center-atom indices)."""
    batch, n_atoms = 2, 10
    atom_type = torch.randint(0, 20, (batch, n_atoms))
    adj = (torch.rand(batch, n_atoms, n_atoms) > 0.7).float()
    adj = adj + adj.transpose(1, 2)
    adj = (adj > 0).float()
    for b in range(batch):
        adj[b].fill_diagonal_(1.0)
    edge_type = torch.randint(0, 6, (batch, n_atoms, n_atoms))
    center_idx = torch.randint(0, n_atoms, (batch,))
    return atom_type, adj, edge_type, center_idx


# ---------------------------------------------------------------------------
# HierVAE -- two-level (atom graph + motif tree) directed message-passing
# encoder fused into a diagonal-Gaussian latent via reparameterization.
# ---------------------------------------------------------------------------


class DirectedMPNEncoder(nn.Module):
    """Directed bond-message GRU-gated MPN, mirroring ``hgraph/encoder.py::MPNEncoder``.

    Node features are combined with dense scatter-add neighbor aggregation
    of bond messages (a traceable substitute for the original's
    ``index_select_ND`` ragged gather), refined for a fixed depth with a
    GRUCell update per message, then projected to node hidden states.
    """

    def __init__(self, node_dim: int, hidden_size: int, depth: int = 3) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.w_i = nn.Linear(node_dim, hidden_size)
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)
        self.w_o = nn.Sequential(nn.Linear(node_dim + hidden_size, hidden_size), nn.ReLU())

    def forward(self, fnode: Tensor, adj: Tensor) -> Tensor:
        """Encode nodes given a dense (batch, n, n) adjacency matrix.

        Parameters
        ----------
        fnode : Tensor
            Node input features, shape ``(batch, n, node_dim)``.
        adj : Tensor
            Dense neighbor adjacency, shape ``(batch, n, n)``.

        Returns
        -------
        Tensor
            Node hidden states, shape ``(batch, n, hidden_size)``.
        """
        batch, n, _ = fnode.shape
        h = torch.tanh(self.w_i(fnode))
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
        for _ in range(self.depth):
            nei_sum = torch.bmm(adj, h) / deg
            h_flat = h.reshape(batch * n, self.hidden_size)
            nei_flat = nei_sum.reshape(batch * n, self.hidden_size)
            h = self.gru_cell(nei_flat, h_flat).reshape(batch, n, self.hidden_size)
        return self.w_o(torch.cat([fnode, h], dim=-1))


class HierMPNEncoderCompact(nn.Module):
    """Two-level (fine atom graph + coarse motif tree) hierarchical MPN encoder.

    Mirrors ``hgraph/encoder.py::HierMPNEncoder``: separately encodes the
    atom-level molecular graph and the motif-level junction tree, then fuses
    the two pooled representations with a tanh-activated linear combiner
    (``W_root``) exactly as the original.
    """

    def __init__(
        self, atom_feat_dim: int = 16, motif_feat_dim: int = 16, hidden_size: int = 32
    ) -> None:
        super().__init__()
        self.graph_encoder = DirectedMPNEncoder(atom_feat_dim, hidden_size, depth=3)
        self.tree_encoder = DirectedMPNEncoder(motif_feat_dim, hidden_size, depth=3)
        self.w_root = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Tanh())

    def forward(
        self, atom_feat: Tensor, atom_adj: Tensor, motif_feat: Tensor, motif_adj: Tensor
    ) -> Tensor:
        """Return the fused root vector per molecule, shape ``(batch, hidden_size)``."""
        graph_h = self.graph_encoder(atom_feat, atom_adj).mean(dim=1)
        tree_h = self.tree_encoder(motif_feat, motif_adj).mean(dim=1)
        return self.w_root(torch.cat([graph_h, tree_h], dim=-1))


class HierVAECompact(nn.Module):
    """Compact HierVAE: hierarchical encoder + diagonal-Gaussian VAE bottleneck.

    Mirrors ``hgraph/hgnn.py::HierVAE.rsample`` for the reparameterization
    step. The autoregressive motif-attachment decoder is not reproduced
    (variable-length, control-flow-heavy); this module captures the paper's
    central encoder mechanism -- read the molecule at two structural
    granularities and fuse them into one latent.
    """

    def __init__(
        self,
        atom_feat_dim: int = 16,
        motif_feat_dim: int = 16,
        hidden_size: int = 32,
        latent_size: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = HierMPNEncoderCompact(atom_feat_dim, motif_feat_dim, hidden_size)
        self.r_mean = nn.Linear(hidden_size, latent_size)
        self.r_var = nn.Linear(hidden_size, latent_size)

    def forward(
        self, atom_feat: Tensor, atom_adj: Tensor, motif_feat: Tensor, motif_adj: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return (sampled latent, KL divergence per batch element)."""
        root_vec = self.encoder(atom_feat, atom_adj, motif_feat, motif_adj)
        z_mean = self.r_mean(root_vec)
        z_log_var = -torch.abs(self.r_var(root_vec))
        kl = -0.5 * torch.sum(1.0 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1)
        eps = torch.randn_like(z_mean)
        z = z_mean + torch.exp(z_log_var / 2) * eps
        return z, kl


def build_hiervae() -> nn.Module:
    """Build a compact HierVAE encoder + VAE bottleneck.

    Returns
    -------
    nn.Module
        ``HierVAECompact`` in eval mode.
    """
    return HierVAECompact(
        atom_feat_dim=16, motif_feat_dim=16, hidden_size=32, latent_size=16
    ).eval()


def example_input_hiervae() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (atom features/adjacency, motif features/adjacency)."""
    batch, n_atoms, n_motifs = 2, 12, 4
    atom_feat = torch.randn(batch, n_atoms, 16)
    atom_adj = (torch.rand(batch, n_atoms, n_atoms) > 0.6).float()
    atom_adj = ((atom_adj + atom_adj.transpose(1, 2)) > 0).float()
    motif_feat = torch.randn(batch, n_motifs, 16)
    motif_adj = (torch.rand(batch, n_motifs, n_motifs) > 0.4).float()
    motif_adj = ((motif_adj + motif_adj.transpose(1, 2)) > 0).float()
    return atom_feat, atom_adj, motif_feat, motif_adj


# ---------------------------------------------------------------------------
# IPDiff -- interaction-prior-fused equivariant point-cloud denoiser for
# protein-pocket / ligand joint diffusion.
# ---------------------------------------------------------------------------


class EquivariantRefineLayer(nn.Module):
    """One E(3)-equivariant message-passing / coordinate-update layer (EGNN-style)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 1)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, h: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
        """Update (features, coordinates) with one equivariant message pass.

        Parameters
        ----------
        h : Tensor
            Node features, shape ``(batch, n, hidden_size)``.
        pos : Tensor
            3D coordinates, shape ``(batch, n, 3)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated (features, coordinates), same shapes as inputs.
        """
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (batch, n, n, 3)
        dist2 = diff.pow(2).sum(dim=-1, keepdim=True)  # (batch, n, n, 1)
        h_i = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
        h_j = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
        edge_feat = self.edge_mlp(torch.cat([h_i, h_j, dist2], dim=-1))
        coord_w = self.coord_mlp(edge_feat)
        pos_update = (diff * coord_w).mean(dim=2)
        pos_new = pos + pos_update
        agg = edge_feat.mean(dim=2)
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        return h_new, pos_new


class InteractionPriorDiffusion(nn.Module):
    """IPDiff single denoising step: prior-fused atom embedding + equivariant refine net.

    Mirrors ``models/molopt_score_model.py::ScorePosNet3D.forward``: protein
    and ligand atoms are embedded, each fused with a precomputed
    "history binding-affinity prior" (HBAP) interaction embedding via a
    shared ``emb_mlp`` (the paper's distinctive interaction-prior guidance),
    concatenated into one composed point cloud, and refined jointly by a
    stack of :class:`EquivariantRefineLayer` layers; a final linear head
    predicts the denoised ligand atom-type logits.
    """

    def __init__(
        self,
        protein_feat_dim: int = 12,
        ligand_feat_dim: int = 10,
        prior_dim: int = 8,
        hidden_size: int = 32,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.protein_emb = nn.Linear(protein_feat_dim, hidden_size)
        self.ligand_emb = nn.Linear(ligand_feat_dim, hidden_size)
        self.emb_mlp = nn.Linear(hidden_size + prior_dim, hidden_size)
        self.layers = nn.ModuleList([EquivariantRefineLayer(hidden_size) for _ in range(n_layers)])
        self.v_head = nn.Linear(hidden_size, ligand_feat_dim)

    def forward(
        self,
        protein_feat: Tensor,
        protein_pos: Tensor,
        protein_prior: Tensor,
        ligand_feat: Tensor,
        ligand_pos: Tensor,
        ligand_prior: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return (predicted ligand atom-type logits, refined ligand coordinates)."""
        h_protein = self.emb_mlp(torch.cat([self.protein_emb(protein_feat), protein_prior], dim=-1))
        h_ligand = self.emb_mlp(torch.cat([self.ligand_emb(ligand_feat), ligand_prior], dim=-1))
        h_all = torch.cat([h_protein, h_ligand], dim=1)
        pos_all = torch.cat([protein_pos, ligand_pos], dim=1)
        for layer in self.layers:
            h_all, pos_all = layer(h_all, pos_all)
        n_ligand = ligand_feat.shape[1]
        h_ligand_final = h_all[:, -n_ligand:]
        pos_ligand_final = pos_all[:, -n_ligand:]
        v_pred = self.v_head(h_ligand_final)
        return v_pred, pos_ligand_final


def build_ipdiff() -> nn.Module:
    """Build a compact IPDiff interaction-prior-guided denoiser.

    Returns
    -------
    nn.Module
        ``InteractionPriorDiffusion`` in eval mode.
    """
    return InteractionPriorDiffusion(
        protein_feat_dim=12, ligand_feat_dim=10, prior_dim=8, hidden_size=32, n_layers=3
    ).eval()


def example_input_ipdiff() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (protein feat/pos/prior, ligand feat/pos/prior)."""
    batch, n_protein, n_ligand = 2, 8, 6
    protein_feat = torch.randn(batch, n_protein, 12)
    protein_pos = torch.randn(batch, n_protein, 3)
    protein_prior = torch.randn(batch, n_protein, 8)
    ligand_feat = torch.randn(batch, n_ligand, 10)
    ligand_pos = torch.randn(batch, n_ligand, 3)
    ligand_prior = torch.randn(batch, n_ligand, 8)
    return protein_feat, protein_pos, protein_prior, ligand_feat, ligand_pos, ligand_prior


# ---------------------------------------------------------------------------
# JANUS -- sigmoid-activated MLP fitness classifier (the sole trained
# nn.Module component of the parallel-tempered SELFIES genetic algorithm).
# ---------------------------------------------------------------------------


class JanusFitnessMLP(nn.Module):
    """All-sigmoid MLP classifier over molecular descriptors.

    Mirrors ``src/janus/network.py::MLP`` exactly: a list of hidden linear
    layers each followed by ``sigmoid``, and a sigmoid-activated output
    layer, trained with ``BCELoss`` to discriminate "high value" vs
    "low value" GA-generated candidate molecules (JANUS's shortlisting
    discriminator).
    """

    def __init__(self, h_sizes: list[int], n_input: int, n_output: int) -> None:
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(n_input, h_sizes[0])])
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        self.predict = nn.Linear(h_sizes[-1], n_output)

    def forward(self, x: Tensor) -> Tensor:
        """Return sigmoid-activated classification score, shape ``(batch, n_output)``."""
        for layer in self.hidden:
            x = torch.sigmoid(layer(x))
        return torch.sigmoid(self.predict(x))


def build_janus() -> nn.Module:
    """Build the JANUS fitness-classifier MLP.

    Returns
    -------
    nn.Module
        ``JanusFitnessMLP`` in eval mode.
    """
    return JanusFitnessMLP(h_sizes=[64, 16], n_input=32, n_output=1).eval()


def example_input_janus() -> Tensor:
    """Return a batch of molecular descriptor feature vectors, shape ``(8, 32)``."""
    return torch.randn(8, 32)


# ---------------------------------------------------------------------------
# KANO -- directed bond-message MPN with a functional-group knowledge-graph
# "prompt" fused into atom hidden states via gated self-attention.
# ---------------------------------------------------------------------------


class KGPromptAttentionLayer(nn.Module):
    """One self-attention block over functional-group states.

    Mirrors ``chemprop/models/model.py::AttentionLayer``: query/key/value
    projections of the functional-group hidden states, scaled dot-product
    attention, and a residual + LayerNorm output block.
    """

    def __init__(self, fg_dim: int = 133, proj_dim: int = 32) -> None:
        super().__init__()
        self.w_q = nn.Linear(fg_dim, proj_dim)
        self.w_k = nn.Linear(fg_dim, proj_dim)
        self.w_v = nn.Linear(fg_dim, proj_dim)
        self.dense = nn.Linear(proj_dim, fg_dim)
        self.norm = nn.LayerNorm(fg_dim, eps=1e-6)

    def forward(self, fg_hiddens: Tensor) -> Tensor:
        """Self-attend over functional-group states, shape ``(batch, n_fg, fg_dim)``."""
        query, key, value = self.w_q(fg_hiddens), self.w_k(fg_hiddens), self.w_v(fg_hiddens)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)
        return self.norm(self.dense(context) + fg_hiddens)


class KGPromptGenerator(nn.Module):
    """Fuses a functional-group knowledge embedding into atom hidden states.

    Mirrors ``chemprop/models/model.py::Prompt_generator``: two stacked
    :class:`KGPromptAttentionLayer` blocks refine the functional-group
    state, a linear head projects it to the atom hidden size, and an
    alpha-gated residual adds it onto the CMPN atom hidden states.
    """

    def __init__(self, fg_dim: int = 133, hidden_size: int = 32) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1,), 0.1))
        self.attention_layer_1 = KGPromptAttentionLayer(fg_dim)
        self.attention_layer_2 = KGPromptAttentionLayer(fg_dim)
        self.linear = nn.Linear(fg_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, atom_hiddens: Tensor, fg_states: Tensor, fg_to_atom: Tensor) -> Tensor:
        """Return atom hidden states with the KG functional-group prompt fused in.

        Parameters
        ----------
        atom_hiddens : Tensor
            CMPN atom hidden states, shape ``(batch, n_atoms, hidden_size)``.
        fg_states : Tensor
            Per-molecule functional-group KG embeddings, shape ``(batch, n_fg, fg_dim)``.
        fg_to_atom : Tensor
            Dense (batch, n_atoms, n_fg) assignment matrix mapping each atom
            to its molecule's pooled functional-group summary (broadcast row
            of ones per atom in the compact/dense reformulation).

        Returns
        -------
        Tensor
            Prompt-fused atom hidden states, same shape as ``atom_hiddens``.
        """
        h = self.attention_layer_1(fg_states)
        h = self.attention_layer_2(h)
        fg_summary = self.linear(h).mean(dim=1, keepdim=True)  # (batch, 1, hidden_size)
        fg_out = self.norm(fg_summary.expand(-1, atom_hiddens.shape[1], -1))
        return atom_hiddens + self.alpha * fg_out


class DMPNNEncoder(nn.Module):
    """Directed bond-message MPN (D-MPNN) with sum-then-max neighbor aggregation.

    Mirrors the core message-passing loop of ``chemprop/models/cmpn.py::
    CMPNEncoder`` (excluding its CUDA-only ragged-index machinery): dense
    neighbor aggregation via sum times max over a fixed-size adjacency mask,
    for a fixed number of depths.
    """

    def __init__(self, atom_feat_dim: int, hidden_size: int, depth: int = 3) -> None:
        super().__init__()
        self.w_i_atom = nn.Linear(atom_feat_dim, hidden_size)
        self.w_h = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth - 1)])
        self.w_o = nn.Linear(hidden_size, hidden_size)
        self.depth = depth

    def forward(self, atom_feat: Tensor, adj: Tensor) -> Tensor:
        """Encode atoms with directed bond-message passing.

        Parameters
        ----------
        atom_feat : Tensor
            Atom input features, shape ``(batch, n_atoms, atom_feat_dim)``.
        adj : Tensor
            Dense bond adjacency mask, shape ``(batch, n_atoms, n_atoms)``.

        Returns
        -------
        Tensor
            Atom hidden states, shape ``(batch, n_atoms, hidden_size)``.
        """
        h = F.relu(self.w_i_atom(atom_feat))
        for w_h in self.w_h:
            nei_sum = torch.bmm(adj, h)
            nei_max = (adj.unsqueeze(-1) * h.unsqueeze(1)).amax(dim=2)
            agg = nei_sum * nei_max
            h = F.relu(h + w_h(agg))
        return self.w_o(h)


class KANOCompact(nn.Module):
    """Compact KANO: D-MPNN atom encoder + knowledge-graph functional-group prompt fusion."""

    def __init__(
        self, atom_feat_dim: int = 16, fg_dim: int = 133, hidden_size: int = 32, depth: int = 3
    ) -> None:
        super().__init__()
        self.encoder = DMPNNEncoder(atom_feat_dim, hidden_size, depth)
        self.prompt_generator = KGPromptGenerator(fg_dim, hidden_size)
        self.readout = nn.Linear(hidden_size, hidden_size)

    def forward(self, atom_feat: Tensor, adj: Tensor, fg_states: Tensor) -> Tensor:
        """Return per-molecule readout embedding, shape ``(batch, hidden_size)``."""
        atom_hiddens = self.encoder(atom_feat, adj)
        dummy_map = torch.ones(
            atom_feat.shape[0], atom_feat.shape[1], fg_states.shape[1], device=atom_feat.device
        )
        fused = self.prompt_generator(atom_hiddens, fg_states, dummy_map)
        return self.readout(fused.mean(dim=1))


def build_kano() -> nn.Module:
    """Build a compact KANO model.

    Returns
    -------
    nn.Module
        ``KANOCompact`` in eval mode.
    """
    return KANOCompact(atom_feat_dim=16, fg_dim=133, hidden_size=32, depth=3).eval()


def example_input_kano() -> tuple[Tensor, Tensor, Tensor]:
    """Return (atom features, dense bond adjacency, functional-group KG states)."""
    batch, n_atoms, n_fg = 2, 10, 5
    atom_feat = torch.randn(batch, n_atoms, 16)
    adj = (torch.rand(batch, n_atoms, n_atoms) > 0.6).float()
    adj = ((adj + adj.transpose(1, 2)) > 0).float()
    fg_states = torch.randn(batch, n_fg, 133)
    return atom_feat, adj, fg_states


# ---------------------------------------------------------------------------
# KV-PLM -- one shared BERT/SciBERT tower encoding both SMILES-as-text and
# biomedical text, with pooled cosine-similarity structure-text matching.
# ---------------------------------------------------------------------------


class SharedBertTower(nn.Module):
    """Compact pre-norm Transformer encoder tower, shared across modalities.

    Mirrors the shared ``BertModel`` in ``modeling.py`` used by both the
    SMILES and biomedical-text branches in ``demo_matching.py``: token +
    position embeddings, a stack of self-attention + feed-forward blocks,
    and CLS-token pooling (``BigModel.forward``'s ``pooler_output`` +
    dropout).
    """

    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_len, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pooler = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.dropout = nn.Dropout(0.1)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return the pooled (CLS) embedding for a token-id sequence, shape ``(batch, hidden_size)``."""
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        h = self.token_embedding(token_ids) + self.position_embedding(positions)
        h = self.encoder(self.layer_norm(h))
        pooled = self.pooler(h[:, 0])
        return self.dropout(pooled)


class KVPLMMatcher(nn.Module):
    """KV-PLM cross-modal matcher: shared tower + cosine-similarity score.

    A single :class:`SharedBertTower` is applied to a SMILES-token sequence
    and a biomedical-text-token sequence (same weights, mirroring KV-PLM's
    single-PLM-for-both-modalities design), and the two pooled embeddings
    are compared with cosine similarity, exactly as ``demo_matching.py``.
    """

    def __init__(
        self, vocab_size: int = 512, hidden_size: int = 64, n_layers: int = 2, n_heads: int = 4
    ) -> None:
        super().__init__()
        self.shared_tower = SharedBertTower(vocab_size, hidden_size, n_layers, n_heads)

    def forward(self, smiles_tokens: Tensor, text_tokens: Tensor) -> Tensor:
        """Return the per-pair cosine-similarity matching score, shape ``(batch,)``."""
        smiles_emb = self.shared_tower(smiles_tokens)
        text_emb = self.shared_tower(text_tokens)
        return F.cosine_similarity(smiles_emb, text_emb, dim=-1)


def build_kvplm() -> nn.Module:
    """Build a compact KV-PLM shared-tower cross-modal matcher.

    Returns
    -------
    nn.Module
        ``KVPLMMatcher`` in eval mode.
    """
    return KVPLMMatcher(vocab_size=512, hidden_size=64, n_layers=2, n_heads=4).eval()


def example_input_kvplm() -> tuple[Tensor, Tensor]:
    """Return (SMILES token-id sequence, biomedical-text token-id sequence)."""
    batch, seq_len = 2, 16
    smiles_tokens = torch.randint(0, 512, (batch, seq_len))
    text_tokens = torch.randint(0, 512, (batch, seq_len))
    return smiles_tokens, text_tokens


# ---------------------------------------------------------------------------

MENAGERIE_ENTRIES = [
    (
        "GROVER-style ContextPred sibling: PretrainGNN",
        "build_pretraingnn_contextpred",
        "example_input_pretraingnn_contextpred",
        "2020",
        "BIO",
    ),
    ("HierVAE", "build_hiervae", "example_input_hiervae", "2020", "BIO"),
    ("IPDiff", "build_ipdiff", "example_input_ipdiff", "2024", "BIO"),
    ("JANUS fitness MLP", "build_janus", "example_input_janus", "2022", "BIO"),
    ("KANO", "build_kano", "example_input_kano", "2023", "BIO"),
    ("KV-PLM", "build_kvplm", "example_input_kvplm", "2022", "BIO"),
]

if __name__ == "__main__":
    print(f"{len(MENAGERIE_ENTRIES)} entries defined; run the smoke-trace gate to verify.")

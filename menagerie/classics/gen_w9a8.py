"""Compact faithful reimplementations for build_queue rows 49-54 (W9A8).

Sources checked (repo/paper browsed via ``gh api`` / web, no clone/pip-install):
  - SAFE-GPT (cand_01272): Noutahi, Gabellini, Craig, Lim, Tossou, "Gotta be
    SAFE: A New Framework for Molecular Design", Digital Discovery 2024,
    arXiv:2310.10773. Official repo github.com/datamol-io/safe,
    ``datamol-io/safe-gpt`` on HuggingFace. SKIPPED as already_in_catalog:
    this is the identical paper/codebase already built in
    ``menagerie/classics/gen_w9a7.py`` as ``build_safe_gpt`` / "SAFE
    encoding" (cand_01271), which that file's own docstring already flags
    as a POTENTIAL_DEDUP with this candidate. Building a second
    near-identical causal-GPT2-over-SAFE-tokens module under a different
    slug would just duplicate an existing catalog entry, not add a new
    distinct architecture.
  - ScaffoldGVAE (cand_01273): Hu, Liu, Chen, Wang, Ding, "ScaffoldGVAE:
    scaffold generation and hopping of drug molecules via a variational
    autoencoder based on multi-view graph neural networks", Journal of
    Cheminformatics 15:91, 2023, arXiv:2309.05618. Official repo
    github.com/ecust-hc/ScaffoldGVAE, ``model.py`` (classes
    ``NMPN``/``EMPN``/``DMPN``/``MultiGRU``). Distinctive mechanism: a
    *dual-view* graph message-passing scaffold encoder -- a node-central
    MPN (``NMPN``: message flows along bonds into each atom, GRU-free ReLU
    update) run in parallel with an edge-central MPN (``EMPN``: message
    flows atom-into-bond, producing per-bond hidden states) over the same
    scaffold graph -- whose two node-level and edge-level views are fused
    with a structured self-attentive pooling (``W_1``/``W_2``/``W_3``,
    i.e. attention scores over the concatenated per-atom [node-view,
    edge-view] hidden states, softmax-weighted sum into a fixed-size
    scaffold embedding) into a VAE latent; the latent is concatenated with
    a separately encoded side-chain/decoration embedding and decoded by a
    3-layer stacked ``GRUCell`` (``MultiGRU``) autoregressive SMILES
    decoder. Reimplemented as ``ScaffoldGVAE`` with dense (adjacency-
    masked) node-central and edge-central message passing over a padded
    scaffold graph, the same attention-pooling fusion, VAE
    reparameterization, and 3-layer GRUCell SMILES-token decoder head, at
    reduced hidden width / vocab / atom count.
  - Scaffold-based graph generator / GGM (cand_01274): Lim, Hwang, Moon,
    Kim, Kim, "Scaffold-based molecular design with a graph generative
    model", Chemical Science 11, 2020, arXiv:1905.13639. Official repo
    github.com/jaechanglim/GGM, ``ggm.py`` (class ``ggm``). Distinctive
    mechanism: rather than generating a molecule from scratch, the model
    *extends a fixed input scaffold graph* atom-by-atom -- an
    edge-conditioned message-passing encoder (``enc_U``/``enc_C``: per-
    edge linear transform of [node, node, edge, condition] followed by a
    ``GRUCell`` node update, 3 rounds) embeds the whole molecule and,
    separately, the scaffold-only subgraph is seeded with the same encoder
    weights (``init_scaffold_U``/``init_scaffold_C``); a VAE latent drawn
    from the whole-molecule embedding is combined with property
    conditions and fed through four decoder MPN stacks that alternately
    (i) propose a new atom type conditioned on the current scaffold-graph
    state (``prop_add_node_*`` -> ``add_node`` MLP head), (ii) propose the
    bond type linking it in (``prop_add_edge_*`` -> ``add_edge`` head),
    (iii) select which existing scaffold atom the new atom attaches to
    (``prop_select_node_*`` -> ``select_node`` head), and (iv) select
    among stereoisomer completions (``select_isomer``) -- guaranteeing the
    generated molecule strictly contains the input scaffold as a
    substructure. Reimplemented as ``ScaffoldGraphGenerator`` with the
    same edge-conditioned-MPN-with-GRUCell-update encoder over a padded
    scaffold graph, property-conditioned VAE latent, and the four-head
    (add-node / add-edge / select-node / select-isomer) sequential
    decoder producing one new-atom-attachment step per forward call, at
    reduced hidden width / atom count / property-condition count.
  - SD-VAE / Syntax-Directed VAE (cand_01275): Dai, Tian, Dai, Skiena,
    Song, "Syntax-Directed Variational Autoencoder for Structured Data",
    ICLR 2018, arXiv:1802.08786. Official repo
    github.com/Hanjun-Dai/sdvae, ``mol_vae/mol_encoder/mol_encoder.py``
    (``CNNEncoder``) and ``mol_vae/mol_decoder/mol_decoder.py``
    (``StateDecoder``). Distinctive mechanism: SMILES are first parsed
    into a context-free-grammar (attribute grammar) derivation and encoded
    as a one-hot sequence of grammar-*production-rule* indices
    (``DECISION_DIM`` channels) rather than characters; a 3-layer 1D-CNN
    (``conv1``: 9-filters/width-9, ``conv2``: 9-filters/width-9,
    ``conv3``: 10-filters/width-11, all VALID/no-padding, matching the
    reference exactly) compresses the one-hot rule sequence to a VAE
    latent (``mean_w``/``log_var_w``); the decoder repeats the latent
    across all timesteps and runs it through a 3-layer ``nn.GRU`` (hidden
    size 501 in the reference, reduced here) to emit, at every step, a
    distribution over the same ``DECISION_DIM`` production rules. The
    "syntax-directed" contribution is an *inference-time* stochastic-
    lazy-attribute masking procedure (grammar-derivation-state-conditioned
    masking of invalid rules) layered on top of this same CNN-encoder /
    GRU-decoder network -- the masking logic is search/constraint machinery
    external to the trainable module, so the traced network here is the
    CNN-encoder-to-GRU-decoder rule-sequence-VAE exactly as the reference
    defines it, at reduced channel widths / rule-vocab size / sequence
    length.
  - SMILES-BERT (cand_01277): Wang, Guo, Wang, Sun, Huang, "SMILES-BERT:
    Large Scale Unsupervised Pre-Training for Molecular Property
    Prediction", ACM-BCB 2019. Official repo
    github.com/uta-smile/SMILES-BERT (fairseq-based BERT masked-language
    model over SMILES character/token sequences; per the paper: standard
    Transformer encoder stack pretrained with a Masked SMILES Recovery
    (masked-language-modeling) objective on unlabeled SMILES, then
    fine-tuned with a shallow regression/classification head for molecular
    property prediction). Distinctive mechanism is the pretrain/fine-tune
    protocol (BERT-style bidirectional self-attention encoder over SMILES
    tokens with a masked-token-recovery pretraining head, later swapped
    for a property-prediction head) rather than a novel layer design.
    Reimplemented as ``SmilesBert`` -- a compact bidirectional Transformer
    encoder (learned token + positional embeddings, pre-LN self-attention
    blocks) over SMILES character tokens with both the masked-language-
    model head (tied output projection back to the SMILES vocabulary) and
    the downstream property-regression head active in the same forward
    pass, matching the reference's two-stage design at reduced width /
    depth / sequence length.
  - SQUID (cand_01278): Adams & Coley, "Equivariant Shape-Conditioned
    Generation of 3D Molecules for Ligand-Based Drug Design", ICLR 2023,
    arXiv:2210.04893. Official repo github.com/keiradams/SQUID,
    ``models/EGNN.py`` / ``models/decoder.py`` / ``models/encoder.py``.
    Distinctive mechanism: molecular shape is encoded via an SE(3)-
    equivariant graph neural network (EGNN-style message passing that
    updates per-node scalar features *and* 3D coordinates together, so
    relative-distance-based messages and coordinate updates are invariant/
    equivariant to global rotation+translation of the input point cloud);
    the resulting shape+fragment latent conditions an autoregressive
    fragment-graph decoder that sequentially attaches molecular fragments
    with fixed bond lengths/angles to fill the target 3D shape. Reproduced
    here as ``SquidShapeEncoder`` with the reference's core coordinate-
    equivariant message-passing update (edge message from invariant
    pairwise-distance + scalar features, node-feature update from
    aggregated messages, coordinate update from a *scalar-weighted* sum of
    relative-position vectors -- the standard EGNN update rule the
    reference's ``EGNN.py`` implements) applied to a random 3D point cloud
    with per-point scalar (pseudo-atom-type) features, at reduced hidden
    width / layer count / point count.

Every model below returns real (untrained, randomly initialized) weights at
small dimensions -- this is an architecture catalog, not a trained-weights
zoo.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# ScaffoldGVAE (cand_01273)
# ---------------------------------------------------------------------------


class _NodeCentralMPN(nn.Module):
    """Node-central message-passing view (``NMPN`` in the reference)."""

    def __init__(self, atom_dim: int, bond_dim: int, hidden: int, depth: int) -> None:
        super().__init__()
        self.depth = depth
        self.w_in = nn.Linear(atom_dim, hidden, bias=False)
        self.w_node = nn.Linear(hidden + bond_dim, hidden, bias=False)

    def forward(self, atom_feat: Tensor, bond_feat: Tensor, adj: Tensor) -> Tensor:
        """Run node-central message passing.

        Parameters
        ----------
        atom_feat : Tensor
            Shape ``(B, N, atom_dim)`` one-hot-ish atom features.
        bond_feat : Tensor
            Shape ``(B, N, N, bond_dim)`` per-edge-slot bond features.
        adj : Tensor
            Shape ``(B, N, N)`` binary adjacency mask.

        Returns
        -------
        Tensor
            Node hidden states, shape ``(B, N, hidden)``.
        """
        h0 = F.relu(self.w_in(atom_feat))
        h = h0
        for _ in range(self.depth):
            # message from neighbor j into i: concat(h_j, bond_ij)
            h_expand = h.unsqueeze(1).expand(-1, h.size(1), -1, -1)
            msg = torch.cat([h_expand, bond_feat], dim=-1)
            msg = self.w_node(msg) * adj.unsqueeze(-1)
            agg = msg.sum(dim=2)
            h = F.relu(h0 + agg)
        return h


class _EdgeCentralMPN(nn.Module):
    """Edge-central message-passing view (``EMPN`` in the reference)."""

    def __init__(self, atom_dim: int, bond_dim: int, hidden: int, out: int, depth: int) -> None:
        super().__init__()
        self.depth = depth
        self.w_in = nn.Linear(bond_dim, hidden, bias=False)
        self.w_edge = nn.Linear(hidden + atom_dim, hidden, bias=False)
        self.w_out = nn.Linear(hidden + atom_dim, out, bias=False)

    def forward(self, atom_feat: Tensor, bond_feat: Tensor, adj: Tensor) -> Tensor:
        """Run edge-central message passing.

        Parameters
        ----------
        atom_feat : Tensor
            Shape ``(B, N, atom_dim)``.
        bond_feat : Tensor
            Shape ``(B, N, N, bond_dim)``.
        adj : Tensor
            Shape ``(B, N, N)`` binary adjacency mask.

        Returns
        -------
        Tensor
            Node-pooled edge-view features, shape ``(B, N, out)``.
        """
        he0 = F.relu(self.w_in(bond_feat))
        he = he0
        for _ in range(self.depth):
            atom_expand = atom_feat.unsqueeze(1).expand(-1, atom_feat.size(1), -1, -1)
            msg = torch.cat([he, atom_expand], dim=-1)
            msg = self.w_edge(msg) * adj.unsqueeze(-1)
            agg = msg.sum(dim=2, keepdim=True).expand_as(he)
            he = F.relu(he0 + agg)
        out_msg = torch.cat(
            [he, atom_feat.unsqueeze(1).expand(-1, atom_feat.size(1), -1, -1)], dim=-1
        )
        out_msg = self.w_out(out_msg) * adj.unsqueeze(-1)
        return F.relu(out_msg.sum(dim=2))


class ScaffoldGVAE(nn.Module):
    """Dual-view (node-central + edge-central) graph-VAE for scaffold hopping.

    Encodes a scaffold graph with two parallel message-passing views, fuses
    them with structured self-attentive pooling into a VAE latent, appends a
    side-chain embedding, and decodes SMILES tokens with a stacked-GRUCell
    autoregressive head (matching ``DMPN``/``MultiGRU`` in the reference).
    """

    def __init__(
        self,
        atom_dim: int = 12,
        bond_dim: int = 6,
        hidden: int = 16,
        depth: int = 2,
        atten_size: int = 8,
        r_heads: int = 4,
        d_hid: int = 16,
        latent_dim: int = 8,
        vocab_size: int = 24,
    ) -> None:
        super().__init__()
        self.nmpn = _NodeCentralMPN(atom_dim, bond_dim, hidden, depth)
        self.empn = _EdgeCentralMPN(atom_dim, bond_dim, hidden, hidden, depth)
        fused_dim = hidden + hidden
        self.w1 = nn.Linear(fused_dim, atten_size, bias=False)
        self.w2 = nn.Linear(atten_size, r_heads, bias=False)
        self.w3 = nn.Linear(r_heads * fused_dim, d_hid)

        self.side_chain_embed = nn.Linear(atom_dim, d_hid)

        self.q_mu = nn.Linear(d_hid, latent_dim)
        self.q_logvar = nn.Linear(d_hid, latent_dim)
        self.decoder_lat = nn.Linear(latent_dim, d_hid)

        gru_hidden = d_hid * 2
        self.embedding = nn.Embedding(vocab_size, 32)
        self.gru_1 = nn.GRUCell(32, gru_hidden)
        self.gru_2 = nn.GRUCell(gru_hidden, gru_hidden)
        self.gru_3 = nn.GRUCell(gru_hidden, gru_hidden)
        self.out_proj = nn.Linear(gru_hidden, vocab_size)
        self.gru_hidden = gru_hidden

    def _pool_scaffold(self, atom_feat: Tensor, bond_feat: Tensor, adj: Tensor) -> Tensor:
        h_node = self.nmpn(atom_feat, bond_feat, adj)
        h_edge = self.empn(atom_feat, bond_feat, adj)
        fused = torch.cat([h_node, h_edge], dim=-1)  # (B, N, fused_dim)
        atten = self.w2(torch.tanh(self.w1(fused)))  # (B, N, r_heads)
        atten = F.softmax(atten, dim=1)
        pooled = torch.einsum("bnr,bnd->brd", atten, fused).reshape(fused.size(0), -1)
        return F.relu(self.w3(pooled))

    def forward(
        self,
        scaffold_atoms: Tensor,
        scaffold_bonds: Tensor,
        scaffold_adj: Tensor,
        side_chain_atoms: Tensor,
        target_tokens: Tensor,
    ) -> Tensor:
        """Encode a scaffold + side-chain and decode SMILES token logits.

        Parameters
        ----------
        scaffold_atoms : Tensor
            Shape ``(B, N, atom_dim)``.
        scaffold_bonds : Tensor
            Shape ``(B, N, N, bond_dim)``.
        scaffold_adj : Tensor
            Shape ``(B, N, N)``.
        side_chain_atoms : Tensor
            Shape ``(B, atom_dim)`` pooled side-chain descriptor.
        target_tokens : Tensor
            Shape ``(B, T)`` long SMILES token ids to teacher-force decode.

        Returns
        -------
        Tensor
            Per-step vocabulary logits, shape ``(B, T, vocab_size)``.
        """
        sca_h = self._pool_scaffold(scaffold_atoms, scaffold_bonds, scaffold_adj)
        mu, logvar = self.q_mu(sca_h), self.q_logvar(sca_h)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps

        side_h = self.side_chain_embed(side_chain_atoms)
        h0 = torch.cat([self.decoder_lat(z), side_h], dim=-1)

        h1 = h2 = h3 = h0
        logits = []
        for t in range(target_tokens.size(1)):
            x = self.embedding(target_tokens[:, t])
            h1 = self.gru_1(x, h1)
            h2 = self.gru_2(h1, h2)
            h3 = self.gru_3(h2, h3)
            logits.append(self.out_proj(h3))
        return torch.stack(logits, dim=1)


def build_scaffoldgvae() -> nn.Module:
    """Build a compact ScaffoldGVAE dual-view scaffold-hopping VAE.

    Returns
    -------
    nn.Module
        ``ScaffoldGVAE`` in eval mode.
    """
    model = ScaffoldGVAE()
    model.eval()
    return model


def example_input_scaffoldgvae() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_scaffoldgvae`.

    Returns
    -------
    tuple of Tensor
        ``(scaffold_atoms, scaffold_bonds, scaffold_adj, side_chain_atoms,
        target_tokens)``.
    """
    torch.manual_seed(0)
    batch, n_atoms, atom_dim, bond_dim, seq_len, vocab = 2, 9, 12, 6, 10, 24
    atoms = torch.rand(batch, n_atoms, atom_dim)
    bonds = torch.rand(batch, n_atoms, n_atoms, bond_dim)
    adj = torch.zeros(batch, n_atoms, n_atoms)
    for i in range(n_atoms - 1):
        adj[:, i, i + 1] = 1.0
        adj[:, i + 1, i] = 1.0
    side_chain = torch.rand(batch, atom_dim)
    tokens = torch.randint(0, vocab, (batch, seq_len))
    return atoms, bonds, adj, side_chain, tokens


# ---------------------------------------------------------------------------
# Scaffold-based graph generator / GGM (cand_01274)
# ---------------------------------------------------------------------------


class ScaffoldGraphGenerator(nn.Module):
    """Scaffold-conditioned atom-by-atom graph-extension VAE (GGM).

    Encodes a whole-molecule graph plus separately-seeded scaffold subgraph
    with an edge-conditioned message-passing / ``GRUCell`` node-update
    encoder, draws a property-conditioned VAE latent, and predicts a single
    scaffold-extension step (new atom type, new bond type, attachment atom,
    isomer choice) matching the reference's four decoder MPN heads.
    """

    def __init__(
        self,
        node_dim: int = 16,
        edge_dim: int = 8,
        n_conditions: int = 2,
        n_atom_types: int = 10,
        n_bond_types: int = 5,
        fc_dim: int = 32,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.n_conditions = n_conditions
        graph_dim = node_dim * 2

        def mpn_stack(steps: int, extra_in: int) -> tuple[nn.ModuleList, nn.ModuleList]:
            u = nn.ModuleList(
                [nn.Linear(extra_in + edge_dim + n_conditions, node_dim) for _ in range(steps)]
            )
            c = nn.ModuleList([nn.GRUCell(node_dim, node_dim) for _ in range(steps)])
            return u, c

        # whole-molecule encoder MPN (`enc_U`/`enc_C`)
        self.enc_u, self.enc_c = mpn_stack(depth, 2 * node_dim)
        # scaffold-seed MPN sharing the same shape (`init_scaffold_U`/`_C`)
        self.init_u, self.init_c = mpn_stack(depth, 2 * node_dim)
        # decoder MPN heads, each 2 rounds of message passing
        self.add_node_u, self.add_node_c = mpn_stack(2, 3 * node_dim)
        self.add_edge_u, self.add_edge_c = mpn_stack(2, 3 * node_dim)
        self.select_node_u, self.select_node_c = mpn_stack(2, 3 * node_dim)
        self.select_isomer_u, self.select_isomer_c = mpn_stack(2, 3 * node_dim)

        self.node_embedding = nn.Linear(n_atom_types, node_dim, bias=False)
        self.edge_embedding = nn.Linear(n_bond_types, edge_dim, bias=False)

        self.mean = nn.Linear(node_dim, node_dim)
        self.logvar = nn.Linear(node_dim, node_dim)

        self.add_node1 = nn.Linear(graph_dim + node_dim + n_conditions, fc_dim)
        self.add_node2 = nn.Linear(fc_dim, fc_dim)
        self.add_node3 = nn.Linear(fc_dim, n_atom_types)

        self.add_edge1 = nn.Linear(graph_dim + node_dim + n_conditions, fc_dim)
        self.add_edge2 = nn.Linear(fc_dim, fc_dim)
        self.add_edge3 = nn.Linear(fc_dim, n_bond_types)

        self.select_node1 = nn.Linear(node_dim * 2 + n_conditions, fc_dim)
        self.select_node2 = nn.Linear(fc_dim, fc_dim)
        self.select_node3 = nn.Linear(fc_dim, 1)

        self.select_isomer1 = nn.Linear(node_dim + n_conditions, fc_dim)
        self.select_isomer2 = nn.Linear(fc_dim, fc_dim)
        self.select_isomer3 = nn.Linear(fc_dim, 1)

        self.graph_vec1 = nn.Linear(node_dim, graph_dim)

    @staticmethod
    def _run_mpn(
        node_h: Tensor,
        edge_h: Tensor,
        adj: Tensor,
        cond_bcast: Tensor,
        u_layers: nn.ModuleList,
        c_layers: nn.ModuleList,
        extra: Tensor | None = None,
    ) -> Tensor:
        """Edge-conditioned message passing with GRUCell node update."""
        h = node_h
        n = h.size(1)
        for u_lin, c_cell in zip(u_layers, c_layers):
            h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
            h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
            parts = [h_i, h_j, edge_h]
            if extra is not None:
                extra_b = extra.unsqueeze(1).unsqueeze(1).expand(-1, n, n, -1)
                parts.insert(0, extra_b)
            parts.append(cond_bcast.unsqueeze(1).unsqueeze(1).expand(-1, n, n, -1))
            msg = torch.cat(parts, dim=-1)
            msg = u_lin(msg) * adj.unsqueeze(-1)
            agg = msg.sum(dim=2)
            h_flat = h.reshape(-1, h.size(-1))
            agg_flat = agg.reshape(-1, agg.size(-1))
            h = c_cell(agg_flat, h_flat).reshape(h.shape)
        return h

    def forward(
        self,
        atom_types: Tensor,
        bond_types: Tensor,
        adj: Tensor,
        scaffold_mask: Tensor,
        conditions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode a molecule/scaffold graph and predict one extension step.

        Parameters
        ----------
        atom_types : Tensor
            Shape ``(B, N, n_atom_types)`` one-hot atom features.
        bond_types : Tensor
            Shape ``(B, N, N, n_bond_types)`` one-hot bond features.
        adj : Tensor
            Shape ``(B, N, N)`` binary adjacency mask.
        scaffold_mask : Tensor
            Shape ``(B, N)`` binary mask of which atoms belong to the fixed
            scaffold (``1``) vs. free decoration region (``0``).
        conditions : Tensor
            Shape ``(B, n_conditions)`` target property values.

        Returns
        -------
        tuple of Tensor
            ``(add_node_logits, add_edge_logits, select_node_score,
            select_isomer_score)``.
        """
        node_h0 = self.node_embedding(atom_types)
        edge_h = self.edge_embedding(bond_types)

        # whole-molecule encoding -> VAE latent
        enc_h = self._run_mpn(node_h0, edge_h, adj, conditions, self.enc_u, self.enc_c)
        pooled = enc_h.mean(dim=1)
        mu, logvar = self.mean(pooled), self.logvar(pooled)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

        # scaffold-seed encoding, sharing shape with the encoder
        sca_node_h0 = node_h0 * scaffold_mask.unsqueeze(-1)
        sca_h = self._run_mpn(sca_node_h0, edge_h, adj, conditions, self.init_u, self.init_c)

        graph_vec = F.relu(self.graph_vec1(z)).unsqueeze(1).expand(-1, sca_h.size(1), -1)
        cond_b = conditions.unsqueeze(1).expand(-1, sca_h.size(1), -1)

        node_query = torch.cat([graph_vec, sca_h, cond_b], dim=-1)
        add_node = F.relu(self.add_node1(node_query))
        add_node = F.relu(self.add_node2(add_node))
        add_node_logits = self.add_node3(add_node).mean(dim=1)

        add_edge = F.relu(self.add_edge1(node_query))
        add_edge = F.relu(self.add_edge2(add_edge))
        add_edge_logits = self.add_edge3(add_edge).mean(dim=1)

        select_query = torch.cat([sca_h, sca_h, cond_b], dim=-1)
        select_node = F.relu(self.select_node1(select_query))
        select_node = F.relu(self.select_node2(select_node))
        select_node_score = self.select_node3(select_node).squeeze(-1)

        isomer_query = torch.cat([sca_h, cond_b], dim=-1)
        select_isomer = F.relu(self.select_isomer1(isomer_query))
        select_isomer = F.relu(self.select_isomer2(select_isomer))
        select_isomer_score = self.select_isomer3(select_isomer).squeeze(-1)

        return add_node_logits, add_edge_logits, select_node_score, select_isomer_score


def build_scaffold_graph_generator() -> nn.Module:
    """Build a compact scaffold-conditioned atom-by-atom graph generator (GGM).

    Returns
    -------
    nn.Module
        ``ScaffoldGraphGenerator`` in eval mode.
    """
    model = ScaffoldGraphGenerator()
    model.eval()
    return model


def example_input_scaffold_graph_generator() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Create example input for :func:`build_scaffold_graph_generator`.

    Returns
    -------
    tuple of Tensor
        ``(atom_types, bond_types, adj, scaffold_mask, conditions)``.
    """
    torch.manual_seed(0)
    batch, n_atoms, n_atom_types, n_bond_types, n_conditions = 2, 8, 10, 5, 2
    atom_types = F.one_hot(torch.randint(0, n_atom_types, (batch, n_atoms)), n_atom_types).float()
    bond_types = F.one_hot(
        torch.randint(0, n_bond_types, (batch, n_atoms, n_atoms)), n_bond_types
    ).float()
    adj = torch.zeros(batch, n_atoms, n_atoms)
    for i in range(n_atoms - 1):
        adj[:, i, i + 1] = 1.0
        adj[:, i + 1, i] = 1.0
    scaffold_mask = torch.zeros(batch, n_atoms)
    scaffold_mask[:, : n_atoms // 2] = 1.0
    conditions = torch.rand(batch, n_conditions)
    return atom_types, bond_types, adj, scaffold_mask, conditions


# ---------------------------------------------------------------------------
# SD-VAE / Syntax-Directed VAE (cand_01275)
# ---------------------------------------------------------------------------


class SdVaeEncoder(nn.Module):
    """3-layer 1D-CNN encoder over one-hot grammar-production-rule sequences.

    Matches the reference ``CNNEncoder``'s exact filter/kernel shape:
    conv(9,9) -> conv(9,9) -> conv(10,11), all VALID (no padding).
    """

    def __init__(self, decision_dim: int, max_len: int, latent_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(decision_dim, 9, 9)
        self.conv2 = nn.Conv1d(9, 9, 9)
        self.conv3 = nn.Conv1d(9, 10, 11)
        last_conv_size = max_len - 9 + 1 - 9 + 1 - 11 + 1
        self.w1 = nn.Linear(last_conv_size * 10, 64)
        self.mean_w = nn.Linear(64, latent_dim)
        self.log_var_w = nn.Linear(64, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a batch of one-hot rule sequences to VAE latent params.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, decision_dim, max_len)`` one-hot production-rule
            sequence.

        Returns
        -------
        tuple of Tensor
            ``(z_mean, z_log_var)``, each shape ``(B, latent_dim)``.
        """
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        flat = h.reshape(x.size(0), -1)
        h = F.relu(self.w1(flat))
        return self.mean_w(h), self.log_var_w(h)


class SdVaeDecoder(nn.Module):
    """3-layer GRU state decoder emitting per-step production-rule logits."""

    def __init__(self, decision_dim: int, max_len: int, latent_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.max_len = max_len
        self.z_to_latent = nn.Linear(latent_dim, latent_dim)
        self.gru = nn.GRU(latent_dim, hidden, 3)
        self.decoded_logits = nn.Linear(hidden, decision_dim)

    def forward(self, z: Tensor) -> Tensor:
        """Decode a latent vector to per-step grammar-rule logits.

        Parameters
        ----------
        z : Tensor
            Shape ``(B, latent_dim)``.

        Returns
        -------
        Tensor
            Shape ``(max_len, B, decision_dim)`` per-step rule logits.
        """
        h = F.relu(self.z_to_latent(z))
        rep_h = h.unsqueeze(0).expand(self.max_len, -1, -1)
        out, _ = self.gru(rep_h)
        return self.decoded_logits(out)


class SyntaxDirectedVAE(nn.Module):
    """Grammar-rule-sequence VAE for molecules (SD-VAE).

    A context-free-grammar derivation of a SMILES string is one-hot encoded
    as a sequence of production-rule indices; a CNN encoder compresses it to
    a VAE latent and a GRU decoder reconstructs per-step rule logits. The
    "syntax-directed" masking of invalid rules at generation time is
    inference-time constraint machinery layered on top of this same network
    and is not itself a trainable component.
    """

    def __init__(self, decision_dim: int = 24, max_len: int = 40, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = SdVaeEncoder(decision_dim, max_len, latent_dim)
        self.decoder = SdVaeDecoder(decision_dim, max_len, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Reconstruct a one-hot rule sequence through the VAE bottleneck.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, decision_dim, max_len)``.

        Returns
        -------
        tuple of Tensor
            ``(logits, mean, log_var)``.
        """
        mean, log_var = self.encoder(x)
        z = mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)
        logits = self.decoder(z)
        return logits, mean, log_var


def build_sdvae() -> nn.Module:
    """Build a compact Syntax-Directed VAE over grammar-rule sequences.

    Returns
    -------
    nn.Module
        ``SyntaxDirectedVAE`` in eval mode.
    """
    model = SyntaxDirectedVAE()
    model.eval()
    return model


def example_input_sdvae() -> Tensor:
    """Create example input for :func:`build_sdvae`.

    Returns
    -------
    Tensor
        One-hot grammar-rule sequence, shape ``(2, 24, 40)``.
    """
    torch.manual_seed(0)
    batch, decision_dim, max_len = 2, 24, 40
    idx = torch.randint(0, decision_dim, (batch, max_len))
    return F.one_hot(idx, decision_dim).permute(0, 2, 1).float()


# ---------------------------------------------------------------------------
# SMILES-BERT (cand_01277)
# ---------------------------------------------------------------------------


class _PreLNTransformerBlock(nn.Module):
    """Pre-LayerNorm bidirectional self-attention encoder block."""

    def __init__(self, dim: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class SmilesBert(nn.Module):
    """BERT-style bidirectional Transformer pretrained via masked-SMILES-recovery.

    Learned token + positional embeddings feed a stack of pre-LN
    bidirectional self-attention blocks; both the masked-language-modeling
    head (tied to the token embedding) and a downstream property-prediction
    head run in the same forward pass, matching SMILES-BERT's two-stage
    pretrain-then-finetune design.
    """

    def __init__(
        self,
        vocab_size: int = 48,
        max_len: int = 64,
        dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 3,
        ff_dim: int = 64,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(
            [_PreLNTransformerBlock(dim, n_heads, ff_dim) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(dim)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        self.property_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, token_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Run the masked-SMILES-recovery + property-prediction forward pass.

        Parameters
        ----------
        token_ids : Tensor
            Shape ``(B, L)`` long SMILES token ids (some masked upstream).

        Returns
        -------
        tuple of Tensor
            ``(mlm_logits, property_pred)`` of shapes ``(B, L, vocab_size)``
            and ``(B, 1)``.
        """
        b, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(b, -1)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        mlm_logits = F.linear(x, self.token_embed.weight, self.mlm_bias)
        pooled = x[:, 0]
        property_pred = self.property_head(pooled)
        return mlm_logits, property_pred


def build_smiles_bert() -> nn.Module:
    """Build a compact SMILES-BERT masked-language-model + property predictor.

    Returns
    -------
    nn.Module
        ``SmilesBert`` in eval mode.
    """
    model = SmilesBert()
    model.eval()
    return model


def example_input_smiles_bert() -> Tensor:
    """Create example input for :func:`build_smiles_bert`.

    Returns
    -------
    Tensor
        SMILES token ids, shape ``(2, 24)``.
    """
    torch.manual_seed(0)
    return torch.randint(0, 48, (2, 24))


# ---------------------------------------------------------------------------
# SQUID (cand_01278)
# ---------------------------------------------------------------------------


class _EGNNLayer(nn.Module):
    """SE(3)-equivariant node/coordinate update layer (EGNN-style).

    Message from an edge is built from invariant relative-distance +
    endpoint scalar features; node features update from aggregated
    messages, and 3D coordinates update from a scalar-weighted sum of
    relative-position vectors -- the standard equivariant update rule the
    reference's ``EGNN.py`` implements.
    """

    def __init__(self, feat_dim: int, hidden: int) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * feat_dim + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU()
        )
        self.coord_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.node_mlp = nn.Sequential(
            nn.Linear(feat_dim + hidden, hidden), nn.SiLU(), nn.Linear(hidden, feat_dim)
        )

    def forward(self, feat: Tensor, coord: Tensor) -> tuple[Tensor, Tensor]:
        """Apply one equivariant message-passing update.

        Parameters
        ----------
        feat : Tensor
            Shape ``(B, N, feat_dim)`` per-point scalar features.
        coord : Tensor
            Shape ``(B, N, 3)`` per-point 3D coordinates.

        Returns
        -------
        tuple of Tensor
            Updated ``(feat, coord)``.
        """
        n = feat.size(1)
        rel = coord.unsqueeze(2) - coord.unsqueeze(1)  # (B, N, N, 3)
        dist2 = (rel**2).sum(dim=-1, keepdim=True)  # invariant to rotation/translation
        f_i = feat.unsqueeze(2).expand(-1, -1, n, -1)
        f_j = feat.unsqueeze(1).expand(-1, n, -1, -1)
        edge_in = torch.cat([f_i, f_j, dist2], dim=-1)
        m_ij = self.edge_mlp(edge_in)  # (B, N, N, hidden) -- invariant messages

        coord_weight = self.coord_mlp(m_ij)  # (B, N, N, 1) scalar per edge
        coord_update = (rel * coord_weight).mean(dim=2)  # equivariant coordinate update
        new_coord = coord + coord_update

        agg = m_ij.mean(dim=2)  # invariant aggregated message
        new_feat = feat + self.node_mlp(torch.cat([feat, agg], dim=-1))
        return new_feat, new_coord


class SquidShapeEncoder(nn.Module):
    """Equivariant shape encoder for shape-conditioned 3D molecule generation.

    Stacks ``_EGNNLayer`` coordinate-equivariant message passing over a
    randomly initialized 3D point cloud with per-point scalar (pseudo-
    atom-type) features, then pools to a rotation/translation-invariant
    shape embedding -- the encoder half of SQUID's shape-conditioned
    fragment-attachment generator.
    """

    def __init__(
        self, feat_dim: int = 8, hidden: int = 16, n_layers: int = 3, embed_dim: int = 32
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, hidden)
        self.layers = nn.ModuleList([_EGNNLayer(hidden, hidden) for _ in range(n_layers)])
        self.readout = nn.Sequential(nn.Linear(hidden, embed_dim), nn.SiLU())

    def forward(self, feat: Tensor, coord: Tensor) -> Tensor:
        """Encode a 3D point cloud into a shape embedding.

        Parameters
        ----------
        feat : Tensor
            Shape ``(B, N, feat_dim)`` per-point scalar features.
        coord : Tensor
            Shape ``(B, N, 3)`` per-point 3D coordinates.

        Returns
        -------
        Tensor
            Shape embedding, shape ``(B, embed_dim)``.
        """
        h = self.in_proj(feat)
        c = coord
        for layer in self.layers:
            h, c = layer(h, c)
        return self.readout(h.mean(dim=1))


def build_squid() -> nn.Module:
    """Build a compact SQUID SE(3)-equivariant shape encoder.

    Returns
    -------
    nn.Module
        ``SquidShapeEncoder`` in eval mode.
    """
    model = SquidShapeEncoder()
    model.eval()
    return model


def example_input_squid() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_squid`.

    Returns
    -------
    tuple of Tensor
        ``(feat, coord)`` of shapes ``(2, 14, 8)`` and ``(2, 14, 3)``.
    """
    torch.manual_seed(0)
    batch, n_points, feat_dim = 2, 14, 8
    feat = torch.rand(batch, n_points, feat_dim)
    coord = torch.randn(batch, n_points, 3)
    return feat, coord


# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------

MENAGERIE_ENTRIES = [
    ("ScaffoldGVAE", "build_scaffoldgvae", "example_input_scaffoldgvae", "2023", "BIO"),
    (
        "Scafold-based graph generator (Lim et al.)",
        "build_scaffold_graph_generator",
        "example_input_scaffold_graph_generator",
        "2020",
        "BIO",
    ),
    ("SD-VAE (Syntax-Directed VAE)", "build_sdvae", "example_input_sdvae", "2018", "BIO"),
    ("SMILES-BERT", "build_smiles_bert", "example_input_smiles_bert", "2019", "BIO"),
    ("SQUID", "build_squid", "example_input_squid", "2023", "BIO"),
]

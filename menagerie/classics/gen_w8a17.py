"""Molecular/materials generative-modeling architecture family: gen_w8a17.

Sources checked (repo_url / desc_source from the build queue, plus web search):
  - DiffSBDD is SKIPPED: https://github.com/arneschneuing/DiffSBDD (Schneuing
    et al., Nature Computational Science 2024) -- this exact SE(3)-equivariant
    DDPM conditioned on a protein pocket (with the joint pocket+ligand
    inpainting mode described in the candidate notes) is already present as
    ``build_diffsbdd`` / ``example_input_diffsbdd`` in
    ``menagerie/classics/gen_w5a5.py``. Same repo, same paper, same mechanism
    -- a genuine duplicate, not a variant.
  - DiGress: https://github.com/cvignac/DiGress; Vignac et al., ICLR 2023
    (arXiv:2209.14734). Confirmed from the paper/``models/transformer_model.py``
    in the official repo: a **discrete** denoising diffusion model over graphs
    with categorical node and edge types. Forward noising uses a Markovian
    transition-matrix process on the discrete node/edge categories (rather than
    Gaussian noise on continuous coordinates) that is built to preserve the
    marginal type distribution at the noise limit. The denoiser is a
    **graph transformer** (`XEyTransformerLayer` in the official code): a stack
    of blocks that jointly update node features X, edge features E, and a
    pooled graph-level feature y via FiLM-style feature-wise modulation between
    the three streams, with edge features gating a masked multi-head
    self-attention over nodes (`Y = QK^T ⊙ (edge-derived gate)`), plus a
    sinusoidal embedding of the diffusion timestep folded into y. Reimplemented
    here as a compact `XEyTransformerLayer`-style stack over a small padded
    graph (dense node type logits X, dense edge type logits E, pooled y),
    predicting denoised categorical logits for X and E from a noised
    categorical graph plus a diffusion timestep.
  - DoG-Gen: https://github.com/john-bradshaw/synthesis-dags; Bradshaw et al.,
    NeurIPS 2020 (arXiv:2012.11522), "Barking up the right tree: an approach to
    search over molecule synthesis DAGs". Confirmed from the paper's decoder
    description (`dogae/model/get_dags.py` / `mol_dag_model.py` structure in
    the official repo): synthesis routes are represented as DAGs-of-graphs
    (DoGs) where each node is a starting reactant/intermediate molecule
    (encoded once by a GNN into a fixed-size embedding) and edges represent
    reaction-combination steps. The decoder is an **autoregressive RNN**
    (GRU) that, at each step, conditions on a running hidden state (initialized
    either from a latent code -- the DoG-VAE variant -- or from a zero/learned
    constant -- the **DoG-Gen** "zero-latent" variant used for RL fine-tuning,
    per the candidate notes) and predicts, in sequence: (i) which action to
    take (add a new reactant leaf / react two existing DAG nodes / stop),
    (ii) which existing reactant embedding(s) or leaf identity to select via an
    attention/pointer readout over the pool of previously-embedded DAG nodes,
    and (iii) which of a small set of reaction templates joins them. Each
    accepted step feeds a GNN embedding of the newly formed intermediate back
    into the running node-embedding pool for future steps. Reimplemented here
    as a compact GNN node-embedder + single-layer GRU decoder with a pointer
    (dot-product attention) head over the growing node-embedding pool plus a
    small action-type and reaction-template classification head, run for a
    fixed number of autoregressive steps (DoG-Gen's zero-latent variant: no
    VAE encoder, hidden state initialized from a learned constant).
  - DRlinker: https://github.com/DaiDaiD/DRlinker; Tan et al., J. Chem. Inf.
    Model. 2022 (PMID 36404642), "DRlinker: Deep Reinforcement Learning for
    Optimization in Fragment Linking Design". Confirmed from the paper and the
    official repo's use of OpenNMT-py 0.4.1: fragment linking is posed as
    sentence completion by a **Transformer encoder-decoder**, where the input
    sequence is the SMILES of two fragments (each carrying an attachment-point
    "*" token) concatenated with a separator, and the output sequence is the
    SMILES of the full linked molecule, generated autoregressively with
    causal self-attention plus encoder-decoder cross-attention. The Prior
    model is pretrained by supervised seq2seq on ChEMBL fragment/molecule
    pairs, then fine-tuned by policy-gradient RL (REINFORCE-style, with the
    Transformer as the stochastic policy over next-token SMILES) against
    property-scoring rewards (linker length, logP, docking score, etc.) --
    the RL loop itself is a training-time procedure, not part of the
    persistent module, so the module built here is the Transformer seq2seq
    policy/prior network. Reimplemented here as a compact standard
    Transformer encoder-decoder over a small SMILES-token vocabulary,
    matching the official architecture (OpenNMT-py's default Transformer:
    sinusoidal position encoding, multi-head self/cross attention, position-
    wise feed-forward).
  - DrugBAN: https://github.com/peizhenbai/DrugBAN; Bai et al., Nature Machine
    Intelligence 2023 (arXiv:2208.02194), "Interpretable bilinear attention
    network with domain adaptation improves drug-target prediction". Confirmed
    from ``models.py`` in the official repo: a drug **graph convolutional
    network** (GCN, `MolecularGCN`) encodes the 2D drug molecular graph into a
    per-atom feature sequence, a protein **1D CNN** (`ProteinCNN`, three
    stacked `Conv1d` + batchnorm blocks) encodes the amino-acid sequence into a
    per-residue feature sequence, and a **bilinear attention network** (BAN,
    `BANLayer`, following Kim et al. 2018's low-rank bilinear pooling with
    learned attention maps) computes pairwise drug-atom x protein-residue
    interaction weights and pools them into a fixed-size interaction
    representation, which an MLP head maps to a binding-affinity/interaction
    logit. An auxiliary conditional domain-adversarial discriminator (a
    gradient-reversal MLP over the interaction representation, per `da.py`)
    supervises domain adaptation during training but is not needed for a
    forward pass in inference mode. Reimplemented here as a compact GCN drug
    encoder + CNN protein encoder + bilinear attention pooling + MLP
    interaction head (the domain-adversarial branch is training-time-only
    and out of scope for a forward-pass module, consistent with how the
    official repo's `predict()` path also skips it).
  - Dual Transformer retrosynthesis ("Tied Two-Way Transformers"):
    https://github.com/MolecularAI/Chemformer (candidate's repo_url points
    here, but the actual originating work is Kim et al., J. Chem. Inf. Model.
    2021, "Valid, Plausible, and Diverse Retrosynthesis Using Tied Two-Way
    Transformers with Latent Variables", official code
    https://github.com/ejklike/tied-twoway-transformer). This is a genuinely
    distinct architecture from the single BART-style Chemformer already in
    ``gen_w7a13.py``: it couples **two** Transformers with **tied
    (parameter-shared) weights** -- a backward (product -> reactants,
    retrosynthesis) Transformer and a forward (reactants -> product, reaction
    prediction) Transformer that share their encoder/decoder parameters -- so
    that a round-trip through backward-then-forward gives a differentiable
    cycle-consistency signal, plus a small set of learned discrete latent
    variables (fed as an extra conditioning token into the backward decoder)
    that let one product map to multiple plausible reactant sets. Reimplemented
    here as two weight-tied Transformer encoder-decoders (backward and
    forward) over a shared small SMILES-token vocabulary, with a learned
    latent-variable embedding table conditioning the backward decoder and an
    explicit round-trip (product -> reactants -> reconstructed product) forward
    pass exercising both directions and the tied weights, matching the tied
    two-way + latent-variable design of the paper.

Every module below is a small, randomly initialized ``nn.Module`` (``.eval()``
mode) built to be TorchLens-traceable: no data-dependent Python control flow
on tensor values, only fixed-size loops over small constants.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# 1. DiGress -- discrete denoising diffusion graph transformer
# ---------------------------------------------------------------------------


class _XEyLayer(nn.Module):
    """One DiGress-style joint X/E/y update block (graph transformer layer)."""

    def __init__(self, node_dim: int, edge_dim: int, y_dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = node_dim // n_heads
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.edge_gate = nn.Linear(edge_dim, n_heads)
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.node_norm1 = nn.LayerNorm(node_dim)
        self.node_ff = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2), nn.ReLU(), nn.Linear(node_dim * 2, node_dim)
        )
        self.node_norm2 = nn.LayerNorm(node_dim)

        self.edge_from_nodes = nn.Linear(node_dim * 2, edge_dim)
        self.edge_film = nn.Linear(y_dim, edge_dim * 2)
        self.edge_norm = nn.LayerNorm(edge_dim)

        self.y_from_nodes = nn.Linear(node_dim, y_dim)
        self.y_from_edges = nn.Linear(edge_dim, y_dim)
        self.y_ff = nn.Sequential(
            nn.Linear(y_dim, y_dim * 2), nn.ReLU(), nn.Linear(y_dim * 2, y_dim)
        )
        self.y_norm = nn.LayerNorm(y_dim)

    def forward(self, x: Tensor, e: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Jointly update node features ``x``, edge features ``e``, graph feature ``y``."""
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, n, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(b, n, self.n_heads, self.head_dim)
        attn_logits = torch.einsum("bihd,bjhd->bijh", q, k) / math.sqrt(self.head_dim)
        edge_bias = self.edge_gate(e)  # (b, n, n, heads) -- edge-conditioned attention gate
        attn = F.softmax(attn_logits + edge_bias, dim=2)
        out = torch.einsum("bijh,bjhd->bihd", attn, v).reshape(b, n, -1)
        x = self.node_norm1(x + self.out_proj(out))
        x = self.node_norm2(x + self.node_ff(x))

        pair = torch.cat(
            [x.unsqueeze(2).expand(-1, -1, n, -1), x.unsqueeze(1).expand(-1, n, -1, -1)], dim=-1
        )
        e_update = self.edge_from_nodes(pair)
        film = self.edge_film(y).unsqueeze(1).unsqueeze(1)
        scale, shift = film.chunk(2, dim=-1)
        e = self.edge_norm(e + e_update * (1.0 + scale) + shift)

        y_update = self.y_from_nodes(x.mean(dim=1)) + self.y_from_edges(e.mean(dim=(1, 2)))
        y = self.y_norm(y + y_update)
        y = y + self.y_ff(y)
        return x, e, y


class DiGress(nn.Module):
    """Compact DiGress: discrete graph-diffusion denoiser over node/edge types."""

    def __init__(
        self,
        n_node_types: int,
        n_edge_types: int,
        node_dim: int = 32,
        edge_dim: int = 16,
        y_dim: int = 16,
        n_layers: int = 3,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.node_in = nn.Linear(n_node_types, node_dim)
        self.edge_in = nn.Linear(n_edge_types, edge_dim)
        self.time_embed = nn.Sequential(nn.Linear(1, y_dim), nn.ReLU(), nn.Linear(y_dim, y_dim))
        self.layers = nn.ModuleList(
            [_XEyLayer(node_dim, edge_dim, y_dim, n_heads) for _ in range(n_layers)]
        )
        self.node_out = nn.Linear(node_dim, n_node_types)
        self.edge_out = nn.Linear(edge_dim, n_edge_types)

    def forward(self, x_noisy: Tensor, e_noisy: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor]:
        """Denoise categorical node/edge logits.

        Parameters
        ----------
        x_noisy : Tensor
            ``(batch, n_nodes, n_node_types)`` one-hot/soft noised node types.
        e_noisy : Tensor
            ``(batch, n_nodes, n_nodes, n_edge_types)`` noised edge types.
        timestep : Tensor
            ``(batch, 1)`` normalized diffusion timestep in ``[0, 1]``.

        Returns
        -------
        tuple of Tensor
            Denoised ``(node_logits, edge_logits)``.
        """
        x = self.node_in(x_noisy)
        e = self.edge_in(e_noisy)
        y = self.time_embed(timestep)
        for layer in self.layers:
            x, e, y = layer(x, e, y)
        return self.node_out(x), self.edge_out(e)


def build_digress() -> nn.Module:
    """Build a compact DiGress discrete graph-diffusion denoiser.

    Returns
    -------
    nn.Module
        ``DiGress`` in eval mode.
    """
    return DiGress(
        n_node_types=8, n_edge_types=4, node_dim=32, edge_dim=16, y_dim=16, n_layers=3, n_heads=4
    ).eval()


def example_input_digress() -> tuple[Tensor, Tensor, Tensor]:
    """Build a small noised-graph batch for ``DiGress``.

    Returns
    -------
    tuple of Tensor
        ``(x_noisy, e_noisy, timestep)``.
    """
    batch, n_nodes, n_node_types, n_edge_types = 2, 7, 8, 4
    x_noisy = F.softmax(torch.randn(batch, n_nodes, n_node_types), dim=-1)
    e_noisy = F.softmax(torch.randn(batch, n_nodes, n_nodes, n_edge_types), dim=-1)
    timestep = torch.rand(batch, 1)
    return x_noisy, e_noisy, timestep


# ---------------------------------------------------------------------------
# 2. DoG-Gen -- autoregressive RNN decoder over synthesis DAGs (zero-latent)
# ---------------------------------------------------------------------------


class _SmallGNNEncoder(nn.Module):
    """Tiny single-hop GNN that embeds a fixed-size molecule feature into a vector."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, atom_feats: Tensor) -> Tensor:
        """Embed ``(batch, n_atoms, in_dim)`` atom features into ``(batch, hidden_dim)``."""
        return self.proj(atom_feats).mean(dim=1)


class DoGGen(nn.Module):
    """Compact DoG-Gen: zero-latent autoregressive RNN decoder over synthesis DAGs.

    Each step reads the running GRU hidden state, predicts an action
    (add-leaf / react-existing / stop), a pointer over the current pool of DAG
    node embeddings (dot-product attention), and a reaction-template id, then
    appends a newly embedded node to the pool for the next step -- mirroring
    the DoG-Gen decoder with the hidden state seeded from a learned constant
    (no VAE latent, per the "zero-latent variant" candidate note).
    """

    def __init__(
        self,
        atom_feat_dim: int = 12,
        hidden_dim: int = 32,
        n_reaction_templates: int = 6,
        n_actions: int = 3,
        n_steps: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.node_encoder = _SmallGNNEncoder(atom_feat_dim, hidden_dim)
        self.init_hidden = nn.Parameter(torch.zeros(1, hidden_dim))
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, n_actions)
        self.template_head = nn.Linear(hidden_dim, n_reaction_templates)
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        self.new_node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, leaf_atom_feats: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Autoregressively build a synthesis DAG over a pool of candidate leaves.

        Parameters
        ----------
        leaf_atom_feats : Tensor
            ``(batch, n_leaves, n_atoms, atom_feat_dim)`` per-leaf atom features
            for a small pool of candidate starting reactants.

        Returns
        -------
        tuple of Tensor
            ``(action_logits, template_logits, pointer_logits)`` stacked over
            decoding steps: ``(batch, n_steps, n_actions)``,
            ``(batch, n_steps, n_reaction_templates)``,
            ``(batch, n_steps, n_leaves + n_steps)`` (pointer logits padded
            to the maximum possible pool size, with ``-inf`` for
            not-yet-created pool slots).
        """
        batch, n_leaves = leaf_atom_feats.shape[0], leaf_atom_feats.shape[1]
        leaf_embeds = torch.stack(
            [self.node_encoder(leaf_atom_feats[:, i]) for i in range(n_leaves)], dim=1
        )  # (batch, n_leaves, hidden_dim)
        pool = leaf_embeds
        h = self.init_hidden.expand(batch, -1)
        max_pool_size = (
            n_leaves + self.n_steps
        )  # fixed-size pointer output (padded for not-yet-created nodes)

        action_logits_list = []
        template_logits_list = []
        pointer_logits_list = []
        for _ in range(self.n_steps):
            pool_summary = pool.mean(dim=1)
            h = self.gru_cell(pool_summary, h)
            action_logits_list.append(self.action_head(h))
            template_logits_list.append(self.template_head(h))

            query = self.pointer_query(h).unsqueeze(1)
            keys = self.pointer_key(pool)
            pointer_scores = torch.einsum("bqd,bkd->bqk", query, keys).squeeze(1) / math.sqrt(
                self.hidden_dim
            )
            pad_width = max_pool_size - pointer_scores.shape[1]
            padded_scores = F.pad(pointer_scores, (0, pad_width), value=float("-inf"))
            pointer_logits_list.append(padded_scores)
            pointer_weights = F.softmax(pointer_scores, dim=-1)
            selected = torch.einsum("bk,bkd->bd", pointer_weights, pool)

            new_node = self.new_node_mlp(torch.cat([h, selected], dim=-1))
            pool = torch.cat([pool, new_node.unsqueeze(1)], dim=1)

        return (
            torch.stack(action_logits_list, dim=1),
            torch.stack(template_logits_list, dim=1),
            torch.stack(pointer_logits_list, dim=1),
        )


def build_dog_gen() -> nn.Module:
    """Build a compact DoG-Gen zero-latent synthesis-DAG decoder.

    Returns
    -------
    nn.Module
        ``DoGGen`` in eval mode.
    """
    return DoGGen(
        atom_feat_dim=12, hidden_dim=32, n_reaction_templates=6, n_actions=3, n_steps=4
    ).eval()


def example_input_dog_gen() -> Tensor:
    """Build a small pool of candidate-leaf atom features for ``DoGGen``.

    Returns
    -------
    Tensor
        ``(batch, n_leaves, n_atoms, atom_feat_dim)``.
    """
    return torch.randn(2, 5, 6, 12)


# ---------------------------------------------------------------------------
# 3. DRlinker -- Transformer seq2seq fragment-linking policy/prior
# ---------------------------------------------------------------------------


class _SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding (OpenNMT-py-style Transformer)."""

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to ``(batch, seq_len, d_model)``."""
        return x + self.pe[:, : x.size(1)]


class DRlinker(nn.Module):
    """Compact DRlinker: Transformer seq2seq fragment-linking policy.

    Encodes the SMILES of two fragments (with attachment-point tokens) and
    autoregressively decodes the SMILES of the full linked molecule, matching
    the OpenNMT-py Transformer prior/policy used before RL fine-tuning.
    """

    def __init__(
        self,
        vocab_size: int = 48,
        d_model: int = 32,
        n_heads: int = 4,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        dim_ff: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encode = _SinusoidalPositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, fragment_tokens: Tensor, target_tokens: Tensor) -> Tensor:
        """Predict next-token logits for the linked-molecule SMILES.

        Parameters
        ----------
        fragment_tokens : Tensor
            ``(batch, src_len)`` token ids of the two-fragment source sequence.
        target_tokens : Tensor
            ``(batch, tgt_len)`` token ids of the (teacher-forced) target
            linked-molecule sequence.

        Returns
        -------
        Tensor
            ``(batch, tgt_len, vocab_size)`` next-token logits.
        """
        src = self.pos_encode(self.token_embed(fragment_tokens) * math.sqrt(self.d_model))
        tgt = self.pos_encode(self.token_embed(target_tokens) * math.sqrt(self.d_model))
        tgt_len = target_tokens.size(1)
        causal_mask = torch.triu(torch.full((tgt_len, tgt_len), float("-inf")), diagonal=1)
        hidden = self.transformer(src, tgt, tgt_mask=causal_mask)
        return self.output_head(hidden)


def build_drlinker() -> nn.Module:
    """Build a compact DRlinker Transformer fragment-linking policy.

    Returns
    -------
    nn.Module
        ``DRlinker`` in eval mode.
    """
    return DRlinker(
        vocab_size=48, d_model=32, n_heads=4, n_enc_layers=2, n_dec_layers=2, dim_ff=64
    ).eval()


def example_input_drlinker() -> tuple[Tensor, Tensor]:
    """Build a small SMILES token batch for ``DRlinker``.

    Returns
    -------
    tuple of Tensor
        ``(fragment_tokens, target_tokens)`` of shape ``(batch, seq_len)``.
    """
    fragment_tokens = torch.randint(0, 48, (2, 14))
    target_tokens = torch.randint(0, 48, (2, 18))
    return fragment_tokens, target_tokens


# ---------------------------------------------------------------------------
# 4. DrugBAN -- GCN drug encoder + CNN protein encoder + bilinear attention
# ---------------------------------------------------------------------------


class _DenseGCNLayer(nn.Module):
    """Dense-adjacency GCN layer (small-molecule graphs, no sparsity needed)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """Propagate ``(batch, n_atoms, in_dim)`` features over ``adj`` (with self-loops)."""
        agg = torch.bmm(adj, x)
        return F.relu(self.linear(agg))


class _MolecularGCN(nn.Module):
    """DrugBAN's drug-side GCN encoder over a 2D molecular graph."""

    def __init__(self, atom_feat_dim: int, hidden_dim: int, n_layers: int) -> None:
        super().__init__()
        dims = [atom_feat_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([_DenseGCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)])

    def forward(self, atom_feats: Tensor, adj: Tensor) -> Tensor:
        """Return per-atom hidden features ``(batch, n_atoms, hidden_dim)``."""
        h = atom_feats
        for layer in self.layers:
            h = layer(h, adj)
        return h


class _ProteinCNN(nn.Module):
    """DrugBAN's protein-side 1D CNN encoder over an amino-acid embedding sequence."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

    def forward(self, residue_ids: Tensor) -> Tensor:
        """Return per-residue hidden features ``(batch, seq_len, hidden_dim)``."""
        x = self.embed(residue_ids).transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x.transpose(1, 2)


class _BilinearAttention(nn.Module):
    """Low-rank bilinear attention network (BAN) pooling drug-atom x protein-residue pairs."""

    def __init__(self, drug_dim: int, protein_dim: int, hidden_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.drug_proj = nn.Linear(drug_dim, hidden_dim * n_heads)
        self.protein_proj = nn.Linear(protein_dim, hidden_dim * n_heads)
        self.hidden_dim = hidden_dim
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, drug_feats: Tensor, protein_feats: Tensor) -> Tensor:
        """Return a pooled interaction representation ``(batch, hidden_dim)``."""
        b, n_atoms, _ = drug_feats.shape
        n_res = protein_feats.shape[1]
        d = self.drug_proj(drug_feats).view(b, n_atoms, self.n_heads, self.hidden_dim)
        p = self.protein_proj(protein_feats).view(b, n_res, self.n_heads, self.hidden_dim)
        # Low-rank bilinear join per head: element-wise product summed over hidden_dim
        # gives an (atom, residue) interaction logit per head (Kim et al. 2018 BAN).
        joint = torch.einsum("bahd,bphd->bahpd", d, p)  # (b, n_atoms, n_heads, n_res, hidden_dim)
        attn_logits = joint.sum(dim=-1)  # (b, n_atoms, n_heads, n_res)
        attn = F.softmax(attn_logits.reshape(b, n_atoms, self.n_heads * n_res), dim=-1)
        attn = attn.view(b, n_atoms, self.n_heads, n_res)
        # Pool the bilinear joint features by the attention map, then average over atoms/heads.
        pooled_per_head = torch.einsum("bahp,bahpd->bhd", attn, joint)
        pooled = pooled_per_head.mean(dim=1)
        return F.relu(self.pool_proj(pooled))


class DrugBAN(nn.Module):
    """Compact DrugBAN: GCN drug encoder + CNN protein encoder + bilinear attention head."""

    def __init__(
        self,
        atom_feat_dim: int = 16,
        protein_vocab: int = 25,
        hidden_dim: int = 32,
        n_gcn_layers: int = 3,
    ) -> None:
        super().__init__()
        self.drug_encoder = _MolecularGCN(atom_feat_dim, hidden_dim, n_gcn_layers)
        self.protein_encoder = _ProteinCNN(protein_vocab, hidden_dim, hidden_dim)
        self.ban = _BilinearAttention(hidden_dim, hidden_dim, hidden_dim, n_heads=4)
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, atom_feats: Tensor, adj: Tensor, residue_ids: Tensor) -> Tensor:
        """Predict a drug-target interaction logit.

        Parameters
        ----------
        atom_feats : Tensor
            ``(batch, n_atoms, atom_feat_dim)`` drug atom features.
        adj : Tensor
            ``(batch, n_atoms, n_atoms)`` drug adjacency (with self-loops).
        residue_ids : Tensor
            ``(batch, seq_len)`` protein amino-acid token ids.

        Returns
        -------
        Tensor
            ``(batch, 1)`` interaction logit.
        """
        drug_h = self.drug_encoder(atom_feats, adj)
        protein_h = self.protein_encoder(residue_ids)
        interaction = self.ban(drug_h, protein_h)
        return self.mlp_head(interaction)


def build_drugban() -> nn.Module:
    """Build a compact DrugBAN drug-target interaction predictor.

    Returns
    -------
    nn.Module
        ``DrugBAN`` in eval mode.
    """
    return DrugBAN(atom_feat_dim=16, protein_vocab=25, hidden_dim=32, n_gcn_layers=3).eval()


def example_input_drugban() -> tuple[Tensor, Tensor, Tensor]:
    """Build a small drug-graph + protein-sequence batch for ``DrugBAN``.

    Returns
    -------
    tuple of Tensor
        ``(atom_feats, adj, residue_ids)``.
    """
    batch, n_atoms, seq_len = 2, 10, 40
    atom_feats = torch.randn(batch, n_atoms, 16)
    adj = torch.eye(n_atoms).unsqueeze(0).expand(batch, -1, -1).clone()
    adj = adj + (torch.rand(batch, n_atoms, n_atoms) > 0.7).float()
    adj = ((adj + adj.transpose(1, 2)) > 0).float()
    residue_ids = torch.randint(0, 25, (batch, seq_len))
    return atom_feats, adj, residue_ids


# ---------------------------------------------------------------------------
# 5. Tied Two-Way Transformer retrosynthesis -- weight-tied dual Transformers
# ---------------------------------------------------------------------------


class TiedTwoWayTransformer(nn.Module):
    """Compact tied two-way transformer for retrosynthesis with latent variables.

    A single Transformer encoder-decoder is used in **both** directions
    (weights literally shared -- ``self.transformer`` is called once for the
    backward product->reactants pass and again for the forward
    reactants->product pass), giving a differentiable round-trip
    cycle-consistency signal. A small learned discrete latent-variable
    embedding table conditions the backward decoding pass, letting one
    product map to multiple plausible reactant sets (per Kim et al. 2021).
    """

    def __init__(
        self,
        vocab_size: int = 48,
        d_model: int = 32,
        n_heads: int = 4,
        n_enc_layers: int = 2,
        n_dec_layers: int = 2,
        dim_ff: int = 64,
        n_latents: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encode = _SinusoidalPositionalEncoding(d_model)
        self.latent_embed = nn.Embedding(n_latents, d_model)
        # Single tied Transformer, reused for both the backward and forward directions.
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_enc_layers,
            num_decoder_layers=n_dec_layers,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.output_head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, length: int) -> Tensor:
        return torch.triu(torch.full((length, length), float("-inf")), diagonal=1)

    def _decode_step(self, src_tokens: Tensor, tgt_tokens: Tensor, latent_id: Tensor) -> Tensor:
        src = self.pos_encode(self.token_embed(src_tokens) * math.sqrt(self.d_model))
        latent = self.latent_embed(latent_id).unsqueeze(
            1
        )  # (batch, 1, d_model) latent conditioning token
        tgt = self.pos_encode(self.token_embed(tgt_tokens) * math.sqrt(self.d_model))
        tgt = torch.cat([latent, tgt], dim=1)
        hidden = self.transformer(src, tgt, tgt_mask=self._causal_mask(tgt.size(1)))
        return self.output_head(hidden[:, 1:])

    def forward(
        self, product_tokens: Tensor, reactant_tokens: Tensor, latent_id: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run the tied backward + forward round trip.

        Parameters
        ----------
        product_tokens : Tensor
            ``(batch, prod_len)`` token ids of the target product SMILES.
        reactant_tokens : Tensor
            ``(batch, react_len)`` token ids of the (teacher-forced) reactant
            SMILES.
        latent_id : Tensor
            ``(batch,)`` discrete latent-variable id conditioning the
            backward decode.

        Returns
        -------
        tuple of Tensor
            ``(backward_logits, forward_logits)``: predicted next-token
            logits for reactants-given-product (backward, retrosynthesis) and
            for product-given-predicted-reactants (forward, the cycle-
            consistency reconstruction), both using the same tied weights.
        """
        # Backward pass: product -> reactants (retrosynthesis), latent-conditioned.
        backward_logits = self._decode_step(product_tokens, reactant_tokens, latent_id)

        # Forward pass (cycle consistency): predicted reactants -> reconstructed product,
        # reusing the identical tied transformer weights with no latent token.
        zero_latent = torch.zeros_like(latent_id)
        forward_logits = self._decode_step(reactant_tokens, product_tokens, zero_latent)
        return backward_logits, forward_logits


def build_tied_twoway_transformer() -> nn.Module:
    """Build a compact tied two-way transformer for retrosynthesis.

    Returns
    -------
    nn.Module
        ``TiedTwoWayTransformer`` in eval mode.
    """
    return TiedTwoWayTransformer(
        vocab_size=48, d_model=32, n_heads=4, n_enc_layers=2, n_dec_layers=2, dim_ff=64, n_latents=4
    ).eval()


def example_input_tied_twoway_transformer() -> tuple[Tensor, Tensor, Tensor]:
    """Build a small product/reactant SMILES token batch for the tied two-way transformer.

    Returns
    -------
    tuple of Tensor
        ``(product_tokens, reactant_tokens, latent_id)``.
    """
    product_tokens = torch.randint(0, 48, (2, 12))
    reactant_tokens = torch.randint(0, 48, (2, 16))
    latent_id = torch.randint(0, 4, (2,))
    return product_tokens, reactant_tokens, latent_id


MENAGERIE_ENTRIES = [
    ("DiGress", "build_digress", "example_input_digress", "2023", "BIO"),
    ("DoG-Gen", "build_dog_gen", "example_input_dog_gen", "2020", "BIO"),
    ("DRlinker", "build_drlinker", "example_input_drlinker", "2022", "BIO"),
    ("DrugBAN", "build_drugban", "example_input_drugban", "2023", "BIO"),
    (
        "Dual Transformer retrosynthesis",
        "build_tied_twoway_transformer",
        "example_input_tied_twoway_transformer",
        "2021",
        "BIO",
    ),
]

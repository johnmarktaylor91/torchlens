"""Compact faithful reimplementations for build_queue rows 85-90 (W9A14).

Sources checked (repo/paper browsed via ``gh api`` / web, no clone/pip-install):
  - RANKITECT (cand_01327): the public realization is Meta's NASRec --
    Zhang, Chen, Zhao, Park, Yang, Wan, "NASRec: Weight Sharing Neural
    Architecture Search for Recommender Systems", ICLR 2023 /
    arXiv:2207.07187. Official repo facebookresearch/NasRec,
    ``nasrec/supernet/modules.py`` / ``supernet.py``. Distinctive
    mechanism: a *heterogeneous single-supernet* NAS search space for
    click-through-rate recommendation that fuses DLRM-style pairwise
    feature interaction with dense-MLP and multi-head self-attention
    "blocks" inside one DAG -- each block reads from *all* prior blocks'
    outputs (concatenated dense stream + concatenated sparse/embedding
    stream), so the network is a directed graph of heterogeneous
    candidate operators (``ElasticLinear`` MLP block, ``DotProduct``
    pairwise-interaction block that dot-products stacked dense+sparse
    embeddings, ``Sum``, and a self-attention block) rather than a fixed
    stack. Reimplemented as ``NasRecBlockDAG`` with a small fixed
    (post-search) subnet: an embedding table for sparse fields, a
    ``DotProductBlock`` (DLRM-style pairwise dot-product interaction of
    projected dense+sparse features), an ``MlpBlock`` (elastic-style
    2-layer ReLU MLP), and a self-attention block, each block consuming
    the *concatenation* of the raw dense input and every prior block's
    dense output (the DAG-style block-to-block wiring that is NASRec's
    signature), followed by a final scoring MLP -- at reduced field
    count / hidden width / block count.
  - RouteNet-Erlang (cand_01329): official repo is BNN-UPC/RouteNet-Fermi
    -- Ferriol-Galmes et al., "RouteNet-Fermi: Network Modeling with
    Graph Neural Networks", IEEE/ACM Transactions on Networking 2023.
    File ``delay_model.py`` (class ``RouteNet_Fermi``, TensorFlow/Keras).
    Distinctive mechanism: a *3-stage heterogeneous message-passing GNN*
    over a path/link/queue tripartite graph -- path hidden states are
    updated by a GRU driven by the sequence of (queue, link) states along
    each path's route; queue states are updated by a GRUCell fed the
    sum of all path-state contributions that traverse that queue; link
    states are updated by a GRU driven by the sequence of queue states
    feeding that link -- iterated for several message-passing rounds,
    then a per-path-per-hop MLP readout predicts per-hop queuing delay
    that is summed with a closed-form transmission-delay term. Reimplemented
    as ``RouteNetFermi`` with the same three GRU/GRUCell-based
    path->queue->link update cycle over a small fixed topology (dense
    padded incidence tensors standing in for the ragged
    path/link/queue-to-* index lists) and the closed-form transmission-
    delay term added to the learned queuing-delay readout, at reduced
    hidden width / iteration count / topology size.
  - RSR / Relational Stock Ranking (cand_01330): Feng, He, Wang, Luo,
    Zha, Chua, "Temporal Relational Ranking for Stock Prediction", ACM
    TOIS 2019, arXiv:1809.09441. Official repo
    fulifeng/Temporal_Relational_Stock_Ranking,
    ``training/relation_rank_lstm.py`` (class ``ReRaLSTM``, TensorFlow).
    Distinctive mechanism: per-stock daily OHLCV sequences are first
    embedded by an LSTM ("Rank_LSTM" sequential encoder) into one
    feature vector per stock per day; a fixed, externally supplied
    stock-relation tensor (sector/industry or Wikidata-derived
    multi-hot relation encoding) is projected to a scalar edge weight
    per stock pair (``rel_weight``, a dense layer over the relation
    encoding) and combined with either an inner-product or an additive
    head/tail attention score computed from the LSTM embeddings; the
    relation mask (zero for unrelated pairs) is added before a
    column-wise softmax, producing a graph-attention propagation matrix
    that aggregates every stock's embedding from its relation-graph
    neighbors; the propagated and original embeddings are concatenated
    (or fused through one more hidden layer, the "flat" variant) and a
    final dense layer regresses a cross-sectional return ratio used for
    ranking. Reimplemented as ``RelationalStockRanking`` with the same
    LSTM sequence encoder, static relation-tensor-conditioned graph
    attention (dense projection of the relation encoding to an edge
    score, masked column-softmax propagation), and return-ratio
    regression head, at reduced stock count / relation-embedding size /
    sequence length.
  - SAFE (cand_01331): Massarelli, Di Luna, Petroni, Baldoni, Querzoni,
    "SAFE: Self-Attentive Function Embeddings for Binary Similarity",
    DIMVA 2019, arXiv:1811.05296. ``facebookresearch/SAFEtorch`` is the
    official PyTorch port (``safetorch/safe_network.py``, class
    ``SAFE``); original TF version at ``gadiluna/SAFE``. Distinctive
    mechanism: assembly-instruction embeddings are fed through a
    bidirectional GRU to produce a per-instruction hidden-state matrix
    ``H``; a *structured self-attention* pooling (Lin et al. 2017 style,
    two learned projection matrices ``WS1``/``WS2``: ``A =
    softmax(WS2 . tanh(WS1 . H^T))`` producing ``attention_hops`` rows of
    attention over the instruction sequence) computes a fixed-size
    embedding matrix ``M = A . H`` regardless of function length; ``M``
    is flattened and passed through a 2-layer dense head with L2
    normalization to produce the final function embedding used for
    binary-similarity comparison. Reimplemented as ``SafeFunctionEmbedder``
    with the identical embedding -> bidirectional-GRU -> structured
    self-attentive pooling (``WS1``/``WS2``) -> flatten -> 2-layer dense
    -> L2-normalize pipeline, at reduced vocab / hidden / hop counts.
  - SySeVR (cand_01333): Li, Zou, Xu, Ou, Jin, Wang, Deng, Zhong,
    "SySeVR: A Framework for Using Deep Learning to Detect Software
    Vulnerabilities", IEEE TDSC 2021, arXiv:1807.06756. Official repo
    SySeVR/SySeVR, ``Implementation/model/bgru.py`` (function
    ``build_model``, Keras). Distinctive mechanism: syntax/semantic
    "code-slice" token sequences are first turned into dense vectors by
    an external word2vec-style embedding (the "syntax-based/semantic-
    based vector representation" step that gives SySeVR its name,
    upstream of the network) and zero-padded; the network itself is a
    ``Masking`` layer (ignore zero-padded timesteps) feeding a stack of
    bidirectional GRU layers (``layers-1`` intermediate BiGRU layers with
    ``return_sequences=True`` + dropout, then a final BiGRU that
    collapses to a single vector) followed by a sigmoid binary
    vulnerability-classification head. Reimplemented as ``SySeVRDetector``
    with the identical masking -> stacked-bidirectional-GRU
    (intermediate layers sequence-returning, final layer vector-
    returning) -> dropout -> sigmoid head pipeline over pre-embedded
    code-slice token vectors, at reduced hidden width / layer count /
    sequence length.
  - TLOB (cand_01335): Berti, "TLOB: A Novel Transformer Model with
    Dual Attention for Stock Price Trend Prediction with Limit Order
    Book Data", arXiv:2502.15757 (2025). Official repo
    LeonardoBerti00/TLOB, ``models/tlob.py`` (class ``TLOB``) +
    ``models/bin.py`` (class ``BiN``). Distinctive mechanism: a *dual-
    attention* transformer that alternates, layer by layer, between
    standard self-attention over the **temporal** axis (attending across
    LOB snapshots at fixed feature channels) and self-attention over the
    **feature** axis (attending across order-book feature channels at a
    fixed timestep) -- implemented by literally transposing
    (``permute``) the sequence and feature dimensions between successive
    identical ``TransformerLayer`` blocks, so the same attention module
    alternately sees "tokens = timesteps" and "tokens = features"; a
    learned bi-normalization layer (``BiN``: a convex combination of
    temporal-axis and feature-axis z-score normalization with learned
    per-position affine parameters and two learned mixing scalars)
    normalizes the raw LOB snapshot before the dual-attention stack, and
    the final flattened representation is projected through a shrinking
    MLP stack to a 3-way (up/stationary/down) trend-prediction head.
    Reimplemented as ``TlobDualAttention`` with the same
    BiN-normalization -> alternating temporal-axis/feature-axis
    ``TransformerLayer`` stack (post-attention linear projection back to
    model width, residual, LayerNorm, feed-forward block) ->
    dimension-shrinking final MLP -> 3-way classification head, at
    reduced hidden width / sequence length / layer count.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# RANKITECT (NASRec) -- heterogeneous block-DAG supernet for CTR recommendation
# ---------------------------------------------------------------------------


class DotProductBlock(nn.Module):
    """DLRM-style pairwise dot-product interaction block.

    Projects dense and sparse (embedding) streams to a common width,
    stacks them as "feature vectors", takes their pairwise dot products
    (lower-triangular, excluding the diagonal), and linearly projects the
    flattened interaction vector to ``out_dim``.

    Parameters
    ----------
    dense_dim : int
        Width of the incoming dense feature stream.
    num_sparse : int
        Number of sparse (embedding-table) fields.
    embed_dim : int
        Width of each sparse embedding / the common projection width.
    out_dim : int
        Output width of this block.
    """

    def __init__(self, dense_dim: int, num_sparse: int, embed_dim: int, out_dim: int) -> None:
        super().__init__()
        self.dense_proj = nn.Linear(dense_dim, embed_dim)
        self.out_dim = out_dim
        n_vecs = num_sparse + 1
        n_pairs = n_vecs * (n_vecs - 1) // 2
        self.out_proj = nn.Linear(n_pairs, out_dim)

    def forward(self, dense_t: torch.Tensor, sparse_t: torch.Tensor) -> torch.Tensor:
        """Interact dense and sparse streams into a fixed-width vector.

        Parameters
        ----------
        dense_t : Tensor
            Shape ``(batch, dense_dim)``.
        sparse_t : Tensor
            Shape ``(batch, num_sparse, embed_dim)``.

        Returns
        -------
        Tensor
            Shape ``(batch, out_dim)``.
        """
        x = self.dense_proj(dense_t).unsqueeze(1)
        stacked = torch.cat([x, sparse_t], dim=1)
        interactions = torch.bmm(stacked, stacked.transpose(1, 2))
        n = interactions.shape[-1]
        li, lj = torch.tril_indices(n, n, offset=-1)
        flat = interactions[:, li, lj]
        return F.relu(self.out_proj(flat))


class MlpBlock(nn.Module):
    """Elastic-style 2-layer ReLU MLP block reading only the dense stream."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, dense_t: torch.Tensor) -> torch.Tensor:
        """Apply the 2-layer ReLU MLP to the dense stream.

        Parameters
        ----------
        dense_t : Tensor
            Shape ``(batch, in_dim)``.

        Returns
        -------
        Tensor
            Shape ``(batch, out_dim)``.
        """
        return F.relu(self.fc2(F.relu(self.fc1(dense_t))))


class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention block over the sparse (embedding) stream."""

    def __init__(self, embed_dim: int, num_heads: int, out_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, sparse_t: torch.Tensor) -> torch.Tensor:
        """Self-attend over sparse fields and pool to a fixed-width vector.

        Parameters
        ----------
        sparse_t : Tensor
            Shape ``(batch, num_sparse, embed_dim)``.

        Returns
        -------
        Tensor
            Shape ``(batch, out_dim)``.
        """
        attended, _ = self.attn(sparse_t, sparse_t, sparse_t, need_weights=False)
        pooled = attended.mean(dim=1)
        return F.relu(self.out_proj(pooled))


class NasRecBlockDAG(nn.Module):
    """Fixed (post-search) NASRec-style heterogeneous block-DAG subnet.

    A small stand-in for the NASRec supernet's search space: sparse
    fields are embedded, then a ``DotProductBlock``, ``MlpBlock``, and
    ``SelfAttentionBlock`` are chained so each block's dense output is
    concatenated onto the running dense stream consumed by the next
    block (the DAG-style "read from all prior outputs" wiring that
    defines the NASRec search space), before a final scoring head.

    Parameters
    ----------
    num_dense : int
        Number of raw dense (continuous) input features.
    num_sparse : int
        Number of sparse categorical fields.
    vocab_size : int
        Shared vocabulary size for the sparse embedding table.
    embed_dim : int
        Sparse embedding width.
    block_dim : int
        Output width of each DAG block.
    """

    def __init__(
        self,
        num_dense: int = 8,
        num_sparse: int = 6,
        vocab_size: int = 64,
        embed_dim: int = 8,
        block_dim: int = 12,
    ) -> None:
        super().__init__()
        self.sparse_embed = nn.Embedding(vocab_size, embed_dim)
        self.dot_block = DotProductBlock(num_dense, num_sparse, embed_dim, block_dim)
        self.mlp_block = MlpBlock(num_dense + block_dim, block_dim * 2, block_dim)
        self.attn_block = SelfAttentionBlock(embed_dim, 2, block_dim)
        final_in = num_dense + block_dim * 3
        self.scorer = nn.Sequential(
            nn.Linear(final_in, block_dim * 2),
            nn.ReLU(),
            nn.Linear(block_dim * 2, 1),
        )

    def forward(self, dense_x: torch.Tensor, sparse_idx: torch.Tensor) -> torch.Tensor:
        """Run the block-DAG subnet.

        Parameters
        ----------
        dense_x : Tensor
            Shape ``(batch, num_dense)``.
        sparse_idx : LongTensor
            Shape ``(batch, num_sparse)``.

        Returns
        -------
        Tensor
            Shape ``(batch, 1)`` click-through-rate logit.
        """
        sparse_t = self.sparse_embed(sparse_idx)
        dot_out = self.dot_block(dense_x, sparse_t)
        stream = torch.cat([dense_x, dot_out], dim=1)
        mlp_out = self.mlp_block(stream)
        attn_out = self.attn_block(sparse_t)
        fused = torch.cat([dense_x, dot_out, mlp_out, attn_out], dim=1)
        return self.scorer(fused)


def build_rankitect() -> nn.Module:
    """Build a compact RANKITECT / NASRec-style block-DAG subnet.

    Returns
    -------
    nn.Module
        ``NasRecBlockDAG`` in eval mode.
    """
    torch.manual_seed(0)
    return NasRecBlockDAG().eval()


def example_input_rankitect() -> tuple[torch.Tensor, torch.Tensor]:
    """Example input for :func:`build_rankitect`.

    Returns
    -------
    tuple of Tensor
        ``(dense_x, sparse_idx)`` of shapes ``(4, 8)`` and ``(4, 6)``.
    """
    torch.manual_seed(0)
    dense_x = torch.randn(4, 8)
    sparse_idx = torch.randint(0, 64, (4, 6))
    return dense_x, sparse_idx


# ---------------------------------------------------------------------------
# RouteNet-Erlang (RouteNet-Fermi) -- 3-stage path/link/queue message-passing GNN
# ---------------------------------------------------------------------------


class RouteNetFermi(nn.Module):
    """Compact RouteNet-Fermi-style path/link/queue message-passing GNN.

    Iteratively updates path, queue, and link hidden states with GRU
    cells driven by each other along a small fixed topology, then reads
    out per-path queuing delay and adds a closed-form transmission-delay
    term, matching the reference's 3-stage message-passing design.

    Parameters
    ----------
    n_paths : int
        Number of paths (flows) in the fixed topology.
    n_links : int
        Number of links.
    n_queues : int
        Number of queues.
    hops : int
        Path length (number of link/queue hops per path).
    state_dim : int
        Hidden width shared by path, link, and queue states.
    iterations : int
        Number of message-passing rounds.
    """

    def __init__(
        self,
        n_paths: int = 5,
        n_links: int = 4,
        n_queues: int = 4,
        hops: int = 3,
        state_dim: int = 8,
        iterations: int = 3,
    ) -> None:
        super().__init__()
        self.n_paths = n_paths
        self.n_links = n_links
        self.n_queues = n_queues
        self.hops = hops
        self.state_dim = state_dim
        self.iterations = iterations

        self.path_embedding = nn.Sequential(
            nn.Linear(3, state_dim), nn.ReLU(), nn.Linear(state_dim, state_dim), nn.ReLU()
        )
        self.link_embedding = nn.Sequential(
            nn.Linear(1, state_dim), nn.ReLU(), nn.Linear(state_dim, state_dim), nn.ReLU()
        )
        self.queue_embedding = nn.Sequential(
            nn.Linear(1, state_dim), nn.ReLU(), nn.Linear(state_dim, state_dim), nn.ReLU()
        )

        self.path_update = nn.GRU(state_dim * 2, state_dim, batch_first=True)
        self.queue_update = nn.GRUCell(state_dim, state_dim)
        self.link_update = nn.GRU(state_dim, state_dim, batch_first=True)

        self.readout = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, 1),
        )

    def forward(
        self,
        path_features: torch.Tensor,
        link_capacity: torch.Tensor,
        queue_size: torch.Tensor,
        path_to_hop_queue: torch.Tensor,
        path_to_hop_link: torch.Tensor,
    ) -> torch.Tensor:
        """Run the 3-stage message-passing GNN.

        Parameters
        ----------
        path_features : Tensor
            Shape ``(n_paths, 3)`` -- raw per-path traffic descriptors.
        link_capacity : Tensor
            Shape ``(n_links, 1)``.
        queue_size : Tensor
            Shape ``(n_queues, 1)``.
        path_to_hop_queue : LongTensor
            Shape ``(n_paths, hops)`` -- queue index visited at each hop.
        path_to_hop_link : LongTensor
            Shape ``(n_paths, hops)`` -- link index visited at each hop.

        Returns
        -------
        Tensor
            Shape ``(n_paths, 1)`` predicted per-path delay.
        """
        path_state = self.path_embedding(path_features)
        link_state = self.link_embedding(link_capacity)
        queue_state = self.queue_embedding(queue_size)

        for _ in range(self.iterations):
            queue_gather = queue_state[path_to_hop_queue]
            link_gather = link_state[path_to_hop_link]
            path_seq_in = torch.cat([queue_gather, link_gather], dim=-1)
            path_seq_out, path_h = self.path_update(path_seq_in, path_state.unsqueeze(0))
            path_state = path_h.squeeze(0)

            queue_contrib = torch.zeros_like(queue_state)
            queue_contrib.index_add_(
                0, path_to_hop_queue.reshape(-1), path_seq_out.reshape(-1, self.state_dim)
            )
            queue_state = self.queue_update(queue_contrib, queue_state)

            link_contrib = torch.zeros(self.n_links, self.hops, self.state_dim)
            link_contrib.scatter_(
                1,
                path_to_hop_link.new_zeros(self.n_links, self.hops)
                .unsqueeze(-1)
                .expand(-1, -1, self.state_dim),
                queue_state.unsqueeze(0).expand(self.n_links, -1, -1)[:, : self.hops],
            )
            _, link_h = self.link_update(link_contrib)
            link_state = link_h.squeeze(0)

        queuing_delay = self.readout(path_seq_out).squeeze(-1).sum(dim=-1, keepdim=True)
        link_cap_per_hop = link_capacity.squeeze(-1)[path_to_hop_link]
        trans_delay = (1.0 / link_cap_per_hop.clamp_min(1e-3)).sum(dim=-1, keepdim=True)
        return queuing_delay + trans_delay


def build_routenet_erlang() -> nn.Module:
    """Build a compact RouteNet-Fermi-style message-passing GNN.

    Returns
    -------
    nn.Module
        ``RouteNetFermi`` in eval mode.
    """
    torch.manual_seed(0)
    return RouteNetFermi().eval()


def example_input_routenet_erlang() -> tuple[torch.Tensor, ...]:
    """Example input for :func:`build_routenet_erlang`.

    Returns
    -------
    tuple of Tensor
        ``(path_features, link_capacity, queue_size, path_to_hop_queue,
        path_to_hop_link)``.
    """
    torch.manual_seed(0)
    n_paths, n_links, n_queues, hops = 5, 4, 4, 3
    path_features = torch.rand(n_paths, 3)
    link_capacity = torch.rand(n_links, 1) + 0.5
    queue_size = torch.rand(n_queues, 1) + 0.5
    path_to_hop_queue = torch.randint(0, n_queues, (n_paths, hops))
    path_to_hop_link = torch.randint(0, n_links, (n_paths, hops))
    return path_features, link_capacity, queue_size, path_to_hop_queue, path_to_hop_link


# ---------------------------------------------------------------------------
# RSR (Relational Stock Ranking) -- LSTM + relation-graph attention propagation
# ---------------------------------------------------------------------------


class RelationalStockRanking(nn.Module):
    """Compact Temporal Relational Stock Ranking model.

    LSTM-encodes each stock's daily feature sequence, then propagates
    information across stocks through a static relation-tensor-
    conditioned graph attention (additive head/tail attention combined
    with a learned projection of the relation encoding, masked
    column-softmax), and regresses a cross-sectional return ratio.

    Parameters
    ----------
    n_stocks : int
        Number of stocks in the cross-section.
    feature_dim : int
        Per-day raw feature width (e.g. OHLCV = 5).
    hidden_dim : int
        LSTM / embedding width.
    relation_dim : int
        Width of the static per-pair relation encoding.
    """

    def __init__(
        self,
        n_stocks: int = 10,
        feature_dim: int = 5,
        hidden_dim: int = 16,
        relation_dim: int = 6,
    ) -> None:
        super().__init__()
        self.n_stocks = n_stocks
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.rel_weight = nn.Linear(relation_dim, 1)
        self.head_weight = nn.Linear(hidden_dim, 1)
        self.tail_weight = nn.Linear(hidden_dim, 1)
        self.predictor = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        price_seq: torch.Tensor,
        relation_encoding: torch.Tensor,
        relation_mask: torch.Tensor,
        base_price: torch.Tensor,
    ) -> torch.Tensor:
        """Run the LSTM encoder + relation-graph-attention propagation.

        Parameters
        ----------
        price_seq : Tensor
            Shape ``(n_stocks, seq_len, feature_dim)``.
        relation_encoding : Tensor
            Shape ``(n_stocks, n_stocks, relation_dim)``.
        relation_mask : Tensor
            Shape ``(n_stocks, n_stocks)``, ``0`` for related pairs and a
            large negative value for unrelated ones (additive mask).
        base_price : Tensor
            Shape ``(n_stocks, 1)``.

        Returns
        -------
        Tensor
            Shape ``(n_stocks, 1)`` predicted return ratio.
        """
        _, (h_n, _) = self.lstm(price_seq)
        feature = h_n.squeeze(0)

        rel_edge = self.rel_weight(relation_encoding).squeeze(-1)
        head = self.head_weight(feature)
        tail = self.tail_weight(feature)
        additive = head + tail.transpose(0, 1)
        weight = additive + rel_edge
        weight_masked = F.softmax(relation_mask + weight, dim=0)

        propagated = weight_masked @ feature
        fused = torch.cat([feature, propagated], dim=-1)
        prediction = F.leaky_relu(self.predictor(fused), negative_slope=0.2)
        return_ratio = (prediction - base_price) / base_price
        return return_ratio


def build_rsr() -> nn.Module:
    """Build a compact Relational-Stock-Ranking model.

    Returns
    -------
    nn.Module
        ``RelationalStockRanking`` in eval mode.
    """
    torch.manual_seed(0)
    return RelationalStockRanking().eval()


def example_input_rsr() -> tuple[torch.Tensor, ...]:
    """Example input for :func:`build_rsr`.

    Returns
    -------
    tuple of Tensor
        ``(price_seq, relation_encoding, relation_mask, base_price)``.
    """
    torch.manual_seed(0)
    n_stocks, seq_len, feature_dim, relation_dim = 10, 12, 5, 6
    price_seq = torch.rand(n_stocks, seq_len, feature_dim)
    relation_encoding = torch.rand(n_stocks, n_stocks, relation_dim)
    related = torch.rand(n_stocks, n_stocks) > 0.5
    related.fill_diagonal_(True)
    relation_mask = torch.where(
        related, torch.zeros(n_stocks, n_stocks), torch.full((n_stocks, n_stocks), -1e9)
    )
    base_price = torch.rand(n_stocks, 1) + 1.0
    return price_seq, relation_encoding, relation_mask, base_price


# ---------------------------------------------------------------------------
# SAFE -- self-attentive function embeddings for binary similarity
# ---------------------------------------------------------------------------


class SafeFunctionEmbedder(nn.Module):
    """Compact SAFE-style self-attentive binary function embedder.

    Embeds a sequence of assembly-instruction token ids, encodes them
    with a bidirectional GRU, pools the per-instruction hidden states
    with a structured self-attention (Lin et al. 2017 style, two
    learned projections producing multiple attention "hops"), and
    projects the flattened pooled matrix through a 2-layer dense head
    with L2 normalization.

    Parameters
    ----------
    vocab_size : int
        Instruction-token vocabulary size.
    embed_dim : int
        Instruction embedding width.
    rnn_hidden : int
        Per-direction GRU hidden width.
    attention_depth : int
        Width of the intermediate attention projection ``WS1``.
    attention_hops : int
        Number of attention rows / hops (``WS2`` output rows).
    dense_dim : int
        Width of the first dense head layer.
    """

    def __init__(
        self,
        vocab_size: int = 200,
        embed_dim: int = 16,
        rnn_hidden: int = 12,
        attention_depth: int = 10,
        attention_hops: int = 4,
        dense_dim: int = 32,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, rnn_hidden, batch_first=True, bidirectional=True)
        self.ws1 = nn.Parameter(torch.randn(attention_depth, 2 * rnn_hidden) * 0.1)
        self.ws2 = nn.Parameter(torch.randn(attention_hops, attention_depth) * 0.1)
        self.dense_1 = nn.Linear(2 * attention_hops * rnn_hidden, dense_dim)
        self.dense_2 = nn.Linear(dense_dim, embed_dim)
        self.attention_hops = attention_hops
        self.rnn_hidden = rnn_hidden

    def forward(self, instructions: torch.Tensor) -> torch.Tensor:
        """Embed a batch of instruction-token sequences.

        Parameters
        ----------
        instructions : LongTensor
            Shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Shape ``(batch, embed_dim)``, L2-normalized function embeddings.
        """
        vectors = self.embedding(instructions)
        h, _ = self.gru(vectors)
        a = torch.softmax(self.ws2 @ torch.tanh(self.ws1 @ h.transpose(1, 2)), dim=2)
        m = a @ h
        flattened = m.reshape(m.shape[0], self.attention_hops * 2 * self.rnn_hidden)
        dense_out = F.relu(self.dense_1(flattened))
        return F.normalize(self.dense_2(dense_out), dim=1, p=2)


def build_safe() -> nn.Module:
    """Build a compact SAFE self-attentive function embedder.

    Returns
    -------
    nn.Module
        ``SafeFunctionEmbedder`` in eval mode.
    """
    torch.manual_seed(0)
    return SafeFunctionEmbedder().eval()


def example_input_safe() -> torch.Tensor:
    """Example input for :func:`build_safe`.

    Returns
    -------
    Tensor
        Shape ``(3, 24)`` instruction-token ids.
    """
    torch.manual_seed(0)
    return torch.randint(0, 200, (3, 24))


# ---------------------------------------------------------------------------
# SySeVR -- masked stacked bidirectional GRU vulnerability detector
# ---------------------------------------------------------------------------


class SySeVRDetector(nn.Module):
    """Compact SySeVR-style stacked bidirectional-GRU vulnerability detector.

    Consumes pre-embedded (word2vec-style) code-slice token vectors,
    masks zero-padded timesteps, runs them through a stack of
    bidirectional GRU layers (intermediate layers keep the full
    sequence, the final layer collapses to one vector), and predicts
    vulnerability presence with a sigmoid head.

    Parameters
    ----------
    vector_dim : int
        Per-token pre-embedded feature width.
    hidden_dim : int
        Per-direction GRU hidden width.
    layers : int
        Number of stacked bidirectional GRU layers.
    dropout : float
        Dropout probability applied after every GRU layer.
    """

    def __init__(
        self,
        vector_dim: int = 20,
        hidden_dim: int = 16,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.grus = nn.ModuleList(
            [
                nn.GRU(
                    vector_dim if i == 0 else hidden_dim * 2,
                    hidden_dim,
                    batch_first=True,
                    bidirectional=True,
                )
                for i in range(layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, token_vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run the masked stacked-BiGRU detector.

        Parameters
        ----------
        token_vectors : Tensor
            Shape ``(batch, seq_len, vector_dim)``, zero-padded.
        mask : Tensor
            Shape ``(batch, seq_len, 1)``, ``1`` for valid timesteps and
            ``0`` for padding (applied as the reference's ``Masking``
            layer would).

        Returns
        -------
        Tensor
            Shape ``(batch, 1)`` vulnerability probability.
        """
        x = token_vectors * mask
        for i, gru in enumerate(self.grus):
            if i != self.layers - 1:
                x, _ = gru(x)
                x = self.dropout(x)
            else:
                _, h_n = gru(x)
                x = torch.cat([h_n[0], h_n[1]], dim=-1)
                x = self.dropout(x)
        return torch.sigmoid(self.classifier(x))


def build_sysevr() -> nn.Module:
    """Build a compact SySeVR masked stacked-BiGRU vulnerability detector.

    Returns
    -------
    nn.Module
        ``SySeVRDetector`` in eval mode.
    """
    torch.manual_seed(0)
    return SySeVRDetector().eval()


def example_input_sysevr() -> tuple[torch.Tensor, torch.Tensor]:
    """Example input for :func:`build_sysevr`.

    Returns
    -------
    tuple of Tensor
        ``(token_vectors, mask)`` of shapes ``(4, 30, 20)`` and
        ``(4, 30, 1)``.
    """
    torch.manual_seed(0)
    token_vectors = torch.randn(4, 30, 20)
    mask = torch.ones(4, 30, 1)
    mask[:, 22:, :] = 0.0
    return token_vectors, mask


# ---------------------------------------------------------------------------
# TLOB -- dual (temporal + feature axis) attention transformer for LOB data
# ---------------------------------------------------------------------------


class BiNormalization(nn.Module):
    """Bi-normalization layer: learned mix of temporal and feature z-scores.

    Parameters
    ----------
    n_features : int
        Number of feature channels.
    seq_len : int
        Sequence length.
    """

    def __init__(self, n_features: int, seq_len: int) -> None:
        super().__init__()
        self.l1 = nn.Parameter(torch.randn(seq_len, 1))
        self.b1 = nn.Parameter(torch.zeros(seq_len, 1))
        self.l2 = nn.Parameter(torch.randn(n_features, 1))
        self.b2 = nn.Parameter(torch.zeros(n_features, 1))
        self.y1 = nn.Parameter(torch.tensor(0.5))
        self.y2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a ``(batch, n_features, seq_len)`` tensor.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, n_features, seq_len)``.

        Returns
        -------
        Tensor
            Same shape as ``x``.
        """
        mean_t = x.mean(dim=2, keepdim=True)
        std_t = x.std(dim=2, keepdim=True).clamp_min(1e-4)
        z_feature = (x - mean_t) / std_t
        x_feature = self.l2.unsqueeze(0) * z_feature + self.b2.unsqueeze(0)

        mean_f = x.mean(dim=1, keepdim=True)
        std_f = x.std(dim=1, keepdim=True).clamp_min(1e-4)
        z_temporal = (x - mean_f) / std_f
        x_temporal = self.l1.transpose(0, 1).unsqueeze(1) * z_temporal + self.b1.transpose(
            0, 1
        ).unsqueeze(1)

        return self.y1 * x_temporal + self.y2 * x_feature


class DualAttentionLayer(nn.Module):
    """One TLOB transformer block: multi-head self-attention + feed-forward.

    Parameters
    ----------
    dim : int
        Token width for this axis (hidden width for the temporal-axis
        pass, sequence length for the feature-axis pass).
    num_heads : int
        Number of attention heads.
    out_dim : int
        Output width of the feed-forward sub-block.
    """

    def __init__(self, dim: int, num_heads: int, out_dim: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(out_dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attend over the current last axis, then feed-forward.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, tokens, dim)``.

        Returns
        -------
        Tensor
            Shape ``(batch, tokens, out_dim)``.
        """
        res = x
        attended, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(attended + res)
        ff = self.fc2(F.gelu(self.fc1(x)))
        if ff.shape[-1] == x.shape[-1]:
            ff = ff + x
        return self.ff_norm(ff)


class TlobDualAttention(nn.Module):
    """Compact TLOB-style dual (temporal/feature axis) attention transformer.

    Alternates transformer blocks between the temporal axis (tokens =
    LOB snapshots) and the feature axis (tokens = order-book feature
    channels) by transposing between successive blocks, after a
    bi-normalization front end, and finishes with a shrinking MLP head
    that predicts a 3-way price-trend label.

    Parameters
    ----------
    n_features : int
        Number of raw LOB feature channels per snapshot.
    seq_len : int
        Number of LOB snapshots in the window.
    hidden_dim : int
        Model width after the embedding projection.
    num_heads : int
        Attention heads per block.
    num_layers : int
        Number of (temporal, feature) block pairs.
    """

    def __init__(
        self,
        n_features: int = 10,
        seq_len: int = 16,
        hidden_dim: int = 16,
        num_heads: int = 2,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.norm_layer = BiNormalization(n_features, seq_len)
        self.emb_layer = nn.Linear(n_features, hidden_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        self.temporal_layers = nn.ModuleList()
        self.feature_layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                self.temporal_layers.append(DualAttentionLayer(hidden_dim, num_heads, hidden_dim))
                self.feature_layers.append(DualAttentionLayer(seq_len, num_heads, seq_len))
            else:
                self.temporal_layers.append(
                    DualAttentionLayer(hidden_dim, num_heads, hidden_dim // 4)
                )
                self.feature_layers.append(DualAttentionLayer(seq_len, num_heads, seq_len // 4))

        final_dim = (hidden_dim // 4) * (seq_len // 4)
        self.final = nn.Sequential(
            nn.Linear(final_dim, max(final_dim // 4, 8)),
            nn.GELU(),
            nn.Linear(max(final_dim // 4, 8), 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the dual-attention transformer.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, seq_len, n_features)`` raw LOB snapshots.

        Returns
        -------
        Tensor
            Shape ``(batch, 3)`` trend logits.
        """
        x = x.permute(0, 2, 1)
        x = self.norm_layer(x)
        x = x.permute(0, 2, 1)
        x = self.emb_layer(x) + self.pos_encoder

        for temporal_block, feature_block in zip(self.temporal_layers, self.feature_layers):
            x = temporal_block(x)
            x = x.permute(0, 2, 1)
            x = feature_block(x)
            x = x.permute(0, 2, 1)

        x = x.reshape(x.shape[0], -1)
        return self.final(x)


def build_tlob() -> nn.Module:
    """Build a compact TLOB dual-attention transformer.

    Returns
    -------
    nn.Module
        ``TlobDualAttention`` in eval mode.
    """
    torch.manual_seed(0)
    return TlobDualAttention().eval()


def example_input_tlob() -> torch.Tensor:
    """Example input for :func:`build_tlob`.

    Returns
    -------
    Tensor
        Shape ``(2, 16, 10)`` LOB snapshot window.
    """
    torch.manual_seed(0)
    return torch.randn(2, 16, 10)


# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------

MENAGERIE_ENTRIES = [
    ("RANKITECT", "build_rankitect", "example_input_rankitect", "2023", "REC"),
    ("RouteNet-Erlang", "build_routenet_erlang", "example_input_routenet_erlang", "2023", "GRAPH"),
    ("RSR (Relational Stock Ranking)", "build_rsr", "example_input_rsr", "2019", "SEQ"),
    ("SAFE", "build_safe", "example_input_safe", "2019", "SEQ"),
    ("SySeVR", "build_sysevr", "example_input_sysevr", "2021", "SEQ"),
    ("TLOB", "build_tlob", "example_input_tlob", "2025", "SEQ"),
]

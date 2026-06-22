"""Continuous-time dynamic-graph (temporal graph) models from DyGLib / TGB.

Library: "Towards Better Dynamic Graph Learning: New Architecture and Unified Library"
  Yu et al., NeurIPS 2023 (DyGLib).  Paper: https://arxiv.org/abs/2303.13047
  Source: https://github.com/yule-BUAA/DyGLib   (TGB: https://tgb.complexdatalab.com)

All of these are continuous-time dynamic-graph link predictors. The common substrate is:
each event (src, dst, time) is encoded with the FUNCTIONAL TIME ENCODING from TGAT --
phi(t) = cos(t * w + b) with learnable frequencies w -- and each node samples a fixed
window of historical temporal neighbors. The models differ in HOW they aggregate that
neighbor history; that aggregation is the distinctive primitive reproduced here:

  - TGAT       -- temporal graph attention: self-attention over (neighbor feat || time
                  encoding) sequences (https://arxiv.org/abs/2002.07962).
  - TGN        -- a per-node MEMORY (GRU) module updated by message passing, read out by
                  a graph-attention embedding (https://arxiv.org/abs/2006.10637).
  - DyRep      -- temporal point process: a two-time-scale recurrent update where the
                  node state evolves via a recurrent cell plus a temporal-attention
                  localized embedding (https://openreview.net/forum?id=HyePrhR5KX).
  - JODIE      -- coupled user/item RNNs + a projection operator that linearly projects
                  an embedding forward in time by elapsed dt (https://arxiv.org/abs/1908.01207).
  - CAWN       -- causal anonymous walks: sample temporal walks, RELABEL nodes
                  anonymously by their position-in-walk counts, encode walks with an RNN,
                  pool (https://arxiv.org/abs/2101.05974).
  - GraphMixer -- an MLP-Mixer over fixed (non-learned) time-encoded neighbor features:
                  token-mixing MLP across neighbors + channel-mixing MLP
                  (https://arxiv.org/abs/2302.11636).
  - DyGFormer  -- patch the per-node neighbor sequence into patches and run a Transformer;
                  adds a NEIGHBOR CO-OCCURRENCE encoding measuring how often src/dst share
                  neighbors (https://arxiv.org/abs/2303.13047).
  - TCL        -- temporal-dependency-aware Transformer with a graph-topology / temporal
                  contrastive objective (https://arxiv.org/abs/2105.07944).
  - NAT        -- neighborhood-aware dictionary: a per-node hashed dictionary of recent
                  neighbor representations aggregated for the target pair.
  - EdgeBank   -- a NON-PARAMETRIC memory heuristic: predicts a link iff the edge is in a
                  remembered edge set. Implemented here as a buffer-lookup module with no
                  learned weights (kept for completeness; it has no neural forward graph).

Each model is a faithful COMPACT random-init core operating on small node/edge counts.
forward() takes a single long tensor of node ids (shape (1, N)); the model internally
splits it into src / dst / neighbor windows and synthesizes timestamps, so every model is
forward-able from one example tensor for the atlas. Output is a link-prediction score.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- small shared dimensions -----
_NUM_NODES = 64
_NBR = 10  # neighbors sampled per node
_DIM = 32
_TIME_DIM = 32


class TimeEncoder(nn.Module):
    """Functional time encoding phi(t) = cos(t * w + b) (TGAT)."""

    def __init__(self, dim: int = _TIME_DIM) -> None:
        super().__init__()
        self.w = nn.Linear(1, dim)
        # Bochner-style log-spaced initial frequencies.
        self.w.weight = nn.Parameter((1.0 / 10 ** torch.linspace(0, 9, dim)).reshape(dim, 1))
        self.w.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.w(t.unsqueeze(-1)))


class _TemporalBase(nn.Module):
    """Shared scaffold: node embedding table + time encoder + neighbor synthesis.

    Splits the single long input into (src, dst) and synthesizes a fixed window of
    temporal neighbors with monotonically decreasing timestamps, so every subclass can be
    driven by one example tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(_NUM_NODES, _DIM)
        self.time_enc = TimeEncoder(_TIME_DIM)

    def _prepare(self, node_ids: torch.Tensor):
        ids = node_ids.view(-1).remainder(_NUM_NODES)
        src = ids[0::2][:_NBR]
        dst = ids[1::2][:_NBR]

        # pad to _NBR
        def pad(t):
            if t.numel() < _NBR:
                t = (
                    torch.cat([t, t[: _NBR - t.numel()]])
                    if t.numel()
                    else torch.zeros(_NBR, dtype=torch.long)
                )
            return t[:_NBR]

        src, dst = pad(src), pad(dst)
        # neighbor ids: outer mix of src/dst, deterministic
        nbr_src = (src.unsqueeze(1) + torch.arange(_NBR).unsqueeze(0)).remainder(_NUM_NODES)
        nbr_dst = (dst.unsqueeze(1) + torch.arange(_NBR).unsqueeze(0)).remainder(_NUM_NODES)
        # decreasing timestamps (recent -> older)
        times = torch.arange(_NBR, dtype=torch.float32).flip(0)
        return src, dst, nbr_src, nbr_dst, times


# ============================================================
# TGAT -- temporal graph attention
# ============================================================


class TGAT(_TemporalBase):
    """Self-attention over (neighbor feature || time encoding) sequences."""

    def __init__(self) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(_DIM + _TIME_DIM, _DIM)
        self.attn = nn.MultiheadAttention(_DIM, 2, batch_first=True)
        self.merge = nn.Linear(_DIM * 2, _DIM)
        self.out = nn.Linear(_DIM * 2, 1)

    def _embed(self, center, nbr, times):
        te = self.time_enc(times).unsqueeze(0).expand(_NBR, -1, -1)  # (N,N,T)
        nbr_feat = self.node_emb(nbr)  # (N, Nbr, D)
        seq = self.feat_proj(torch.cat([nbr_feat, te], dim=-1))  # (N, Nbr, D)
        q = self.node_emb(center).unsqueeze(1)  # (N,1,D)
        att, _ = self.attn(q, seq, seq)
        return self.merge(torch.cat([self.node_emb(center), att.squeeze(1)], dim=-1))

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        src, dst, nbr_src, nbr_dst, times = self._prepare(node_ids)
        h_src = self._embed(src, nbr_src, times)
        h_dst = self._embed(dst, nbr_dst, times)
        return self.out(torch.cat([h_src, h_dst], dim=-1))


# ============================================================
# TGN -- memory (GRU) module + attention embedding
# ============================================================


class TGN(_TemporalBase):
    """Per-node GRU memory updated by messages, read by an attention embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.msg_fn = nn.Linear(_DIM * 2 + _TIME_DIM, _DIM)
        self.memory_gru = nn.GRUCell(_DIM, _DIM)
        self.embed_proj = nn.Linear(_DIM + _TIME_DIM, _DIM)
        self.attn = nn.MultiheadAttention(_DIM, 2, batch_first=True)
        self.out = nn.Linear(_DIM * 2, 1)

    def _update_memory(self, center, nbr, times):
        prev = self.node_emb(center)  # (N, D) as previous memory
        te = self.time_enc(times)  # (Nbr, T)
        nbr_feat = self.node_emb(nbr)  # (N, Nbr, D)
        # aggregate messages from neighbors
        msg_in = torch.cat(
            [
                prev.unsqueeze(1).expand(-1, _NBR, -1),
                nbr_feat,
                te.unsqueeze(0).expand(_NBR, -1, -1),
            ],
            dim=-1,
        )
        msg = self.msg_fn(msg_in).mean(dim=1)  # (N, D)
        return self.memory_gru(msg, prev)

    def _embed(self, mem, nbr, times):
        te = self.time_enc(times).unsqueeze(0).expand(_NBR, -1, -1)
        nbr_feat = self.node_emb(nbr)
        seq = self.embed_proj(torch.cat([nbr_feat, te], dim=-1))
        att, _ = self.attn(mem.unsqueeze(1), seq, seq)
        return att.squeeze(1)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        src, dst, nbr_src, nbr_dst, times = self._prepare(node_ids)
        m_src = self._update_memory(src, nbr_src, times)
        m_dst = self._update_memory(dst, nbr_dst, times)
        e_src = self._embed(m_src, nbr_src, times)
        e_dst = self._embed(m_dst, nbr_dst, times)
        return self.out(torch.cat([e_src, e_dst], dim=-1))


# ============================================================
# DyRep -- temporal point process, two-time-scale recurrent update
# ============================================================


class DyRep(_TemporalBase):
    """Two-time-scale recurrent node update + localized temporal-attention embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.recurrent = nn.GRUCell(_DIM + _TIME_DIM, _DIM)
        self.localized_attn = nn.Linear(_DIM + _TIME_DIM, 1)
        self.intensity = nn.Linear(_DIM * 2, 1)  # point-process intensity

    def _evolve(self, center, nbr, times):
        prev = self.node_emb(center)
        te = self.time_enc(times)  # (Nbr, T)
        nbr_feat = self.node_emb(nbr)  # (N, Nbr, D)
        ctx = torch.cat([nbr_feat, te.unsqueeze(0).expand(_NBR, -1, -1)], dim=-1)
        # localized temporal attention over neighbors
        a = torch.softmax(self.localized_attn(ctx), dim=1)  # (N, Nbr, 1)
        agg = (a * nbr_feat).sum(dim=1)  # (N, D)
        rec_in = torch.cat([agg, te.mean(dim=0, keepdim=True).expand(prev.size(0), -1)], dim=-1)
        return self.recurrent(rec_in, prev)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        src, dst, nbr_src, nbr_dst, times = self._prepare(node_ids)
        h_src = self._evolve(src, nbr_src, times)
        h_dst = self._evolve(dst, nbr_dst, times)
        # conditional intensity of the (src,dst) event = point-process score
        return self.intensity(torch.cat([h_src, h_dst], dim=-1))


# ============================================================
# JODIE -- coupled RNNs + temporal projection
# ============================================================


class JODIE(_TemporalBase):
    """Coupled user/item RNNs + linear time-projection of embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.user_rnn = nn.RNNCell(_DIM + _TIME_DIM, _DIM)
        self.item_rnn = nn.RNNCell(_DIM + _TIME_DIM, _DIM)
        # projection operator: scales embedding by (1 + W * dt)
        self.project = nn.Linear(1, _DIM)
        self.out = nn.Linear(_DIM * 2, 1)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        src, dst, nbr_src, nbr_dst, times = self._prepare(node_ids)
        te = self.time_enc(times).mean(dim=0, keepdim=True)  # (1, T)
        u = self.node_emb(src)  # users
        v = self.node_emb(dst)  # items
        # coupled update: user state uses item state and vice versa
        u_next = self.user_rnn(torch.cat([v, te.expand(u.size(0), -1)], dim=-1), u)
        v_next = self.item_rnn(torch.cat([u, te.expand(v.size(0), -1)], dim=-1), v)
        # JODIE projection: project user embedding forward by elapsed dt
        dt = times[:1].view(1, 1).expand(u.size(0), 1)
        u_proj = u_next * (1.0 + self.project(dt))
        return self.out(torch.cat([u_proj, v_next], dim=-1))


# ============================================================
# CAWN -- causal anonymous walks
# ============================================================


class CAWN(_TemporalBase):
    """Causal anonymous walks: anonymized positional encoding + walk RNN + pooling."""

    def __init__(self, walk_len: int = 4) -> None:
        super().__init__()
        self.walk_len = walk_len
        # anonymous relabeling: position-count one-hot of size walk_len
        self.pos_enc = nn.Linear(walk_len, _DIM)
        self.walk_rnn = nn.GRU(_DIM + _TIME_DIM, _DIM, batch_first=True)
        self.out = nn.Linear(_DIM * 2, 1)

    def _encode_walks(self, center, nbr, times):
        # build walk_len-step walks per node from the neighbor window
        n = center.size(0)
        steps = []
        for k in range(self.walk_len):
            node_k = nbr[:, k % _NBR]  # (N,)
            anon = F.one_hot(
                torch.full((n,), k, dtype=torch.long), num_classes=self.walk_len
            ).float()
            pe = self.pos_enc(anon)  # anonymous positional id
            feat = self.node_emb(node_k) + pe  # (N, D)
            te = self.time_enc(times[k % _NBR].view(1)).expand(n, -1)  # (N, T)
            steps.append(torch.cat([feat, te], dim=-1))
        seq = torch.stack(steps, dim=1)  # (N, walk_len, D+T)
        _, h = self.walk_rnn(seq)
        return h.squeeze(0)  # (N, D)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        src, dst, nbr_src, nbr_dst, times = self._prepare(node_ids)
        h_src = self._encode_walks(src, nbr_src, times)
        h_dst = self._encode_walks(dst, nbr_dst, times)
        return self.out(torch.cat([h_src, h_dst], dim=-1))


# ============================================================
# GraphMixer -- MLP-Mixer over fixed time-encoded neighbors
# ============================================================


class _FixedTimeEncoder(nn.Module):
    """NON-learnable cos time encoding (GraphMixer fixes the time encoder)."""

    def __init__(self, dim: int = _TIME_DIM) -> None:
        super().__init__()
        freqs = 1.0 / 10 ** torch.linspace(0, 9, dim)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(t.unsqueeze(-1) * self.freqs)


class GraphMixer(_TemporalBase):
    """MLP-Mixer: token-mixing across neighbors + channel-mixing MLP (fixed time enc)."""

    def __init__(self) -> None:
        super().__init__()
        self.time_enc = _FixedTimeEncoder(_TIME_DIM)  # FIXED, not learned
        self.in_proj = nn.Linear(_DIM + _TIME_DIM, _DIM)
        # token-mixing MLP (mixes across the _NBR neighbor tokens)
        self.token_mlp = nn.Sequential(
            nn.Linear(_NBR, _NBR * 2), nn.GELU(), nn.Linear(_NBR * 2, _NBR)
        )
        # channel-mixing MLP (mixes across feature channels)
        self.channel_mlp = nn.Sequential(
            nn.Linear(_DIM, _DIM * 2), nn.GELU(), nn.Linear(_DIM * 2, _DIM)
        )
        self.norm1 = nn.LayerNorm(_DIM)
        self.norm2 = nn.LayerNorm(_DIM)
        self.out = nn.Linear(_DIM * 2, 1)

    def _mix(self, nbr, times):
        te = self.time_enc(times).unsqueeze(0).expand(_NBR, -1, -1)  # (N,Nbr,T)
        x = self.in_proj(torch.cat([self.node_emb(nbr), te], dim=-1))  # (N, Nbr, D)
        # token mixing
        y = self.norm1(x).transpose(1, 2)  # (N, D, Nbr)
        x = x + self.token_mlp(y).transpose(1, 2)
        # channel mixing
        x = x + self.channel_mlp(self.norm2(x))
        return x.mean(dim=1)  # pool over neighbors -> (N, D)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        src, dst, nbr_src, nbr_dst, times = self._prepare(node_ids)
        return self.out(torch.cat([self._mix(nbr_src, times), self._mix(nbr_dst, times)], dim=-1))


# ============================================================
# DyGFormer -- patching transformer + neighbor co-occurrence encoding
# ============================================================


class DyGFormer(_TemporalBase):
    """Patch the neighbor sequence, add co-occurrence encoding, run a Transformer."""

    def __init__(self, patch_size: int = 2) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.cooc_enc = nn.Linear(1, _DIM)  # neighbor co-occurrence encoding
        in_dim = (_DIM + _TIME_DIM + _DIM) * patch_size
        self.patch_proj = nn.Linear(in_dim, _DIM)
        enc_layer = nn.TransformerEncoderLayer(_DIM, 2, _DIM * 2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.out = nn.Linear(_DIM * 2, 1)

    def _cooccurrence(self, nbr_a, nbr_b):
        # count, for each neighbor of A, how many times it appears in B's neighbors.
        # (N, Nbr) counts -> co-occurrence feature.
        eq = (nbr_a.unsqueeze(2) == nbr_b.unsqueeze(1)).float()  # (N, Nbr, Nbr)
        return eq.sum(dim=2, keepdim=True)  # (N, Nbr, 1)

    def _encode(self, nbr, times, cooc):
        te = self.time_enc(times).unsqueeze(0).expand(_NBR, -1, -1)  # (N,Nbr,T)
        feat = self.node_emb(nbr)  # (N, Nbr, D)
        cooc_feat = self.cooc_enc(cooc)  # (N, Nbr, D)
        x = torch.cat([feat, te, cooc_feat], dim=-1)  # (N, Nbr, D+T+D)
        # patching: group consecutive neighbors into patches
        n, nbr_n, c = x.shape
        x = x.view(n, nbr_n // self.patch_size, self.patch_size * c)
        x = self.patch_proj(x)  # (N, num_patches, D)
        x = self.transformer(x)
        return x.mean(dim=1)  # (N, D)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        src, dst, nbr_src, nbr_dst, times = self._prepare(node_ids)
        cooc_s = self._cooccurrence(nbr_src, nbr_dst)
        cooc_d = self._cooccurrence(nbr_dst, nbr_src)
        h_src = self._encode(nbr_src, times, cooc_s)
        h_dst = self._encode(nbr_dst, times, cooc_d)
        return self.out(torch.cat([h_src, h_dst], dim=-1))


# ============================================================
# TCL -- temporal-dependency-aware Transformer
# ============================================================


class TCL(_TemporalBase):
    """Temporal-dependency Transformer with a cross-attention pair interaction (TCL)."""

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(_DIM + _TIME_DIM, _DIM)
        enc_layer = nn.TransformerEncoderLayer(_DIM, 2, _DIM * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.cross = nn.MultiheadAttention(_DIM, 2, batch_first=True)
        self.proj_head = nn.Linear(_DIM, _DIM)  # contrastive projection head
        self.out = nn.Linear(_DIM * 2, 1)

    def _seq(self, nbr, times):
        te = self.time_enc(times).unsqueeze(0).expand(_NBR, -1, -1)
        x = self.in_proj(torch.cat([self.node_emb(nbr), te], dim=-1))
        return self.encoder(x)  # (N, Nbr, D)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        src, dst, nbr_src, nbr_dst, times = self._prepare(node_ids)
        s_seq = self._seq(nbr_src, times)
        d_seq = self._seq(nbr_dst, times)
        # cross-attention between the two interaction histories
        s_att, _ = self.cross(s_seq, d_seq, d_seq)
        d_att, _ = self.cross(d_seq, s_seq, s_seq)
        s = self.proj_head(s_att.mean(dim=1))
        d = self.proj_head(d_att.mean(dim=1))
        return self.out(torch.cat([s, d], dim=-1))


# ============================================================
# NAT -- neighborhood-aware dictionary
# ============================================================


class NAT(_TemporalBase):
    """Neighborhood-aware dictionary: per-node cached neighbor representations + GRU."""

    def __init__(self, dict_size: int = 8) -> None:
        super().__init__()
        self.dict_size = dict_size
        # a learned dictionary of neighbor-slot representations (hashed by neighbor id)
        self.dictionary = nn.Embedding(dict_size, _DIM)
        self.slot_gru = nn.GRUCell(_DIM + _TIME_DIM, _DIM)
        self.aggr = nn.Linear(_DIM, _DIM)
        self.out = nn.Linear(_DIM * 2, 1)

    def _dict_repr(self, center, nbr, times):
        te = self.time_enc(times)  # (Nbr, T)
        # hash neighbor ids into dictionary slots
        slots = nbr.remainder(self.dict_size)  # (N, Nbr)
        slot_feat = self.dictionary(slots)  # (N, Nbr, D)
        nbr_feat = self.node_emb(nbr)  # (N, Nbr, D)
        merged = slot_feat + nbr_feat  # neighbor-aware cached representation
        # update via GRU over the cached slots
        h = self.node_emb(center)
        for k in range(_NBR):
            inp = torch.cat([merged[:, k], te[k].unsqueeze(0).expand(h.size(0), -1)], dim=-1)
            h = self.slot_gru(inp, h)
        return self.aggr(h)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        src, dst, nbr_src, nbr_dst, times = self._prepare(node_ids)
        h_src = self._dict_repr(src, nbr_src, times)
        h_dst = self._dict_repr(dst, nbr_dst, times)
        return self.out(torch.cat([h_src, h_dst], dim=-1))


# ============================================================
# EdgeBank -- non-parametric memory heuristic
# ============================================================


class EdgeBank(nn.Module):
    """Non-parametric temporal edge-memory heuristic: predicts a link iff remembered.

    EdgeBank has NO trainable weights -- it stores seen (src,dst) edges in a memory buffer
    and predicts positive iff the queried edge is in memory. This compact module keeps a
    fixed random memory mask buffer so it produces a forward graph (lookup + compare),
    faithfully reflecting that EdgeBank is a pure memory baseline, not a neural network.
    """

    def __init__(self, num_nodes: int = _NUM_NODES) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        # remembered-edge memory as a fixed boolean buffer (no gradients).
        mem = (torch.rand(num_nodes, num_nodes) > 0.5).float()
        self.register_buffer("memory", mem)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        ids = node_ids.view(-1).remainder(self.num_nodes)
        src = ids[0::2][:_NBR]
        dst = ids[1::2][:_NBR]
        n = min(src.numel(), dst.numel())
        src, dst = src[:n], dst[:n]
        # memory lookup: is edge (src,dst) remembered?
        scores = self.memory[src, dst].unsqueeze(-1)  # (n, 1)
        return scores


# ============================================================
# Menagerie wiring
# ============================================================


def _example_edges() -> torch.Tensor:
    """Example node-id stream ``(1, 100)`` (long) for a temporal-graph link query."""
    return torch.arange(100, dtype=torch.long).remainder(_NUM_NODES).unsqueeze(0)


def example_input() -> torch.Tensor:
    """Example temporal-graph event stream ``(1, 100)`` long node ids."""
    return _example_edges()


def build_tgat() -> nn.Module:
    """Build TGAT (temporal graph attention + functional time encoding)."""
    return TGAT().eval()


def build_tgn() -> nn.Module:
    """Build TGN (GRU memory module + attention embedding)."""
    return TGN().eval()


def build_dyrep() -> nn.Module:
    """Build DyRep (temporal point process, two-time-scale recurrent update)."""
    return DyRep().eval()


def build_jodie() -> nn.Module:
    """Build JODIE (coupled user/item RNNs + temporal projection)."""
    return JODIE().eval()


def build_cawn() -> nn.Module:
    """Build CAWN (causal anonymous walks + walk RNN)."""
    return CAWN().eval()


def build_graphmixer() -> nn.Module:
    """Build GraphMixer (MLP-Mixer over fixed time-encoded neighbors)."""
    return GraphMixer().eval()


def build_dygformer() -> nn.Module:
    """Build DyGFormer (patching transformer + neighbor co-occurrence encoding)."""
    return DyGFormer().eval()


def build_tcl() -> nn.Module:
    """Build TCL (temporal-dependency Transformer + cross-attention)."""
    return TCL().eval()


def build_nat() -> nn.Module:
    """Build NAT (neighborhood-aware dictionary + GRU aggregation)."""
    return NAT().eval()


def build_edgebank() -> nn.Module:
    """Build EdgeBank (non-parametric temporal edge-memory heuristic)."""
    return EdgeBank().eval()


MENAGERIE_ENTRIES = [
    (
        "TGAT (temporal graph attention + functional time encoding)",
        "build_tgat",
        "example_input",
        "2020",
        "DC",
    ),
    ("TGN (memory GRU module + attention embedding)", "build_tgn", "example_input", "2020", "DC"),
    (
        "DyRep (temporal point-process two-time-scale recurrence)",
        "build_dyrep",
        "example_input",
        "2019",
        "DC",
    ),
    (
        "JODIE (coupled user/item RNNs + temporal projection)",
        "build_jodie",
        "example_input",
        "2019",
        "DC",
    ),
    ("CAWN (causal anonymous walks + walk RNN)", "build_cawn", "example_input", "2021", "DC"),
    (
        "GraphMixer (MLP-Mixer over fixed time-encoded neighbors)",
        "build_graphmixer",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "DyGFormer (patching transformer + neighbor co-occurrence)",
        "build_dygformer",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "TCL (temporal-dependency transformer + cross-attention)",
        "build_tcl",
        "example_input",
        "2021",
        "DC",
    ),
    ("NAT (neighborhood-aware dictionary + GRU)", "build_nat", "example_input", "2022", "DC"),
    (
        "EdgeBank (non-parametric temporal edge-memory heuristic)",
        "build_edgebank",
        "example_input",
        "2022",
        "DC",
    ),
]

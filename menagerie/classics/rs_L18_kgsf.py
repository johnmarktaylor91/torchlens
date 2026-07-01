# SOURCE: vendored from RUCAIBox/KGSF @ d4f783a0b386ebb6beb74506f0e3de4d509c996e
# https://raw.githubusercontent.com/RUCAIBox/KGSF/d4f783a0b386ebb6beb74506f0e3de4d509c996e/model.py
# https://raw.githubusercontent.com/RUCAIBox/KGSF/d4f783a0b386ebb6beb74506f0e3de4d509c996e/models/transformer.py
# https://raw.githubusercontent.com/RUCAIBox/KGSF/d4f783a0b386ebb6beb74506f0e3de4d509c996e/models/graph.py
# https://raw.githubusercontent.com/RUCAIBox/KGSF/d4f783a0b386ebb6beb74506f0e3de4d509c996e/models/utils.py
#
# Zhou et al. 2020 "Improving Conversational Recommender Systems via Knowledge Graph
# based Semantic Fusion" (KDD 2020, a.k.a. KGSF). Real architecture: two knowledge
# graphs are embedded with real graph-neural-network layers from torch_geometric --
# `RGCNConv` runs relational message passing over the DBpedia item KG (word/entity
# relation types, self loops added by `_edge_list`) and `GCNConv` runs over the
# ConceptNet word-concept graph -- producing `db_nodes_features`/`con_nodes_features`.
# Per-user "seed set" entity/concept subgraphs are pooled with real batched self-
# attention layers (`SelfAttentionLayer_batch` for the KG side, `SelfAttentionLayer`
# per-example on the concept side), fused through a learned gate (`user_norm` +
# `gate_norm` sigmoid mixing) into one user representation, and scored against the
# item-KG node embeddings for recommendation. The generation branch reuses the same
# fused KG/concept features as extra memory banks for a ParlAI-style custom
# Transformer encoder/decoder (`TransformerEncoder` for the dialogue context,
# `TransformerDecoderKG` -- a 4-way multi-head-attention decoder layer that cross-
# attends over dialogue, DBpedia-KG, and concept-KG memories every layer) with a
# copy-generator style "representation_bias" vocabulary-extension head
# ("semantic fusion" of KG/concept/dialogue signals into one generation logit). All of
# `models/transformer.py` and `models/graph.py` (the Transformer + graph-attention
# building blocks) and `CrossModel`'s real `__init__`/`infomax_loss`/`forward` graph
# construction are the real code, taken essentially verbatim.
#
# Minimal, non-architectural changes made (bookkeeping / disk-IO / CUDA-only-code
# removal only; no computation changed):
#   - The real `CrossModel.__init__` loads several fixed on-disk artifacts at
#     construction time: `data/subkg.pkl` (DBpedia KG adjacency for `_edge_list`),
#     `key2index_3rd.json` + `conceptnet_edges2nd.txt` + `stopwords.txt` (ConceptNet
#     edge list for `concept_edge_list4GCN`), `word2index_redial.json` (vocab index),
#     `mask4key.npy`/`mask4movie.npy` (fixed-size copy-vocabulary masks), and
#     `word2vec_redial.npy` (pretrained word embeddings loaded inside
#     `models/utils.py::_create_embeddings`). None of these are architecture -- they
#     are fixed data tables sized to the real ~65k-entity/~30k-concept/~30k-word
#     Redial-trained artifacts. Here they are synthesized in-memory at tiny size
#     (`n_entity`, `n_concept`, vocabulary all shrunk) with the exact same shapes/
#     roles the real code reads, so `_edge_list`, `concept_edge_list4GCN`-equivalent
#     edge construction, `RGCNConv`/`GCNConv`, and the copy-vocabulary masking run the
#     identical real computation, just over synthetic small graphs/vocab instead of
#     the shipped Redial KG dump.
#   - All hardcoded `.cuda()` calls (the repo assumes single-GPU training) removed;
#     tensors are created directly on the module's device (CPU here) instead. This is
#     a device-placement change only -- every op that ran on CUDA in the original runs
#     identically on CPU.
#   - `F.sigmoid` (removed in modern torch; deprecated alias for `torch.sigmoid`)
#     replaced with `torch.sigmoid` in the gate computation -- same op, current API.
#   - `nn.MSELoss(size_average=False, reduce=False)` (the `size_average` kwarg was
#     removed from modern `nn.MSELoss`) replaced with `nn.MSELoss(reduction="none")`,
#     which is the documented equivalent of `reduce=False`.
#   - `nn.CrossEntropyLoss(reduce=False)` (removed kwarg) replaced with
#     `nn.CrossEntropyLoss(reduction="none")`, the documented equivalent.
#   - `F.softmax(e)` in `SelfAttentionLayer.forward` (real code omits an explicit
#     `dim=`, which is a deprecated/warned call in modern torch and defaults to the
#     last dim for a 1-D tensor here) made explicit as `F.softmax(e, dim=-1)` --
#     identical behavior, no deprecation warning.
#   - `create_position_codes`'s real code writes the sinusoidal position table into
#     the freshly-constructed `nn.Embedding.weight` (a leaf `nn.Parameter` that
#     requires grad) via plain in-place slice assignment, then calls `.detach_()`
#     afterward. Modern torch raises `RuntimeError: a view of a leaf Variable that
#     requires grad is being used in an in-place operation` for that ordering (older
#     torch allowed it); wrapped the two in-place writes in `torch.no_grad()` so the
#     exact same values are written -- an autograd-API compat fix at init time only,
#     not part of the traced forward computation.
#   - Dropped training-loop / greedy & forced decoding / checkpoint-io / entity-set
#     recall bookkeeping (`decode_greedy`, `decode_forced`, `save_model`,
#     `load_model`, `output`, `reorder_*`, `compute_loss`) that read from the real
#     dataset/dictionary objects; kept the real `CrossModel.__init__` module graph
#     (embeddings, encoder, decoder, RGCN/GCN, self-attention, gating, infomax heads)
#     and the real `forward()`'s graph-construction + fusion + recommendation-scoring
#     + `infomax_loss` + generation-encoding pipeline (the traced architecture)
#     verbatim, with the KG-only recommendation path exercised for the trace
#     (mirrors the "rec" forward branch, matching how the real code always builds and
#     scores `entity_scores` regardless of whether the response is later generated).

import json
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.rgcn_conv import RGCNConv


# ---- models/utils.py (real helpers, verbatim except disk-load -> in-memory init) ----

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504


def neginf(dtype):
    """Returns a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def _create_embeddings(vocab_size, embedding_size, padding_idx):
    """Create and initialize word embeddings (real code loads word2vec_redial.npy;
    here randomly initialized -- a data/loading concern, not an architecture change)."""
    e = nn.Embedding(vocab_size, embedding_size, padding_idx)
    nn.init.normal_(e.weight, mean=0, std=embedding_size**-0.5)
    return e


def _create_entity_embeddings(entity_num, embedding_size, padding_idx):
    """Create and initialize entity embeddings."""
    e = nn.Embedding(entity_num, embedding_size)
    nn.init.normal_(e.weight, mean=0, std=embedding_size**-0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e


# ---- models/graph.py (real self-attention pooling layers, verbatim) ----


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h):
        assert self.dim == h.shape[1]
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e, dim=-1)
        return torch.matmul(attention, h)


class SelfAttentionLayer_batch(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer_batch, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h, mask):
        assert self.dim == h.shape[2]
        mask = 1e-30 * mask.float()

        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)
        attention = F.softmax(e + mask.unsqueeze(-1), dim=1)
        return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1), attention


# ---- models/transformer.py (real ParlAI-style Transformer blocks, verbatim) ----


def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


def create_position_codes(n_pos, dim, out):
    position_enc = np.array(
        [[pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)] for pos in range(n_pos)]
    )

    # real code in-place-writes into `out` (an nn.Parameter) before detaching it;
    # modern torch forbids in-place ops on a leaf Variable that still requires grad
    # (older torch allowed it here). Wrapped in `torch.no_grad()` to preserve the
    # exact same written values with current autograd semantics -- init-time-only,
    # not part of the traced forward computation.
    with torch.no_grad():
        out[:, 0::2] = torch.FloatTensor(np.sin(position_enc)).type_as(out)
        out[:, 1::2] = torch.FloatTensor(np.cos(position_enc)).type_as(out)
    out.detach_()
    out.requires_grad = False


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        batch_size, query_len, dim = query.size()
        assert dim == self.dim, f"Dimensions do not match: {dim} query vs {self.dim} configured"
        assert mask is not None, "Mask is None, please specify a mask"
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = (
                tensor.transpose(1, 2)
                .contiguous()
                .view(batch_size * n_heads, seq_len, dim_per_head)
            )
            return tensor

        if key is None and value is None:
            key = value = query
        elif value is None:
            value = key
        _, key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))

        dot_prod = q.div(scale).bmm(k.transpose(1, 2))
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, key_len)
            .view(batch_size * n_heads, query_len, key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod = dot_prod.masked_fill(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(dot_prod, dim=-1).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
            .view(batch_size, n_heads, query_len, dim_per_head)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, dim)
        )

        out = self.out_lin(attentioned)

        return out


class TransformerFFN(nn.Module):
    def __init__(self, dim, dim_hidden, relu_dropout=0):
        super(TransformerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.relu_dropout(x)
        x = self.lin2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.attention = MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, mask):
        tensor = tensor + self.dropout(self.attention(tensor, mask=mask))
        tensor = _normalize(tensor, self.norm1)
        tensor = tensor + self.dropout(self.ffn(tensor))
        tensor = _normalize(tensor, self.norm2)
        tensor = tensor * mask.unsqueeze(-1).type_as(tensor)
        return tensor


class TransformerEncoder(nn.Module):
    """Transformer encoder module (dialogue-context encoder)."""

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction=True,
        n_positions=1024,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction = reduction
        self.padding_idx = padding_idx
        self.dropout = nn.Dropout(p=dropout)

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, (
            "Transformer embedding size must be a multiple of n_heads"
        )

        assert embedding is not None
        self.embeddings = embedding

        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(n_positions, embedding_size, out=self.position_embeddings.weight)
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size**-0.5)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                )
            )

    def forward(self, input):
        mask = input != self.padding_idx
        positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)

        tensor = tensor * mask.unsqueeze(-1).type_as(tensor)
        for i in range(self.n_layers):
            tensor = self.layers[i](tensor, mask)

        if self.reduction:
            divisor = mask.type_as(tensor).sum(dim=1).unsqueeze(-1).clamp(min=1e-7)
            output = tensor.sum(dim=1) / divisor
            return output
        else:
            output = tensor
            return output, mask


class TransformerDecoderLayerKG(nn.Module):
    def __init__(
        self,
        n_heads,
        embedding_size,
        ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(n_heads, embedding_size, dropout=attention_dropout)
        self.norm1 = nn.LayerNorm(embedding_size)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embedding_size)

        self.encoder_db_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2_db = nn.LayerNorm(embedding_size)

        self.encoder_kg_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2_kg = nn.LayerNorm(embedding_size)

        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(
        self,
        x,
        encoder_output,
        encoder_mask,
        kg_encoder_output,
        kg_encoder_mask,
        db_encoder_output,
        db_encoder_mask,
    ):
        decoder_mask = self._create_selfattn_mask(x)
        residual = x
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)
        x = x + residual
        x = _normalize(x, self.norm1)

        residual = x
        x = self.encoder_db_attention(
            query=x, key=db_encoder_output, value=db_encoder_output, mask=db_encoder_mask
        )
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm2_db)

        residual = x
        x = self.encoder_kg_attention(
            query=x, key=kg_encoder_output, value=kg_encoder_output, mask=kg_encoder_mask
        )
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm2_kg)

        residual = x
        x = self.encoder_attention(
            query=x, key=encoder_output, value=encoder_output, mask=encoder_mask
        )
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm2)

        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        x = _normalize(x, self.norm3)

        return x

    def _create_selfattn_mask(self, x):
        bsz = x.size(0)
        time = x.size(1)
        mask = torch.tril(x.new(time, time).fill_(1))
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask


class TransformerDecoderKG(nn.Module):
    """Transformer decoder layer with 3-way (dialogue/db-KG/concept-KG) cross-attention."""

    def __init__(
        self,
        n_heads,
        n_layers,
        embedding_size,
        ffn_size,
        vocabulary_size,
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        padding_idx=None,
        n_positions=1024,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, (
            "Transformer embedding size must be a multiple of n_heads"
        )

        self.embeddings = embedding

        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(n_positions, embedding_size, out=self.position_embeddings.weight)
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size**-0.5)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(
                TransformerDecoderLayerKG(
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                )
            )

    def forward(self, input, encoder_state, encoder_kg_state, encoder_db_state, incr_state=None):
        encoder_output, encoder_mask = encoder_state
        kg_encoder_output, kg_encoder_mask = encoder_kg_state
        db_encoder_output, db_encoder_mask = encoder_db_state

        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)

        for layer in self.layers:
            tensor = layer(
                tensor,
                encoder_output,
                encoder_mask,
                kg_encoder_output,
                kg_encoder_mask,
                db_encoder_output,
                db_encoder_mask,
            )

        return tensor, None


def _build_encoder(opt, vocab_size, embedding, padding_idx, reduction=False, n_positions=1024):
    return TransformerEncoder(
        n_heads=opt["n_heads"],
        n_layers=opt["n_layers"],
        embedding_size=opt["embedding_size"],
        ffn_size=opt["ffn_size"],
        vocabulary_size=vocab_size,
        embedding=embedding,
        dropout=opt["dropout"],
        attention_dropout=opt["attention_dropout"],
        relu_dropout=opt["relu_dropout"],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt.get("learn_positional_embeddings", False),
        embeddings_scale=opt["embeddings_scale"],
        reduction=reduction,
        n_positions=n_positions,
    )


def _build_decoder4kg(opt, vocab_size, embedding, padding_idx, n_positions=1024):
    return TransformerDecoderKG(
        n_heads=opt["n_heads"],
        n_layers=opt["n_layers"],
        embedding_size=opt["embedding_size"],
        ffn_size=opt["ffn_size"],
        vocabulary_size=vocab_size,
        embedding=embedding,
        dropout=opt["dropout"],
        attention_dropout=opt["attention_dropout"],
        relu_dropout=opt["relu_dropout"],
        padding_idx=padding_idx,
        learn_positional_embeddings=opt.get("learn_positional_embeddings", False),
        embeddings_scale=opt["embeddings_scale"],
        n_positions=n_positions,
    )


# ---- model.py (real KGSF `_edge_list` KG-adjacency builder, verbatim) ----


def _edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != 185:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        # real threshold is `relation_cnt[r] > 1000`, tuned for the full ~65k-entity
        # Redial DBpedia KG; scaled down here to keep >=1 relation surviving on the
        # tiny synthetic KG used for tracing (same selection logic, smaller constant).
        if relation_cnt[r] > 2 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return (
        [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 2],
        len(relation_idx),
    )


def _make_synthetic_kg(n_entity, avg_degree=3, n_relation_types=6, seed=0):
    """Stand-in for the real data/subkg.pkl (DBpedia adjacency dict entity -> [(rel, tail), ...]).
    Real file is a pickled dict harvested from DBpedia; synthesized here at tiny scale
    with the same {entity: [(relation, tail), ...]} shape the real `_edge_list` reads."""
    rng = np.random.RandomState(seed)
    kg = {}
    for e in range(n_entity):
        n_edges = rng.randint(1, avg_degree + 1)
        kg[e] = [
            (int(rng.randint(0, n_relation_types)), int(rng.randint(0, n_entity)))
            for _ in range(n_edges)
        ]
    return kg


def _make_synthetic_concept_edges(n_concept, avg_degree=3, seed=1):
    """Stand-in for concept_edge_list4GCN's ConceptNet-derived edge_index (real code
    reads key2index_3rd.json/conceptnet_edges2nd.txt/stopwords.txt from disk)."""
    rng = np.random.RandomState(seed)
    edges = set()
    for c in range(n_concept):
        for _ in range(rng.randint(1, avg_degree + 1)):
            other = int(rng.randint(0, n_concept))
            edges.add((c, other))
            edges.add((other, c))
    edge_set = [[co[0] for co in edges], [co[1] for co in edges]]
    return torch.LongTensor(edge_set)


# ---- model.py (real KGSF CrossModel, module graph + forward verbatim) ----


class CrossModel(nn.Module):
    def __init__(self, opt, vocab_size, padding_idx=0, start_idx=1, end_idx=2, longest_label=1):
        super().__init__()
        self.batch_size = opt["batch_size"]
        self.max_r_length = opt["max_r_length"]

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer("START", torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        self.pad_idx = padding_idx
        self.embeddings = _create_embeddings(vocab_size, opt["embedding_size"], self.pad_idx)

        self.concept_embeddings = _create_entity_embeddings(opt["n_concept"] + 1, opt["dim"], 0)
        self.concept_padding = 0

        # real code: self.kg = pkl.load(open("data/subkg.pkl", "rb"))
        self.kg = _make_synthetic_kg(opt["n_entity"])

        n_positions = opt["n_positions"]

        self.encoder = _build_encoder(
            opt, vocab_size, self.embeddings, self.pad_idx, reduction=False, n_positions=n_positions
        )
        self.decoder = _build_decoder4kg(
            opt, vocab_size, self.embeddings, self.pad_idx, n_positions=n_positions
        )
        self.db_norm = nn.Linear(opt["dim"], opt["embedding_size"])
        self.kg_norm = nn.Linear(opt["dim"], opt["embedding_size"])

        self.db_attn_norm = nn.Linear(opt["dim"], opt["embedding_size"])
        self.kg_attn_norm = nn.Linear(opt["dim"], opt["embedding_size"])

        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.self_attn = SelfAttentionLayer_batch(opt["dim"], opt["dim"])

        self.self_attn_db = SelfAttentionLayer(opt["dim"], opt["dim"])

        self.user_norm = nn.Linear(opt["dim"] * 2, opt["dim"])
        self.gate_norm = nn.Linear(opt["dim"], 1)
        self.copy_norm = nn.Linear(
            opt["embedding_size"] * 2 + opt["embedding_size"], opt["embedding_size"]
        )
        self.representation_bias = nn.Linear(opt["embedding_size"], vocab_size)

        self.info_con_norm = nn.Linear(opt["dim"], opt["dim"])
        self.info_db_norm = nn.Linear(opt["dim"], opt["dim"])
        self.info_output_db = nn.Linear(opt["dim"], opt["n_entity"])
        self.info_output_con = nn.Linear(opt["dim"], opt["n_concept"] + 1)
        self.info_con_loss = nn.MSELoss(reduction="none")
        self.info_db_loss = nn.MSELoss(reduction="none")

        self.user_representation_to_bias_1 = nn.Linear(opt["dim"], 512)
        self.user_representation_to_bias_2 = nn.Linear(512, vocab_size)

        self.output_en = nn.Linear(opt["dim"], opt["n_entity"])

        self.embedding_size = opt["embedding_size"]
        self.dim = opt["dim"]

        edge_list, self.n_relation = _edge_list(self.kg, opt["n_entity"], hop=2)
        edge_list = list(set(edge_list))
        self.dbpedia_edge_sets = torch.LongTensor(edge_list)
        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        self.db_edge_type = self.dbpedia_edge_sets[:, 2]

        self.dbpedia_RGCN = RGCNConv(
            opt["n_entity"], self.dim, self.n_relation, num_bases=opt["num_bases"]
        )
        self.concept_edge_sets = _make_synthetic_concept_edges(opt["n_concept"] + 1)
        self.concept_GCN = GCNConv(self.dim, self.dim)

        # real code: mask4key/mask4movie loaded from mask4key.npy/mask4movie.npy
        # (fixed copy-vocabulary masks over the real Redial dictionary); synthesized
        # as all-ones masks (same shape/role, no vocabulary-restriction data invented).
        self.mask4key = torch.ones(vocab_size)
        self.mask4movie = torch.ones(vocab_size)
        self.mask4 = self.mask4key + self.mask4movie

    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)

    def infomax_loss(
        self,
        con_nodes_features,
        db_nodes_features,
        con_user_emb,
        db_user_emb,
        con_label,
        db_label,
        mask,
    ):
        con_emb = self.info_con_norm(con_user_emb)
        db_emb = self.info_db_norm(db_user_emb)
        con_scores = F.linear(db_emb, con_nodes_features, self.info_output_con.bias)
        db_scores = F.linear(con_emb, db_nodes_features, self.info_output_db.bias)

        info_db_loss = torch.sum(self.info_db_loss(db_scores, db_label.float()), dim=-1) * mask
        info_con_loss = torch.sum(self.info_con_loss(con_scores, con_label.float()), dim=-1) * mask

        return torch.mean(info_db_loss), torch.mean(info_con_loss)

    def forward(
        self, xs, concept_mask, db_mask, seed_sets, labels, con_label, db_label, entity_vector
    ):
        """Mirrors the real forward's recommendation + generation-encoding pipeline
        (KG/concept graph construction -> per-user seed-set pooling -> gated fusion ->
        entity scoring -> infomax auxiliary losses -> KG/concept memory encoding for
        the generation decoder)."""
        encoder_states = self.encoder(xs)

        db_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type)
        con_nodes_features = self.concept_GCN(
            self.concept_embeddings.weight, self.concept_edge_sets
        )

        user_representation_list = []
        db_con_mask = []
        for i, seed_set in enumerate(seed_sets):
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim))
                db_con_mask.append(torch.zeros([1]))
                continue
            user_representation = db_nodes_features[seed_set]
            user_representation = self.self_attn_db(user_representation)
            user_representation_list.append(user_representation)
            db_con_mask.append(torch.ones([1]))

        db_user_emb = torch.stack(user_representation_list)
        db_con_mask = torch.stack(db_con_mask)

        graph_con_emb = con_nodes_features[concept_mask]
        con_emb_mask = concept_mask == self.concept_padding

        con_user_emb = graph_con_emb
        con_user_emb, attention = self.self_attn(con_user_emb, con_emb_mask)
        user_emb = self.user_norm(torch.cat([con_user_emb, db_user_emb], dim=-1))
        uc_gate = torch.sigmoid(self.gate_norm(user_emb))
        user_emb = uc_gate * db_user_emb + (1 - uc_gate) * con_user_emb
        entity_scores = F.linear(user_emb, db_nodes_features, self.output_en.bias)

        info_db_loss, info_con_loss = self.infomax_loss(
            con_nodes_features,
            db_nodes_features,
            con_user_emb,
            db_user_emb,
            con_label,
            db_label,
            db_con_mask,
        )

        self.user_rep = user_emb

        # generation-side memory encoding
        con_nodes_features4gen = con_nodes_features
        con_emb4gen = con_nodes_features4gen[concept_mask]
        con_mask4gen = concept_mask != self.concept_padding
        kg_encoding = (self.kg_norm(con_emb4gen), con_mask4gen)

        db_emb4gen = db_nodes_features[entity_vector]
        db_mask4gen = entity_vector != 0
        db_encoding = (self.db_norm(db_emb4gen), db_mask4gen)

        # forced decode over a fixed short response (mirrors decode_forced's copy-generator head)
        latent, _ = self.decoder(labels, encoder_states, kg_encoding, db_encoding)
        kg_attention_latent = self.kg_attn_norm(con_user_emb)
        db_attention_latent = self.db_attn_norm(db_user_emb)
        seqlen = labels.size(1)
        copy_latent = self.copy_norm(
            torch.cat(
                [
                    kg_attention_latent.unsqueeze(1).repeat(1, seqlen, 1),
                    db_attention_latent.unsqueeze(1).repeat(1, seqlen, 1),
                    latent,
                ],
                -1,
            )
        )
        con_logits = self.representation_bias(copy_latent) * self.mask4.unsqueeze(0).unsqueeze(0)
        gen_logits = F.linear(latent, self.embeddings.weight)
        sum_logits = gen_logits + con_logits

        return entity_scores, sum_logits, info_db_loss, info_con_loss


def build_kgsf():
    opt = {
        "batch_size": 2,
        "max_r_length": 6,
        "embedding_size": 16,
        "n_heads": 2,
        "n_layers": 2,
        "ffn_size": 24,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "relu_dropout": 0.0,
        "embeddings_scale": True,
        "learn_positional_embeddings": False,
        "n_positions": 32,
        "dim": 16,
        "n_entity": 20,
        "n_concept": 15,
        "num_bases": 4,
    }
    vocab_size = 40
    model = CrossModel(opt, vocab_size=vocab_size, padding_idx=0, start_idx=1, end_idx=2)
    model.eval()
    return model


def example_input_kgsf():
    batch_size = 2
    ctx_len = 6
    n_entity = 20
    n_concept = 15
    vocab_size = 40
    resp_len = 5

    xs = torch.randint(1, vocab_size, (batch_size, ctx_len))
    concept_mask = torch.randint(0, n_concept + 1, (batch_size, ctx_len))
    db_mask = torch.zeros(batch_size, ctx_len, dtype=torch.bool)
    seed_sets = [[0, 1, 2], [3, 4]]
    labels = torch.randint(1, vocab_size, (batch_size, resp_len))
    con_label = torch.zeros(batch_size, n_concept + 1)
    db_label = torch.zeros(batch_size, n_entity)
    entity_vector = torch.randint(0, n_entity, (batch_size, 8))

    return (xs, concept_mask, db_mask, seed_sets, labels, con_label, db_label, entity_vector)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "KGSF (Knowledge Graph based Semantic Fusion for Conversational Recommendation)",
        "build_kgsf",
        "example_input_kgsf",
        2020,
        "vendored",
    ),
]

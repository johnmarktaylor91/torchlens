# SOURCE: vendored from RUCAIBox/CRSLab @ 6497938 (main)
# (crslab/model/crs/kbrd/kbrd.py, crslab/model/utils/modules/attention.py,
#  crslab/model/utils/modules/transformer.py, crslab/model/utils/functions.py)
#
# KBRD (Chen, Lin, Zhang, Ren, "Towards Knowledge-Based Recommender Dialog
# System", EMNLP 2019), as implemented in the CRSLab conversational-
# recommendation toolkit (ACL 2021 demo). Real architecture: an R-GCN
# encoder over a knowledge graph of entities, a self-attention pooling layer
# that turns a user's mentioned-entity set into a "user embedding", and a
# ParlAI-style Transformer encoder/decoder dialogue module whose output
# logits are additively biased by a projection of the user embedding (so
# recommendation-relevant entities are boosted in the generated response).
# This is real, distinct architecture code (KBRD's own KG-biased decoder),
# vendored (rung 2) rather than recipe'd since no installed base library
# ships this fusion.
#
# Vendoring notes (imports/config fixes only, architecture untouched):
#   - `KBRDModel` originally subclassed `crslab.model.base.BaseModel`, whose
#     `__init__` calls `crslab.download.build(...)` when a `resource` dict
#     is supplied (network fetch of pretrained CRSLab checkpoints/data) --
#     not used here; `BaseModel` is reproduced as a minimal local
#     `nn.Module` base (`abstractmethod build_model`, same `__init__`
#     signature) with the `resource`/`build()` download branch dropped
#     (the original also only calls `build()` `if resource is not None`,
#     so passing `resource=None`, the traced entry's default, exercises
#     identical code either way).
#   - `KBRDModel.__init__` reads `opt`/`vocab`/`side_data` dicts (CRSLab's
#     config/data-loading convention) and computes `self.edge_idx,
#     self.edge_type = edge_to_pyg_format(entity_kg['edge'], 'RGCN')` from a
#     real knowledge-graph edge list; the traced entry builds tiny
#     synthetic `vocab`/`side_data` dicts of the same shapes (random KG
#     edges, tiny vocab) instead of loading CRSLab's ReDial/TG-ReDial
#     corpus + ConceptNet-derived KG files, per the menagerie "tiny
#     config, random init" convention -- `edge_to_pyg_format` itself
#     (copied verbatim from `crslab/model/utils/functions.py`) is exercised
#     unchanged.
#   - `KBRDModel.forward`/`recommend`/`converse`/`decode_forced` are copied
#     verbatim (including the additive `token_logits + user_logits` KG-bias
#     mechanism); `decode_greedy`/`decode_beam_search` (autoregressive
#     inference-time decoding loops, not exercised by `mode='train'`
#     teacher forcing) and `freeze_parameters` are dropped as dead code for
#     the traced path.
#   - `TransformerEncoder`/`TransformerDecoder`/`MultiHeadAttention`/
#     `TransformerFFN`/`create_position_codes` are copied verbatim from
#     `crslab/model/utils/modules/transformer.py` (CRSLab's own vendored,
#     lightly-adapted copy of a ParlAI-style Transformer, pure torch/numpy,
#     no additional dependency).
#   - `SelfAttentionBatch` is copied verbatim from
#     `crslab/model/utils/modules/attention.py`; `SelfAttentionSeq` (unused
#     by KBRD) is dropped.
#   - `logger.debug(...)` calls (from CRSLab's `loguru` logger) are dropped
#     since `loguru` is not an installed base lib and the calls are pure
#     side-effect logging, not part of the computation graph.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

# ---------------------------------------------------------------------------
# crslab/model/base.py (vendored: minimal, download-free BaseModel)
# ---------------------------------------------------------------------------


class BaseModel(nn.Module):
    """Base class for CRSLab models (download/resource-fetch branch dropped;
    see module header)."""

    def __init__(self, opt, device, dpath=None, resource=None):
        super().__init__()
        self.opt = opt
        self.device = device
        self.build_model()

    def build_model(self, *args, **kwargs):
        raise NotImplementedError

    def recommend(self, batch, mode):
        pass

    def converse(self, batch, mode):
        pass

    def guide(self, batch, mode):
        pass


# ---------------------------------------------------------------------------
# crslab/model/utils/functions.py (vendored verbatim, RGCN branch)
# ---------------------------------------------------------------------------


def edge_to_pyg_format(edge, type="RGCN"):
    if type == "RGCN":
        edge_sets = torch.as_tensor(edge, dtype=torch.long)
        edge_idx = edge_sets[:, :2].t()
        edge_type = edge_sets[:, 2]
        return edge_idx, edge_type
    elif type == "GCN":
        edge_set = [[co[0] for co in edge], [co[1] for co in edge]]
        return torch.as_tensor(edge_set, dtype=torch.long)
    else:
        raise NotImplementedError(f"type {type} has not been implemented")


# ---------------------------------------------------------------------------
# crslab/model/utils/modules/attention.py (vendored: SelfAttentionBatch)
# ---------------------------------------------------------------------------


class SelfAttentionBatch(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super().__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(self.da, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h):
        # h: (N, dim)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(dim=1)
        attention = F.softmax(e, dim=0)  # (N)
        return torch.matmul(attention, h)  # (dim)


# ---------------------------------------------------------------------------
# crslab/model/utils/modules/transformer.py (vendored verbatim)
# ---------------------------------------------------------------------------

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504


def neginf(dtype):
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def create_position_codes(n_pos, dim, out):
    position_enc = np.array(
        [[pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)] for pos in range(n_pos)]
    )
    out.data[:, 0::2] = torch.as_tensor(np.sin(position_enc))
    out.data[:, 1::2] = torch.as_tensor(np.cos(position_enc))
    out.detach_()
    out.requires_grad = False


def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0.0):
        super().__init__()
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

        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, key_len)
            .reshape(batch_size * n_heads, query_len, key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

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
    def __init__(self, dim, dim_hidden, relu_dropout=0.0):
        super().__init__()
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
    """ParlAI-style Transformer encoder (vendored verbatim)."""

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
        super().__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction = reduction
        self.padding_idx = padding_idx
        self.dropout = nn.Dropout(dropout)
        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, (
            "Transformer embedding size must be a multiple of n_heads"
        )

        assert embedding is not None, "vendored entry always supplies a shared embedding"
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


class TransformerDecoderLayer(nn.Module):
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

        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, encoder_output, encoder_mask):
        decoder_mask = self._create_selfattn_mask(x)
        residual = x
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)
        x = x + residual
        x = _normalize(x, self.norm1)

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


class TransformerDecoder(nn.Module):
    """ParlAI-style Transformer decoder (vendored verbatim)."""

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
                TransformerDecoderLayer(
                    n_heads,
                    embedding_size,
                    ffn_size,
                    attention_dropout=attention_dropout,
                    relu_dropout=relu_dropout,
                    dropout=dropout,
                )
            )

    def forward(self, input, encoder_state, incr_state=None):
        encoder_output, encoder_mask = encoder_state

        seq_len = input.shape[1]
        positions = input.new_empty(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)

        for layer in self.layers:
            tensor = layer(tensor, encoder_output, encoder_mask)

        return tensor, None


# ---------------------------------------------------------------------------
# crslab/model/crs/kbrd/kbrd.py (vendored: KBRDModel, minus decode_greedy/
# decode_beam_search/freeze_parameters -- see module header)
# ---------------------------------------------------------------------------


class KBRDModel(BaseModel):
    def __init__(self, opt, device, vocab, side_data):
        self.device = device
        self.gpu = opt.get("gpu", [-1])
        self.pad_token_idx = vocab["pad"]
        self.start_token_idx = vocab["start"]
        self.end_token_idx = vocab["end"]
        self.vocab_size = vocab["vocab_size"]
        self.token_emb_dim = opt.get("token_emb_dim", 300)
        self.pretrain_embedding = side_data.get("embedding", None)

        self.n_entity = vocab["n_entity"]
        entity_kg = side_data["entity_kg"]
        self.n_relation = entity_kg["n_relation"]
        self.edge_idx, self.edge_type = edge_to_pyg_format(entity_kg["edge"], "RGCN")
        self.edge_idx = self.edge_idx.to(device)
        self.edge_type = self.edge_type.to(device)
        self.num_bases = opt.get("num_bases", 8)
        self.kg_emb_dim = opt.get("kg_emb_dim", 300)
        self.user_emb_dim = self.kg_emb_dim

        self.n_heads = opt.get("n_heads", 2)
        self.n_layers = opt.get("n_layers", 2)
        self.ffn_size = opt.get("ffn_size", 300)
        self.dropout = opt.get("dropout", 0.1)
        self.attention_dropout = opt.get("attention_dropout", 0.0)
        self.relu_dropout = opt.get("relu_dropout", 0.1)
        self.embeddings_scale = opt.get("embedding_scale", True)
        self.learn_positional_embeddings = opt.get("learn_positional_embeddings", False)
        self.reduction = opt.get("reduction", False)
        self.n_positions = opt.get("n_positions", 1024)
        self.longest_label = opt.get("longest_label", 1)
        self.user_proj_dim = opt.get("user_proj_dim", 512)

        super().__init__(opt, device)

    def build_model(self, *args, **kwargs):
        self._build_embedding()
        self._build_kg_layer()
        self._build_recommendation_layer()
        self._build_conversation_layer()

    def _build_embedding(self):
        if self.pretrain_embedding is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrain_embedding, dtype=torch.float),
                freeze=False,
                padding_idx=self.pad_token_idx,
            )
        else:
            self.token_embedding = nn.Embedding(
                self.vocab_size, self.token_emb_dim, self.pad_token_idx
            )
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim**-0.5)
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)

    def _build_kg_layer(self):
        self.kg_encoder = RGCNConv(
            self.n_entity, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases
        )
        self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)

    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()

    def _build_conversation_layer(self):
        self.register_buffer("START", torch.tensor([self.start_token_idx], dtype=torch.long))
        self.dialog_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions,
        )
        self.decoder = TransformerDecoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.embeddings_scale,
            self.learn_positional_embeddings,
            self.pad_token_idx,
            self.n_positions,
        )
        self.user_proj_1 = nn.Linear(self.user_emb_dim, self.user_proj_dim)
        self.user_proj_2 = nn.Linear(self.user_proj_dim, self.vocab_size)
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)

    def encode_user(self, entity_lists, kg_embedding):
        user_repr_list = []
        for entity_list in entity_lists:
            if entity_list is None:
                user_repr_list.append(torch.zeros(self.user_emb_dim, device=self.device))
                continue
            user_repr = kg_embedding[entity_list]
            user_repr = self.kg_attn(user_repr)
            user_repr_list.append(user_repr)
        return torch.stack(user_repr_list, dim=0)  # (bs, dim)

    def recommend(self, batch, mode):
        context_entities, item = batch["context_entities"], batch["item"]
        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)
        user_embedding = self.encode_user(context_entities, kg_embedding)
        scores = F.linear(user_embedding, kg_embedding, self.rec_bias.bias)
        loss = self.rec_loss(scores, item)
        return loss, scores

    def _starts(self, batch_size):
        return self.START.detach().expand(batch_size, 1)

    def decode_forced(self, encoder_states, user_embedding, resp):
        bsz = resp.size(0)
        seqlen = resp.size(1)
        inputs = resp.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states)
        token_logits = F.linear(latent, self.token_embedding.weight)
        user_logits = self.user_proj_2(torch.relu(self.user_proj_1(user_embedding))).unsqueeze(1)
        sum_logits = token_logits + user_logits
        _, preds = sum_logits.max(dim=-1)
        return sum_logits, preds

    def converse(self, batch, mode):
        context_tokens, context_entities, response = (
            batch["context_tokens"],
            batch["context_entities"],
            batch["response"],
        )
        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)
        user_embedding = self.encode_user(context_entities, kg_embedding)
        encoder_state = self.dialog_encoder(context_tokens)
        self.longest_label = max(self.longest_label, response.shape[1])
        logits, preds = self.decode_forced(encoder_state, user_embedding, response)
        logits = logits.view(-1, logits.shape[-1])
        labels = response.view(-1)
        return self.conv_loss(logits, labels), preds

    def forward(self, batch, mode, stage):
        if stage == "conv":
            return self.converse(batch, mode)
        if stage == "rec":
            return self.recommend(batch, mode)


MENAGERIE_ZOO = "vendored-pytorch"

_N_ENTITY = 24
_N_RELATION = 5
_NUM_BASES = 2
_KG_EMB_DIM = 8
_VOCAB_SIZE = 32
_TOKEN_EMB_DIM = 8
_FFN_SIZE = 16
_N_HEADS = 2
_N_LAYERS = 1
_N_POSITIONS = 16
_USER_PROJ_DIM = 12
_BATCH = 2
_CTX_LEN = 5
_RESP_LEN = 4
_PAD, _START, _END = 0, 1, 2


def _build_opt():
    return {
        "gpu": [-1],
        "token_emb_dim": _TOKEN_EMB_DIM,
        "num_bases": _NUM_BASES,
        "kg_emb_dim": _KG_EMB_DIM,
        "n_heads": _N_HEADS,
        "n_layers": _N_LAYERS,
        "ffn_size": _FFN_SIZE,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "relu_dropout": 0.0,
        "embedding_scale": True,
        "learn_positional_embeddings": False,
        "reduction": False,
        "n_positions": _N_POSITIONS,
        "longest_label": _RESP_LEN,
        "user_proj_dim": _USER_PROJ_DIM,
    }


def _build_vocab():
    return {
        "pad": _PAD,
        "start": _START,
        "end": _END,
        "vocab_size": _VOCAB_SIZE,
        "n_entity": _N_ENTITY,
    }


def _build_side_data():
    n_edges = 40
    src = torch.randint(0, _N_ENTITY, (n_edges, 1))
    dst = torch.randint(0, _N_ENTITY, (n_edges, 1))
    rel = torch.randint(0, _N_RELATION, (n_edges, 1))
    edge = torch.cat([src, dst, rel], dim=1).tolist()
    return {
        "embedding": None,
        "entity_kg": {"n_relation": _N_RELATION, "edge": edge},
    }


def build_kbrd():
    model = KBRDModel(_build_opt(), torch.device("cpu"), _build_vocab(), _build_side_data())
    model.eval()
    return model


class _KBRDConverseEntry(nn.Module):
    """Traces `KBRDModel.forward(batch, mode='train', stage='conv')` --
    the KG-biased Transformer dialogue decoder (teacher-forced)."""

    def __init__(self):
        super().__init__()
        self.kbrd = build_kbrd()

    def forward(self, context_tokens, context_entities, response):
        batch = {
            "context_tokens": context_tokens,
            "context_entities": context_entities,
            "response": response,
        }
        return self.kbrd.forward(batch, mode="train", stage="conv")


def build_kbrd_converse():
    m = _KBRDConverseEntry()
    m.eval()
    return m


def example_input_kbrd_converse():
    context_tokens = torch.randint(1, _VOCAB_SIZE, (_BATCH, _CTX_LEN))
    context_entities = [torch.randint(0, _N_ENTITY, (3,)) for _ in range(_BATCH)]
    response = torch.randint(1, _VOCAB_SIZE, (_BATCH, _RESP_LEN))
    return (context_tokens, context_entities, response)


class _KBRDRecommendEntry(nn.Module):
    """Traces `KBRDModel.forward(batch, mode='train', stage='rec')` -- the
    R-GCN + self-attention user-encoder recommendation head."""

    def __init__(self):
        super().__init__()
        self.kbrd = build_kbrd()

    def forward(self, context_entities, item):
        batch = {"context_entities": context_entities, "item": item}
        return self.kbrd.forward(batch, mode="train", stage="rec")


def build_kbrd_recommend():
    m = _KBRDRecommendEntry()
    m.eval()
    return m


def example_input_kbrd_recommend():
    context_entities = [torch.randint(0, _N_ENTITY, (3,)) for _ in range(_BATCH)]
    item = torch.randint(0, _N_ENTITY, (_BATCH,))
    return (context_entities, item)


MENAGERIE_ENTRIES = [
    (
        "KBRD (KG-Biased Transformer Decoder)",
        build_kbrd_converse,
        example_input_kbrd_converse,
        2019,
        "vendored-pytorch",
    ),
    (
        "KBRD (R-GCN Recommendation Head)",
        build_kbrd_recommend,
        example_input_kbrd_recommend,
        2019,
        "vendored-pytorch",
    ),
]

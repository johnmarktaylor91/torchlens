# SOURCE: vendored from https://github.com/uta-smile/SMILES-BERT @ master
# Files: fairseq/models/bert.py (BertModel, TransformerEncoder, TransformerEncoderLayer,
# BertMLMHead, BertPooler), fairseq/modules/multihead_attention.py (MultiheadAttention),
# fairseq/modules/sinusoidal_positional_embedding.py (SinusoidalPositionalEmbedding),
# fairseq/modules/learned_positional_embedding.py (LearnedPositionalEmbedding) -- all
# architecture classes copied verbatim from the repo's vendored fairseq fork (SMILES-BERT
# forks fairseq wholesale and adds bert.py on top of it). Only the small `fairseq.utils`
# helper functions actually used (make_positions, get/set_incremental_state) are localized
# here since they are pure-torch utilities with no third-party dependency; the fairseq
# `@register_model` decorator machinery and CLI arg parsing (build_model/add_args) are
# dropped since they only wire training-script plumbing, not the network architecture.
#
# SMILES-BERT (Wang et al. 2019, "SMILES-BERT: Large Scale Unsupervised Pre-Training for
# Molecular Property Prediction") is a BERT-style Transformer encoder pretrained via masked
# language modeling directly on SMILES token sequences. The forward path here mirrors the
# original BertModel.forward's pretraining branch (encoder -> BertMLMHead), which is the
# model's headline contribution; the optional prop_predict fine-tuning branch (BertPooler +
# PredNet) is a small linear head added downstream of the same encoder and is omitted for a
# single-input trace (prop_predict=False in the original code's default configuration).
#
# One minimal non-architectural robustness fix: MultiheadAttention.forward's `q *= self.scaling`
# (in-place multiply on a view returned by `chunk()`) is changed to the out-of-place
# `q = q * self.scaling`; this is exactly what later stock fairseq releases did too, and is
# required for autograd-aware tracing since PyTorch now rejects in-place writes to multi-view
# chunk() outputs. No architectural change.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# --- fairseq/utils.py (small pure-torch helpers only, localized) ---
from collections import defaultdict

_INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__
    if not hasattr(module_instance, "_fairseq_instance_id"):
        _INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = _INCREMENTAL_STATE_INSTANCE_ID[module_name]
    return "{}.{}.{}".format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def make_positions(tensor, padding_idx, left_pad, onnx_trace=False):
    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, "range_buf"):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[: tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])


# --- fairseq/modules/multihead_attention.py (MultiheadAttention, verbatim) ---
class MultiheadAttention(nn.Module):
    """Multi-headed attention. See "Attention Is All You Need" for more details."""

    def __init__(
        self, embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )
        self.scaling = self.head_dim**-0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False,
        attn_mask=None,
    ):
        """Input shape: Time x Batch x Channel"""

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if "prev_key" in saved_state:
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)],
                    dim=1,
                )

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = (
                attn_weights.float()
                .masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float("-inf"),
                )
                .type_as(attn_weights)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return get_incremental_state(self, incremental_state, "attn_state") or {}

    def _set_input_buffer(self, incremental_state, buffer):
        set_incremental_state(self, incremental_state, "attn_state", buffer)


# --- fairseq/modules/sinusoidal_positional_embedding.py (verbatim, ONNX branch dropped) ---
class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, embedding_dim, padding_idx, left_pad, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size(0), input.size(1)
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.type_as(self._float_tensor)

        if incremental_state is not None:
            pos = (timestep.int() + 1).long() if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        return int(1e5)


# --- fairseq/modules/learned_positional_embedding.py (verbatim) ---
class LearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx, left_pad):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.left_pad = left_pad

    def forward(self, input, incremental_state=None):
        if incremental_state is not None:
            positions = input.data.new(1, 1).fill_(self.padding_idx + input.size(1))
        else:
            positions = make_positions(input.data, self.padding_idx, self.left_pad)
        return super().forward(positions)

    def max_positions(self):
        return self.num_embeddings - self.padding_idx - 1


# --- fairseq/models/bert.py helper factory functions (verbatim) ---
def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim):
    return nn.LayerNorm(embedding_dim)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def gelu(x):
    """Implementation of the gelu activation function. See https://arxiv.org/abs/1606.08415"""
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(
            num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad
        )
        nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(
            embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1
        )
    return m


# --- fairseq/models/bert.py architecture classes (verbatim) ---
class TransformerEncoderLayer(nn.Module):
    """Encoder layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        dropout,
        attention_dropout,
        relu_dropout,
        normalize_before,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.normalize_before = normalize_before
        self.fc1 = Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = Linear(ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = gelu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerEncoder(nn.Module):
    """Transformer encoder consisting of *encoder_layers* layers (BERT encoder stack)."""

    def __init__(
        self,
        embed_tokens,
        max_source_positions=1024,
        encoder_layers=12,
        encoder_attention_heads=12,
        encoder_ffn_embed_dim=3072,
        dropout=0.1,
        attention_dropout=0.0,
        relu_dropout=0.0,
        encoder_normalize_before=False,
        encoder_learned_pos=False,
        no_token_positional_embeddings=False,
        left_pad=False,
    ):
        super().__init__()
        self.dropout = dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = (
            PositionalEmbedding(
                max_source_positions,
                embed_dim,
                self.padding_idx,
                left_pad=left_pad,
                learned=encoder_learned_pos,
            )
            if not no_token_positional_embeddings
            else None
        )

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerEncoderLayer(
                    embed_dim,
                    encoder_ffn_embed_dim,
                    encoder_attention_heads,
                    dropout,
                    attention_dropout,
                    relu_dropout,
                    encoder_normalize_before,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.register_buffer("version", torch.Tensor([2]))
        self.normalize = encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        x = x.transpose(0, 1)
        return {
            "encoder_out": x,  # B x T x C
            "encoder_padding_mask": encoder_padding_mask,  # B x T
        }


class BertMLMHead(nn.Module):
    """BERT MLM Head for pretraining."""

    def __init__(self, embed_dim, bert_model_embedding_weights):
        super().__init__()
        self.embed_dim = embed_dim
        self.dense = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm = LayerNorm(self.embed_dim)
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, x):
        x = gelu(self.dense(x))
        x = self.layer_norm(x)
        x = self.decoder(x) + self.bias
        return x


class BertPooler(nn.Module):
    """BERT pooler."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.tanh = nn.Tanh()
        self.dense = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        x = x[:, 0, :]
        x = self.tanh(self.dense(x))
        return x


class BertModel(nn.Module):
    """BERT model from arxiv.org/abs/1810.04805, as adapted by SMILES-BERT for SMILES token
    sequences. This vendored version wires up the same modules the original build_model()
    factory constructs (TransformerEncoder + BertMLMHead), with __init__ taking config values
    directly instead of through fairseq's argparse Namespace + task/dictionary machinery.
    """

    def __init__(
        self,
        vocab_size=64,
        padding_idx=0,
        max_source_positions=64,
        encoder_embed_dim=128,
        encoder_ffn_embed_dim=512,
        encoder_layers=4,
        encoder_attention_heads=4,
    ):
        super().__init__()
        embed_tokens = Embedding(vocab_size, encoder_embed_dim, padding_idx)
        self.encoder = TransformerEncoder(
            embed_tokens,
            max_source_positions=max_source_positions,
            encoder_layers=encoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            encoder_ffn_embed_dim=encoder_ffn_embed_dim,
        )
        self.pretrain_head = BertMLMHead(encoder_embed_dim, embed_tokens.weight)

    def forward(self, src_tokens, src_lengths):
        encoder_out = self.encoder(src_tokens, src_lengths)
        # Pre-training stage (masked language modeling).
        x = self.pretrain_head(encoder_out["encoder_out"])
        return x


MENAGERIE_ZOO = "vendored-pytorch"


def build_smilesbert():
    return BertModel(
        vocab_size=64,
        padding_idx=0,
        max_source_positions=64,
        encoder_embed_dim=128,
        encoder_ffn_embed_dim=512,
        encoder_layers=4,
        encoder_attention_heads=4,
    )


def example_input_smilesbert():
    src_tokens = torch.randint(1, 64, (2, 32), dtype=torch.long)
    src_lengths = torch.tensor([32, 32], dtype=torch.long)
    return (src_tokens, src_lengths)


MENAGERIE_ENTRIES = [
    ("SMILES-BERT", build_smilesbert, example_input_smilesbert, 2019, "SOURCE_AVAILABLE"),
]

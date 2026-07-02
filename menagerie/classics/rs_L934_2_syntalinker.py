# SOURCE: vendored from https://github.com/YuYaoYang2333/SyntaLinker @ master
# (onmt/models/model.py, onmt/encoders/{encoder,transformer}.py,
#  onmt/decoders/{decoder,transformer}.py, onmt/modules/{multi_headed_attn,
#  position_ffn,embeddings,util_class}.py, onmt/utils/misc.py:sequence_mask)
"""SyntaLinker: scaffold-constrained deep learning approach for generating
structural diverse molecules using deep reinforcement learning
(Yang et al. 2020, J. Cheminform / arXiv:2011.08483). SyntaLinker vendors
OpenNMT-py wholesale and trains a plain Transformer encoder-decoder
(Vaswani et al. "Attention is All You Need") to translate a fragment-pair
SMILES sequence into a linker SMILES sequence -- i.e. it is architecturally
the standard OpenNMT-py `NMTModel(TransformerEncoder, TransformerDecoder)`
seq2seq Transformer, with SyntaLinker's actual contribution being the
molecular linker-design *task* (scaffold decoration via fragment growing
constrained by a random fragmentation strategy) and RL fine-tuning loop, not
a new layer type. This module vendors the real `NMTModel` /
`TransformerEncoder` / `TransformerDecoder` / `MultiHeadedAttention` /
`PositionwiseFeedForward` / `Embeddings` code verbatim and constructs the
real model classes directly (bypassing the CLI `opt`/`torchtext.Field`/
vocab plumbing in `onmt.model_builder.build_base_model`, which is
option-parsing/data-loading infrastructure, not architecture) at tiny
size for a fast trace.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# onmt/utils/misc.py (sequence_mask only)
# ---------------------------------------------------------------------------
def sequence_mask(lengths, max_len=None):
    """Creates a boolean mask from sequence lengths."""
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)
        .repeat(batch_size, 1)
        .lt(lengths.unsqueeze(1))
    )


# ---------------------------------------------------------------------------
# onmt/modules/util_class.py
# ---------------------------------------------------------------------------
class Elementwise(nn.ModuleList):
    """A simple network container: parameters are a list of modules; inputs
    are a 3d Tensor whose last dim is the same length as the list."""

    def __init__(self, merge=None, *args):
        assert merge in [None, "first", "concat", "sum", "mlp"]
        self.merge = merge
        super().__init__(*args)

    def forward(self, inputs):
        inputs_ = [feat.squeeze(2) for feat in inputs.split(1, dim=2)]
        assert len(self) == len(inputs_)
        outputs = [f(x) for f, x in zip(self, inputs_)]
        if self.merge == "first":
            return outputs[0]
        elif self.merge == "concat" or self.merge == "mlp":
            return torch.cat(outputs, 2)
        elif self.merge == "sum":
            return sum(outputs)
        else:
            return outputs


# ---------------------------------------------------------------------------
# onmt/modules/embeddings.py
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks."""

    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with odd dim (got dim={dim})")
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super().__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[: emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb


class Embeddings(nn.Module):
    """Words embeddings for encoder/decoder (feature-lookup machinery kept
    intact, exercised with zero extra features -- matches SyntaLinker's
    real training config which uses plain SMILES-token vocab, no feats)."""

    def __init__(
        self,
        word_vec_size,
        word_vocab_size,
        word_padding_idx,
        position_encoding=False,
        feat_merge="concat",
        feat_vec_exponent=0.7,
        feat_vec_size=-1,
        feat_padding_idx=None,
        feat_vocab_sizes=None,
        dropout=0,
        sparse=False,
        fix_word_vecs=False,
    ):
        feat_padding_idx = feat_padding_idx or []
        feat_vocab_sizes = feat_vocab_sizes or []
        self.word_padding_idx = word_padding_idx
        self.word_vec_size = word_vec_size

        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        if feat_merge == "sum":
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab**feat_vec_exponent) for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [
            nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse)
            for vocab, dim, pad in emb_params
        ]
        emb_luts = Elementwise(feat_merge, embeddings)

        self.embedding_size = sum(emb_dims) if feat_merge == "concat" else word_vec_size

        super().__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module("emb_luts", emb_luts)

        if feat_merge == "mlp" and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            mlp = nn.Sequential(nn.Linear(in_dim, word_vec_size), nn.ReLU())
            self.make_embedding.add_module("mlp", mlp)

        self.position_encoding = position_encoding
        if self.position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module("pe", pe)

        if fix_word_vecs:
            self.word_lut.weight.requires_grad = False

    @property
    def word_lut(self):
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        return self.make_embedding[0]

    def forward(self, source, step=None):
        if self.position_encoding:
            for i, module in enumerate(self.make_embedding._modules.values()):
                if i == len(self.make_embedding._modules.values()) - 1:
                    source = module(source, step=step)
                else:
                    source = module(source)
        else:
            source = self.make_embedding(source)
        return source


# ---------------------------------------------------------------------------
# onmt/modules/multi_headed_attn.py
# ---------------------------------------------------------------------------
class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need".
    (max_relative_positions path omitted -- SyntaLinker's default config
    uses max_relative_positions=0, i.e. plain absolute positional encoding.)
    """

    def __init__(self, head_count, model_dim, dropout=0.1, max_relative_positions=0):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super().__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)
        self.max_relative_positions = max_relative_positions

    def forward(self, key, value, query, mask=None, layer_cache=None, attn_type=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        if layer_cache is not None:
            if attn_type == "self":
                query, key, value = (
                    self.linear_query(query),
                    self.linear_keys(query),
                    self.linear_values(query),
                )
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat((layer_cache["self_keys"], key), dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat((layer_cache["self_values"], value), dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key), self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"], layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)
        key_len = key.size(2)
        query_len = query.size(2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)

        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)
        context_original = torch.matmul(drop_attn, value)
        context = unshape(context_original)
        output = self.final_linear(context)

        top_attn = attn.view(batch_size, head_count, query_len, key_len)[:, 0, :, :].contiguous()
        return output, top_attn


# ---------------------------------------------------------------------------
# onmt/modules/position_ffn.py
# ---------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """A two-layer Feed-Forward-Network with residual layer norm."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


# ---------------------------------------------------------------------------
# onmt/encoders/{encoder,transformer}.py
# ---------------------------------------------------------------------------
class EncoderBase(nn.Module):
    """Base encoder class."""

    def forward(self, src, lengths=None):
        raise NotImplementedError


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout, max_relative_positions=0):
        super().__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout, max_relative_positions=max_relative_positions
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask, attn_type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"."""

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        embeddings,
        max_relative_positions,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    max_relative_positions=max_relative_positions,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, lengths=None):
        emb = self.embeddings(src)
        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)
        return emb, out.transpose(0, 1).contiguous(), lengths


# ---------------------------------------------------------------------------
# onmt/decoders/{decoder,transformer}.py
# ---------------------------------------------------------------------------
class DecoderBase(nn.Module):
    def __init__(self, attentional=True):
        super().__init__()
        self.attentional = attentional


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        self_attn_type="scaled-dot",
        max_relative_positions=0,
    ):
        super().__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout, max_relative_positions=max_relative_positions
        )
        self.context_attn = MultiHeadedAttention(heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, layer_cache=None, step=None):
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len], device=tgt_pad_mask.device, dtype=torch.uint8
            )
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            future_mask = future_mask.bool()
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        input_norm = self.layer_norm_1(inputs)
        query, _attn = self.self_attn(
            input_norm,
            input_norm,
            input_norm,
            mask=dec_mask,
            layer_cache=layer_cache,
            attn_type="self",
        )
        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(
            memory_bank,
            memory_bank,
            query_norm,
            mask=src_pad_mask,
            layer_cache=layer_cache,
            attn_type="context",
        )
        output = self.feed_forward(self.drop(mid) + query)
        return output, attn


class TransformerDecoder(DecoderBase):
    """The Transformer decoder from "Attention is All You Need"."""

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        copy_attn,
        self_attn_type,
        dropout,
        attention_dropout,
        embeddings,
        max_relative_positions,
        aan_useffn,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.state = {}
        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    self_attn_type=self_attn_type,
                    max_relative_positions=max_relative_positions,
                )
                for _ in range(num_layers)
            ]
        )
        self._copy = copy_attn
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def init_state(self, src, memory_bank, enc_hidden):
        self.state["src"] = src
        self.state["cache"] = None

    def forward(self, tgt, memory_bank, step=None, **kwargs):
        if step == 0:
            self._init_cache(memory_bank)

        tgt_words = tgt[:, :, 0].transpose(0, 1)
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        pad_idx = self.embeddings.word_padding_idx
        src_lens = kwargs["memory_lengths"]
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1)
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)

        attn = None
        for i, layer in enumerate(self.transformer_layers):
            layer_cache = self.state["cache"][f"layer_{i}"] if step is not None else None
            output, attn = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
            )

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn
        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        for i, _layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.state["cache"][f"layer_{i}"] = layer_cache


# ---------------------------------------------------------------------------
# onmt/models/model.py
# ---------------------------------------------------------------------------
class NMTModel(nn.Module):
    """Core trainable object in OpenNMT: generic encoder + decoder model."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, bptt=False):
        tgt = tgt[:-1]  # exclude last target from inputs
        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank, memory_lengths=lengths)
        return dec_out, attns


# ---------------------------------------------------------------------------
# Builders (tiny SMILES-vocab config, matching SyntaLinker's real
# transformer training defaults: 6 layers real config; scaled down here)
# ---------------------------------------------------------------------------
_VOCAB_SIZE = 48  # tiny SMILES-token vocab (real: ~a few dozen tokens)
_D_MODEL = 32
_HEADS = 4
_D_FF = 64
_N_LAYERS = 2
_PAD_IDX = 0


def _build_embeddings():
    return Embeddings(
        word_vec_size=_D_MODEL,
        word_vocab_size=_VOCAB_SIZE,
        word_padding_idx=_PAD_IDX,
        position_encoding=True,
        dropout=0.1,
    )


def build_syntalinker():
    encoder = TransformerEncoder(
        num_layers=_N_LAYERS,
        d_model=_D_MODEL,
        heads=_HEADS,
        d_ff=_D_FF,
        dropout=0.1,
        attention_dropout=0.1,
        embeddings=_build_embeddings(),
        max_relative_positions=0,
    )
    decoder = TransformerDecoder(
        num_layers=_N_LAYERS,
        d_model=_D_MODEL,
        heads=_HEADS,
        d_ff=_D_FF,
        copy_attn=False,
        self_attn_type="scaled-dot",
        dropout=0.1,
        attention_dropout=0.1,
        embeddings=_build_embeddings(),
        max_relative_positions=0,
        aan_useffn=False,
    )
    model = NMTModel(encoder, decoder)
    return model.eval()


def example_input_syntalinker():
    # (src, tgt, lengths): src/tgt are (seq_len, batch, 1) LongTensors of
    # token indices (nfeat=1, no extra features), lengths is (batch,).
    src_len, tgt_len, batch = 9, 7, 2
    src = torch.randint(1, _VOCAB_SIZE, (src_len, batch, 1))
    tgt = torch.randint(1, _VOCAB_SIZE, (tgt_len, batch, 1))
    lengths = torch.full((batch,), src_len, dtype=torch.long)
    return (src, tgt, lengths)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SyntaLinker", "build_syntalinker", "example_input_syntalinker", 2020, "vendored"),
]

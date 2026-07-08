# SOURCE: vendored from https://github.com/wangxr0526/RetroPrime @ master
# Files vendored (near-verbatim; the repo's own `retroprime/transformer_model/onmt/`
# package -- a self-contained fork of legacy OpenNMT-py, NOT the pip `OpenNMT-py`
# package, so there is nothing to "not install": this vendored `onmt` IS the model
# source): onmt/modules/util_class.py (LayerNorm, Elementwise), onmt/modules/
# multi_headed_attn.py (MultiHeadedAttention), onmt/modules/position_ffn.py
# (PositionwiseFeedForward), onmt/modules/embeddings.py (PositionalEncoding,
# Embeddings), onmt/encoders/transformer.py (TransformerEncoderLayer,
# TransformerEncoder), onmt/decoders/transformer.py (TransformerDecoderLayer,
# TransformerDecoder, TransformerDecoderState), onmt/decoders/decoder.py
# (DecoderState base), onmt/models/model.py (NMTModel).
#
# RetroPrime (Wang et al., 2021) is a two-stage retrosynthesis pipeline
# (Prediction of Reaction Center via "P2S" then "S2R" synthon-to-reactant
# completion); BOTH stages are trained with this exact same OpenNMT-py-style
# Transformer seq2seq architecture, only the SMILES-token vocabularies/data differ
# -- so this staging module traces the shared encoder-decoder-generator backbone.
from __future__ import annotations

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# onmt/modules/util_class.py
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Elementwise(nn.ModuleList):
    """A simple network container: applies a list of modules elementwise
    over the feature dimension of a 3d Tensor and optionally merges."""

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
        return outputs


# ---------------------------------------------------------------------------
# onmt/modules/multi_headed_attn.py
# ---------------------------------------------------------------------------
class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"."""

    def __init__(self, head_count, model_dim, dropout=0.1):
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

    def forward(self, key, value, query, mask=None, layer_cache=None, type=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        if layer_cache is not None:
            if type == "self":
                query, key, value = (
                    self.linear_query(query),
                    self.linear_keys(query),
                    self.linear_values(query),
                )

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat((layer_cache["self_keys"].to(device), key), dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat((layer_cache["self_values"].to(device), value), dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
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

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value))

        output = self.final_linear(context)

        top_attn = attn.view(batch_size, head_count, query_len, key_len)[:, 0, :, :].contiguous()

        return output, top_attn


# ---------------------------------------------------------------------------
# onmt/modules/position_ffn.py
# ---------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


# ---------------------------------------------------------------------------
# onmt/modules/embeddings.py
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)
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
    """Word embeddings for encoder/decoder with optional sinusoidal positional encoding."""

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
    ):
        if feat_padding_idx is None:
            feat_padding_idx = []
        if feat_vocab_sizes is None:
            feat_vocab_sizes = []
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
            out_dim = word_vec_size
            mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            self.make_embedding.add_module("mlp", mlp)

        self.position_encoding = position_encoding

        if self.position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module("pe", pe)

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
# onmt/encoders/encoder.py + onmt/encoders/transformer.py
# ---------------------------------------------------------------------------
class EncoderBase(nn.Module):
    """Base encoder class (arg-count assertion `_check_args` from upstream `aeq`
    dropped -- it is a debug-only shape check with no computational effect)."""

    def _check_args(self, src, lengths=None, hidden=None):
        pass


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"."""

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
        super().__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.layer_norm = LayerNorm(d_model)

    def forward(self, src, lengths=None):
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1).expand(w_batch, w_len, w_len)
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths


# ---------------------------------------------------------------------------
# onmt/decoders/decoder.py (DecoderState base) + onmt/decoders/transformer.py
# ---------------------------------------------------------------------------
class DecoderState:
    """Interface for grouping together the current state of a decoder."""

    def detach(self):
        raise NotImplementedError()

    def map_batch_fn(self, fn):
        raise NotImplementedError()


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, self_attn_type="scaled-dot"):
        super().__init__()
        self.self_attn_type = self_attn_type
        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.context_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(5000)
        self.register_buffer("mask", mask)

    def forward(
        self,
        inputs,
        memory_bank,
        src_pad_mask,
        tgt_pad_mask,
        previous_input=None,
        layer_cache=None,
        step=None,
    ):
        dec_mask = torch.gt(
            tgt_pad_mask + self.mask[:, : tgt_pad_mask.size(1), : tgt_pad_mask.size(1)], 0
        )
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        query, attn = self.self_attn(
            all_input, all_input, input_norm, mask=dec_mask, layer_cache=layer_cache, type="self"
        )

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(
            memory_bank,
            memory_bank,
            query_norm,
            mask=src_pad_mask,
            layer_cache=layer_cache,
            type="context",
        )
        output = self.feed_forward(self.drop(mid) + query)

        return output, attn, all_input

    def _get_attn_subsequent_mask(self, size):
        import numpy as np

        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoderState(DecoderState):
    def __init__(self, src):
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def map_batch_fn(self, fn):
        self.src = fn(self.src, 1)


class TransformerDecoder(nn.Module):
    """The Transformer decoder from "Attention is All You Need"."""

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        attn_type,
        copy_attn,
        self_attn_type,
        dropout,
        embeddings,
    ):
        super().__init__()
        self.decoder_type = "transformer"
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.self_attn_type = self_attn_type

        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model, heads, d_ff, dropout, self_attn_type=self_attn_type
                )
                for _ in range(num_layers)
            ]
        )

        self._copy = False
        self.layer_norm = LayerNorm(d_model)

    def forward(self, tgt, memory_bank, state, memory_lengths=None, step=None, cache=None):
        src = state.src
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        attns = {"std": []}

        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        src_pad_mask = (
            src_words.data.eq(padding_idx).unsqueeze(1).expand(src_batch, tgt_len, src_len)
        )
        tgt_pad_mask = (
            tgt_words.data.eq(padding_idx).unsqueeze(1).expand(tgt_batch, tgt_len, tgt_len)
        )

        if state.cache is None:
            saved_inputs = []

        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, attn, all_input = self.transformer_layers[i](
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                previous_input=prev_layer_input,
                layer_cache=state.cache["layer_{}".format(i)] if state.cache is not None else None,
                step=step,
            )
            if state.cache is None:
                saved_inputs.append(all_input)

        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        outputs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns["std"] = attn

        if state.cache is None:
            state = state.update_state(tgt, saved_inputs)

        return outputs, state, attns

    def init_decoder_state(self, src, memory_bank, enc_hidden, with_cache=False):
        return TransformerDecoderState(src)


# ---------------------------------------------------------------------------
# onmt/models/model.py :: NMTModel
# ---------------------------------------------------------------------------
class NMTModel(nn.Module):
    """Core trainable object in OpenNMT: generic encoder + decoder model."""

    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        tgt = tgt[:-1]

        enc_final, memory_bank, lengths = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = self.decoder(
            tgt, memory_bank, enc_state if dec_state is None else dec_state, memory_lengths=lengths
        )
        if self.multigpu:
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state


# ---------------------------------------------------------------------------
# Staging harness: real NMTModel + real generator head (as wired in
# onmt/model_builder.py build_base_model), wrapped for a plain forward() trace.
# ---------------------------------------------------------------------------
class RetroPrimeSeq2Seq(nn.Module):
    def __init__(self, model: NMTModel, generator: nn.Module):
        super().__init__()
        self.model = model
        self.generator = generator

    def forward(self, src, tgt, lengths):
        decoder_outputs, attns, dec_state = self.model(src, tgt, lengths)
        scores = self.generator(decoder_outputs)
        return scores


def build_retroprime():
    d_model = 16
    heads = 4
    d_ff = 32
    vocab_size = 48
    padding_idx = 1

    src_embeddings = Embeddings(
        word_vec_size=d_model,
        word_vocab_size=vocab_size,
        word_padding_idx=padding_idx,
        position_encoding=True,
        dropout=0.0,
    )
    tgt_embeddings = Embeddings(
        word_vec_size=d_model,
        word_vocab_size=vocab_size,
        word_padding_idx=padding_idx,
        position_encoding=True,
        dropout=0.0,
    )

    encoder = TransformerEncoder(
        num_layers=2,
        d_model=d_model,
        heads=heads,
        d_ff=d_ff,
        dropout=0.0,
        embeddings=src_embeddings,
    )
    decoder = TransformerDecoder(
        num_layers=2,
        d_model=d_model,
        heads=heads,
        d_ff=d_ff,
        attn_type="general",
        copy_attn=False,
        self_attn_type="scaled-dot",
        dropout=0.0,
        embeddings=tgt_embeddings,
    )
    model = NMTModel(encoder, decoder)
    generator = nn.Sequential(nn.Linear(d_model, vocab_size), nn.LogSoftmax(dim=-1))

    return RetroPrimeSeq2Seq(model, generator).eval()


def example_input_retroprime():
    src_len, tgt_len, batch = 7, 5, 2
    vocab_size = 48
    src = torch.randint(2, vocab_size, (src_len, batch, 1), dtype=torch.long)
    tgt = torch.randint(2, vocab_size, (tgt_len, batch, 1), dtype=torch.long)
    lengths = torch.full((batch,), src_len, dtype=torch.long)
    return (src, tgt, lengths)


MENAGERIE_ENTRIES = [
    ("RetroPrime", build_retroprime, example_input_retroprime, 2021, MENAGERIE_ZOO),
]

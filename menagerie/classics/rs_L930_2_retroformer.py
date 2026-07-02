# SOURCE: vendored from https://github.com/yuewan2/Retroformer @ master
# Files vendored (near-verbatim; only the `retroformer.models.*` import prefixes were
# collapsed since everything lives in this one file): retroformer/models/embedding.py
# (TokenEmbedding/PositionalEmbedding/Embedding), retroformer/models/module.py
# (SSP/PositionwiseFeedForward/MultiHeadedAttention/LayerNorm), retroformer/models/
# encoder.py (TransformerEncoderLayer/TransformerEncoder), retroformer/models/decoder.py
# (TransformerDecoderLayer/TransformerDecoder), retroformer/models/model.py (RetroModel).
#
# Retroformer (Wan et al., ICML 2022) is a Transformer for single-step retrosynthesis
# prediction with a "local-global attention" mechanism: the encoder augments standard
# self-attention with a molecular-graph bond-edge-conditioned local head (reaction-center
# aware), and reaction-center identification scores gate a local decoder attention head.
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# retroformer/models/embedding.py
# ---------------------------------------------------------------------------
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512, padding_idx=1):
        super().__init__(vocab_size, embed_size, padding_idx=padding_idx)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(0)].transpose(0, 1)


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1, padding_idx=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.token = TokenEmbedding(
            vocab_size=vocab_size, embed_size=embed_size, padding_idx=padding_idx
        )
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=512)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.word_padding_idx = padding_idx

    def forward(self, sequence, step=None):
        output = self.token(sequence) + self.position(sequence)
        if step is None:
            return self.dropout(output)
        return self.dropout(output)[step].unsqueeze(0)


# ---------------------------------------------------------------------------
# retroformer/models/module.py
# ---------------------------------------------------------------------------
class SSP(nn.Softplus):
    def __init__(self, beta=1, threshold=20):
        super().__init__(beta, threshold)

    def forward(self, input):
        sp0 = F.softplus(torch.zeros(1), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0


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


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from OpenNMT, extended with a local-global
    edge-conditioned attention head (Retroformer's reaction-center mechanism)."""

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

        self.edge_project = nn.Sequential(
            nn.Linear(model_dim, model_dim), SSP(), nn.Linear(model_dim, model_dim // 2)
        )
        self.edge_update = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim), SSP(), nn.Linear(model_dim, model_dim)
        )

    def forward(
        self,
        key,
        value,
        query,
        mask=None,
        additional_mask=None,
        layer_cache=None,
        type=None,
        edge_feature=None,
        pair_indices=None,
    ):
        global query_projected, key_shaped, value_shaped
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
                query_projected, key_projected, value_projected = (
                    self.linear_query(query),
                    self.linear_keys(query),
                    self.linear_values(query),
                )

                key_shaped = shape(key_projected)
                value_shaped = shape(value_projected)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key_shaped = torch.cat(
                            (layer_cache["self_keys"].to(device), key_shaped), dim=2
                        )
                    if layer_cache["self_values"] is not None:
                        value_shaped = torch.cat(
                            (layer_cache["self_values"].to(device), value_shaped), dim=2
                        )
                    layer_cache["self_keys"] = key_shaped
                    layer_cache["self_values"] = value_shaped
            elif type == "context":
                query_projected = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key_projected, value_projected = (
                            self.linear_keys(key),
                            self.linear_values(value),
                        )
                        key_shaped = shape(key_projected)
                        value_shaped = shape(value_projected)
                    else:
                        key_shaped, value_shaped = (
                            layer_cache["memory_keys"],
                            layer_cache["memory_values"],
                        )
                    layer_cache["memory_keys"] = key_shaped
                    layer_cache["memory_values"] = value_shaped
                else:
                    key_projected, value_projected = (
                        self.linear_keys(key),
                        self.linear_values(value),
                    )
                    key_shaped = shape(key_projected)
                    value_shaped = shape(value_projected)
        else:
            key_projected = self.linear_keys(key)
            value_projected = self.linear_values(value)
            query_projected = self.linear_query(query)
            key_shaped = shape(key_projected)
            value_shaped = shape(value_projected)

        query_shaped = shape(query_projected)
        key_len = key_shaped.size(2)
        query_len = query_shaped.size(2)

        if edge_feature is None and additional_mask is not None:
            query_shaped = query_shaped / math.sqrt(dim_per_head)
            query_shaped_global, query_shaped_local = (
                query_shaped[:, : head_count // 2],
                query_shaped[:, head_count // 2 :],
            )
            key_shaped_global, key_shaped_local = (
                key_shaped[:, : head_count // 2],
                key_shaped[:, head_count // 2 :],
            )
            value_shaped_global, value_shaped_local = (
                value_shaped[:, : head_count // 2],
                value_shaped[:, head_count // 2 :],
            )

            score_global = torch.matmul(query_shaped_global, key_shaped_global.transpose(2, 3))
            top_score = score_global.view(batch_size, score_global.shape[1], query_len, key_len)[
                :, 0, :, :
            ].contiguous()
            if mask is not None:
                mask = mask.unsqueeze(1).expand_as(score_global).clone()
                score_global = score_global.masked_fill(mask, -1e18)
            attn = self.softmax(score_global)
            drop_attn = self.dropout(attn)
            global_context = torch.matmul(drop_attn, value_shaped_global)

            score_local = torch.matmul(query_shaped_local, key_shaped_local.transpose(2, 3))
            if additional_mask is not None:
                additional_mask = (
                    additional_mask.unsqueeze(1).unsqueeze(2).expand_as(score_local).clone()
                )
                score_local = score_local.masked_fill(additional_mask, -1e18)
            attn = self.softmax(score_local)
            drop_attn = self.dropout(attn)
            local_context = torch.matmul(drop_attn, value_shaped_local)

            context = torch.cat([global_context, local_context], dim=1)
            context = unshape(context)

        elif edge_feature is not None:
            edge_feature_shaped = self.edge_project(edge_feature).view(
                -1, head_count // 2, dim_per_head
            )
            key_shaped_local = key_shaped[pair_indices[0], head_count // 2 :, pair_indices[2]]
            query_shaped_local = query_shaped[pair_indices[0], head_count // 2 :, pair_indices[1]]
            value_shaped_local = value_shaped[:, head_count // 2 :]

            key_shaped_local = key_shaped_local * edge_feature_shaped
            query_shaped_local = query_shaped_local / math.sqrt(dim_per_head)

            scores_local = torch.matmul(
                query_shaped_local.unsqueeze(2), key_shaped_local.unsqueeze(3)
            ).view(edge_feature.shape[0], head_count // 2)

            score_expand_local = scores_local.new_full(
                (value.shape[0], value.shape[1], value.shape[1], head_count // 2), -float("inf")
            )
            score_expand_local[pair_indices] = scores_local
            score_expand_local = score_expand_local.transpose(1, 3).transpose(2, 3)

            attn_local = self.softmax(score_expand_local)
            attn_local = attn_local.masked_fill(score_expand_local < -10000, 0)
            drop_attn_local = self.dropout(attn_local)
            local_context = torch.matmul(drop_attn_local, value_shaped_local)

            query_shaped_global = query_shaped[:, : head_count // 2]
            key_shaped_global = key_shaped[:, : head_count // 2]
            value_shaped_global = value_shaped[:, : head_count // 2]

            query_shaped_global = query_shaped_global / math.sqrt(dim_per_head)
            score_global = torch.matmul(query_shaped_global, key_shaped_global.transpose(2, 3))
            top_score = score_global.view(batch_size, score_global.shape[1], query_len, key_len)[
                :, 0, :, :
            ].contiguous()
            if mask is not None:
                mask = mask.unsqueeze(1).expand_as(score_global).clone()
                score_global = score_global.masked_fill(mask, -1e18)

            attn = self.softmax(score_global)
            drop_attn = self.dropout(attn)
            global_context = torch.matmul(drop_attn, value_shaped_global)

            context = torch.cat([global_context, local_context], dim=1)
            context = unshape(context)

        else:
            query_shaped = query_shaped / math.sqrt(dim_per_head)
            scores = torch.matmul(query_shaped, key_shaped.transpose(2, 3))
            top_score = scores.view(batch_size, scores.shape[1], query_len, key_len)[
                :, 0, :, :
            ].contiguous()
            if mask is not None:
                mask = mask.unsqueeze(1).expand_as(scores).clone()
                if additional_mask is not None:
                    additional_mask = additional_mask.unsqueeze(1).expand(
                        (batch_size, head_count // 2, query_len, key_len)
                    )
                    mask[:, mask.shape[1] // 2 :] = additional_mask
                scores = scores.masked_fill(mask, -1e18)
            attn = self.softmax(scores)
            drop_attn = self.dropout(attn)
            context = torch.matmul(drop_attn, value_shaped)
            context = unshape(context)

        output = self.final_linear(context)

        if edge_feature is not None:
            node_feature_updated = output
            node_features = torch.cat(
                [
                    node_feature_updated[pair_indices[0], pair_indices[1]],
                    node_feature_updated[pair_indices[0], pair_indices[2]],
                ],
                dim=-1,
            )
            edge_feature_updated = self.edge_update(node_features)
            return output, top_score, edge_feature_updated
        return output, top_score, None


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


# ---------------------------------------------------------------------------
# retroformer/models/encoder.py
# ---------------------------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attn):
        super().__init__()
        self.self_attn = attn
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask, edge_feature, pair_indices):
        input_norm = self.layer_norm(inputs)
        context, attn, edge_feature_updated = self.self_attn(
            input_norm,
            input_norm,
            input_norm,
            mask=mask,
            edge_feature=edge_feature,
            pair_indices=pair_indices,
        )

        out = self.dropout(context) + inputs
        if edge_feature is not None:
            edge_feature = self.layer_norm(edge_feature + edge_feature_updated)
        return self.feed_forward(out), attn, edge_feature


class TransformerEncoder(nn.Module):
    def __init__(
        self, num_layers, d_model, heads, d_ff, dropout, embeddings, embeddings_bond, attn_modules
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.embeddings_bond = embeddings_bond
        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, heads, d_ff, dropout, attn_modules[i])
                for i in range(num_layers)
            ]
        )
        self.layer_norm = LayerNorm(d_model)

    def forward(self, src, bond=None):
        """
        :param src: [src_len, batch_size]
        :param bond: [batch_size, src_len, src_len, 7]
        """
        global node_feature
        emb = self.embeddings(src)
        out = emb.transpose(0, 1).contiguous()

        if bond is not None:
            pair_indices = torch.where(bond.sum(-1) > 0)
            valid_bond = bond[bond.sum(-1) > 0]
            edge_feature = self.embeddings_bond(valid_bond.float())
        else:
            pair_indices, edge_feature = None, None

        words = src.transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1).expand(w_batch, w_len, w_len)

        for i in range(self.num_layers):
            out, attn, edge_feature = self.transformer[i](out, mask, edge_feature, pair_indices)

        out = self.layer_norm(out)
        out = out.transpose(0, 1).contiguous()
        edge_out = self.layer_norm(edge_feature) if edge_feature is not None else None
        return out, edge_out


# ---------------------------------------------------------------------------
# retroformer/models/decoder.py
# ---------------------------------------------------------------------------
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, self_attn, context_attn):
        super().__init__()
        self.self_attn = self_attn
        self.context_attn = context_attn
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
        nonreactive_mask_input=None,
        layer_input=None,
        layer_cache=None,
    ):
        dec_mask = torch.gt(
            tgt_pad_mask + self.mask[:, : tgt_pad_mask.size(1), : tgt_pad_mask.size(1)], 0
        )
        input_norm = self.layer_norm_1(inputs)

        all_input = input_norm
        if layer_input is not None:
            all_input = torch.cat((layer_input, input_norm), dim=1)
            dec_mask = None
        query, self_attn, _ = self.self_attn(
            all_input, all_input, input_norm, mask=dec_mask, type="self", layer_cache=layer_cache
        )
        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)

        mid, context_attn, _ = self.context_attn(
            memory_bank,
            memory_bank,
            query_norm,
            mask=src_pad_mask,
            additional_mask=nonreactive_mask_input,
            type="context",
            layer_cache=layer_cache,
        )
        output = self.feed_forward(self.drop(mid) + query)

        return output, context_attn, all_input

    def _get_attn_subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, self_attn_modules):
        super().__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

        context_attn_modules = nn.ModuleList(
            [MultiHeadedAttention(heads, d_model, dropout=dropout) for _ in range(num_layers)]
        )

        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model, heads, d_ff, dropout, self_attn_modules[i], context_attn_modules[i]
                )
                for i in range(num_layers)
            ]
        )

        self.layer_norm_0 = LayerNorm(d_model)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, src, tgt, memory_bank, nonreactive_mask=None, state_cache=None, step=None):
        if nonreactive_mask is not None:
            nonreactive_mask[0] = False

        src_words = src.transpose(0, 1)
        tgt_words = tgt.transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        outputs = []

        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3
        if step is not None:
            tgt_words = tgt[-1].unsqueeze(0).transpose(0, 1)
            tgt_batch, tgt_len = tgt_words.size()

        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        src_pad_mask = (
            src_words.data.eq(padding_idx).unsqueeze(1).expand(src_batch, tgt_len, src_len)
        )
        tgt_pad_mask = (
            tgt_words.data.eq(padding_idx).unsqueeze(1).expand(tgt_batch, tgt_len, tgt_len)
        )

        nonreactive_mask_input = (
            nonreactive_mask.transpose(0, 1) if nonreactive_mask is not None else None
        )
        top_context_attns = []
        for i in range(self.num_layers):
            layer_input = None
            layer_cache = {
                "self_keys": None,
                "self_values": None,
                "memory_keys": None,
                "memory_values": None,
            }
            if state_cache is not None:
                layer_cache = state_cache.get("layer_cache_{}".format(i), layer_cache)
                layer_input = state_cache.get("layer_input_{}".format(i), layer_input)

            output, top_context_attn, all_input = self.transformer_layers[i](
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_input=layer_input,
                layer_cache=layer_cache,
                nonreactive_mask_input=nonreactive_mask_input,
            )

            top_context_attns.append(top_context_attn)
            if state_cache is not None:
                state_cache["layer_cache_{}".format(i)] = layer_cache
                state_cache["layer_input_{}".format(i)] = all_input

        output = self.layer_norm(output)
        outputs = output.transpose(0, 1).contiguous()

        return outputs, top_context_attns


# ---------------------------------------------------------------------------
# retroformer/models/model.py :: RetroModel
# ---------------------------------------------------------------------------
class RetroModel(nn.Module):
    def __init__(
        self,
        encoder_num_layers,
        decoder_num_layers,
        d_model,
        heads,
        d_ff,
        dropout,
        vocab_size_src,
        vocab_size_tgt,
        shared_vocab,
        num_bonds=5,
        shared_encoder=False,
        src_pad_idx=1,
        tgt_pad_idx=1,
    ):
        super().__init__()
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.shared_vocab = shared_vocab
        self.shared_encoder = shared_encoder
        if shared_vocab:
            assert vocab_size_src == vocab_size_tgt and src_pad_idx == tgt_pad_idx
            self.embedding_src = self.embedding_tgt = Embedding(
                vocab_size=vocab_size_src + 1, embed_size=d_model, padding_idx=src_pad_idx
            )
        else:
            self.embedding_src = Embedding(
                vocab_size=vocab_size_src + 1, embed_size=d_model, padding_idx=src_pad_idx
            )
            self.embedding_tgt = Embedding(
                vocab_size=vocab_size_tgt + 1, embed_size=d_model, padding_idx=tgt_pad_idx
            )

        self.embedding_bond = nn.Linear(7, d_model)

        multihead_attn_modules_en = nn.ModuleList(
            [
                MultiHeadedAttention(heads, d_model, dropout=dropout)
                for _ in range(encoder_num_layers)
            ]
        )
        if shared_encoder:
            assert encoder_num_layers == decoder_num_layers
            multihead_attn_modules_de = multihead_attn_modules_en
        else:
            multihead_attn_modules_de = nn.ModuleList(
                [
                    MultiHeadedAttention(heads, d_model, dropout=dropout)
                    for _ in range(decoder_num_layers)
                ]
            )

        self.encoder = TransformerEncoder(
            num_layers=encoder_num_layers,
            d_model=d_model,
            heads=heads,
            d_ff=d_ff,
            dropout=dropout,
            embeddings=self.embedding_src,
            embeddings_bond=self.embedding_bond,
            attn_modules=multihead_attn_modules_en,
        )

        self.decoder = TransformerDecoder(
            num_layers=decoder_num_layers,
            d_model=d_model,
            heads=heads,
            d_ff=d_ff,
            dropout=dropout,
            embeddings=self.embedding_tgt,
            self_attn_modules=multihead_attn_modules_de,
        )

        self.atom_rc_identifier = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.bond_rc_identifier = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        self.generator = nn.Sequential(nn.Linear(d_model, vocab_size_tgt), nn.LogSoftmax(dim=-1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt, bond=None, teacher_mask=None):
        encoder_out, edge_feature = self.encoder(src, bond)

        atom_rc_scores = self.atom_rc_identifier(encoder_out)
        bond_rc_scores = self.bond_rc_identifier(edge_feature) if edge_feature is not None else None

        if teacher_mask is None:
            student_mask = self.infer_reaction_center_mask(bond, atom_rc_scores, bond_rc_scores)
            decoder_out, top_aligns = self.decoder(src, tgt[:-1], encoder_out, student_mask.clone())
        else:
            decoder_out, top_aligns = self.decoder(src, tgt[:-1], encoder_out, teacher_mask.clone())

        generative_scores = self.generator(decoder_out)

        return generative_scores, atom_rc_scores, bond_rc_scores, top_aligns

    @staticmethod
    def infer_reaction_center_mask(bond, atom_rc_scores, bond_rc_scores=None):
        atom_rc_scores = atom_rc_scores.squeeze(2)
        if bond_rc_scores is not None:
            bond_rc_scores = bond_rc_scores.squeeze(1)
            bond_indicator = (
                torch.zeros((bond.shape[0], bond.shape[1], bond.shape[2])).bool().to(bond.device)
            )
            bond_indicator[bond.sum(-1) > 0] = bond_rc_scores > 0.5

            result = (
                ~(bond_indicator.sum(dim=1).bool())
                + ~(bond_indicator.sum(dim=2).bool())
                + (atom_rc_scores.transpose(0, 1) < 0.5)
            ).transpose(0, 1)
        else:
            result = (atom_rc_scores.transpose(0, 1) < 0.5).transpose(0, 1)
        return result


# ---------------------------------------------------------------------------
# Staging harness (tiny random-init construction + example input)
# ---------------------------------------------------------------------------
def build_retroformer():
    return RetroModel(
        encoder_num_layers=2,
        decoder_num_layers=2,
        d_model=16,
        heads=4,
        d_ff=32,
        dropout=0.0,
        vocab_size_src=32,
        vocab_size_tgt=32,
        shared_vocab=True,
        src_pad_idx=1,
        tgt_pad_idx=1,
    ).eval()


def example_input_retroformer():
    src_len, tgt_len, batch = 6, 5, 2
    vocab_size = 32
    src = torch.randint(2, vocab_size, (src_len, batch), dtype=torch.long)
    tgt = torch.randint(2, vocab_size, (tgt_len, batch), dtype=torch.long)
    bond = torch.zeros(batch, src_len, src_len, 7)
    # sparsely activate a few real bonds so the encoder's edge-conditioned attention path runs
    bond[:, 0, 1, 1] = 1.0
    bond[:, 1, 0, 1] = 1.0
    bond[:, 2, 3, 2] = 1.0
    bond[:, 3, 2, 2] = 1.0
    return (src, tgt, bond)


MENAGERIE_ENTRIES = [
    ("Retroformer", build_retroformer, example_input_retroformer, 2022, MENAGERIE_ZOO),
]

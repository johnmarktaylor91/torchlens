# SOURCE: vendored from wenhuchen/HDSA-Dialog @ 88e2604adb5dc38ae32205410b15b2ac39116ecd
# https://raw.githubusercontent.com/wenhuchen/HDSA-Dialog/88e2604adb5dc38ae32205410b15b2ac39116ecd/transformer/Transformer.py
# https://raw.githubusercontent.com/wenhuchen/HDSA-Dialog/88e2604adb5dc38ae32205410b15b2ac39116ecd/transformer/Constants.py
# https://raw.githubusercontent.com/wenhuchen/HDSA-Dialog/88e2604adb5dc38ae32205410b15b2ac39116ecd/train_generator.py (decoder selection)
#
# Chen, Chen, Qin, Yan, Wang 2019 (ACL) "Semantically Conditioned Dialog Response
# Generation via Hierarchical Disentangled Self-Attention" (HDSA). The response
# GENERATOR is `TableSemanticDecoder` (train_generator.py: `decoder =
# TableSemanticDecoder(...)` when `args.field` is set), which conditions
# transformer decoding on a predicted dialog-act vector split into three
# semantic LEVELS -- domain / function / argument -- and threads them through
# three successive disentangled cross-attention blocks (`AvgDecoderLayer`,
# using `AverageHeadAttention` to average, rather than concatenate, per-head
# attention outputs, so each head specializes to one act slice) before a
# final ordinary `DecoderLayer`. This hierarchical disentangled self-attention
# stack over the dialog-act table is what distinguishes HDSA from a plain
# Transformer decoder; the (separate, BERT-based) `train_predictor.py` script
# that predicts the act vector in the first place is out of scope here (that
# component is stock `BertForSequenceClassification`, not part of HDSA's own
# architectural contribution).
#
# `Transformer`/`TransformerDecoder`/`TableSemanticDecoder`/`AvgDecoderLayer`/
# `AverageHeadAttention`/`MultiHeadAttention`/`EncoderLayer`/`DecoderLayer`/
# `PositionwiseFeedForward`/`ScaledDotProductAttention`/`PositionalEmbedding`
# are copied verbatim from `transformer/Transformer.py` (only `TableSemanticDecoder`
# plus the `Transformer` act-conditioned utterance encoder from the same file are
# exercised below; `Transformer` is included because `TableSemanticDecoder`'s own
# forward duplicates its architecture inline rather than instantiating it, so it is
# not itself part of the traced call graph, but is kept for source-completeness of
# the module since the same file also exports `TransformerDecoder`/beam-search
# helpers used elsewhere in the repo).
#
# No architectural changes were made; only mechanical fixes for import isolation:
#   - `transformer/Constants.py` loads `data/act_ontology.json` and
#     `data/belief_state.json` from the repo's data directory at import time,
#     which is unavailable/irrelevant outside the original repo checkout. Only
#     the handful of pure-Python constants this file's traced code path actually
#     needs (`PAD`, `domains`, `functions`, `arguments`) are reproduced verbatim
#     below as a local `Constants` shim (`used_levels`/`act_len` derived exactly
#     as upstream); the JSON-backed `act_ontology`/`belief_state`/`act_to_vectors`
#     symbols are omitted since `TableSemanticDecoder.forward` never touches them.
#   - `transformer/Beam.py` (`from .Beam import Beam`) is only used by
#     `translate_batch` (beam-search decoding, not the traced training-mode
#     forward pass); omitted here since it pulls in a whole extra beam-search
#     module for a code path this recipe does not exercise, and `Beam`'s own
#     import (`from . import Constants`) would otherwise need the same shim.
#     `TableSemanticDecoder.translate_batch` is therefore dropped from this
#     vendored copy; `forward` (the actual disentangled-self-attention decoder
#     computation) is unchanged.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Constants:
    """Minimal shim: only the plain-Python constants TableSemanticDecoder.forward
    and its layer stack need, copied verbatim from transformer/Constants.py.
    The JSON-file-backed act_ontology/belief_state lookups are dropped (unused
    on this file's traced call path -- see header note)."""

    PAD = 0

    domains = [
        "restaurant",
        "hotel",
        "attraction",
        "train",
        "taxi",
        "hospital",
        "police",
        "bus",
        "booking",
        "general",
    ]
    functions = ["inform", "request", "recommend", "book", "select", "sorry", "none"]
    arguments = [
        "pricerange",
        "id",
        "address",
        "postcode",
        "type",
        "food",
        "phone",
        "name",
        "area",
        "choice",
        "price",
        "time",
        "reference",
        "none",
        "parking",
        "stars",
        "internet",
        "day",
        "arriveby",
        "departure",
        "destination",
        "leaveat",
        "duration",
        "trainid",
        "people",
        "department",
        "stay",
    ]

    used_levels = domains + functions + arguments
    act_len = len(used_levels)


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """Sinusoid position encoding table"""

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    """For masking out the padding part of key sequence."""

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1
    )
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class AverageHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(AverageHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, a, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        a = a.permute(1, 0).contiguous()[:, :, None, None]

        # output = output * a
        output = torch.sum(output * a, 0)
        output = output.view(sz_b, len_q, -1)
        # output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class Transformer(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(
        self,
        n_src_vocab,
        len_max_seq,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        embedding,
        dropout=0.1,
    ):
        super(Transformer, self).__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        # self.src_word_emb = nn.Embedding.from_pretrained(embedding, freeze=False)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0), freeze=True
        )

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, src_pos, act_vocab_id):
        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward Word Embedding
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        # -- Forward Ontology Embedding
        ontology_embedding = self.src_word_emb(act_vocab_id)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask
            )

        dot_prod = torch.sum(enc_output[:, :, None, :] * ontology_embedding[None, None, :, :], -1)
        pooled_dot_prod = dot_prod[:, 0, :]
        pooling_likelihood = torch.sigmoid(pooled_dot_prod)
        return pooling_likelihood, enc_output


class AvgDecoderLayer(nn.Module):
    """Compose with three layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, n_head_enc, dropout=0.1):
        super(AvgDecoderLayer, self).__init__()
        self.slf_attn = AverageHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(
            n_head_enc, d_model, d_model // n_head_enc, d_model // n_head_enc, dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
        self,
        act_vecs,
        dec_input,
        enc_output,
        non_pad_mask=None,
        slf_attn_mask=None,
        dec_enc_attn_mask=None,
    ):
        dec_output, dec_slf_attn = self.slf_attn(
            act_vecs, dec_input, dec_input, dec_input, mask=slf_attn_mask
        )
        dec_output *= non_pad_mask
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask
        )
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, None


class DecoderLayer(nn.Module):
    """Compose with three layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
        self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None
    ):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask
        )
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask
        )
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn


class TableSemanticDecoder(nn.Module):
    def __init__(self, vocab_size, d_word_vec, n_layers, d_model, n_head, dropout=0.1):
        super(TableSemanticDecoder, self).__init__()
        self.take_domain = True

        self.tgt_word_emb = nn.Embedding(vocab_size, d_word_vec, padding_idx=Constants.PAD)
        self.post_word_emb = PositionalEmbedding(d_model=d_word_vec)

        d_inner = d_model * 4
        d_k, d_v = d_model // n_head, d_model // n_head

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        d_inner = d_model * 4
        d_k, d_v = d_model // n_head, d_model // n_head
        if self.take_domain:
            self.prior_layer_stack = AvgDecoderLayer(
                d_model,
                d_inner,
                len(Constants.domains),
                d_k,
                d_v,
                n_head_enc=n_head,
                dropout=dropout,
            )
            self.middle_layer_stack = AvgDecoderLayer(
                d_model,
                d_inner,
                len(Constants.functions),
                d_k,
                d_v,
                n_head_enc=n_head,
                dropout=dropout,
            )
            self.post_layer_stack = AvgDecoderLayer(
                d_model,
                d_inner,
                len(Constants.arguments),
                d_k,
                d_v,
                n_head_enc=n_head,
                dropout=dropout,
            )
            self.final_layer_stack = DecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout
            )
        else:
            self.prior_layer_stack = AvgDecoderLayer(
                d_model,
                d_inner,
                len(Constants.functions),
                d_k,
                d_v,
                n_head_enc=n_head,
                dropout=dropout,
            )
            self.middle_layer_stack = AvgDecoderLayer(
                d_model,
                d_inner,
                len(Constants.arguments),
                d_k,
                d_v,
                n_head_enc=n_head,
                dropout=dropout,
            )
            self.post_layer_stack = DecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout
            )

        self.tgt_word_prj = nn.Linear(d_model, vocab_size, bias=False)
        self.softmax = nn.Softmax(-1)

    def forward(self, tgt_seq, src_seq, act_vecs):
        # -- Encode source
        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        enc_inp = self.tgt_word_emb(src_seq) + self.post_word_emb(src_seq)

        for layer in self.layer_stack:
            enc_inp, _ = layer(enc_inp, non_pad_mask, slf_attn_mask)
        enc_output = enc_inp

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_inp = self.tgt_word_emb(tgt_seq) + self.post_word_emb(tgt_seq)
        domain_vecs = act_vecs[:, : len(Constants.domains)]
        function_vecs = act_vecs[
            :, len(Constants.domains) : len(Constants.domains) + len(Constants.functions)
        ]
        argument_vecs = act_vecs[:, len(Constants.domains) + len(Constants.functions) :]
        if self.take_domain:
            dec_inp, _, _ = self.prior_layer_stack(
                domain_vecs,
                dec_inp,
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask,
            )
            dec_inp, _, _ = self.middle_layer_stack(
                function_vecs,
                dec_inp,
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask,
            )
            dec_inp, _, _ = self.post_layer_stack(
                argument_vecs,
                dec_inp,
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask,
            )
            dec_inp, _, _ = self.final_layer_stack(
                dec_inp,
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask,
            )
        else:
            dec_inp, _, _ = self.prior_layer_stack(
                function_vecs,
                dec_inp,
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask,
            )
            dec_inp, _, _ = self.middle_layer_stack(
                argument_vecs,
                dec_inp,
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask,
            )
            dec_inp, _, _ = self.post_layer_stack(
                dec_inp,
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask,
            )
        logits = self.tgt_word_prj(dec_inp)
        return logits


def build_hdsa():
    vocab_size = 200
    d_word_vec = 32
    n_layers = 2
    d_model = 32
    n_head = 4
    return TableSemanticDecoder(
        vocab_size=vocab_size,
        d_word_vec=d_word_vec,
        n_layers=n_layers,
        d_model=d_model,
        n_head=n_head,
        dropout=0.1,
    )


def example_input_hdsa():
    batch = 2
    src_len = 10
    tgt_len = 8
    act_len = Constants.act_len  # 10 domains + 7 functions + 27 arguments = 44
    src_seq = torch.randint(1, 200, (batch, src_len))
    tgt_seq = torch.randint(1, 200, (batch, tgt_len))
    act_vecs = torch.randint(0, 2, (batch, act_len)).float()
    return (tgt_seq, src_seq, act_vecs)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("HDSA", "build_hdsa", "example_input_hdsa", 2019, "vendored"),
]

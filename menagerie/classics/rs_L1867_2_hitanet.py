# SOURCE: vendored from machinelearning4health/HiTANet @ 284cc6ae0325
# (models/transformer.py::TransformerTimeMix, the model class selected by
# `model_choice = 'TransformerTimeMix'` in the repo's own train_hitanet.py entry
# point, plus its full real dependency chain: TimeEncoder, EncoderNew,
# EncoderLayer, MultiHeadAttention, PositionalWiseFeedForward,
# ScaledDotProductAttention, PositionalEncoding, Embedding, padding_mask; and
# the real numpy batch-padding helpers pad_time/pad_matrix_new from
# rnn_tools.py that TransformerTimeMix.forward calls internally). HiTANet is a
# hierarchical time-aware transformer for EHR risk prediction (KDD 2020):
# a per-visit multi-head-attention encoder over diagnosis codes, fused with a
# learned time-decay attention (TimeEncoder) and a self-attention branch,
# mixed via a softmax gate (quiry_weight_layer) into a final binary risk
# classifier. All class bodies below are copied verbatim from the real repo
# files; only the module-level ICD-9 comorbidity code lists (unused by the
# traced forward path) were dropped and the `rnn_tools` module functions this
# file needs were copied in directly to keep the staging file self-contained
# (no relative-import restructuring of the model architecture itself).
"""Vendored HiTANet TransformerTimeMix (machinelearning4health/HiTANet)."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# rnn_tools.py (real repo helpers used inside TransformerTimeMix.forward)
# ---------------------------------------------------------------------------


def pad_time(seq_time_step, options):
    lengths = np.array([len(seq) for seq in seq_time_step])
    maxlen = np.max(lengths)
    for k in range(len(seq_time_step)):
        while len(seq_time_step[k]) < maxlen:
            seq_time_step[k].append(100000)
    return seq_time_step


def pad_matrix_new(seq_diagnosis_codes, seq_labels, options):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    n_diagnosis_codes = options["n_diagnosis_codes"]  # noqa: F841 (unused in original repo code too)
    maxlen = np.max(lengths)
    lengths_code = []
    for seq in seq_diagnosis_codes:
        for code_set in seq:
            lengths_code.append(len(code_set))
    lengths_code = np.array(lengths_code)
    maxcode = np.max(lengths_code)

    batch_diagnosis_codes = (
        np.zeros((n_samples, maxlen, maxcode), dtype=np.int64) + options["n_diagnosis_codes"]
    )
    batch_mask = np.zeros((n_samples, maxlen), dtype=np.float32)
    batch_mask_code = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32)
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)

    for bid, seq in enumerate(seq_diagnosis_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                batch_diagnosis_codes[bid, pid, tid] = code
                batch_mask_code[bid, pid, tid] = 1

    for i in range(n_samples):
        batch_mask[i, 0 : lengths[i] - 1] = 1
        max_visit = lengths[i] - 1
        batch_mask_final[i, max_visit] = 1

    batch_labels = np.array(seq_labels, dtype=np.int64)

    return batch_diagnosis_codes, batch_labels, batch_mask, batch_mask_final, batch_mask_code


# ---------------------------------------------------------------------------
# models/transformer.py (real repo classes, verbatim)
# ---------------------------------------------------------------------------


class Embedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
    ):
        super(Embedding, self).__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
        )

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        position_encoding = np.array(
            [
                [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
                for pos in range(max_seq_len)
            ]
        )
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, input_len):
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        pos = np.zeros([len(input_len), max_len])
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind
        input_pos = tensor(pos)
        return self.position_encoding(input_pos), input_pos


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(torch.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))
        output = self.layer_norm(x + output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        context = context.view(batch_size, -1, dim_per_head * num_heads)

        output = self.linear_final(context)

        output = self.dropout(output)

        output = self.layer_norm(residual + output)

        return output, attention


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        output = self.feed_forward(context)

        return output, attention


def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


class EncoderNew(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        num_layers=1,
        model_dim=256,
        num_heads=4,
        ffn_dim=1024,
        dropout=0.0,
    ):
        super(EncoderNew, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.pre_embedding = Embedding(vocab_size, model_dim)
        self.bias_embedding = torch.nn.Parameter(torch.Tensor(model_dim))
        bound = 1 / math.sqrt(vocab_size)
        init.uniform_(self.bias_embedding, -bound, bound)

        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.time_layer = torch.nn.Linear(64, 256)
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, diagnosis_codes, mask, mask_code, seq_time_step, input_len):
        seq_time_step = torch.Tensor(seq_time_step).cuda().unsqueeze(2) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        time_feature = self.time_layer(time_feature)
        output = (self.pre_embedding(diagnosis_codes) * mask_code).sum(dim=2) + self.bias_embedding
        output += time_feature
        output_pos, ind_pos = self.pos_embedding(input_len.unsqueeze(1))
        output += output_pos
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
        return output


class TimeEncoder(nn.Module):
    def __init__(self, batch_size):
        super(TimeEncoder, self).__init__()
        self.batch_size = batch_size
        self.selection_layer = torch.nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.weight_layer = torch.nn.Linear(64, 64)

    def forward(self, seq_time_step, final_queries, options, mask):
        if options["use_gpu"]:
            seq_time_step = torch.Tensor(seq_time_step).unsqueeze(2).cuda() / 180
        else:
            seq_time_step = torch.Tensor(seq_time_step).unsqueeze(2) / 180
        selection_feature = 1 - self.tanh(torch.pow(self.selection_layer(seq_time_step), 2))
        selection_feature = self.relu(self.weight_layer(selection_feature))
        selection_feature = torch.sum(selection_feature * final_queries, 2, keepdim=True) / 8
        selection_feature = selection_feature.masked_fill_(mask, -np.inf)
        return torch.softmax(selection_feature, 1)


class TransformerTimeMix(nn.Module):
    def __init__(self, n_diagnosis_codes, batch_size, options):
        super(TransformerTimeMix, self).__init__()
        self.time_encoder = TimeEncoder(batch_size)
        self.feature_encoder = EncoderNew(
            options["n_diagnosis_codes"] + 1, 51, num_layers=options["layer"]
        )
        self.self_layer = torch.nn.Linear(256, 1)
        self.classify_layer = torch.nn.Linear(256 * 2, 2)
        self.quiry_layer = torch.nn.Linear(256, 64)
        self.quiry_weight_layer = torch.nn.Linear(256, 2)
        self.relu = nn.ReLU(inplace=True)
        dropout_rate = options["dropout_rate"]
        self.dropout = nn.Dropout(dropout_rate)

    def get_self_attention(self, features, query, mask):
        attention = torch.softmax(self.self_layer(features).masked_fill(mask, -np.inf), dim=1)
        return attention

    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen):
        seq_time_step = np.array(list(pad_time(seq_time_step, options)))
        lengths = torch.from_numpy(np.array([len(seq) for seq in seq_dignosis_codes])).cuda()
        diagnosis_codes, labels, mask, mask_final, mask_code = pad_matrix_new(
            seq_dignosis_codes, batch_labels, options
        )
        if options["use_gpu"]:
            diagnosis_codes = torch.LongTensor(diagnosis_codes).cuda()
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2).cuda()
            mask_final = torch.Tensor(mask_final).unsqueeze(2).cuda()
            mask_code = torch.Tensor(mask_code).unsqueeze(3).cuda()
        else:
            diagnosis_codes = torch.LongTensor(diagnosis_codes)
            mask_mult = torch.BoolTensor(1 - mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code).unsqueeze(3)
        features = self.feature_encoder(
            diagnosis_codes, mask_mult, mask_code, seq_time_step, lengths
        )
        final_statues = features * mask_final
        final_statues = final_statues.sum(1, keepdim=True)

        quiryes = self.relu(self.quiry_layer(final_statues))

        self_weight = self.get_self_attention(features, quiryes, mask_mult)
        time_weight = self.time_encoder(seq_time_step, quiryes, options, mask_mult)
        attention_weight = torch.softmax(self.quiry_weight_layer(final_statues), 2)

        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-5)
        weighted_features = features * total_weight
        averaged_features = torch.sum(weighted_features, 1)
        mix_features = torch.cat([averaged_features, final_statues.squeeze()], 1)
        averaged_features = self.dropout(mix_features)
        predictions = self.classify_layer(mix_features)
        labels = torch.LongTensor(labels)
        if options["use_gpu"]:
            labels = labels.cuda()
        return predictions, labels, self_weight


# ---------------------------------------------------------------------------
# Staging build/example helpers. TransformerTimeMix.forward consumes raw
# nested Python lists (variable-length per-patient visit sequences of
# diagnosis-code lists) plus a plain `options` dict, exactly like the real
# `train_hitanet.py` training loop feeds it -- not plain tensors -- so this
# is a MODULE (multi-argument, non-tensor input contract), not a tensor-only
# recipe. `use_gpu=True` is required because EncoderNew.forward hardcodes a
# `.cuda()` call unconditionally in the real repo code (this box has CUDA).
# Vocabulary/embedding sizes are shrunk from the paper's real ICD-9 vocab
# (n_diagnosis_codes~4894+) to a tiny random-init probe; the visit/code
# structure (nested variable-length code lists per visit, per patient) and
# every model dimension inside TransformerTimeMix are left untouched.
# ---------------------------------------------------------------------------


def build_hitanet():
    n_diagnosis_codes = 50
    options = {
        "n_diagnosis_codes": n_diagnosis_codes,
        "dropout_rate": 0.5,
        "layer": 1,
        "use_gpu": True,
    }
    model = TransformerTimeMix(n_diagnosis_codes, batch_size=4, options=options)
    model.cuda()
    model.eval()
    return model


def example_input_hitanet():
    n_diagnosis_codes = 50
    options = {
        "n_diagnosis_codes": n_diagnosis_codes,
        "dropout_rate": 0.5,
        "layer": 1,
        "use_gpu": True,
        "batch_size": 4,
    }
    rng = np.random.RandomState(0)
    batch_size = 4
    seq_dignosis_codes = []
    seq_time_step = []
    batch_labels = []
    for _ in range(batch_size):
        n_visits = int(rng.randint(2, 5))
        visits = []
        times = []
        for v in range(n_visits):
            n_codes = int(rng.randint(1, 4))
            codes = [int(c) for c in rng.randint(0, n_diagnosis_codes, size=n_codes)]
            visits.append(codes)
            times.append(int(rng.randint(0, 365)))
        seq_dignosis_codes.append(visits)
        seq_time_step.append(times)
        batch_labels.append(int(rng.randint(0, 2)))
    maxlen = max(len(seq) for seq in seq_dignosis_codes)
    return (seq_dignosis_codes, seq_time_step, batch_labels, options, maxlen)


MENAGERIE_ENTRIES = [
    ("HiTANet", "build_hitanet", "example_input_hitanet", 2020, "vendored-pytorch"),
]

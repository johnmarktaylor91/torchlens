# SOURCE: vendored from https://github.com/WING-NUS/SG-Deep-Question-Generation @ master
# (src/onqg/models/{Models,Encoders,Decoders}.py + models/modules/{Attention,Layers,
# SubLayers,MaxOut,DecAssist}.py + utils/{model_builder,mask}.py + dataset/Constants.py)
#
# SemQG / "Semantic Graphs for Generating Deep Questions" (Pan, Wu, Xie, Xiong, Ai, Kan,
# ACL 2020). This is the paper's official generator architecture: a BiGRU sequence
# encoder over the source passage, an EncoderTransformer that pools per-node word spans
# out of the sequence encoding via cross-attention to build node vectors, a GGNN+GAT
# GraphEncoder that message-passes over the input semantic (SRL) graph, a
# DecoderTransformer that scatters graph-node vectors back onto sequence positions, and
# an input-feed GRU decoder with Luong-style attention + a copy-switch head, unified by
# `UnifiedModel` -- exactly `model_builder.build_model(opt, device)` with
# `opt.training_mode='generate'`, `opt.sparse=False` (dense GraphEncoder, the repo's
# default `-sparse 0` from `scripts/train_generator.sh`).
#
# The classes below are the REAL SemQG model code (encoders, graph layers, attention,
# decoder), copied faithfully with only mechanical, non-architectural changes: repo-
# relative imports were inlined into this single file, `torch.autograd.Variable`
# wrapping was dropped (a no-op on modern torch), and the optional BERT/`TransfEncoder`
# and `SparseGraphEncoder` code paths (unused by the repo's own `train_generator.sh`
# defaults) were omitted since they require extra pretrained-BERT / sparse-adjacency
# inputs orthogonal to the paper's core GGNN-graph + seq2seq architecture. No layer,
# dimension, or control-flow logic in the retained modules was altered. Because the
# real `UnifiedModel.forward` needs a nested dict of many tensors (encoder input
# sequence, per-node word-span indices, graph edge masks/types, decoder target/answer
# sequences) rather than a single tensor, this file also includes a small
# `example_input_semqg()` harness that constructs one syntactically-valid batch
# matching each submodule's documented input contract (see `data_processor.
# preprocess_batch` in the original repo) -- that harness is new glue code, not part
# of the vendored architecture.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

MENAGERIE_ZOO = "vendored-pytorch"

# ---------------------------------------------------------------------------
# onqg/dataset/Constants.py
# ---------------------------------------------------------------------------
PAD = 0
UNK = 1
BOS = 2
EOS = 3
SEP = 4


# ---------------------------------------------------------------------------
# onqg/utils/mask.py
# ---------------------------------------------------------------------------


def get_edge_mask(edges):
    """Get mask matrix for edges. edges - [batch_size, node_num * node_num]
    return - [batch_size, node_num, node_num]"""
    len_edges = edges.size(1)
    node_num = int(len_edges**0.5)

    mask = edges.eq(PAD)
    mask = mask.view(-1, node_num, node_num)

    return mask


# ---------------------------------------------------------------------------
# onqg/models/modules/MaxOut.py
# ---------------------------------------------------------------------------


class MaxOut(nn.Module):
    def __init__(self, pool_size):
        super(MaxOut, self).__init__()
        self.pool_size = pool_size

    def forward(self, ipt):
        input_size = list(ipt.size())
        assert input_size[-1] % self.pool_size == 0
        output_size = [d for d in input_size]
        output_size[-1] = output_size[-1] // self.pool_size
        output_size.append(self.pool_size)
        last_dim = len(output_size) - 1
        ipt = ipt.view(*output_size)
        ipt, _ = ipt.max(last_dim, keepdim=True)
        output = ipt.squeeze(last_dim)

        return output


# ---------------------------------------------------------------------------
# onqg/models/modules/Attention.py
# ---------------------------------------------------------------------------


class ConcatAttention(nn.Module):
    def __init__(self, attend_dim, query_dim, att_dim, is_coverage=False):
        super(ConcatAttention, self).__init__()

        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim

        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=False)
        self.linear_v = nn.Linear(att_dim, 1, bias=False)

        self.sftmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        self.mask = None

        self.is_coverage = is_coverage
        if is_coverage:
            self.linear_cov = nn.Linear(1, att_dim, bias=False)

    def apply_mask(self, mask):
        self.mask = mask

    def forward(
        self, input, context, precompute=None, coverage=None, feat_inputs=None, feature=False
    ):
        """input: batch x dim; context: batch x sourceL x dim"""
        enc_output = torch.cat((context, feat_inputs), dim=2) if feature else context
        if precompute is None:
            precompute = self.linear_pre(enc_output)  # batch x sourceL x att_dim
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim

        tmp_sum = precompute + targetT.repeat(1, precompute.size(1), 1)  # batch x sourceL x att_dim

        if self.is_coverage:
            weighted_coverage = self.linear_cov(coverage.unsqueeze(2))  # batch x sourceL x att_dim
            tmp_sum = tmp_sum + weighted_coverage

        tmp_activated = self.tanh(tmp_sum)  # batch x sourceL x att_dim
        energy = self.linear_v(tmp_activated).view(
            tmp_sum.size(0), tmp_sum.size(1)
        )  # batch x sourceL
        if self.mask is not None:
            energy = energy * (1 - self.mask) + self.mask * (-1000000)

        score = self.sftmax(energy)  # batch x sourceL

        weightedContext = torch.bmm(score.unsqueeze(1), context).squeeze(1)  # batch x dim

        if self.is_coverage:
            coverage = coverage + score  # batch x sourceL
            return weightedContext, score, precompute, coverage

        return weightedContext, score, precompute


class GatedSelfAttention(nn.Module):
    def __init__(self, dim, attn_dim=64, dropout=0.1):
        super(GatedSelfAttention, self).__init__()

        self.m_translate = nn.Linear(dim, attn_dim)
        self.q_translate = nn.Linear(dim, attn_dim)

        self.update = nn.Linear(2 * dim, dim, bias=False)

        self.gate = nn.Linear(2 * dim, dim, bias=False)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        self.has_dropout = True if dropout > 0 else False

    def forward(self, query, mask):
        raw = query

        memory = self.m_translate(query)  # b_sz x src_len x 64
        query = self.q_translate(query)

        energy = torch.bmm(query, memory.transpose(1, 2))  # b_sz x src_len x src_len
        energy = energy.masked_fill(mask, value=-1e12)

        score = torch.softmax(energy, dim=2)
        if self.has_dropout:
            score = self.dropout(score)
        context = torch.bmm(score, raw)

        inputs = torch.cat((raw, context), dim=2)

        f_t = torch.tanh(self.update(inputs))
        g_t = torch.sigmoid(self.gate(inputs))

        output = g_t * f_t + (1 - g_t) * raw

        return output, score


class GraphAttention(nn.Module):
    def __init__(self, d_q, d_v, alpha, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

        self.attention = nn.Linear(d_q + d_v, 1)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, query, value, mask):
        """query - [batch_size, node_num * node_num, d_hidden]
        value - [batch_size, node_num * node_num, d_model]
        mask - [batch_size, node_num, node_num]"""
        node_num = int(query.size(1) ** 0.5)
        query = query.view(-1, node_num, query.size(2))
        value = value.view(
            -1, node_num, value.size(2)
        )  # (batch_size * node_num) x node_num x d_model

        pre_attention = torch.cat([query, value], dim=2)
        energy = self.leaky_relu(
            self.attention(pre_attention).squeeze(2)
        )  # (batch_size * node_num) x node_num

        mask = mask.reshape(-1, node_num)
        zero_vec = -9e15 * torch.ones_like(energy)
        attention = torch.where(mask > 0, energy, zero_vec)

        scores = torch.softmax(attention, dim=1)  # (batch_size * node_num) x node_num
        scores = self.dropout(scores)

        value = torch.bmm(scores.unsqueeze(1), value).squeeze(
            1
        )  # (batch_size * node_num) x d_model
        value = value.view(-1, node_num, value.size(-1))

        return value


# ---------------------------------------------------------------------------
# onqg/models/modules/SubLayers.py (Propagator only; MultiHeadAttention/
# PositionwiseFeedForward are unused by RNNEncoder/GraphEncoder with
# slf_attn='gated' or None, so they are omitted -- see header note)
# ---------------------------------------------------------------------------


class Propagator(nn.Module):
    def __init__(self, state_dim, dropout=0.1):
        super(Propagator, self).__init__()

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim), nn.Sigmoid(), nn.Dropout(dropout)
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim), nn.Sigmoid(), nn.Dropout(dropout)
        )
        self.transform = nn.Sequential(nn.Linear(state_dim * 3, state_dim), nn.Tanh())

    def forward(self, cur_state, in_vec, out_vec):
        """cur_state/in_vec/out_vec - [batch_size, node_num, d_model]"""
        a = torch.cat([in_vec, out_vec, cur_state], dim=2)
        r = self.reset_gate(a)
        z = self.update_gate(a)

        joined_input = torch.cat([in_vec, out_vec, r * cur_state], dim=2)
        h_hat = self.transform(joined_input)

        output = (1 - z) * cur_state + z * h_hat
        return output


# ---------------------------------------------------------------------------
# onqg/models/modules/Layers.py (GraphEncoderLayer only)
# ---------------------------------------------------------------------------


class GraphEncoderLayer(nn.Module):
    """GGNN & GAT Layer"""

    def __init__(self, d_hidden, d_model, alpha, feature=False, dropout=0.1, attn_dropout=0.1):
        super(GraphEncoderLayer, self).__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.feature = feature

        self.edge_num = 3  # TODO: fix this magic number (matches upstream)
        bias_list = [False, False, False]
        self.edge_in_list = nn.ModuleList(
            [nn.Linear(d_hidden, d_model, bias=bias_list[i]) for i in range(self.edge_num)]
        )
        self.edge_out_list = nn.ModuleList(
            [nn.Linear(d_hidden, d_model, bias=bias_list[i]) for i in range(self.edge_num)]
        )

        self.graph_in_attention = GraphAttention(d_hidden, d_model, alpha, dropout=attn_dropout)
        self.graph_out_attention = GraphAttention(d_hidden, d_model, alpha, dropout=attn_dropout)
        self.output_gate = Propagator(d_model, dropout=dropout)

    def forward(self, nodes, mask, node_type, feat_hidden=None):
        node_hidden = nodes  # batch_size x node_num x d_model
        in_masks = [
            (node_type == tag).float().unsqueeze(2).repeat(1, 1, self.d_model).to(nodes.device)
            for tag in range(2, 2 + self.edge_num)
        ]
        node_in_hidden = torch.sum(
            torch.stack(
                [
                    in_emb(node_hidden) * in_masks[idx]
                    for idx, in_emb in enumerate(self.edge_in_list)
                ],
                dim=0,
            ),
            dim=0,
        )
        out_masks = [
            (node_type == tag).float().unsqueeze(2).repeat(1, 1, self.d_model).to(nodes.device)
            for tag in range(2, 2 + self.edge_num)
        ]
        node_out_hidden = torch.sum(
            torch.stack(
                [
                    out_emb(node_hidden) * out_masks[idx]
                    for idx, out_emb in enumerate(self.edge_out_list)
                ],
                dim=0,
            ),
            dim=0,
        )
        node_hidden = (
            node_hidden.unsqueeze(2)
            .repeat(1, 1, nodes.size(1), 1)
            .view(nodes.size(0), -1, self.d_hidden)
        )
        node_in_hidden = self.graph_in_attention(
            node_hidden, node_in_hidden.repeat(1, nodes.size(1), 1), mask[0]
        )
        node_out_hidden = self.graph_out_attention(
            node_hidden, node_out_hidden.repeat(1, nodes.size(1), 1), mask[1]
        )
        node_output = self.output_gate(nodes, node_in_hidden, node_out_hidden)

        return node_output


# ---------------------------------------------------------------------------
# onqg/models/modules/DecAssist.py
# ---------------------------------------------------------------------------


class DecInit(nn.Module):
    def __init__(self, d_enc, d_dec, n_enc_layer):
        self.d_enc_model = d_enc
        self.n_enc_layer = n_enc_layer
        self.d_dec_model = d_dec

        super(DecInit, self).__init__()

        self.initer = nn.Linear(self.d_enc_model * self.n_enc_layer, self.d_dec_model)
        self.tanh = nn.Tanh()

    def forward(self, hidden):
        if isinstance(hidden, tuple) or isinstance(hidden, list) or hidden.dim() == 3:
            hidden = [h for h in hidden]
            hidden = torch.cat(hidden, dim=1)
        hidden = hidden.contiguous().view(hidden.size(0), -1)
        return self.tanh(self.initer(hidden))


class StackedRNN(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout, rnn="lstm"):
        self.dropout = dropout
        self.num_layers = num_layers

        super(StackedRNN, self).__init__()

        self.layers = nn.ModuleList()
        self.name = rnn

        for _ in range(num_layers):
            if rnn == "lstm":
                self.layers.append(nn.LSTMCell(input_size, rnn_size))
            elif rnn == "gru":
                self.layers.append(nn.GRUCell(input_size, rnn_size))
            else:
                raise ValueError("Supported StackedRNN: LSTM, GRU")
            input_size = rnn_size

    def forward(self, inputs, hidden):
        if self.name == "lstm":
            h_0, c_0 = hidden
        elif self.name == "gru":
            h_0 = hidden
        h_1, c_1 = [], []

        for i, layer in enumerate(self.layers):
            if self.name == "lstm":
                h_1_i, c_1_i = layer(inputs, (h_0[i], c_0[i]))
            elif self.name == "gru":
                h_1_i = layer(inputs, h_0[i])
            inputs = h_1_i
            if i + 1 != self.num_layers:
                inputs = self.dropout(inputs)
            h_1.append(h_1_i)
            if self.name == "lstm":
                c_1.append(c_1_i)

        h_1 = torch.stack(h_1)
        if self.name == "lstm":
            c_1 = torch.stack(c_1)
            h_1 = (h_1, c_1)

        return inputs, h_1


# ---------------------------------------------------------------------------
# onqg/models/Encoders.py (RNNEncoder, GraphEncoder, EncoderTransformer)
# ---------------------------------------------------------------------------


class RNNEncoder(nn.Module):
    """Input: (1) inputs['src_seq'] (2) inputs['lengths'] (3) inputs['feat_seqs']
    Output: (1) enc_output (2) hidden"""

    def __init__(
        self,
        n_vocab,
        d_word_vec,
        d_model,
        n_layer,
        brnn,
        rnn,
        feat_vocab,
        d_feat_vec,
        slf_attn,
        dropout,
    ):
        self.name = "rnn"

        self.n_layer = n_layer
        self.num_directions = 2 if brnn else 1
        assert d_model % self.num_directions == 0, "d_model = hidden_size x direction_num"
        self.hidden_size = d_model // self.num_directions
        self.d_enc_model = d_model

        super(RNNEncoder, self).__init__()

        self.word_emb = nn.Embedding(n_vocab, d_word_vec, padding_idx=PAD)
        input_size = d_word_vec

        self.feature = False if not feat_vocab else True
        if self.feature:
            self.feat_embs = nn.ModuleList(
                [nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=PAD) for n_f_vocab in feat_vocab]
            )
            input_size += len(feat_vocab) * d_feat_vec

        self.slf_attn = slf_attn
        if slf_attn:
            self.gated_slf_attn = GatedSelfAttention(d_model)

        if rnn == "lstm":
            self.rnn = nn.LSTM(
                input_size,
                self.hidden_size,
                num_layers=n_layer,
                dropout=dropout,
                bidirectional=brnn,
                batch_first=True,
            )
        elif rnn == "gru":
            self.rnn = nn.GRU(
                input_size,
                self.hidden_size,
                num_layers=n_layer,
                dropout=dropout,
                bidirectional=brnn,
                batch_first=True,
            )
        else:
            raise ValueError("Only support 'LSTM' and 'GRU' for RNN-based Encoder ")

    def forward(self, inputs):
        src_seq, lengths, feat_seqs = inputs["src_seq"], inputs["lengths"], inputs["feat_seqs"]
        lengths = torch.LongTensor(lengths.data.view(-1).tolist())

        enc_input = self.word_emb(src_seq)
        if self.feature:
            feat_outputs = [
                feat_emb(feat_seq) for feat_seq, feat_emb in zip(feat_seqs, self.feat_embs)
            ]
            feat_outputs = torch.cat(feat_outputs, dim=2)
            enc_input = torch.cat((enc_input, feat_outputs), dim=-1)

        enc_input = pack(enc_input, lengths, batch_first=True, enforce_sorted=False)
        enc_output, hidden = self.rnn(enc_input, None)
        enc_output = unpack(enc_output, batch_first=True)[0]

        if self.slf_attn:
            mask = (src_seq == PAD).bool()
            enc_output = self.gated_slf_attn(enc_output, mask)

        return enc_output, hidden


class GraphEncoder(nn.Module):
    """Combine GGNN (Gated Graph Neural Network) and GAT (Graph Attention Network)
    Input: (1) nodes - [batch_size, node_num, d_model]
           (2) edges - ([batch_size, node_num*node_num], [batch_size, node_num*node_num])
           (3) mask - ([batch_size, node_num, node_num], [batch_size, node_num, node_num])
           (4) node_feats - list of [batch_size, node_num]"""

    def __init__(
        self,
        n_edge_type,
        d_model,
        n_layer,
        alpha,
        d_feat_vec,
        feat_vocab,
        layer_attn,
        dropout,
        attn_dropout,
    ):
        self.name = "graph"
        super(GraphEncoder, self).__init__()
        self.layer_attn = layer_attn

        self.hidden_size = d_model
        self.d_model = d_model
        self.feature = True if feat_vocab else False
        if self.feature:
            self.feat_embs = nn.ModuleList(
                [nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=PAD) for n_f_vocab in feat_vocab]
            )
            self.feature_transform = nn.Linear(
                self.hidden_size + d_feat_vec * len(feat_vocab), self.hidden_size
            )
        self.layer_stack = nn.ModuleList(
            [
                GraphEncoderLayer(
                    self.hidden_size,
                    d_model,
                    alpha,
                    feature=self.feature,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.gate = nn.Linear(2 * d_model, d_model, bias=False)
        # `activate` is assigned externally by model_builder.build_model (Sequential
        # linear+tanh when d_seq_enc_model != d_graph_enc_model, else plain Tanh).
        self.activate = nn.Tanh()

    def gated_output(self, outputs, inputs):
        concatenation = torch.cat((outputs, inputs), dim=2)
        g_t = torch.sigmoid(self.gate(concatenation))

        output = g_t * outputs + (1 - g_t) * inputs
        return output

    def forward(self, inputs):
        nodes, mask = inputs["nodes"], inputs["mask"]
        node_feats, node_type = inputs["feat_seqs"], inputs["type"]
        nodes = self.activate(nodes)
        node_output = nodes  # batch_size x node_num x d_model
        feat_hidden = None
        if self.feature:
            feat_hidden = [
                feat_emb(node_feat) for node_feat, feat_emb in zip(node_feats, self.feat_embs)
            ]
            feat_hidden = torch.cat(feat_hidden, dim=2)
            node_output = self.feature_transform(torch.cat((node_output, feat_hidden), dim=-1))
        node_outputs = []
        for enc_layer in self.layer_stack:
            node_output = enc_layer(node_output, mask, node_type, feat_hidden=feat_hidden)
            node_outputs.append(node_output)

        node_output = self.gated_output(node_output, nodes)
        node_outputs[-1] = node_output

        hidden = [layer_output.transpose(0, 1)[0] for layer_output in node_outputs]

        if self.layer_attn:
            node_output = node_outputs

        return node_output, hidden


class EncoderTransformer(nn.Module):
    """Transform RNN-Encoder's output to Graph-Encoder's input
    Input: seq_output - [batch_size, seq_length, rnn_enc_dim] (tensor)
           indexes_list - [batch_size, node_num, index_num] (list)
           node_sizes - [batch_size * node_num, 1] (list)"""

    def __init__(self, d_model, d_k=64, device=None):
        super(EncoderTransformer, self).__init__()
        self.device = device
        self.d_k = d_k

        self.attn = ConcatAttention(d_model, d_model, d_k)

    def forward(self, inputs, max_length):
        def pad(vectors, data_length, max_length=None):
            hidden_size = (max_length, vectors.size(1))
            out = torch.zeros(hidden_size, device=self.device)
            out.narrow(0, 0, data_length).copy_(vectors)  # bag_size x rnn_enc_dim
            return out

        seq_output, hidden, indexes_list = inputs["seq_output"], inputs["hidden"], inputs["index"]

        if isinstance(hidden, tuple) or isinstance(hidden, list) or hidden.dim() == 3:
            hidden = [h for h in hidden]
            hidden = torch.cat(hidden, dim=1)
        hidden = hidden.contiguous().view(hidden.size(0), -1)

        node_sizes, node_lengths = inputs["lengths"], inputs["node_lengths"]
        max_length = max(node_sizes)
        roots, bags, cnt = [], [], 0
        for sample_idx, indexes in enumerate(indexes_list):
            for indexes_idx in indexes:
                roots.append(hidden[sample_idx])
                bag = pad(
                    torch.stack([seq_output[sample_idx][idx] for idx in indexes_idx], dim=0),
                    node_sizes[cnt],
                    max_length,
                )
                bags.append(bag)
                cnt += 1
        roots = torch.stack(roots, dim=0)  # all_node_num x rnn_enc_dim
        bags = torch.stack(bags, dim=0)  # all_node_num x bag_size x rnn_enc_dim
        context, *_ = self.attn(roots, bags)  # all_node_num x rnn_enc_dim
        max_length = max(node_lengths)
        nodes = []
        for node_length in node_lengths:
            nodes.append(pad(context[:node_length], node_length, max_length))
            context = context[node_length:]

        nodes = torch.stack(nodes, dim=0)  # batch_size x node_num x d_model

        return nodes, hidden


# ---------------------------------------------------------------------------
# onqg/models/Decoders.py (RNNDecoder, DecoderTransformer)
# ---------------------------------------------------------------------------


class RNNDecoder(nn.Module):
    """Input: (1) inputs['tgt_seq'] (2) inputs['src_seq'] (3) inputs['ans_seq']
    (4) inputs['enc_output'] (5) inputs['hidden'] (6) inputs['feat_seqs']
    Output: rst['pred'], rst['attn'], rst['context'], (copy), (coverage)"""

    def __init__(
        self,
        n_vocab,
        ans_n_vocab,
        d_word_vec,
        d_model,
        n_layer,
        n_rnn_enc_layer,
        rnn,
        d_k,
        feat_vocab,
        d_feat_vec,
        d_rnn_enc_model,
        d_enc_model,
        n_enc_layer,
        input_feed,
        copy,
        answer,
        coverage,
        layer_attn,
        maxout_pool_size,
        dropout,
        device=None,
    ):
        self.name = "rnn"

        super(RNNDecoder, self).__init__()

        self.n_layer = n_layer
        self.layer_attn = layer_attn
        self.coverage = coverage
        self.copy = copy
        self.maxout_pool_size = maxout_pool_size
        input_size = d_word_vec

        self.input_feed = input_feed
        if input_feed:
            input_size += d_rnn_enc_model + d_enc_model

        self.ans_emb = nn.Embedding(ans_n_vocab, d_word_vec, padding_idx=PAD)

        self.answer = answer
        tmp_in = d_word_vec if answer else d_rnn_enc_model
        self.decInit = DecInit(d_enc=tmp_in, d_dec=d_model, n_enc_layer=n_rnn_enc_layer)

        self.feature = False if not feat_vocab else True
        if self.feature:
            self.feat_embs = nn.ModuleList(
                [nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=PAD) for n_f_vocab in feat_vocab]
            )
        feat_size = len(feat_vocab) * d_feat_vec if self.feature else 0

        self.d_enc_model = d_rnn_enc_model + d_enc_model

        self.word_emb = nn.Embedding(n_vocab, d_word_vec, padding_idx=PAD)
        self.rnn = StackedRNN(n_layer, input_size, d_model, dropout, rnn=rnn)
        self.attn = ConcatAttention(self.d_enc_model + feat_size, d_model, d_k, coverage)

        self.readout = nn.Linear((d_word_vec + d_model + self.d_enc_model), d_model)
        self.maxout = MaxOut(maxout_pool_size)

        if copy:
            self.copy_switch = nn.Linear(self.d_enc_model + d_model, 1)

        self.hidden_size = d_model
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def attn_init(self, context):
        if isinstance(context, list):
            context = context[-1]
        if isinstance(context, tuple):
            context = torch.cat(context, dim=-1)
        batch_size = context.size(0)
        hidden_sizes = (batch_size, self.d_enc_model)
        return context.new_zeros(*hidden_sizes)

    def forward(self, inputs, max_length=300):
        tgt_seq, src_seq, ans_seq = inputs["tgt_seq"], inputs["src_seq"], inputs["ans_seq"]
        enc_output, hidden = inputs["enc_output"], inputs["hidden"]
        feat_seqs = inputs["feat_seqs"]

        src_pad_mask = src_seq.data.eq(PAD).float()
        if self.layer_attn:
            n_enc_layer = len(enc_output)
            src_pad_mask = src_pad_mask.repeat(1, n_enc_layer)
            enc_output = torch.cat(enc_output, dim=1)

        feat_inputs = None
        if self.feature:
            feat_inputs = [
                feat_emb(feat_seq) for feat_seq, feat_emb in zip(feat_seqs, self.feat_embs)
            ]
            feat_inputs = torch.cat(feat_inputs, dim=2)
            if self.layer_attn:
                feat_inputs = feat_inputs.repeat(1, n_enc_layer, 1)

        dec_outputs, coverage_output, copy_output, copy_gate_output = [], [], [], []
        cur_context = self.attn_init(enc_output)

        if self.answer:
            ans_words = torch.sum(self.ans_emb(ans_seq), dim=1)
            hidden = self.decInit(ans_words).unsqueeze(0)
        else:
            hidden = self.decInit(hidden).unsqueeze(0)
        tmp_context, tmp_coverage = None, None

        dec_input = self.word_emb(tgt_seq)

        self.attn.apply_mask(src_pad_mask)

        attention_scores = None
        tag = False

        dec_input = dec_input.transpose(0, 1)
        for seq_idx, dec_input_emb in enumerate(dec_input.split(1)):
            dec_input_emb = dec_input_emb.squeeze(0)
            raw_dec_input_emb = dec_input_emb
            if self.input_feed:
                dec_input_emb = torch.cat((dec_input_emb, cur_context), dim=1)
            dec_output, hidden = self.rnn(dec_input_emb, hidden)

            if self.coverage:
                if tmp_coverage is None:
                    tmp_coverage = torch.zeros(
                        (enc_output.size(0), enc_output.size(1)), device=enc_output.device
                    )
                cur_context, attn, tmp_context, next_coverage = self.attn(
                    dec_output,
                    enc_output,
                    precompute=tmp_context,
                    coverage=tmp_coverage,
                    feat_inputs=feat_inputs,
                    feature=self.feature,
                )
                avg_tmp_coverage = tmp_coverage / max(1, seq_idx)
                coverage_loss = torch.sum(torch.min(attn, avg_tmp_coverage), dim=1)
                tmp_coverage = next_coverage
                coverage_output.append(coverage_loss)
            else:
                cur_context, attn, tmp_context = self.attn(
                    dec_output,
                    enc_output,
                    precompute=tmp_context,
                    feat_inputs=feat_inputs,
                    feature=self.feature,
                )

            attention_scores = attn if not tag else attn + attention_scores
            tag = True

            if self.copy:
                copy_prob = self.copy_switch(torch.cat((dec_output, cur_context), dim=1))
                copy_prob = torch.sigmoid(copy_prob)

                if self.layer_attn:
                    attn = attn.view(attn.size(0), n_enc_layer, -1)
                    attn = attn.sum(1)

                copy_output.append(attn)
                copy_gate_output.append(copy_prob)

            readout = self.readout(torch.cat((raw_dec_input_emb, dec_output, cur_context), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)

            dec_outputs.append(output)

        dec_output = torch.stack(dec_outputs).transpose(0, 1)

        sum_attention_scores = torch.sum(attention_scores, dim=1, keepdim=True) + 1e-8
        attention_scores = attention_scores / sum_attention_scores

        rst = {}
        rst["pred"], rst["attn"], rst["context"] = dec_output, attn, cur_context
        rst["attention_scores"] = (attention_scores, inputs.get("scores"))
        if self.copy:
            copy_output = torch.stack(copy_output).transpose(0, 1)
            copy_gate_output = torch.stack(copy_gate_output).transpose(0, 1)
            rst["copy_pred"], rst["copy_gate"] = copy_output, copy_gate_output
        if self.coverage:
            coverage_output = torch.stack(coverage_output).transpose(0, 1)
            rst["coverage_pred"] = coverage_output
        return rst


class DecoderTransformer(nn.Module):
    """seq_output - [batch_size, seq_length, dim_seq_enc]
    graph_output - [batch_size, node_num, dim_graph_enc]
    indexes_list - [batch_size, node_num, index_num] (list)"""

    def __init__(self, layer_attn, device=None):
        super(DecoderTransformer, self).__init__()
        self.layer_attn = layer_attn
        self.device = device

    def forward(self, inputs):
        seq_output, hidden = inputs["seq_output"], inputs["hidden"]
        graph_output, indexes_list = inputs["graph_output"], inputs["index"]

        batch_size, seq_length = seq_output.size(0), seq_output.size(1)
        dim_graph_enc = graph_output.size(-1) if not self.layer_attn else graph_output[-1].size(-1)
        if "scores" in inputs and inputs["scores"] is not None:
            scores = inputs["scores"]
            distribution = torch.full((batch_size, seq_length), 1e-8, device=self.device)
        else:
            scores = None

        if self.layer_attn:
            graph_hidden_states = [
                torch.full((batch_size, seq_length, dim_graph_enc), 1e-8, device=self.device)
                for _ in range(len(graph_output))
            ]
        else:
            graph_hidden_states = torch.full(
                (batch_size, seq_length, dim_graph_enc), 1e-8, device=self.device
            )

        graph_node_sizes = torch.full((batch_size, seq_length), 0, device=self.device)

        for sample_idx, indexes in enumerate(indexes_list):
            for node_idx, index in enumerate(indexes):
                for i in index:
                    if self.layer_attn:
                        for idx in range(len(graph_hidden_states)):
                            graph_hidden_states[idx][sample_idx].narrow(0, i, 1).add_(
                                graph_output[idx][sample_idx][node_idx]
                            )
                    else:
                        graph_hidden_states[sample_idx].narrow(0, i, 1).add_(
                            graph_output[sample_idx][node_idx]
                        )
                    graph_node_sizes[sample_idx][i] += 1
                    if scores is not None:
                        distribution[sample_idx][i] = scores[sample_idx][node_idx]

        for i in range(batch_size):
            for j in range(seq_length):
                if graph_node_sizes[i][j].item() < 1:
                    graph_node_sizes[i][j] = 1

        if self.layer_attn:
            graph_hidden_states = [
                x / graph_node_sizes.unsqueeze(2).repeat(1, 1, dim_graph_enc)
                for x in graph_hidden_states
            ]
        else:
            graph_hidden_states = graph_hidden_states / graph_node_sizes.unsqueeze(2).repeat(
                1, 1, dim_graph_enc
            )
        if scores is not None:
            distribution = distribution / graph_node_sizes

        if isinstance(hidden, tuple) or isinstance(hidden, list) or hidden.dim() == 3:
            hidden = [h for h in hidden]
            hidden = torch.cat(hidden, dim=1)
        hidden = hidden.contiguous().view(hidden.size(0), -1)

        distribution = distribution if scores is not None else None
        if self.layer_attn:
            enc_output = [
                torch.cat((graph_output, seq_output), dim=-1)
                for graph_output in graph_hidden_states
            ]
        else:
            enc_output = torch.cat((graph_hidden_states, seq_output), dim=-1)

        return enc_output, distribution, hidden


# ---------------------------------------------------------------------------
# onqg/models/Models.py
# ---------------------------------------------------------------------------


class UnifiedModel(nn.Module):
    """Unify Sequence-Encoder and Graph-Encoder

    Input:  seq-encoder: src_seq, lengths, feat_seqs
            graph-encoder: edges
            encoder-transform: index, lengths, root
            decoder: tgt_seq, src_seq, feat_seqs
            answer-encoder: src_seq, lengths, feat_seqs

    Output: results output from the Decoder (type: dict)"""

    def __init__(
        self,
        model_type,
        seq_encoder,
        graph_encoder,
        encoder_transformer,
        decoder,
        decoder_transformer,
    ):
        super(UnifiedModel, self).__init__()

        self.model_type = model_type

        self.seq_encoder = seq_encoder

        self.encoder_transformer = encoder_transformer
        self.graph_encoder = graph_encoder

        self.decoder_transformer = decoder_transformer
        self.decoder = decoder

    def forward(self, inputs, max_length=None):
        ## RNN encode ##
        seq_output, hidden = self.seq_encoder(inputs["seq-encoder"])
        ## encoder transform ##
        inputs["encoder-transform"]["seq_output"] = seq_output
        inputs["encoder-transform"]["hidden"] = hidden
        node_input, hidden = self.encoder_transformer(inputs["encoder-transform"], max_length)
        ## graph encode ##
        inputs["graph-encoder"]["nodes"] = node_input
        node_output, _ = self.graph_encoder(inputs["graph-encoder"])

        outputs = {}

        # ========== classify =========#
        if self.model_type != "generate":
            scores = (
                self.classifier(node_output)
                if not self.decoder.layer_attn
                else self.classifier(node_output[-1])
            )
            inputs["decoder-transform"]["scores"] = scores
            outputs["classification"] = scores
        # ========== generate =========#
        inputs["decoder-transform"]["graph_output"] = node_output
        inputs["decoder-transform"]["seq_output"] = seq_output
        inputs["decoder-transform"]["hidden"] = hidden
        inputs["decoder"]["enc_output"], inputs["decoder"]["scores"], hidden = (
            self.decoder_transformer(inputs["decoder-transform"])
        )
        inputs["decoder"]["hidden"] = hidden
        dec_output = self.decoder(inputs["decoder"])
        outputs["generation"] = dec_output
        if self.model_type != "classify":
            outputs["generation"]["pred"] = self.generator(dec_output["pred"])

        return outputs


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------
# Config below mirrors scripts/train_generator.sh (training_mode='generate',
# sparse=0 -> dense GraphEncoder, copy=True, node_feature off (kept off only to
# avoid needing an extra node-feature-vocab tensor; the GGNN/GAT/attention/
# input-feed architecture used is otherwise identical), scaled down to tiny dims.

_D_WORD_VEC = 12
_D_SEQ_ENC = 16  # brnn -> hidden per direction = 8
_D_GRAPH_ENC = 16
_D_DEC = 16
_D_K = 8
_N_GRAPH_LAYER = 2
_SRC_VOCAB = 40
_TGT_VOCAB = 36
_EDGE_VOCAB = 8


class SemQGGenerator(nn.Module):
    """Thin wrapper exposing UnifiedModel + its `generator` head as one nn.Module,
    matching model_builder.build_model's construction (generator/classifier are
    attached to the model instance after construction in the original code)."""

    def __init__(self):
        super().__init__()
        seq_encoder = RNNEncoder(
            n_vocab=_SRC_VOCAB,
            d_word_vec=_D_WORD_VEC,
            d_model=_D_SEQ_ENC,
            n_layer=1,
            brnn=True,
            rnn="gru",
            feat_vocab=None,
            d_feat_vec=0,
            slf_attn=False,
            dropout=0.0,
        )
        encoder_transformer = EncoderTransformer(_D_SEQ_ENC, d_k=_D_K, device=None)
        graph_encoder = GraphEncoder(
            n_edge_type=_EDGE_VOCAB,
            d_model=_D_GRAPH_ENC,
            n_layer=_N_GRAPH_LAYER,
            alpha=0.2,
            d_feat_vec=0,
            feat_vocab=None,
            layer_attn=False,
            dropout=0.0,
            attn_dropout=0.0,
        )
        if _D_SEQ_ENC != _D_GRAPH_ENC:
            graph_encoder.activate = nn.Sequential(
                nn.Linear(_D_SEQ_ENC, _D_GRAPH_ENC, bias=False), nn.Tanh()
            )
        else:
            graph_encoder.activate = nn.Tanh()
        decoder_transformer = DecoderTransformer(layer_attn=False, device=None)
        decoder = RNNDecoder(
            n_vocab=_TGT_VOCAB,
            ans_n_vocab=_SRC_VOCAB,
            d_word_vec=_D_WORD_VEC,
            d_model=_D_DEC,
            n_layer=1,
            n_rnn_enc_layer=1,
            rnn="gru",
            d_k=_D_K,
            feat_vocab=None,
            d_feat_vec=0,
            d_rnn_enc_model=_D_SEQ_ENC,
            d_enc_model=_D_GRAPH_ENC,
            n_enc_layer=_N_GRAPH_LAYER,
            input_feed=True,
            copy=True,
            answer=False,
            coverage=True,
            layer_attn=False,
            maxout_pool_size=2,
            dropout=0.0,
            device=None,
        )

        self.model = UnifiedModel(
            "generate",
            seq_encoder,
            graph_encoder,
            encoder_transformer,
            decoder,
            decoder_transformer,
        )
        self.model.generator = nn.Linear(_D_DEC // 2, _TGT_VOCAB, bias=False)

    def forward(self, inputs):
        return self.model(inputs)


def build_semqg():
    model = SemQGGenerator()
    model.eval()
    return model


def example_input_semqg():
    """Builds one syntactically-valid batch matching preprocess_batch's documented
    inputs contract: 2 source sentences (lengths 7 and 5), 4 semantic-graph nodes
    per sample (each spanning >=1 source token), a fully-connected toy edge/mask
    structure, and a 5-token target/answer sequence."""
    batch_size = 2
    src_len = 7
    node_num = 4
    tgt_len = 6

    src_seq = torch.randint(low=1, high=_SRC_VOCAB, size=(batch_size, src_len), dtype=torch.long)
    src_seq[:, -2:] = PAD  # last two source tokens padded -> lengths 5
    lengths = torch.tensor([src_len, src_len - 2], dtype=torch.long)

    # encoder-transform: each of the node_num nodes covers a contiguous 1-token
    # span of the (unpadded) source; node_sizes has batch_size*node_num entries.
    indexes_list = [[[i] for i in range(node_num)] for _ in range(batch_size)]
    node_sizes = [1 for _ in range(batch_size * node_num)]
    node_lengths = [node_num for _ in range(batch_size)]

    # graph-encoder: dense edge mask/type over a small fully-connected toy graph.
    # edges[k] eq PAD -> masked out; use all-nonzero (fully connected) edge ids.
    edges_in = torch.ones(batch_size, node_num * node_num, dtype=torch.long)
    edges_out = torch.ones(batch_size, node_num * node_num, dtype=torch.long)
    node_type = torch.full(
        (batch_size, node_num), 2, dtype=torch.long
    )  # edge_num tag range [2, 2+3)

    tgt_seq = torch.randint(low=1, high=_TGT_VOCAB, size=(batch_size, tgt_len), dtype=torch.long)
    ans_seq = torch.randint(low=1, high=_SRC_VOCAB, size=(batch_size, 3), dtype=torch.long)

    inputs = {
        "seq-encoder": {"src_seq": src_seq, "lengths": lengths, "feat_seqs": None},
        "encoder-transform": {
            "index": indexes_list,
            "lengths": node_sizes,
            "node_lengths": node_lengths,
        },
        "graph-encoder": {
            "edges": (get_edge_mask(edges_in), get_edge_mask(edges_out)),
            "mask": (get_edge_mask(edges_in), get_edge_mask(edges_out)),
            "type": node_type,
            "feat_seqs": None,
        },
        "decoder-transform": {"index": indexes_list},
        "decoder": {
            "tgt_seq": tgt_seq[:, :-1],
            "src_seq": src_seq,
            "ans_seq": ans_seq,
            "feat_seqs": None,
        },
    }
    return (inputs,)


MENAGERIE_ENTRIES = [
    ("SemQG", "build_semqg", "example_input_semqg", 2020, MENAGERIE_ZOO),
]

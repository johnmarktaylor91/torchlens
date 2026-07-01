# SOURCE: vendored from LooperXX/AGIF @ master (models/module.py, utils/process.py::normalize_adj)
"""AGIF: Adaptive Graph-Interactive Framework for joint multiple intent detection
and slot filling.

Paper: "A Co-Interactive Transformer for Joint Slot Filling and Intent Detection"
predecessor / "AGIF: An Adaptive Graph-Interactive Framework for Joint Multiple
Intent Detection and Slot Filling", Qin et al., EMNLP 2020 Findings.
Official repo: https://github.com/LooperXX/AGIF (models/module.py @ master).

The model jointly predicts multiple sentence-level intents and per-token slot
labels. A shared BiLSTM + self-attention encoder feeds an intent decoder (MLP)
and a slot decoder. The slot decoder is a token-by-token LSTM whose input at
each step is augmented, via a small Graph Attention Network (GAT), with
representations of the intents predicted for the sentence -- this per-token
label-aware graph interaction is the paper's core contribution ("adaptive
graph-interactive").

The classes below (``GraphAttentionLayer``, ``GAT``, ``LSTMEncoder``,
``LSTMDecoder``, ``QKVAttention``, ``SelfAttention``, ``UnflatSelfAttention``,
``Encoder``, ``ModelManager``) are copied verbatim from the official repo's
``models/module.py``, with only two minimal, non-architectural changes:

1. ``from utils.process import normalize_adj`` (a relative-import-only path
   dependency) is replaced by vendoring the single ``normalize_adj`` helper
   function directly (copied unmodified from ``utils/process.py``).
2. The training-only ``args`` argparse Namespace from ``utils/config.py`` is
   replaced by a small local ``_default_args()`` builder using the same
   default hyperparameter values declared in ``utils/config.py``, since the
   original script parses ``sys.argv`` at import time which is unsuitable for
   library use. No architectural values were altered from the repo's own
   argparse defaults.

No layer, module, or forward-pass logic was modified.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def normalize_adj(mx):
    """Row-normalize matrix  D^{-1}A

    torch.diag_embed: https://github.com/pytorch/pytorch/pull/12447

    Copied verbatim from AGIF's ``utils/process.py``.
    """
    mx = mx.float()
    rowsum = mx.sum(2)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.0
    r_mat_inv = torch.diag_embed(r_inv, 0)
    mx = r_mat_inv.matmul(mx)
    return mx


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        B, N = h.size()[0], h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(
            B, N, -1, 2 * self.out_features
        )
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nheads = nheads
        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ]
        for i, attention in enumerate(self.attentions):
            self.add_module("attention_{}".format(i), attention)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                for j in range(self.nheads):
                    self.add_module(
                        "attention_{}_{}".format(i + 1, j),
                        GraphAttentionLayer(
                            nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True
                        ),
                    )

        self.out_att = GraphAttentionLayer(
            nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False
        )

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        input = x
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                temp = []
                x = F.dropout(x, self.dropout, training=self.training)
                cur_input = x
                for j in range(self.nheads):
                    temp.append(self.__getattr__("attention_{}_{}".format(i + 1, j))(x, adj))
                x = torch.cat(temp, dim=2) + cur_input
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x + input


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.__args = args

        # Initialize an LSTM Encoder object.
        self.__encoder = LSTMEncoder(
            self.__args.word_embedding_dim, self.__args.encoder_hidden_dim, self.__args.dropout_rate
        )

        # Initialize an self-attention layer.
        self.__attention = SelfAttention(
            self.__args.word_embedding_dim,
            self.__args.attention_hidden_dim,
            self.__args.attention_output_dim,
            self.__args.dropout_rate,
        )

        self.__sentattention = UnflatSelfAttention(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.dropout_rate,
        )

    def forward(self, word_tensor, seq_lens):
        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=2)
        c = self.__sentattention(hiddens, seq_lens)
        return hiddens, c


class ModelManager(nn.Module):
    def __init__(self, args, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()

        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        # Initialize an embedding object.
        self.__embedding = nn.Embedding(self.__num_word, self.__args.word_embedding_dim)

        self.G_encoder = Encoder(args)
        # Initialize an Decoder object for intent.
        self.__intent_decoder = nn.Sequential(
            nn.Linear(
                self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
                self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            ),
            nn.LeakyReLU(args.alpha),
            nn.Linear(
                self.__args.encoder_hidden_dim + self.__args.attention_output_dim, self.__num_intent
            ),
        )

        self.__intent_embedding = nn.Parameter(
            torch.FloatTensor(self.__num_intent, self.__args.intent_embedding_dim)
        )  # 191, 32
        nn.init.normal_(self.__intent_embedding.data)

        # Initialize an Decoder object for slot.
        self.__slot_decoder = LSTMDecoder(
            args,
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.slot_decoder_hidden_dim,
            self.__num_slot,
            self.__args.dropout_rate,
            embedding_dim=self.__args.slot_embedding_dim,
        )

    def show_summary(self):
        """
        print the abstract of the defined model.
        """

        print("Model parameters are listed as follows:\n")

        print("\tnumber of word:                            {};".format(self.__num_word))
        print("\tnumber of slot:                            {};".format(self.__num_slot))
        print("\tnumber of intent:						    {};".format(self.__num_intent))
        print(
            "\tword embedding dimension:				    {};".format(
                self.__args.word_embedding_dim
            )
        )
        print(
            "\tencoder hidden dimension:				    {};".format(
                self.__args.encoder_hidden_dim
            )
        )
        print(
            "\tdimension of intent embedding:		    	{};".format(
                self.__args.intent_embedding_dim
            )
        )
        print(
            "\tdimension of slot embedding:			    {};".format(
                self.__args.slot_embedding_dim
            )
        )
        print(
            "\tdimension of slot decoder hidden:  	    {};".format(
                self.__args.slot_decoder_hidden_dim
            )
        )
        print(
            "\thidden dimension of self-attention:        {};".format(
                self.__args.attention_hidden_dim
            )
        )
        print(
            "\toutput dimension of self-attention:        {};".format(
                self.__args.attention_output_dim
            )
        )

        print("\nEnd of parameters show. Now training begins.\n\n")

    def generate_adj_gat(self, index, batch):
        intent_idx_ = [[torch.tensor(0)] for i in range(batch)]
        for item in index:
            intent_idx_[item[0]].append(item[1] + 1)
        intent_idx = intent_idx_
        adj = torch.cat([torch.eye(self.__num_intent + 1).unsqueeze(0) for i in range(batch)])
        for i in range(batch):
            for j in intent_idx[i]:
                adj[i, j, intent_idx[i]] = 1.0
        if self.__args.row_normalized:
            adj = normalize_adj(adj)
        if self.__args.gpu:
            adj = adj.cuda()
        return adj

    def forward(self, text, seq_lens, n_predicts=None, forced_slot=None, forced_intent=None):
        word_tensor = self.__embedding(text)
        g_hiddens, g_c = self.G_encoder(word_tensor, seq_lens)
        pred_intent = self.__intent_decoder(g_c)
        intent_index = (torch.sigmoid(pred_intent) > self.__args.threshold).nonzero()
        adj = self.generate_adj_gat(intent_index, len(pred_intent))

        pred_slot = self.__slot_decoder(
            g_hiddens,
            seq_lens,
            forced_input=forced_slot,
            adj=adj,
            intent_embedding=self.__intent_embedding,
        )

        if n_predicts is None:
            return F.log_softmax(pred_slot, dim=1), pred_intent
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            intent_index = (torch.sigmoid(pred_intent) > self.__args.threshold).nonzero()

            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist()


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

        # Network attributes.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=self.__embedding_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.__dropout_rate,
            num_layers=1,
        )

    def forward(self, embedded_text, seq_lens):
        """Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)
        -> (total_word_num, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.__dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        return padded_hiddens


class LSTMDecoder(nn.Module):
    """
    Decoder structure based on unidirectional LSTM.
    """

    def __init__(
        self,
        args,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_rate,
        embedding_dim=None,
        extra_dim=None,
    ):
        """Construction function for Decoder.

        :param input_dim: input dimension of Decoder. In fact, it's encoder hidden size.
        :param hidden_dim: hidden dimension of iterative LSTM.
        :param output_dim: output dimension of Decoder. In fact, it's total number of intent or slot.
        :param dropout_rate: dropout rate of network which is only useful for embedding.
        :param embedding_dim: if it's not None, the input and output are relevant.
        :param extra_dim: if it's not None, the decoder receives information tensors.
        """

        super(LSTMDecoder, self).__init__()
        self.__args = args
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(
                torch.randn(1, self.__embedding_dim), requires_grad=True
            )

        # Make sure the input dimension of iterative LSTM.
        if self.__extra_dim is not None and self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim + self.__embedding_dim
        elif self.__extra_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim
        elif self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__embedding_dim
        else:
            lstm_input_dim = self.__input_dim

        # Network parameter definition.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.__dropout_rate,
            num_layers=1,
        )

        self.__graph = GAT(
            self.__hidden_dim,
            self.__args.decoder_gat_hidden_dim,
            self.__hidden_dim,
            self.__args.gat_dropout_rate,
            self.__args.alpha,
            self.__args.n_heads,
            self.__args.n_layers_decoder,
        )

        self.__linear_layer = nn.Linear(self.__hidden_dim, self.__output_dim)

    def forward(
        self,
        encoded_hiddens,
        seq_lens,
        forced_input=None,  # extra_input=None,
        adj=None,
        intent_embedding=None,
    ):
        """Forward process for decoder.

        :param encoded_hiddens: is encoded hidden tensors produced by encoder.
        :param seq_lens: is a list containing lengths of sentence.
        :param forced_input: is truth values of label, provided by teacher forcing.
        :param extra_input: comes from another decoder as information tensor.
        :return: is distribution of prediction labels.
        """

        input_tensor = encoded_hiddens
        output_tensor_list, sent_start_pos = [], 0
        if self.__embedding_dim is not None and forced_input is not None:
            forced_tensor = self.__embedding_layer(forced_input)[:, :-1]
            prev_tensor = torch.cat(
                (self.__init_tensor.unsqueeze(0).repeat(len(forced_tensor), 1, 1), forced_tensor),
                dim=1,
            )
            combined_input = torch.cat([input_tensor, prev_tensor], dim=2)
            dropout_input = self.__dropout_layer(combined_input)
            packed_input = pack_padded_sequence(dropout_input, seq_lens, batch_first=True)
            lstm_out, _ = self.__lstm_layer(packed_input)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            for sent_i in range(0, len(seq_lens)):
                if adj is not None:
                    lstm_out_i = torch.cat(
                        (
                            lstm_out[sent_i][: seq_lens[sent_i]].unsqueeze(1),
                            intent_embedding.unsqueeze(0).repeat(seq_lens[sent_i], 1, 1),
                        ),
                        dim=1,
                    )
                    lstm_out_i = self.__graph(
                        lstm_out_i, adj[sent_i].unsqueeze(0).repeat(seq_lens[sent_i], 1, 1)
                    )[:, 0]
                else:
                    lstm_out_i = lstm_out[sent_i][: seq_lens[sent_i]]
                linear_out = self.__linear_layer(lstm_out_i)
                output_tensor_list.append(linear_out)
        else:
            prev_tensor = self.__init_tensor.unsqueeze(0).repeat(len(seq_lens), 1, 1)
            last_h, last_c = None, None
            for word_i in range(seq_lens[0]):
                combined_input = torch.cat(
                    (input_tensor[:, word_i].unsqueeze(1), prev_tensor), dim=2
                )
                dropout_input = self.__dropout_layer(combined_input)
                if last_h is None and last_c is None:
                    lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input)
                else:
                    lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input, (last_h, last_c))

                if adj is not None:
                    lstm_out = torch.cat(
                        (lstm_out, intent_embedding.unsqueeze(0).repeat(len(lstm_out), 1, 1)),
                        dim=1,
                    )
                    lstm_out = self.__graph(lstm_out, adj)[:, 0]

                lstm_out = self.__linear_layer(lstm_out.squeeze(1))
                output_tensor_list.append(lstm_out)

                _, index = lstm_out.topk(1, dim=1)
                prev_tensor = self.__embedding_layer(index.squeeze(1)).unsqueeze(1)
            output_tensor = torch.stack(output_tensor_list)
            output_tensor_list = [output_tensor[:length, i] for i, length in enumerate(seq_lens)]

        return torch.cat(output_tensor_list, dim=0)


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Declare network structures.
        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        """The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(
            torch.matmul(linear_query, linear_key.transpose(-2, -1)), dim=-1
        ) / math.sqrt(self.__hidden_dim)
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        # Record network parameters.
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__attention_layer = QKVAttention(
            self.__input_dim,
            self.__input_dim,
            self.__input_dim,
            self.__hidden_dim,
            self.__output_dim,
            self.__dropout_rate,
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.__dropout_layer(input_x)
        attention_x = self.__attention_layer(dropout_x, dropout_x, dropout_x)

        return attention_x


class UnflatSelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.0):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context


def _default_args() -> SimpleNamespace:
    """Build an args namespace using AGIF's own ``utils/config.py`` argparse
    defaults, scaled down for a tiny example instantiation (encoder/attention
    widths shrunk; hyperparameter *values* -- dropout rates, alpha,
    n_heads, n_layers_decoder, row_normalized, threshold -- are the repo's
    literal defaults, unmodified).
    """
    return SimpleNamespace(
        gpu=False,
        dropout_rate=0.4,
        gat_dropout_rate=0.4,
        threshold=0.5,
        row_normalized=True,
        n_heads=4,
        alpha=0.2,
        decoder_gat_hidden_dim=8,
        n_layers_decoder=2,
        word_embedding_dim=16,
        intent_embedding_dim=16,
        slot_embedding_dim=8,
        encoder_hidden_dim=16,
        slot_decoder_hidden_dim=16,
        attention_hidden_dim=32,
        attention_output_dim=16,
    )


def build_agif() -> nn.Module:
    """Build a tiny AGIF ``ModelManager`` (num_word=64, num_slot=12, num_intent=6).

    Returns
    -------
    nn.Module
        Random-initialized AGIF joint intent-slot model.
    """
    args = _default_args()
    model = ModelManager(args, num_word=64, num_slot=12, num_intent=6)
    model.eval()
    return model


def example_input_agif():
    """Return an example (text, seq_lens) pair for AGIF.

    Sequence lengths must be sorted descending (required by
    ``pack_padded_sequence`` with ``enforce_sorted`` default True), and the
    forward pass indexes ``seq_lens[0]`` as the max length for the
    autoregressive slot decoder loop.

    Returns
    -------
    tuple[torch.Tensor, list[int]]
        ``text``: Long tensor with shape ``(batch, max_len)``.
        ``seq_lens``: Python list of per-example sequence lengths, descending.
    """
    seq_lens = [6, 5, 4]
    max_len = seq_lens[0]
    text = torch.randint(0, 64, (len(seq_lens), max_len), dtype=torch.long)
    return text, seq_lens


MENAGERIE_ENTRIES = [
    (
        "AGIF",
        "build_agif",
        "example_input_agif",
        "2020",
        "NL",
    )
]

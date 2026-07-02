# SOURCE: vendored from https://github.com/aehrc/LAAT @ master (src/models/rnn.py,
# src/models/embeddings/embedding_layer.py, src/models/attentions/attention_layer.py,
# src/models/attentions/util.py)
# LAAT / JointLAAT: bi-LSTM + label-attention text classifier for ICD coding, with a
# hierarchical joint-learning extension across label levels (Vu, Nguyen & Yip, IJCAI
# 2020, arXiv:2007.06351). The official repo ships one model implementation (RNN +
# AttentionLayer) that becomes "LAAT" with `--joint_mode flat` and "JointLAAT" with
# `--joint_mode hierarchical` (the repo's own default, per src/args_parser.py and the
# example run.sh: `--attention_mode label --joint_mode hierarchical RNN --rnn_model
# LSTM`). The classes `RNN`, `EmbeddingLayer`, `AttentionLayer` and the module-level
# functions `init_attention_layer`/`perform_attention` below are copied verbatim from
# the official repo. Only two changes were made, both non-architectural: (1) the real
# `Vocab` class (src/data_helpers/vocab.py) requires `gensim` to load pretrained
# word2vec/fastText embeddings from MIMIC-III training data; since we run
# `embedding_mode="rand"` (random-init embeddings, one of the repo's own supported
# modes) that dependency is never exercised, so a `_TinyVocab` stand-in exposing only
# the vocab interface the model code actually calls (`n_words()`, `all_n_labels()`,
# `n_level()`, `n_labels(level)`, `.word_embeddings`) replaces it -- no model logic is
# touched; (2) deprecated torch calls (`torch.nn.init.normal(...)` positional-std,
# `F.tanh`, `F.sigmoid`) were updated to their modern non-deprecated equivalents
# (`torch.nn.init.normal_(..., std=...)`, `torch.tanh`, `torch.sigmoid`) since the
# legacy forms raise/warn on current torch; the numerical behavior is identical.
"""Vendored LAAT / JointLAAT model definition (RNN backbone + hierarchical label attention)."""

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# src/models/embeddings/embedding_layer.py (verbatim)
# ---------------------------------------------------------------------------
class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        embedding_mode: str,
        pretrained_word_embeddings: torch.Tensor,
        vocab_size: int,
        embedding_size: int,
    ):
        """
        Init function
        :param embedding_mode: it can be "rand", "static", "non_static" or "multichannel".
            With "rand" mode, the embeddings are initialised randomly and fine tuned with the model training
            With "static" mode, the embeddings are initialised with pretrained embeddings and fixed
            With "non_static" mode, the embeddings are initialised with pretrained embeddings and fine tuned with the model training
            With "multichannel" mode, there are two versions of embeddings ("static" and "non_static")
        :param pretrained_word_embeddings: Pretrained word embeddings
        :param vocab_size: The size of the word vocab
        :param embedding_size: The embedding size
        """

        self.embedding_mode = embedding_mode
        if pretrained_word_embeddings is not None:
            embedding_size = pretrained_word_embeddings.size(-1)

        super(EmbeddingLayer, self).__init__()
        if embedding_mode.lower() == "rand" or pretrained_word_embeddings is None:
            self.embeddings = nn.Embedding(vocab_size, embedding_size)
            self.output_size = embedding_size

        elif (
            embedding_mode.lower() in ["static", "non_static"]
            and pretrained_word_embeddings is not None
        ):
            requires_grad = False if embedding_mode == "static" else True
            self.embeddings = nn.Embedding(vocab_size, embedding_size)
            self.embeddings.weight = nn.Parameter(
                copy.deepcopy(pretrained_word_embeddings), requires_grad=requires_grad
            )
            self.output_size = embedding_size

        elif embedding_mode.lower() == "multichannel":
            self.static_embeddings = nn.Embedding(vocab_size, embedding_size)
            self.static_embeddings.weight = nn.Parameter(
                copy.deepcopy(pretrained_word_embeddings), requires_grad=False
            )

            self.non_static_embeddings = nn.Embedding(vocab_size, embedding_size)
            self.non_static_embeddings.weight = nn.Parameter(
                copy.deepcopy(pretrained_word_embeddings), requires_grad=True
            )
            self.output_size = 2 * embedding_size
        else:
            raise NotImplementedError

    def forward(self, batch_data: torch.LongTensor):
        """
        :param batch_data: [batch_size x max_len]
        :return: [batch_size x max_len x embedding_size]
            embedding_size = word_embedding_size + char_embedding_size if using char level word embeddings
        """

        if self.embedding_mode.lower() == "multichannel":
            static_embeds = self.static_embeddings(batch_data)
            non_static_embeds = self.non_static_embeddings(batch_data)
            embeds = torch.cat([static_embeds, non_static_embeds], dim=2)
        else:
            embeds = self.embeddings(batch_data)  # [batch_size x max_seq_size x embedding_size]

        return embeds


def init_embedding_layer(args, vocab):
    embedding_layer = EmbeddingLayer(
        embedding_mode=args.mode,
        embedding_size=args.embedding_size,
        pretrained_word_embeddings=vocab.word_embeddings,
        vocab_size=vocab.n_words(),
    )

    return embedding_layer


# ---------------------------------------------------------------------------
# src/models/attentions/attention_layer.py (verbatim, F.tanh/F.sigmoid ->
# torch.tanh/torch.sigmoid and torch.nn.init.normal -> normal_ for modern torch)
# ---------------------------------------------------------------------------
class AttentionLayer(nn.Module):
    def __init__(
        self, args, size: int, level_projection_size: int = 0, n_labels=None, n_level: int = 1
    ):
        """
        The init function
        :param args: the input parameters from commandline
        :param size: the input size of the layer, it is normally the output size of other DNN models,
            such as CNN, RNN
        """
        super(AttentionLayer, self).__init__()
        self.attention_mode = args.attention_mode

        self.size = size
        # For self-attention: d_a and r are the dimension of the dense layer and the number of attention-hops
        # d_a is the output size of the first linear layer
        self.d_a = args.d_a if args.d_a > 0 else self.size

        # r is the number of attention heads

        self.n_labels = n_labels
        self.n_level = n_level
        self.r = [args.r if args.r > 0 else n_labels[label_lvl] for label_lvl in range(n_level)]

        self.level_projection_size = level_projection_size

        self.linear = nn.Linear(self.size, self.size, bias=False)
        if self.attention_mode == "hard":
            self.first_linears = nn.ModuleList(
                [nn.Linear(self.size, self.size, bias=True) for _ in range(self.n_level)]
            )
            self.second_linears = nn.ModuleList(
                [nn.Linear(self.size, 1, bias=False) for _ in range(self.n_level)]
            )

        elif self.attention_mode == "self":
            self.first_linears = nn.ModuleList(
                [nn.Linear(self.size, self.d_a, bias=False) for _ in range(self.n_level)]
            )
            self.second_linears = nn.ModuleList(
                [
                    nn.Linear(self.d_a, self.r[label_lvl], bias=False)
                    for label_lvl in range(self.n_level)
                ]
            )

        elif self.attention_mode == "label" or self.attention_mode == "caml":
            if self.attention_mode == "caml":
                self.d_a = self.size

            self.first_linears = nn.ModuleList(
                [nn.Linear(self.size, self.d_a, bias=False) for _ in range(self.n_level)]
            )
            self.second_linears = nn.ModuleList(
                [
                    nn.Linear(self.d_a, self.n_labels[label_lvl], bias=False)
                    for label_lvl in range(self.n_level)
                ]
            )
            self.third_linears = nn.ModuleList(
                [
                    nn.Linear(
                        self.size + (self.level_projection_size if label_lvl > 0 else 0),
                        self.n_labels[label_lvl],
                        bias=True,
                    )
                    for label_lvl in range(self.n_level)
                ]
            )
        else:
            raise NotImplementedError
        self._init_weights(mean=0.0, std=0.03)

    def _init_weights(self, mean=0.0, std=0.03) -> None:
        """
        Initialise the weights
        :param mean:
        :param std:
        :return: None
        """
        for first_linear in self.first_linears:
            torch.nn.init.normal_(first_linear.weight, mean, std)
            if first_linear.bias is not None:
                first_linear.bias.data.fill_(0)

        for linear in self.second_linears:
            torch.nn.init.normal_(linear.weight, mean, std)
            if linear.bias is not None:
                linear.bias.data.fill_(0)
        if self.attention_mode == "label" or self.attention_mode == "caml":
            for linear in self.third_linears:
                torch.nn.init.normal_(linear.weight, mean, std)

    def forward(self, x, previous_level_projection=None, label_level=0):
        """
        :param x: [batch_size x max_len x dim (i.e., self.size)]

        :param previous_level_projection: the embeddings for the previous level output
        :param label_level: the current label level
        :return:
            Weighted average output: [batch_size x dim (i.e., self.size)]
            Attention weights
        """
        if self.attention_mode == "caml":
            weights = torch.tanh(x)
        else:
            weights = torch.tanh(self.first_linears[label_level](x))

        att_weights = self.second_linears[label_level](weights)
        att_weights = F.softmax(att_weights, 1).transpose(1, 2)
        if len(att_weights.size()) != len(x.size()):
            att_weights = att_weights.squeeze()
        weighted_output = att_weights @ x

        if self.attention_mode == "label" or self.attention_mode == "caml":
            batch_size = weighted_output.size(0)

            if previous_level_projection is not None:
                temp = [
                    weighted_output,
                    previous_level_projection.repeat(1, self.n_labels[label_level]).view(
                        batch_size, self.n_labels[label_level], -1
                    ),
                ]
                weighted_output = torch.cat(temp, dim=2)

            weighted_output = (
                self.third_linears[label_level]
                .weight.mul(weighted_output)
                .sum(dim=2)
                .add(self.third_linears[label_level].bias)
            )

        else:
            weighted_output = torch.sum(weighted_output, 1) / self.r[label_level]
            if previous_level_projection is not None:
                temp = [weighted_output, previous_level_projection]
                weighted_output = torch.cat(temp, dim=1)

        return weighted_output, att_weights

    # Using when use_regularisation = True
    @staticmethod
    def l2_matrix_norm(m):
        """
        Frobenius norm calculation
        :param m: {Variable} ||AAT - I||
        :return: regularized value
        """
        return torch.sum(torch.sum(torch.sum(m**2, 1), 1) ** 0.5)


# ---------------------------------------------------------------------------
# src/models/attentions/util.py (verbatim, F.sigmoid -> torch.sigmoid)
# ---------------------------------------------------------------------------
def init_attention_layer(model):
    if model.args.joint_mode == "flat":
        if model.attention_mode is not None:
            model.attention = AttentionLayer(
                args=model.args,
                size=model.output_size,
                n_labels=model.vocab.all_n_labels(),
                n_level=model.vocab.n_level(),
            )

        model.linears = nn.ModuleList(
            [
                nn.Linear(
                    model.output_size + model.vocab.n_labels(level), model.vocab.n_labels(level)
                )
                for level in range(model.vocab.n_level())
            ]
        )

    elif model.args.joint_mode == "hierarchical":
        model.level_projection_size = model.args.level_projection_size
        if model.attention_mode is not None:
            model.attention = AttentionLayer(
                args=model.args,
                size=model.output_size,
                level_projection_size=model.level_projection_size,
                n_labels=model.vocab.all_n_labels(),
                n_level=model.vocab.n_level(),
            )
        linears = []
        projection_linears = []
        for level in range(model.vocab.n_level()):
            level_projection_size = 0 if level == 0 else model.level_projection_size
            linears.append(
                nn.Linear(model.output_size + level_projection_size, model.vocab.n_labels(level))
            )
            projection_linears.append(
                nn.Linear(model.vocab.n_labels(level), model.level_projection_size, bias=False)
            )
        model.linears = nn.ModuleList(linears)
        model.projection_linears = nn.ModuleList(projection_linears)
    else:
        raise NotImplementedError
    if model.attention_mode is not None:
        model.r = model.attention.r


def perform_attention(model, all_output, last_output):
    attention_weights = None
    if model.args.joint_mode == "flat":
        if model.attention_mode is not None:
            attention_outputs = [
                model.attention(all_output, label_level=label_lvl)
                for label_lvl in range(model.vocab.n_level())
            ]
            weighted_outputs = [
                attention_outputs[label_lvl][0] for label_lvl in range(model.vocab.n_level())
            ]
            attention_weights = [
                attention_outputs[label_lvl][1] for label_lvl in range(model.vocab.n_level())
            ]

            if model.attention_mode not in ["label", "caml"]:
                if model.use_dropout:
                    for label_lvl in range(model.vocab.n_level()):
                        weighted_outputs[label_lvl] = model.dropout(weighted_outputs[label_lvl])
                for label_lvl in range(model.vocab.n_level()):
                    weighted_outputs[label_lvl] = model.linears[label_lvl](
                        weighted_outputs[label_lvl]
                    )
        else:
            weighted_outputs = []
            if model.use_dropout:
                for label_lvl in range(model.vocab.n_level()):
                    weighted_outputs.append(model.dropout(last_output))
            else:
                weighted_outputs = [last_output] * model.vocab.n_level()

            for label_lvl in range(model.vocab.n_level()):
                weighted_outputs[label_lvl] = model.linears[label_lvl](weighted_outputs[label_lvl])

    elif model.args.joint_mode == "hierarchical":
        previous_level_projection = None
        if model.attention_mode is not None:
            weighted_outputs = []
            attention_weights = []
            for level in range(model.vocab.n_level()):
                weighted_output, attention_weight = model.attention(
                    all_output, previous_level_projection, label_level=level
                )
                if model.attention_mode not in ["label", "caml"]:
                    if model.use_dropout:
                        weighted_output = model.dropout(weighted_output)
                    weighted_output = model.linears[level](weighted_output)

                previous_level_projection = model.projection_linears[level](
                    torch.sigmoid(weighted_output)
                    if model.attention_mode in ["label", "caml"]
                    else torch.softmax(weighted_output, 1)
                )
                previous_level_projection = torch.sigmoid(previous_level_projection)
                weighted_outputs.append(weighted_output)
                attention_weights.append(attention_weight)
        else:
            weighted_outputs = []
            attention_weights = None
            previous_level_projection = None
            for level in range(model.vocab.n_level()):
                if previous_level_projection is not None:
                    last_output = [last_output, previous_level_projection]
                    last_output = torch.cat(last_output, dim=1)

                output = last_output
                if model.use_dropout:
                    output = model.dropout(last_output)

                output = model.linears[level](output)
                weighted_outputs.append(output)
                previous_level_projection = model.projection_linears[level](
                    torch.softmax(output, 1)
                )
    return weighted_outputs, attention_weights


# ---------------------------------------------------------------------------
# src/models/rnn.py (verbatim; `from torch.autograd import Variable` calls kept
# as-is -- Variable is a harmless no-op alias on modern torch)
# ---------------------------------------------------------------------------
class RNN(nn.Module):
    def __init__(self, vocab, args):
        """

        :param vocab: Vocab
            The vocabulary normally built on the training data
        :param args:
            mode: rand/static/non-static/multichannel the mode of initialising embeddings
            hidden_size: (int) The size of the hidden layer
            n_layers: (int) The number of hidden layers
            bidirectional: (bool) Whether or not using bidirectional connection
            dropout: (float) The dropout parameter for RNN (GRU or LSTM)
        """

        super(RNN, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state
        self.mode = args.mode
        self.n_layers = args.n_layers
        self.hidden_size = args.hidden_size
        self.bidirectional = bool(args.bidirectional)
        self.n_directions = int(self.bidirectional) + 1
        self.attention_mode = args.attention_mode
        self.output_size = self.hidden_size * self.n_directions
        self.rnn_model = args.rnn_model

        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)

        if self.rnn_model.lower() == "gru":
            self.rnn = nn.GRU(
                self.embedding.output_size,
                self.hidden_size,
                num_layers=self.n_layers,
                bidirectional=self.bidirectional,
                dropout=self.dropout if self.n_layers > 1 else 0,
            )
        else:
            self.rnn = nn.LSTM(
                self.embedding.output_size,
                self.hidden_size,
                num_layers=self.n_layers,
                bidirectional=self.bidirectional,
                dropout=self.dropout if self.n_layers > 1 else 0,
            )

        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)

    def init_hidden(self, batch_size: int = 1):
        """
        Initialise the hidden layer
        :param batch_size: int
            The batch size
        :return: Variable
            The initialised hidden layer
        """
        # [(n_layers x n_directions) x batch_size x hidden_size]
        device = (
            self.embedding.embeddings.weight.device
            if hasattr(self.embedding, "embeddings")
            else torch.device("cpu")
        )
        h = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size).to(device)
        if self.rnn_model.lower() == "gru":
            return h
        return h, c

    def forward(self, batch_data: torch.LongTensor, lengths: torch.LongTensor) -> tuple:
        """

        :param batch_data: torch.LongTensor
            [batch_size x max_seq_len]
        :param lengths: torch.LongTensor
            [batch_size x 1]
        :return: output [batch_size x n_classes]
            attention_weights
        """

        batch_size = batch_data.size()[0]
        hidden = self.init_hidden(batch_size)

        embeds = self.embedding(batch_data)

        if self.use_dropout:
            embeds = self.dropout(embeds)

        self.rnn.flatten_parameters()
        embeds = pack_padded_sequence(embeds, lengths, batch_first=True)

        rnn_output, hidden = self.rnn(embeds, hidden)
        if self.rnn_model.lower() == "lstm":
            hidden = hidden[0]

        rnn_output = pad_packed_sequence(rnn_output)[0]

        rnn_output = rnn_output.permute(1, 0, 2)

        weighted_outputs, attention_weights = perform_attention(
            self, rnn_output, self.get_last_hidden_output(hidden)
        )
        return weighted_outputs, attention_weights

    def get_last_hidden_output(self, hidden):
        if self.bidirectional:
            hidden_forward = hidden[-1]
            hidden_backward = hidden[0]
            if len(hidden_backward.shape) > 2:
                hidden_forward = hidden_forward.squeeze(0)
                hidden_backward = hidden_backward.squeeze(0)
            last_rnn_output = torch.cat((hidden_forward, hidden_backward), 1)
        else:
            last_rnn_output = hidden[-1]
            if len(hidden.shape) > 2:
                last_rnn_output = last_rnn_output.squeeze(0)

        return last_rnn_output


# ---------------------------------------------------------------------------
# Staging build/example helpers. `_TinyVocab` implements only the vocab
# interface the real model code calls (n_words/all_n_labels/n_level/
# n_labels/word_embeddings) so we can construct the real RNN+JointLAAT stack
# with embedding_mode="rand" (one of the repo's own supported embedding
# modes) without pulling in gensim + real MIMIC-III label vocabularies.
# Hyperparameters below mirror the repo's own example run.sh (LSTM,
# bidirectional, attention_mode="label", joint_mode="hierarchical" i.e.
# JointLAAT), just at a tiny scale for tracing.
# ---------------------------------------------------------------------------
class _TinyVocab:
    def __init__(self, vocab_size, label_counts):
        self._vocab_size = vocab_size
        self._label_counts = label_counts  # e.g. [n_labels_level0, n_labels_level1]
        self.word_embeddings = None  # forces embedding_mode="rand" path

    def n_words(self):
        return self._vocab_size

    def n_level(self):
        return len(self._label_counts)

    def n_labels(self, level):
        return self._label_counts[level]

    def all_n_labels(self):
        return list(self._label_counts)


@dataclass
class _JointLAATArgs:
    # Embedding
    mode: str = "rand"
    embedding_size: int = 16
    # RNN backbone (matches run.sh's RNN/--rnn_model LSTM subcommand)
    use_last_hidden_state: int = 0
    n_layers: int = 1
    hidden_size: int = 24
    bidirectional: int = 1
    rnn_model: str = "LSTM"
    dropout: float = 0.3
    # Attention (matches run.sh: --attention_mode label --d_a 512, tiny d_a here)
    attention_mode: str = "label"
    d_a: int = 20
    r: int = -1
    # Joint mode: "hierarchical" is JointLAAT (the repo's own default)
    joint_mode: str = "hierarchical"
    level_projection_size: int = 12


_VOCAB_SIZE = 40
_LABEL_COUNTS = [5, 9]  # two label levels, matches mimic-iii "2_full" (3-char + full)
_SEQ_LEN = 12
_BATCH = 2


def build_jointlaat():
    vocab = _TinyVocab(_VOCAB_SIZE, _LABEL_COUNTS)
    args = _JointLAATArgs()
    return RNN(vocab, args)


def example_input_jointlaat():
    batch_data = torch.randint(low=0, high=_VOCAB_SIZE, size=(_BATCH, _SEQ_LEN), dtype=torch.long)
    lengths = torch.tensor([_SEQ_LEN, _SEQ_LEN - 3], dtype=torch.long)
    return (batch_data, lengths)


MENAGERIE_ENTRIES = [
    ("JointLAAT", "build_jointlaat", "example_input_jointlaat", 2020, "vendored-pytorch"),
]

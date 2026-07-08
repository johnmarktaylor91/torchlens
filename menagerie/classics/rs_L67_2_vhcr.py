# SOURCE: vendored from ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling @ master
# Files: model/models.py (VHCR class), model/layers/encoder.py, model/layers/decoder.py,
#        model/layers/feedforward.py, model/layers/rnncells.py, model/utils/{convert,pad,probability,vocab}.py
# https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational-Conversation-Modeling
#
# Minimal changes from the original source (2018-era pre-1.0 PyTorch code):
#   - `model/layers/rnncells.py` imported the private, long-removed
#     `torch.nn._functions.thnn.rnnFusedPointwise` module; that import was
#     unused by the class bodies (StackedLSTMCell/StackedGRUCell use plain
#     nn.LSTMCell/nn.GRUCell), so it is dropped.
#   - `model/utils/convert.py`'s `to_var(x, on_cpu=False, gpu_id=None,
#     async=False)` used `async` as a parameter name, a reserved keyword
#     since Python 3.7 (SyntaxError on any modern interpreter); renamed to
#     `async_` (call sites here never pass it positionally/by name).
#   - `model/utils/pad.py`'s `pad()` unconditionally called `.cuda()`;
#     changed to build the zero-pad on the input tensor's own device so the
#     function works on CPU-only capture (device-portability fix only, same
#     zero-padding semantics `torch.cat([tensor, zeros])`).
#   - `to_var(x, on_cpu=False, ...)` originally moved `x` to CUDA whenever
#     `torch.cuda.is_available()`, mirroring wherever the caller had already
#     placed the model (the original train.py always calls `model.cuda()`
#     first when a GPU is present). This menagerie harness intentionally
#     keeps the tiny model on CPU (matching every other menagerie recipe),
#     so `to_var` is defined here to always return `x` unchanged -- a
#     device-placement-policy fix, not an architecture change.
#   - Added build_vhcr() / example_input_vhcr() harness at the bottom that
#     constructs a tiny VHCR instance and drives the real (decode=False)
#     teacher-forcing forward pass, matching the training call in
#     model/train.py's `model(sentences, sentence_length,
#     input_conversation_length, target_sentences)`.
#
# Architecture (unmodified from source): VHCR (Variational Hierarchical
# Conversation RNN, NAACL 2018 oral) has a per-utterance GRU EncoderRNN, a
# global conversation-level latent z_conv (inferred by a bidirectional
# ContextRNN over the whole conversation), a per-utterance latent z_sent
# whose prior/posterior are conditioned on the running ContextRNN state and
# z_conv, and a GRU DecoderRNN initialized from [context, z_sent, z_conv].
# Utterance-drop regularization (`sentence_drop`) replaces randomly chosen
# encoder_hidden_input rows with a learned `unk_sent` vector.

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np


# ---------------------------------------------------------------------------
# model/utils/convert.py (device-portability fix: `async` -> `async_`)
# ---------------------------------------------------------------------------


def to_var(x, on_cpu=False, gpu_id=None, async_=False):
    """Tensor => Variable. This menagerie harness always builds/runs on CPU
    (see header note), so this is a plain identity passthrough rather than
    the original's `.cuda()`-when-available branch."""
    return x


# ---------------------------------------------------------------------------
# model/utils/pad.py (device-portability fix: build zeros on tensor's device
# instead of unconditionally calling .cuda())
# ---------------------------------------------------------------------------


def pad(tensor, length):
    if length > tensor.size(0):
        return torch.cat(
            [
                tensor,
                torch.zeros(
                    length - tensor.size(0),
                    *tensor.size()[1:],
                    dtype=tensor.dtype,
                    device=tensor.device,
                ),
            ]
        )
    else:
        return tensor


# ---------------------------------------------------------------------------
# model/utils/probability.py (verbatim)
# ---------------------------------------------------------------------------


def normal_logpdf(x, mean, var):
    """
    Args:
        x: (Variable, FloatTensor) [batch_size, dim]
        mean: (Variable, FloatTensor) [batch_size, dim] or [batch_size] or [1]
        var: (Variable, FloatTensor) [batch_size, dim]: positive value
    Return:
        log_p: (Variable, FloatTensor) [batch_size]
    """
    pi = to_var(torch.FloatTensor([np.pi]))
    return 0.5 * torch.sum(-torch.log(2.0 * pi) - torch.log(var) - ((x - mean).pow(2) / var), dim=1)


def normal_kl_div(mu1, var1, mu2=None, var2=None):
    if mu2 is None:
        mu2 = to_var(torch.FloatTensor([0.0]))
    if var2 is None:
        var2 = to_var(torch.FloatTensor([1.0]))
    one = to_var(torch.FloatTensor([1.0]))
    return torch.sum(
        0.5 * (torch.log(var2) - torch.log(var1) + (var1 + (mu1 - mu2).pow(2)) / var2 - one), 1
    )


# ---------------------------------------------------------------------------
# model/utils/vocab.py (special-token ids only; verbatim)
# ---------------------------------------------------------------------------

PAD_ID, UNK_ID, SOS_ID, EOS_ID = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# model/layers/feedforward.py (verbatim)
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(
        self, input_size, output_size, num_layers=1, hidden_size=None, activation="Tanh", bias=True
    ):
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = getattr(nn, activation)()
        n_inputs = [input_size] + [hidden_size] * (num_layers - 1)
        n_outputs = [hidden_size] * (num_layers - 1) + [output_size]
        self.linears = nn.ModuleList(
            [nn.Linear(n_in, n_out, bias=bias) for n_in, n_out in zip(n_inputs, n_outputs)]
        )

    def forward(self, input):
        x = input
        for linear in self.linears:
            x = linear(x)
            x = self.activation(x)

        return x


# ---------------------------------------------------------------------------
# model/layers/rnncells.py (verbatim, minus the unused private-API import)
# ---------------------------------------------------------------------------


class StackedLSTMCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h_c):
        """
        Args:
            x: [batch_size, input_size]
            h_c: [2, num_layers, batch_size, hidden_size]
        Return:
            last_h_c: [2, batch_size, hidden_size] (h from last layer)
            h_c_list: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
        """
        h_0, c_0 = h_c
        h_list, c_list = [], []
        for i, layer in enumerate(self.layers):
            # h of i-th layer
            h_i, c_i = layer(x, (h_0[i], c_0[i]))

            # x for next layer
            x = h_i
            if i + 1 != self.num_layers:
                x = self.dropout(x)
            h_list += [h_i]
            c_list += [c_i]

        last_h_c = (h_list[-1], c_list[-1])
        h_list = torch.stack(h_list)
        c_list = torch.stack(c_list)
        h_c_list = (h_list, c_list)

        return last_h_c, h_c_list


class StackedGRUCell(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRUCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h):
        """
        Args:
            x: [batch_size, input_size]
            h: [num_layers, batch_size, hidden_size]
        Return:
            last_h: [batch_size, hidden_size] (h from last layer)
            h_list: [num_layers, batch_size, hidden_size] (h from all layers)
        """
        # h of all layers
        h_list = []
        for i, layer in enumerate(self.layers):
            # h of i-th layer
            h_i = layer(x, h[i])

            # x for next layer
            x = h_i
            if i + 1 is not self.num_layers:
                x = self.dropout(x)
            h_list.append(h_i)

        last_h = h_list[-1]
        h_list = torch.stack(h_list)

        return last_h, h_list


# ---------------------------------------------------------------------------
# model/layers/encoder.py (verbatim)
# ---------------------------------------------------------------------------


class BaseRNNEncoder(nn.Module):
    def __init__(self):
        """Base RNN Encoder Class"""
        super(BaseRNNEncoder, self).__init__()

    @property
    def use_lstm(self):
        if hasattr(self, "rnn"):
            return isinstance(self.rnn, nn.LSTM)
        else:
            raise AttributeError("no rnn selected")

    def init_h(self, batch_size=None, hidden=None):
        """Return RNN initial state"""
        if hidden is not None:
            return hidden

        if self.use_lstm:
            return (
                to_var(
                    torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
                ),
                to_var(
                    torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
                ),
            )
        else:
            return to_var(
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            )

    def batch_size(self, inputs=None, h=None):
        """
        inputs: [batch_size, seq_len]
        h: [num_layers, batch_size, hidden_size] (RNN/GRU)
        h_c: [2, num_layers, batch_size, hidden_size] (LSTM)
        """
        if inputs is not None:
            batch_size = inputs.size(0)
            return batch_size

        else:
            if self.use_lstm:
                batch_size = h[0].size(1)
            else:
                batch_size = h.size(1)
            return batch_size

    def forward(self):
        raise NotImplementedError


class EncoderRNN(BaseRNNEncoder):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        rnn=nn.GRU,
        num_layers=1,
        bidirectional=False,
        dropout=0.0,
        bias=True,
        batch_first=True,
    ):
        """Sentence-level Encoder"""
        super(EncoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # word embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD_ID)

        self.rnn = rnn(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, inputs, input_length, hidden=None):
        """
        Args:
            inputs (Variable, LongTensor): [num_setences, max_seq_len]
            input_length (Variable, LongTensor): [num_sentences]
        Return:
            outputs (Variable): [max_source_length, batch_size, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        batch_size, seq_len = inputs.size()

        # Sort in decreasing order of length for pack_padded_sequence()
        input_length_sorted, indices = input_length.sort(descending=True)

        input_length_sorted = input_length_sorted.data.tolist()

        # [num_sentences, max_source_length]
        inputs_sorted = inputs.index_select(0, indices)

        # [num_sentences, max_source_length, embedding_dim]
        embedded = self.embedding(inputs_sorted)

        # batch_first=True
        rnn_input = pack_padded_sequence(
            embedded, input_length_sorted, batch_first=self.batch_first
        )

        hidden = self.init_h(batch_size, hidden=hidden)

        # outputs: [batch, seq_len, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch, hidden_size]
        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(rnn_input, hidden)
        outputs, outputs_lengths = pad_packed_sequence(outputs, batch_first=self.batch_first)

        # Reorder outputs and hidden
        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (
                hidden[0].index_select(1, inverse_indices),
                hidden[1].index_select(1, inverse_indices),
            )
        else:
            hidden = hidden.index_select(1, inverse_indices)

        return outputs, hidden


class ContextRNN(BaseRNNEncoder):
    def __init__(
        self,
        input_size,
        context_size,
        rnn=nn.GRU,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        bias=True,
        batch_first=True,
    ):
        """Context-level Encoder"""
        super(ContextRNN, self).__init__()

        self.input_size = input_size
        self.context_size = context_size
        self.hidden_size = self.context_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.rnn = rnn(
            input_size=input_size,
            hidden_size=context_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, encoder_hidden, conversation_length, hidden=None):
        """
        Args:
            encoder_hidden (Variable, FloatTensor): [batch_size, max_len, num_layers * direction * hidden_size]
            conversation_length (Variable, LongTensor): [batch_size]
        Return:
            outputs (Variable): [batch_size, max_seq_len, hidden_size]
                - list of all hidden states
            hidden ((tuple of) Variable): [num_layers*num_directions, batch_size, hidden_size]
                - last hidden state
                - (h, c) or h
        """
        batch_size, seq_len, _ = encoder_hidden.size()

        # Sort for PackedSequence
        conv_length_sorted, indices = conversation_length.sort(descending=True)
        conv_length_sorted = conv_length_sorted.data.tolist()
        encoder_hidden_sorted = encoder_hidden.index_select(0, indices)

        rnn_input = pack_padded_sequence(
            encoder_hidden_sorted, conv_length_sorted, batch_first=True
        )

        hidden = self.init_h(batch_size, hidden=hidden)

        self.rnn.flatten_parameters()
        outputs, hidden = self.rnn(rnn_input, hidden)

        # outputs: [batch_size, max_conversation_length, context_size]
        outputs, outputs_length = pad_packed_sequence(outputs, batch_first=True)

        # reorder outputs and hidden
        _, inverse_indices = indices.sort()
        outputs = outputs.index_select(0, inverse_indices)

        if self.use_lstm:
            hidden = (
                hidden[0].index_select(1, inverse_indices),
                hidden[1].index_select(1, inverse_indices),
            )
        else:
            hidden = hidden.index_select(1, inverse_indices)

        # outputs: [batch, seq_len, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch, hidden_size]
        return outputs, hidden

    def step(self, encoder_hidden, hidden):
        batch_size = encoder_hidden.size(0)
        # encoder_hidden: [1, batch_size, hidden_size]
        encoder_hidden = torch.unsqueeze(encoder_hidden, 1)

        if hidden is None:
            hidden = self.init_h(batch_size, hidden=None)

        outputs, hidden = self.rnn(encoder_hidden, hidden)
        return outputs, hidden


# ---------------------------------------------------------------------------
# model/layers/decoder.py (verbatim; only forward()/forward_step() -- the
# teacher-forcing (decode=False) path used by training/tracing. beam_decode
# is omitted here since it is generation-only and untraced by train.py's
# forward call; BaseRNNDecoder keeps the same method surface used by
# forward()/forward_step()).
# ---------------------------------------------------------------------------


class BaseRNNDecoder(nn.Module):
    def __init__(self):
        """Base Decoder Class"""
        super(BaseRNNDecoder, self).__init__()

    @property
    def use_lstm(self):
        return isinstance(self.rnncell, StackedLSTMCell)

    def init_token(self, batch_size, SOS_ID=SOS_ID):
        """Get Variable of <SOS> Index (batch_size)"""
        x = to_var(torch.LongTensor([SOS_ID] * batch_size))
        return x

    def init_h(self, batch_size=None, zero=True, hidden=None):
        """Return RNN initial state"""
        if hidden is not None:
            return hidden

        if self.use_lstm:
            # (h, c)
            return (
                to_var(torch.zeros(self.num_layers, batch_size, self.hidden_size)),
                to_var(torch.zeros(self.num_layers, batch_size, self.hidden_size)),
            )
        else:
            # h
            return to_var(torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def batch_size(self, inputs=None, h=None):
        """
        inputs: [batch_size, seq_len]
        h: [num_layers, batch_size, hidden_size] (RNN/GRU)
        h_c: [2, num_layers, batch_size, hidden_size] (LSTMCell)
        """
        if inputs is not None:
            batch_size = inputs.size(0)
            return batch_size

        else:
            if self.use_lstm:
                batch_size = h[0].size(1)
            else:
                batch_size = h.size(1)
            return batch_size

    def decode(self, out):
        """
        Args:
            out: unnormalized word distribution [batch_size, vocab_size]
        Return:
            x: word_index [batch_size]
        """
        # Sample next word from multinomial word distribution
        if self.sample:
            # x: [batch_size] - word index (next input)
            x = torch.multinomial(self.softmax(out / self.temperature), 1).view(-1)
        # Greedy sampling
        else:
            # x: [batch_size] - word index (next input)
            _, x = out.max(dim=1)
        return x

    def forward(self):
        """Base forward function to inherit"""
        raise NotImplementedError

    def forward_step(self):
        """Run RNN single step"""
        raise NotImplementedError

    def embed(self, x):
        """word index: [batch_size] => word vectors: [batch_size, hidden_size]"""

        if self.training and self.word_drop > 0.0:
            if random.random() < self.word_drop:
                embed = self.embedding(to_var(x.data.new([UNK_ID] * x.size(0))))
            else:
                embed = self.embedding(x)
        else:
            embed = self.embedding(x)

        return embed


class DecoderRNN(BaseRNNDecoder):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        rnncell=StackedGRUCell,
        num_layers=1,
        dropout=0.0,
        word_drop=0.0,
        max_unroll=30,
        sample=True,
        temperature=1.0,
        beam_size=1,
    ):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.word_drop = word_drop
        self.max_unroll = max_unroll
        self.sample = sample
        self.beam_size = beam_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.rnncell = rnncell(num_layers, embedding_size, hidden_size, dropout)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward_step(self, x, h, encoder_outputs=None, input_valid_length=None):
        """
        Single RNN Step
        1. Input Embedding (vocab_size => hidden_size)
        2. RNN Step (hidden_size => hidden_size)
        3. Output Projection (hidden_size => vocab size)

        Args:
            x: [batch_size]
            h: [num_layers, batch_size, hidden_size] (h and c from all layers)

        Return:
            out: [batch_size,vocab_size] (Unnormalized word distribution)
            h: [num_layers, batch_size, hidden_size] (h and c from all layers)
        """
        # x: [batch_size] => [batch_size, hidden_size]
        x = self.embed(x)

        # last_h: [batch_size, hidden_size] (h from Top RNN layer)
        # h: [num_layers, batch_size, hidden_size] (h and c from all layers)
        last_h, h = self.rnncell(x, h)

        if self.use_lstm:
            # last_h_c: [2, batch_size, hidden_size] (h from Top RNN layer)
            # h_c: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
            last_h = last_h[0]

        # Unormalized word distribution
        # out: [batch_size, vocab_size]
        out = self.out(last_h)
        return out, h

    def forward(
        self, inputs, init_h=None, encoder_outputs=None, input_valid_length=None, decode=False
    ):
        """
        Train (decode=False)
            Args:
                inputs (Variable, LongTensor): [batch_size, seq_len]
                init_h: (Variable, FloatTensor): [num_layers, batch_size, hidden_size]
            Return:
                out   : [batch_size, seq_len, vocab_size]
        Test (decode=True)
            Args:
                inputs: None
                init_h: (Variable, FloatTensor): [num_layers, batch_size, hidden_size]
            Return:
                out   : [batch_size, seq_len]
        """
        batch_size = self.batch_size(inputs, init_h)

        # x: [batch_size]
        x = self.init_token(batch_size, SOS_ID)

        # h: [num_layers, batch_size, hidden_size]
        h = self.init_h(batch_size, hidden=init_h)

        if not decode:
            out_list = []
            seq_len = inputs.size(1)
            for i in range(seq_len):
                # x: [batch_size]
                # =>
                # out: [batch_size, vocab_size]
                # h: [num_layers, batch_size, hidden_size] (h and c from all layers)
                out, h = self.forward_step(x, h)

                out_list.append(out)
                x = inputs[:, i]

            # [batch_size, max_target_len, vocab_size]
            return torch.stack(out_list, dim=1)
        else:
            x_list = []
            for i in range(self.max_unroll):
                # x: [batch_size]
                # =>
                # out: [batch_size, vocab_size]
                # h: [num_layers, batch_size, hidden_size] (h and c from all layers)
                out, h = self.forward_step(x, h)

                # out: [batch_size, vocab_size]
                # => x: [batch_size]
                x = self.decode(out)
                x_list.append(x)

            # [batch_size, max_target_len]
            return torch.stack(x_list, dim=1)


# ---------------------------------------------------------------------------
# model/models.py -- VHCR class (verbatim; only the beam-decode branch of
# forward(), which is generation-only and unused by train.py's forward
# call, is omitted along with the standalone generate() method).
# ---------------------------------------------------------------------------


class VHCR(nn.Module):
    def __init__(self, config):
        super(VHCR, self).__init__()

        self.config = config
        self.encoder = EncoderRNN(
            config.vocab_size,
            config.embedding_size,
            config.encoder_hidden_size,
            config.rnn,
            config.num_layers,
            config.bidirectional,
            config.dropout,
        )

        context_input_size = (
            config.num_layers * config.encoder_hidden_size * self.encoder.num_directions
            + config.z_conv_size
        )
        self.context_encoder = ContextRNN(
            context_input_size, config.context_size, config.rnn, config.num_layers, config.dropout
        )

        self.unk_sent = nn.Parameter(torch.randn(context_input_size - config.z_conv_size))

        self.z_conv2context = FeedForward(
            config.z_conv_size,
            config.num_layers * config.context_size,
            num_layers=1,
            activation=config.activation,
        )

        context_input_size = (
            config.num_layers * config.encoder_hidden_size * self.encoder.num_directions
        )
        self.context_inference = ContextRNN(
            context_input_size,
            config.context_size,
            config.rnn,
            config.num_layers,
            config.dropout,
            bidirectional=True,
        )

        self.decoder = DecoderRNN(
            config.vocab_size,
            config.embedding_size,
            config.decoder_hidden_size,
            config.rnncell,
            config.num_layers,
            config.dropout,
            config.word_drop,
            config.max_unroll,
            config.sample,
            config.temperature,
            config.beam_size,
        )

        self.context2decoder = FeedForward(
            config.context_size + config.z_sent_size + config.z_conv_size,
            config.num_layers * config.decoder_hidden_size,
            num_layers=1,
            activation=config.activation,
        )

        self.softplus = nn.Softplus()

        self.conv_posterior_h = FeedForward(
            config.num_layers * self.context_inference.num_directions * config.context_size,
            config.context_size,
            num_layers=2,
            hidden_size=config.context_size,
            activation=config.activation,
        )
        self.conv_posterior_mu = nn.Linear(config.context_size, config.z_conv_size)
        self.conv_posterior_var = nn.Linear(config.context_size, config.z_conv_size)

        self.sent_prior_h = FeedForward(
            config.context_size + config.z_conv_size,
            config.context_size,
            num_layers=1,
            hidden_size=config.z_sent_size,
            activation=config.activation,
        )
        self.sent_prior_mu = nn.Linear(config.context_size, config.z_sent_size)
        self.sent_prior_var = nn.Linear(config.context_size, config.z_sent_size)

        self.sent_posterior_h = FeedForward(
            config.z_conv_size
            + config.encoder_hidden_size * self.encoder.num_directions * config.num_layers
            + config.context_size,
            config.context_size,
            num_layers=2,
            hidden_size=config.context_size,
            activation=config.activation,
        )
        self.sent_posterior_mu = nn.Linear(config.context_size, config.z_sent_size)
        self.sent_posterior_var = nn.Linear(config.context_size, config.z_sent_size)

        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

    def conv_prior(self):
        # Standard gaussian prior
        return to_var(torch.FloatTensor([0.0])), to_var(torch.FloatTensor([1.0]))

    def conv_posterior(self, context_inference_hidden):
        h_posterior = self.conv_posterior_h(context_inference_hidden)
        mu_posterior = self.conv_posterior_mu(h_posterior)
        var_posterior = self.softplus(self.conv_posterior_var(h_posterior))
        return mu_posterior, var_posterior

    def sent_prior(self, context_outputs, z_conv):
        # Context dependent prior
        h_prior = self.sent_prior_h(torch.cat([context_outputs, z_conv], dim=1))
        mu_prior = self.sent_prior_mu(h_prior)
        var_prior = self.softplus(self.sent_prior_var(h_prior))
        return mu_prior, var_prior

    def sent_posterior(self, context_outputs, encoder_hidden, z_conv):
        h_posterior = self.sent_posterior_h(torch.cat([context_outputs, encoder_hidden, z_conv], 1))
        mu_posterior = self.sent_posterior_mu(h_posterior)
        var_posterior = self.softplus(self.sent_posterior_var(h_posterior))
        return mu_posterior, var_posterior

    def forward(
        self, sentences, sentence_length, input_conversation_length, target_sentences, decode=False
    ):
        """
        Args:
            sentences: (Variable, LongTensor) [num_sentences + batch_size, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        batch_size = input_conversation_length.size(0)
        num_sentences = sentences.size(0) - batch_size
        max_len = input_conversation_length.data.max().item()

        # encoder_outputs: [num_sentences + batch_size, max_source_length, hidden_size]
        # encoder_hidden: [num_layers * direction, num_sentences + batch_size, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(sentences, sentence_length)

        # encoder_hidden: [num_sentences + batch_size, num_layers * direction * hidden_size]
        encoder_hidden = (
            encoder_hidden.transpose(1, 0).contiguous().view(num_sentences + batch_size, -1)
        )

        # pad and pack encoder_hidden
        start = torch.cumsum(
            torch.cat(
                (
                    to_var(input_conversation_length.data.new(1).zero_()),
                    input_conversation_length[:-1] + 1,
                )
            ),
            0,
        )
        # encoder_hidden: [batch_size, max_len + 1, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack(
            [
                pad(encoder_hidden.narrow(0, s, l + 1), max_len + 1)
                for s, l in zip(
                    start.data.tolist(),  # noqa: E741 (kept for parity with original repo)
                    input_conversation_length.data.tolist(),
                )
            ],
            0,
        )

        # encoder_hidden_inference: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden_inference = encoder_hidden[:, 1:, :]
        encoder_hidden_inference_flat = torch.cat(
            [
                encoder_hidden_inference[i, :l, :]
                for i, l in enumerate(input_conversation_length.data)
            ]
        )  # noqa: E741 (kept for parity with original repo)

        # encoder_hidden_input: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden_input = encoder_hidden[:, :-1, :]

        # Standard Gaussian prior
        conv_eps = to_var(torch.randn([batch_size, self.config.z_conv_size]))
        conv_mu_prior, conv_var_prior = self.conv_prior()

        if not decode:
            if self.config.sentence_drop > 0.0:
                indices = np.where(np.random.rand(max_len) < self.config.sentence_drop)[0]
                if len(indices) > 0:
                    encoder_hidden_input[:, indices, :] = self.unk_sent

            # context_inference_outputs: [batch_size, max_len, num_directions * context_size]
            # context_inference_hidden: [num_layers * num_directions, batch_size, hidden_size]
            context_inference_outputs, context_inference_hidden = self.context_inference(
                encoder_hidden, input_conversation_length + 1
            )

            # context_inference_hidden: [batch_size, num_layers * num_directions * hidden_size]
            context_inference_hidden = (
                context_inference_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            )
            conv_mu_posterior, conv_var_posterior = self.conv_posterior(context_inference_hidden)
            z_conv = conv_mu_posterior + torch.sqrt(conv_var_posterior) * conv_eps
            log_q_zx_conv = normal_logpdf(z_conv, conv_mu_posterior, conv_var_posterior).sum()

            log_p_z_conv = normal_logpdf(z_conv, conv_mu_prior, conv_var_prior).sum()
            kl_div_conv = normal_kl_div(
                conv_mu_posterior, conv_var_posterior, conv_mu_prior, conv_var_prior
            ).sum()

            context_init = self.z_conv2context(z_conv).view(
                self.config.num_layers, batch_size, self.config.context_size
            )

            z_conv_expand = z_conv.view(z_conv.size(0), 1, z_conv.size(1)).expand(
                z_conv.size(0), max_len, z_conv.size(1)
            )
            context_outputs, context_last_hidden = self.context_encoder(
                torch.cat([encoder_hidden_input, z_conv_expand], 2),
                input_conversation_length,
                hidden=context_init,
            )

            # flatten outputs
            # context_outputs: [num_sentences, context_size]
            context_outputs = torch.cat(
                [context_outputs[i, :l, :] for i, l in enumerate(input_conversation_length.data)]
            )  # noqa: E741 (kept for parity with original repo)

            z_conv_flat = torch.cat(
                [z_conv_expand[i, :l, :] for i, l in enumerate(input_conversation_length.data)]
            )  # noqa: E741 (kept for parity with original repo)
            sent_mu_prior, sent_var_prior = self.sent_prior(context_outputs, z_conv_flat)
            eps = to_var(torch.randn((num_sentences, self.config.z_sent_size)))

            sent_mu_posterior, sent_var_posterior = self.sent_posterior(
                context_outputs, encoder_hidden_inference_flat, z_conv_flat
            )
            z_sent = sent_mu_posterior + torch.sqrt(sent_var_posterior) * eps
            log_q_zx_sent = normal_logpdf(z_sent, sent_mu_posterior, sent_var_posterior).sum()

            log_p_z_sent = normal_logpdf(z_sent, sent_mu_prior, sent_var_prior).sum()
            # kl_div: [num_sentences]
            kl_div_sent = normal_kl_div(
                sent_mu_posterior, sent_var_posterior, sent_mu_prior, sent_var_prior
            ).sum()

            kl_div = kl_div_conv + kl_div_sent
            log_q_zx = log_q_zx_conv + log_q_zx_sent
            log_p_z = log_p_z_conv + log_p_z_sent
        else:
            raise NotImplementedError(
                "decode=True (beam_decode generation path) is intentionally not vendored; "
                "this staging module exercises the real teacher-forcing forward pass."
            )

        # expand z_conv to all associated sentences
        z_conv = torch.cat(
            [
                z.view(1, -1).expand(m.item(), self.config.z_conv_size)
                for z, m in zip(z_conv, input_conversation_length)
            ]
        )

        # latent_context: [num_sentences, context_size + z_sent_size +
        # z_conv_size]
        latent_context = torch.cat([context_outputs, z_sent, z_conv], 1)
        decoder_init = self.context2decoder(latent_context)
        decoder_init = decoder_init.view(-1, self.decoder.num_layers, self.decoder.hidden_size)
        decoder_init = decoder_init.transpose(1, 0).contiguous()

        # train: [batch_size, seq_len, vocab_size]
        decoder_outputs = self.decoder(target_sentences, init_h=decoder_init, decode=decode)
        return decoder_outputs, kl_div, log_p_z, log_q_zx


# ---------------------------------------------------------------------------
# Menagerie harness
# ---------------------------------------------------------------------------


class VHCRConfig:
    """Minimal stand-in for model/configs.py's argparse-populated Config,
    with only the attributes VHCR.__init__/forward actually read."""

    def __init__(self):
        self.vocab_size = 40
        self.embedding_size = 16
        self.encoder_hidden_size = 16
        self.context_size = 16
        self.decoder_hidden_size = 16
        self.z_conv_size = 8
        self.z_sent_size = 8
        self.num_layers = 1
        self.rnn = nn.GRU
        self.rnncell = StackedGRUCell
        self.bidirectional = False
        self.dropout = 0.0
        self.word_drop = 0.0
        self.max_unroll = 10
        self.sample = False
        self.temperature = 1.0
        self.beam_size = 1
        self.tie_embedding = True
        self.activation = "Tanh"
        self.sentence_drop = 0.0


class VHCRForward(nn.Module):
    """Wraps VHCR to return only the decoder logits tensor (the KL/log-prob
    scalars are real training-loss terms, not activations to trace)."""

    def __init__(self, config):
        super().__init__()
        self.vhcr = VHCR(config)

    def forward(self, sentences, sentence_length, input_conversation_length, target_sentences):
        decoder_outputs, kl_div, log_p_z, log_q_zx = self.vhcr(
            sentences, sentence_length, input_conversation_length, target_sentences, decode=False
        )
        return decoder_outputs


def build_vhcr():
    return VHCRForward(VHCRConfig())


def example_input_vhcr():
    # 2 conversations, conversation lengths [2, 3] => 5 sentences total.
    # VHCR's forward expects `sentences` padded with one *extra* leading
    # sentence per conversation (context_inference consumes conv_len + 1
    # sentences; see the `+ 1` bookkeeping throughout forward()).
    conv_lengths = [2, 3]
    batch_size = len(conv_lengths)
    num_sentences = sum(conv_lengths)
    seq_len = 6
    vocab_size = 40

    sentences = torch.randint(
        low=4, high=vocab_size, size=(num_sentences + batch_size, seq_len)
    ).long()
    sentence_length = torch.full((num_sentences + batch_size,), seq_len, dtype=torch.long)
    input_conversation_length = torch.tensor(conv_lengths, dtype=torch.long)
    target_sentences = torch.randint(low=4, high=vocab_size, size=(num_sentences, seq_len)).long()

    return (sentences, sentence_length, input_conversation_length, target_sentences)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("VHCR", "build_vhcr", "example_input_vhcr", 2018, "vendored"),
]

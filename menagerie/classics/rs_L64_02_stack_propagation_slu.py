# SOURCE: vendored from LeePleased/StackPropagation-SLU @ master
# https://raw.githubusercontent.com/LeePleased/StackPropagation-SLU/master/utils/module.py
#
# Qin et al. 2019 (EMNLP) "A Stack-Propagation Framework with Token-Level Intent
# Detection for Spoken Language Understanding" -- official repo. `utils/module.py`
# defines the real model: `EmbeddingCollection` (word embedding), `LSTMEncoder`
# (bidirectional LSTM over the packed/padded sequence), `SelfAttention`/
# `QKVAttention` (scaled dot-product self-attention over the word embeddings),
# and `LSTMDecoder` (a unidirectional, token-by-token autoregressive LSTM with a
# learned start token and label-embedding feedback -- used twice: once for
# intent detection, once for slot filling, with the slot decoder additionally
# consuming the intent decoder's output as an extra per-token input -- this is
# the paper's namesake "stack-propagation" of intent into slot filling).
# `ModelManager` wires all of the above together exactly as in the original
# `forward()`.
#
# Copied verbatim from `utils/module.py` with NO architecture changes -- only
# the double-underscore (name-mangled) private attributes were left exactly as
# written (Python name-mangles `self.__foo` inside the class body regardless of
# which module the class lives in, so behavior is unchanged), and Python 2-style
# `super(ClassName, self)` calls were left as-is since they still work under
# modern Python 3. No non-base imports are required (`torch`, `torch.nn`,
# `torch.nn.functional`, `torch.nn.utils.rnn` only).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class ModelManager(nn.Module):
    def __init__(self, args, num_word, num_slot, num_intent):
        super(ModelManager, self).__init__()

        self.__num_word = num_word
        self.__num_slot = num_slot
        self.__num_intent = num_intent
        self.__args = args

        # Initialize an embedding object.
        self.__embedding = EmbeddingCollection(self.__num_word, self.__args.word_embedding_dim)

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

        # Initialize an Decoder object for intent.
        self.__intent_decoder = LSTMDecoder(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.intent_decoder_hidden_dim,
            self.__num_intent,
            self.__args.dropout_rate,
            embedding_dim=self.__args.intent_embedding_dim,
        )
        # Initialize an Decoder object for slot.
        self.__slot_decoder = LSTMDecoder(
            self.__args.encoder_hidden_dim + self.__args.attention_output_dim,
            self.__args.slot_decoder_hidden_dim,
            self.__num_slot,
            self.__args.dropout_rate,
            embedding_dim=self.__args.slot_embedding_dim,
            extra_dim=self.__num_intent,
        )

        # One-hot encoding for augment data feed.
        self.__intent_embedding = nn.Embedding(self.__num_intent, self.__num_intent)
        self.__intent_embedding.weight.data = torch.eye(self.__num_intent)
        self.__intent_embedding.weight.requires_grad = False

    def forward(self, text, seq_lens, n_predicts=None, forced_slot=None, forced_intent=None):
        word_tensor, _ = self.__embedding(text)

        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=1)

        pred_intent = self.__intent_decoder(hiddens, seq_lens, forced_input=forced_intent)

        if not self.__args.differentiable:
            _, idx_intent = pred_intent.topk(1, dim=-1)
            feed_intent = self.__intent_embedding(idx_intent.squeeze(1))
        else:
            feed_intent = pred_intent

        pred_slot = self.__slot_decoder(
            hiddens, seq_lens, forced_input=forced_slot, extra_input=feed_intent
        )

        if n_predicts is None:
            return F.log_softmax(pred_slot, dim=1), F.log_softmax(pred_intent, dim=1)
        else:
            _, slot_index = pred_slot.topk(n_predicts, dim=1)
            _, intent_index = pred_intent.topk(n_predicts, dim=1)

            return slot_index.cpu().data.numpy().tolist(), intent_index.cpu().data.numpy().tolist()

    def golden_intent_predict_slot(self, text, seq_lens, golden_intent, n_predicts=1):
        word_tensor, _ = self.__embedding(text)
        embed_intent = self.__intent_embedding(golden_intent)

        lstm_hiddens = self.__encoder(word_tensor, seq_lens)
        attention_hiddens = self.__attention(word_tensor, seq_lens)
        hiddens = torch.cat([attention_hiddens, lstm_hiddens], dim=1)

        pred_slot = self.__slot_decoder(hiddens, seq_lens, extra_input=embed_intent)
        _, slot_index = pred_slot.topk(n_predicts, dim=-1)

        return slot_index.cpu().data.numpy().tolist()


class EmbeddingCollection(nn.Module):
    """Provide word vector encoding."""

    def __init__(self, input_dim, embedding_dim, max_len=5000):
        super(EmbeddingCollection, self).__init__()

        self.__input_dim = input_dim
        self.__embedding_dim = embedding_dim
        self.__max_len = max_len

        self.__embedding_layer = nn.Embedding(self.__input_dim, self.__embedding_dim)

    def forward(self, input_x):
        embedding_x = self.__embedding_layer(input_x)
        return embedding_x, embedding_x


class LSTMEncoder(nn.Module):
    """Encoder structure based on bidirectional LSTM."""

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        self.__embedding_dim = embedding_dim
        self.__hidden_dim = hidden_dim // 2
        self.__dropout_rate = dropout_rate

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
        dropout_text = self.__dropout_layer(embedded_text)

        packed_text = pack_padded_sequence(dropout_text, seq_lens, batch_first=True)
        lstm_hiddens, (h_last, c_last) = self.__lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)

        return torch.cat(
            [padded_hiddens[i][: seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0
        )


class LSTMDecoder(nn.Module):
    """Decoder structure based on unidirectional LSTM."""

    def __init__(
        self, input_dim, hidden_dim, output_dim, dropout_rate, embedding_dim=None, extra_dim=None
    ):
        super(LSTMDecoder, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate
        self.__embedding_dim = embedding_dim
        self.__extra_dim = extra_dim

        if self.__embedding_dim is not None:
            self.__embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.__init_tensor = nn.Parameter(
                torch.randn(1, self.__embedding_dim), requires_grad=True
            )

        if self.__extra_dim is not None and self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim + self.__embedding_dim
        elif self.__extra_dim is not None:
            lstm_input_dim = self.__input_dim + self.__extra_dim
        elif self.__embedding_dim is not None:
            lstm_input_dim = self.__input_dim + self.__embedding_dim
        else:
            lstm_input_dim = self.__input_dim

        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__lstm_layer = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.__hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.__dropout_rate,
            num_layers=1,
        )
        self.__linear_layer = nn.Linear(self.__hidden_dim, self.__output_dim)

    def forward(self, encoded_hiddens, seq_lens, forced_input=None, extra_input=None):
        if extra_input is not None:
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=1)
        else:
            input_tensor = encoded_hiddens

        output_tensor_list, sent_start_pos = [], 0
        if self.__embedding_dim is None or forced_input is not None:
            for sent_i in range(0, len(seq_lens)):
                sent_end_pos = sent_start_pos + seq_lens[sent_i]

                seg_hiddens = input_tensor[sent_start_pos:sent_end_pos, :]

                if self.__embedding_dim is not None and forced_input is not None:
                    if seq_lens[sent_i] > 1:
                        seg_forced_input = forced_input[sent_start_pos:sent_end_pos]
                        seg_forced_tensor = self.__embedding_layer(seg_forced_input).view(
                            seq_lens[sent_i], -1
                        )
                        seg_prev_tensor = torch.cat(
                            [self.__init_tensor, seg_forced_tensor[:-1, :]], dim=0
                        )
                    else:
                        seg_prev_tensor = self.__init_tensor

                    combined_input = torch.cat([seg_hiddens, seg_prev_tensor], dim=1)
                else:
                    combined_input = seg_hiddens
                dropout_input = self.__dropout_layer(combined_input)

                lstm_out, _ = self.__lstm_layer(dropout_input.view(1, seq_lens[sent_i], -1))
                linear_out = self.__linear_layer(lstm_out.view(seq_lens[sent_i], -1))

                output_tensor_list.append(linear_out)
                sent_start_pos = sent_end_pos
        else:
            for sent_i in range(0, len(seq_lens)):
                prev_tensor = self.__init_tensor

                last_h, last_c = None, None

                sent_end_pos = sent_start_pos + seq_lens[sent_i]
                for word_i in range(sent_start_pos, sent_end_pos):
                    seg_input = input_tensor[[word_i], :]
                    combined_input = torch.cat([seg_input, prev_tensor], dim=1)
                    dropout_input = self.__dropout_layer(combined_input).view(1, 1, -1)

                    if last_h is None and last_c is None:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(dropout_input)
                    else:
                        lstm_out, (last_h, last_c) = self.__lstm_layer(
                            dropout_input, (last_h, last_c)
                        )

                    lstm_out = self.__linear_layer(lstm_out.view(1, -1))
                    output_tensor_list.append(lstm_out)

                    _, index = lstm_out.topk(1, dim=1)
                    prev_tensor = self.__embedding_layer(index).view(1, -1)
                sent_start_pos = sent_end_pos

        return torch.cat(output_tensor_list, dim=0)


class QKVAttention(nn.Module):
    """Attention mechanism based on Query-Key-Value architecture."""

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        self.__query_dim = query_dim
        self.__key_dim = key_dim
        self.__value_dim = value_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

        self.__query_layer = nn.Linear(self.__query_dim, self.__hidden_dim)
        self.__key_layer = nn.Linear(self.__key_dim, self.__hidden_dim)
        self.__value_layer = nn.Linear(self.__value_dim, self.__output_dim)
        self.__dropout_layer = nn.Dropout(p=self.__dropout_rate)

    def forward(self, input_query, input_key, input_value):
        linear_query = self.__query_layer(input_query)
        linear_key = self.__key_layer(input_key)
        linear_value = self.__value_layer(input_value)

        score_tensor = F.softmax(
            torch.matmul(linear_query, linear_key.transpose(-2, -1)) / math.sqrt(self.__hidden_dim),
            dim=-1,
        )
        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.__dropout_layer(forced_tensor)

        return forced_tensor


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        self.__input_dim = input_dim
        self.__hidden_dim = hidden_dim
        self.__output_dim = output_dim
        self.__dropout_rate = dropout_rate

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

        flat_x = torch.cat(
            [attention_x[i][: seq_lens[i], :] for i in range(0, len(seq_lens))], dim=0
        )
        return flat_x


MENAGERIE_ZOO = "vendored-pytorch"


class _StackPropArgs:
    """Minimal stand-in for the original repo's argparse Namespace -- same
    field names/values the real ModelManager.__init__ reads, just supplied
    directly instead of via CLI flags."""

    def __init__(self):
        self.word_embedding_dim = 16
        self.encoder_hidden_dim = 16
        self.attention_hidden_dim = 16
        self.attention_output_dim = 16
        self.dropout_rate = 0.0
        self.intent_decoder_hidden_dim = 16
        self.slot_decoder_hidden_dim = 16
        self.intent_embedding_dim = 4
        self.slot_embedding_dim = 4
        self.differentiable = True


def build_stack_propagation_slu():
    args = _StackPropArgs()
    model = ModelManager(args, num_word=64, num_slot=8, num_intent=5)
    model.eval()
    return model


def example_input_stack_propagation_slu():
    # seq_lens must be sorted descending (pack_padded_sequence default enforce_sorted=True).
    seq_lens = [6, 4]
    text = torch.randint(0, 64, (len(seq_lens), max(seq_lens)))
    return (text, seq_lens)


MENAGERIE_ENTRIES = [
    (
        "Stack-Propagation SLU",
        build_stack_propagation_slu,
        example_input_stack_propagation_slu,
        2019,
        "vendored-pytorch",
    ),
]

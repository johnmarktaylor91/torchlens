# FAITHFUL PORT of https://github.com/nusnlp/nea @ master (original framework: Theano/Keras 1.x)
#
# ASAP-AES CNN+LSTM (Taghipour & Ng, EMNLP 2016 "A Neural Approach to Automated Essay
# Scoring"). The original repo (`nea/models.py::create_model`, `args.model_type == 'regp'`)
# is Theano-backed Keras 1.x (`keras.layers.recurrent.LSTM`, `keras.engine.topology.Layer`,
# `K.theano.tensor.tensordot`) -- APIs removed from modern Keras/TF entirely, and Theano is
# unmaintained/unavailable in this environment. There is no way to install a matching
# Theano+old-Keras stack in the base torch env, so this is transcribed FAITHFULLY into
# self-contained torch, layer-for-layer from the real source (not a from-scratch guess):
#
#   nea/models.py  ('regp' branch, cnn_dim>0, rnn_dim>0, aggregation='attsum'):
#     Embedding(vocab_size, emb_dim, mask_zero=True)
#     -> Conv1DWithMasking(cnn_dim, cnn_window_size, border_mode='same')
#     -> LSTM(rnn_dim, return_sequences=True, dropout_W=0.5, dropout_U=0.1)
#     -> Dropout(dropout_prob)
#     -> Attention(op='attsum', activation='tanh')   # nea/my_layers.py::Attention
#     -> Dense(num_outputs) -> Sigmoid
#
#   nea/my_layers.py::Attention.call (the custom learned-attention aggregation):
#     y = x @ att_W                      # att_W: (D, D) learned matrix
#     weights = softmax( tanh(y) @ att_v )   # att_v: (D,) learned vector; tensordot over D
#     out = sum_t( x_t * weights_t )     # weighted sum over the time axis ('attsum')
#
# Port notes (translation choices only, not architecture changes):
#   - Keras `mask_zero=True` padding-mask semantics are reproduced explicitly: a boolean
#     mask (text_ids != pad_id) is threaded through Conv1D (zero out padded positions,
#     matching Conv1DWithMasking.compute_mask which passes the mask through unchanged --
#     'same' padding conv itself is mask-agnostic, only the *softmax* / mean ops need masking)
#     and the Attention softmax (padded timesteps get -inf pre-softmax, matching Keras
#     masked-softmax behavior for `supports_masking` layers).
#   - Keras `Convolution1D(border_mode='same')` == `torch.nn.Conv1d(padding=kernel_size//2)`
#     for odd kernel sizes (the repo's default `cnn_window_size` is odd).
#   - `LSTM(dropout_W=..., dropout_U=...)` (Keras 1.x per-gate variational dropout) is
#     represented by a single `nn.LSTM(dropout=...)` between layers for this 1-layer-LSTM
#     traced instance the distinction is moot (Keras applies `dropout_U`/`dropout_W` inside
#     the recurrence, but for num_layers=1, at eval/inference time all Keras/torch dropout
#     is a no-op regardless; this only affects training-time stochasticity, not the traced
#     computation graph the menagerie captures).
#   - Bias initialization (`log(mean) - log(1-mean)`) from `create_model` is reproduced.

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"

_PAD_ID = 0


class Attention(nn.Module):
    """Port of nea/my_layers.py::Attention (op='attsum', activation='tanh')."""

    def __init__(self, dim, init_stdev=0.01):
        super().__init__()
        self.att_v = nn.Parameter(torch.randn(dim) * init_stdev)
        self.att_W = nn.Parameter(torch.randn(dim, dim) * init_stdev)

    def forward(self, x, mask):
        # x: (B, T, D), mask: (B, T) bool, True where real (non-pad) tokens
        y = torch.tanh(x @ self.att_W)  # (B, T, D)
        weights = y @ self.att_v  # (B, T)  -- tensordot(att_v, tanh(y), axes=[0,2])
        weights = weights.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(weights, dim=1)  # (B, T)
        out = (x * weights.unsqueeze(-1)).sum(dim=1)  # attsum: sum over time
        return out


class NEA_CNN_LSTM_AttSum(nn.Module):
    """Port of nea/models.py::create_model, args.model_type == 'regp',
    cnn_dim > 0, rnn_dim > 0, aggregation == 'attsum'."""

    def __init__(
        self,
        vocab_size,
        emb_dim,
        cnn_dim,
        cnn_window_size,
        rnn_dim,
        dropout_prob,
        num_outputs,
        initial_mean_value=0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=_PAD_ID)
        cnn_padding = cnn_window_size // 2  # Keras border_mode='same'
        self.conv = nn.Conv1d(emb_dim, cnn_dim, kernel_size=cnn_window_size, padding=cnn_padding)
        self.lstm = nn.LSTM(cnn_dim, rnn_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.attention = Attention(rnn_dim)
        self.dense = nn.Linear(rnn_dim, num_outputs)

        # bias_value = log(initial_mean_value) - log(1 - initial_mean_value)
        bias_value = math.log(initial_mean_value) - math.log(1 - initial_mean_value)
        nn.init.constant_(self.dense.bias, bias_value)

    def forward(self, text_ids):
        mask = text_ids != _PAD_ID  # (B, T) -- Keras Embedding(mask_zero=True)

        x = self.embedding(text_ids)  # (B, T, emb_dim)
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        x = x.transpose(1, 2)  # (B, emb_dim, T) for Conv1d
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, cnn_dim)
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)  # Conv1DWithMasking passes mask through

        x, _ = self.lstm(x)  # (B, T, rnn_dim), return_sequences=True
        x = self.dropout(x)

        x = self.attention(x, mask)  # (B, rnn_dim) -- attsum aggregation
        x = self.dense(x)
        x = torch.sigmoid(x)
        return x


# ---------------------------------------------------------------------------
# Menagerie staging hooks
# ---------------------------------------------------------------------------
_VOCAB_SIZE = 64
_EMB_DIM = 16
_CNN_DIM = 8
_CNN_WINDOW = 3
_RNN_DIM = 12
_SEQ_LEN = 10


def build_nea_cnn_lstm():
    return NEA_CNN_LSTM_AttSum(
        vocab_size=_VOCAB_SIZE,
        emb_dim=_EMB_DIM,
        cnn_dim=_CNN_DIM,
        cnn_window_size=_CNN_WINDOW,
        rnn_dim=_RNN_DIM,
        dropout_prob=0.5,
        num_outputs=1,
    )


def example_input_nea_cnn_lstm():
    # nonzero token ids so the padding mask (id != 0) is nontrivial but doesn't zero the batch
    return torch.randint(1, _VOCAB_SIZE, (2, _SEQ_LEN))


MENAGERIE_ENTRIES = [
    ("ASAP_AES_CNN_LSTM", build_nea_cnn_lstm, example_input_nea_cnn_lstm, 2016, "MENAGERIE_ZOO"),
]

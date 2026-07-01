# FAITHFUL PORT of usadellab/Helixer @ d17bb496b8842542590644b2437798711c0125d5
# (helixer/prediction/HybridModel.py::HybridModel.model / model_hat, original framework: TensorFlow/Keras)
"""Helixer HybridModel: the hybrid CNN-BiLSTM gene-structure predictor used by
the Helixer eukaryotic gene-annotation pipeline (Stiehler et al. 2020,
Bioinformatics; https://github.com/usadellab/Helixer). Given a one-hot-encoded
DNA sequence window, a stack of 1D convolutions extracts local sequence motifs,
consecutive conv-output timesteps are folded together ("pool" via reshape, not
real pooling), a bidirectional LSTM stack models long-range dependencies over
the folded sequence, and a small dense "hat" head predicts, per output
position, a softmax distribution over 4 genic classes (main) and, in the
actually-deployed configuration (``predict_phase=True``, hardcoded in
``HelixerModel.parse_args``), a second softmax distribution over 4
reading-frame-phase classes.

The real ``HybridModel`` is TensorFlow/Keras (``tf.keras.layers.Conv1D`` /
``Bidirectional(LSTM(...))`` / ``tf.split`` / ``tf.concat``) and its
``model()``/``model_hat()`` methods live on a class whose ``__init__`` chain
(``HelixerModel`` -> argparse CLI parsing, HDF5 dataset iteration) is not
runnable outside the full Helixer package + TensorFlow, neither of which is
installed in this environment. This module is a faithful line-by-line PORT of
``HybridModel.model()``/``model_hat()`` (the actual architecture-construction
code, read directly from the real repo) into self-contained PyTorch, with
every layer/mechanism preserved:

  * ``Conv1D(filters, kernel_size, padding="same", activation="relu")`` over
    ``main_input`` -> ``nn.Conv1d(in, filter_depth, kernel_size,
    padding=kernel_size // 2) + ReLU`` (channels_last -> channels_first
    layout; Keras "same" padding for odd kernel sizes == symmetric
    ``kernel_size // 2`` padding in torch, matching this model's default
    ``kernel_size=26`` -> even kernel, Keras pads asymmetrically (13 left, 12
    right); torch's symmetric padding is architecturally equivalent up to
    that one-sample edge-alignment nuance, called out explicitly here rather
    than silently glossed over).
  * Extra CNN layers (``cnn_layers - 1``): ``BatchNorm1d`` -> ``Conv1d`` +
    ``ReLU``, matching the unconditional-BN-before-conv order in the real
    loop.
  * ``pool_size`` reshape-fold: ``Reshape((-1, pool_size * filter_depth))``
    groups ``pool_size`` consecutive conv-output timesteps into one wider
    feature vector along the channel axis (this is NOT real pooling --
    Helixer's own code comments out the ``MaxPooling1D`` alternative).
  * Optional ``dropout1`` before the LSTM stack.
  * ``Bidirectional(LSTM(units, return_sequences=True))`` x ``lstm_layers``
    -> ``nn.LSTM(units, bidirectional=True, batch_first=True)`` stacked
    ``lstm_layers`` times (kept as separate layers, matching the real
    per-iteration ``Bidirectional(LSTM(...))`` construction rather than
    torch's fused multi-layer LSTM, since each Keras call re-instantiates a
    fresh bidirectional layer).
  * Optional ``dropout2`` after the LSTM stack.
  * ``model_hat``: optional coverage-concat + dense (only used when
    ``input_coverage=True``, off by default and not wired into this staging
    module's single-input example), then (since ``predict_phase=True`` is
    hardcoded by ``HelixerModel.parse_args``) a single ``Dense(pool_size * 4
    * 2)`` split in half into genic/phase logits, each reshaped to
    ``(..., pool_size, 4)`` and softmaxed independently.

Default hyperparameters (``cnn_layers=1``, ``lstm_layers=1``, ``units=32``,
``filter_depth=32``, ``kernel_size=26``, ``pool_size=9``, ``dropout1=0.0``,
``dropout2=0.0``) are HybridModel's real argparse defaults; ``input_coverage``
defaults to ``False`` (an ``action='store_true'`` flag) and
``predict_phase=True`` is force-set in ``HelixerModel.parse_args`` regardless
of CLI args, so both are baked in below rather than left as knobs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class HybridModel(nn.Module):
    """Faithful port of Helixer's HybridModel.model()/model_hat() (predict_phase
    branch, no coverage input -- the actually-deployed configuration)."""

    def __init__(
        self,
        cnn_layers: int = 1,
        lstm_layers: int = 1,
        units: int = 32,
        filter_depth: int = 32,
        kernel_size: int = 26,
        pool_size: int = 9,
        dropout1: float = 0.0,
        dropout2: float = 0.0,
        values_per_bp: int = 4,
    ):
        super().__init__()
        self.cnn_layers = cnn_layers
        self.lstm_layers = lstm_layers
        self.units = units
        self.filter_depth = filter_depth
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout1_p = dropout1
        self.dropout2_p = dropout2

        # First Conv1D(filters=filter_depth, kernel_size, padding="same", activation="relu")
        self.conv0 = nn.Conv1d(values_per_bp, filter_depth, kernel_size, padding=kernel_size // 2)

        # Extra CNN layers: BatchNormalization() -> Conv1D(..., activation="relu")
        self.extra_bns = nn.ModuleList()
        self.extra_convs = nn.ModuleList()
        for _ in range(cnn_layers - 1):
            self.extra_bns.append(nn.BatchNorm1d(filter_depth))
            self.extra_convs.append(
                nn.Conv1d(filter_depth, filter_depth, kernel_size, padding=kernel_size // 2)
            )

        if dropout1 > 0.0:
            self.dropout1 = nn.Dropout(dropout1)
        else:
            self.dropout1 = None

        # Bidirectional(LSTM(units, return_sequences=True)) x lstm_layers.
        # First layer's input width is pool_size * filter_depth after the
        # reshape-fold (or filter_depth if pool_size <= 1); subsequent layers
        # consume the bidirectional output width (2 * units).
        lstm_in = pool_size * filter_depth if pool_size > 1 else filter_depth
        self.lstms = nn.ModuleList()
        for i in range(lstm_layers):
            in_dim = lstm_in if i == 0 else 2 * units
            self.lstms.append(nn.LSTM(in_dim, units, batch_first=True, bidirectional=True))

        if dropout2 > 0.0:
            self.dropout2 = nn.Dropout(dropout2)
        else:
            self.dropout2 = None

        # model_hat, predict_phase=True branch: Dense(pool_size * 4 * 2), then
        # split into genic/phase halves.
        self.hat_dense = nn.Linear(2 * units, pool_size * 4 * 2)

    def forward(self, main_input: torch.Tensor):
        # main_input: (batch, seq_len, values_per_bp) -- Keras channels_last
        # layout, matched here so callers pass tensors in the same shape as
        # the real model's Input(shape=(None, values_per_bp)).
        x = main_input.transpose(1, 2)  # -> (batch, values_per_bp, seq_len)
        x = F.relu(self.conv0(x))

        for bn, conv in zip(self.extra_bns, self.extra_convs):
            x = bn(x)
            x = F.relu(conv(x))

        x = x.transpose(1, 2)  # -> (batch, seq_len, filter_depth), channels_last again

        if self.pool_size > 1:
            batch, seq_len, filt = x.shape
            folded_len = seq_len // self.pool_size
            x = x[:, : folded_len * self.pool_size, :]
            x = x.reshape(batch, folded_len, self.pool_size * filt)

        if self.dropout1 is not None:
            x = self.dropout1(x)

        for lstm in self.lstms:
            x, _ = lstm(x)

        if self.dropout2 is not None:
            x = self.dropout2(x)

        return self.model_hat(x)

    def model_hat(self, x: torch.Tensor):
        # predict_phase=True branch (no input_coverage): Dense(pool_size*4*2)
        # split into genic/phase halves, each reshaped + softmaxed.
        x = self.hat_dense(x)
        x_genic, x_phase = torch.chunk(x, 2, dim=-1)

        batch, folded_len, _ = x_genic.shape
        x_genic = x_genic.reshape(batch, folded_len, self.pool_size, 4)
        x_genic = F.softmax(x_genic, dim=-1)

        x_phase = x_phase.reshape(batch, folded_len, self.pool_size, 4)
        x_phase = F.softmax(x_phase, dim=-1)

        return [x_genic, x_phase]


# --- staging harness -------------------------------------------------------


def build_helixer():
    # Real HybridModel argparse defaults: cnn_layers=1, lstm_layers=1,
    # units=32, filter_depth=32, kernel_size=26, pool_size=9, dropout1=0.0,
    # dropout2=0.0; predict_phase=True is hardcoded in HelixerModel.parse_args.
    return HybridModel(
        cnn_layers=1,
        lstm_layers=1,
        units=32,
        filter_depth=32,
        kernel_size=26,
        pool_size=9,
        dropout1=0.0,
        dropout2=0.0,
    ).eval()


def example_input_helixer():
    # One-hot DNA-window batch: (batch, seq_len, 4) matching HybridModel's
    # Input(shape=(None, 4), name='main_input'); seq_len chosen as a multiple
    # of pool_size (9) so the reshape-fold divides evenly.
    batch = 2
    seq_len = 9 * 20  # 180bp window
    return (torch.rand(batch, seq_len, 4),)


MENAGERIE_ENTRIES = [
    ("Helixer_HybridModel", "build_helixer", "example_input_helixer", 2020, "ported"),
]

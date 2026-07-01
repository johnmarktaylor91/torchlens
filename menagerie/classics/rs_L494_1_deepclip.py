# FAITHFUL PORT of deepclip/deepclip @ master (original framework: Theano/Lasagne)
#
# Source files transcribed:
#   - network.py (Network.build_model, setup_conv_layers)
#   - oned_convlayer_rectify.py (setup_conv_layers)
#   - slice_n_pad.py (shape_convolutions2)
#   - custom_layers.py (Sum_ax1, Sum_last_ax)
#   - FnB_LSTM.py (FnB_LSTMtan_N)
#
# DeepCLIP (Gronning et al., Nucleic Acids Research 2020) is a CNN-BiLSTM network
# for predicting RNA-binding-protein binding preferences from CLIP-seq data. The
# original implementation is Theano + Lasagne (both long deprecated / not
# reasonably installable alongside a modern torch stack), so this module is a
# faithful architectural transcription into self-contained torch, preserving
# every mechanism of the original graph:
#
#   1. For each filter size in FILTER_SIZES: a 1D "one-hot stride" convolution
#      (kernel = filter_size over the flattened one-hot sequence, stride =
#      len(VOCAB), i.e. one conv window per nucleotide position) with ReLU,
#      followed by a max-over-filters pooling (FeaturePoolLayer/pool_function=max)
#      and a winner-take-all pooling across the position axis
#      (FeatureWTALayer), producing per-filter-size max/argmax "attribution"
#      pools that get broadcast-multiplied back onto a copy of the one-hot
#      input to produce a masked positional attribution ("done"/"done2").
#   2. All per-filter-size attributions are summed over the position axis
#      (Sum_ax1) and concatenated with the reshaped raw one-hot input to form
#      the LSTM input sequence.
#   3. A forward and a backward LSTM (custom gate biases per FnB_LSTMtan_N,
#      tanh cell nonlinearity, sigmoid gates) run over that sequence and are
#      concatenated ("l_sumz").
#   4. The concatenated LSTM output is summed over the last axis, broadcast
#      back onto the one-hot input again (a second masking / attribution
#      pass), summed over the vocab axis, and passed through dropout to
#      produce the "profile" (a per-nucleotide-position score vector).
#   5. A final single dense layer (no bias, sigmoid) maps the profile to the
#      scalar binding-probability output.
#
# All architectural constants below (VOCAB=4, FILTER_SIZES, MINI_BATCH_SIZE,
# NUM_FILTERS, N_LSTM, dropout rates) come directly from constants.py (the
# original repo's shipped defaults).

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 4  # constants.VOCAB = ['a', 'c', 'g', 'tu']
FILTER_SIZES = [4, 5, 6, 7, 8]  # constants.FILTER_SIZES
NUM_FILTERS = 1  # constants.NUM_FILTERS
N_LSTM = 10  # constants.LSTM_NODES
DROPOUT_LSTM = 0.1  # constants.LSTM_DROPOUT
DROPOUT_OUT = 0.0  # constants.DROPOUT_OUT


class _FilterSizeBranch(nn.Module):
    """One per-filter-size branch of `setup_conv_layers`.

    Faithful transcription of the Lasagne Conv1DLayer (stride=len(VOCAB),
    rectify nonlinearity, no bias, constant weight init 0.01) followed by a
    max-over-filters FeaturePoolLayer and a winner-take-all pool over the
    output-position axis, then broadcast back onto the raw one-hot input via
    an elementwise multiply ("done"/"done2" in `setup_conv_layers`).
    """

    def __init__(self, filter_size: int, seq_size: int, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.filter_size = filter_size
        self.seq_size = seq_size
        self.vocab_size = vocab_size
        # Lasagne Conv1DLayer with stride=len(VOCAB) over the flattened
        # (1, VOCAB*SEQ_SIZE) one-hot input == a Conv1d with kernel_size =
        # filter_size, stride = vocab_size, no bias.
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=NUM_FILTERS,
            kernel_size=filter_size,
            stride=vocab_size,
            bias=False,
        )
        nn.init.constant_(self.conv.weight, 0.01)
        # number of output conv positions ("FS[i]" in the original)
        self.n_positions = (seq_size * vocab_size - filter_size) // vocab_size + 1

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        # x_flat: (batch, 1, VOCAB*SEQ_SIZE)
        conv_out = F.relu(self.conv(x_flat))  # (batch, NUM_FILTERS, n_positions)
        # FeaturePoolLayer(pool_function=max) over the filter axis (NUM_FILTERS=1
        # so this is a no-op reduction that keeps the same tensor shape here).
        pooled_max = conv_out.amax(dim=1, keepdim=True)  # (batch, 1, n_positions)
        # FeatureWTALayer: winner-take-all over the position axis -- zero out
        # everything except the position(s) achieving the per-filter maximum.
        wta = torch.zeros_like(pooled_max)
        max_val, max_idx = pooled_max.max(dim=2, keepdim=True)
        wta.scatter_(2, max_idx, max_val)
        # max = ElemwiseMerge(max, wta, add) ; max = ElemwiseMerge(max, max, mul)
        merged = pooled_max + wta
        merged = merged * merged

        # shape_convolutions2: broadcast each of the `filter_size` positions of
        # the pooled response back onto the SEQ_SIZE*VOCAB one-hot input,
        # producing one masked copy of the input per conv output position,
        # then multiply elementwise with the raw one-hot input and sum.
        batch = x_flat.shape[0]
        merged_rep = merged.expand(
            batch, self.filter_size, self.n_positions
        )  # (B, filter_size, n_positions)
        merged_rep = merged_rep.permute(0, 2, 1).reshape(
            batch, 1, self.n_positions * self.filter_size
        )

        pad_total = self.seq_size * self.vocab_size
        mini_seqs = []
        for i in range(self.n_positions):
            start = i * self.filter_size
            sl = merged_rep[:, :, start : start + self.filter_size]
            left_pad = i * self.vocab_size
            right_pad = pad_total - left_pad - self.filter_size
            padded = F.pad(sl, (left_pad, right_pad))
            mini_seqs.append(padded)
        pad = torch.cat(mini_seqs, dim=1)  # (B, n_positions, VOCAB*SEQ_SIZE)

        temp = pad.sum(dim=1, keepdim=True)  # Sum_ax1 -> (B, 1, VOCAB*SEQ_SIZE)
        temp = temp * x_flat  # elementwise mask with raw one-hot input

        done2 = temp.reshape(batch, self.seq_size, self.vocab_size)
        return done2


class DeepCLIP(nn.Module):
    """CNN-BiLSTM network for RNA-binding-protein binding-preference prediction.

    Faithful port of `network.Network.build_model` (deepclip/deepclip). Input
    is a one-hot-encoded RNA/DNA sequence of shape (batch, seq_size, VOCAB).
    """

    def __init__(
        self,
        seq_size: int = 50,
        vocab_size: int = VOCAB_SIZE,
        filter_sizes: list[int] | None = None,
        n_lstm: int = N_LSTM,
        dropout_lstm: float = DROPOUT_LSTM,
        dropout_out: float = DROPOUT_OUT,
    ):
        super().__init__()
        self.seq_size = seq_size
        self.vocab_size = vocab_size
        self.filter_sizes = filter_sizes if filter_sizes is not None else list(FILTER_SIZES)
        self.n_lstm = n_lstm

        self.branches = nn.ModuleList(
            [_FilterSizeBranch(fs, seq_size, vocab_size) for fs in self.filter_sizes]
        )

        lstm_in_dim = (
            len(self.filter_sizes) * vocab_size + vocab_size
        )  # cn_layers concat + raw input concat
        # Forward and backward LSTMs (FnB_LSTMtan_N): separate uni-directional
        # LSTM layers concatenated afterwards, matching the original's two
        # independently-parameterised LSTMLayer calls (not a single
        # bidirectional layer).
        self.lstm_forward = nn.LSTM(input_size=lstm_in_dim, hidden_size=n_lstm, batch_first=True)
        self.lstm_backward = nn.LSTM(input_size=lstm_in_dim, hidden_size=n_lstm, batch_first=True)
        self.dropout_lstm_f = nn.Dropout(dropout_lstm)
        self.dropout_lstm_b = nn.Dropout(dropout_lstm)
        self.dropout_out = nn.Dropout(dropout_out)

        # Final DenseLayer(num_units=1, nonlinearity=sigmoid, W=Constant(1.0), b=None)
        self.output = nn.Linear(seq_size, 1, bias=False)
        nn.init.constant_(self.output.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_size, vocab_size) one-hot encoded sequence
        batch = x.shape[0]
        x_flat = x.reshape(batch, 1, self.seq_size * self.vocab_size)

        branch_outputs = [branch(x_flat) for branch in self.branches]
        cn_layers = torch.cat(
            branch_outputs, dim=2
        )  # concat along vocab axis -> (B, seq_size, n_filters*vocab)

        raw_reshaped = x  # network['inp']: reshape of l_in back to (B, seq_size, vocab)
        lstm_in = torch.cat([cn_layers, raw_reshaped], dim=2)  # l_lstmin

        fwd_out, _ = self.lstm_forward(lstm_in)
        fwd_out = self.dropout_lstm_f(fwd_out)

        lstm_in_rev = torch.flip(lstm_in, dims=[1])
        bwd_out, _ = self.lstm_backward(lstm_in_rev)
        bwd_out = torch.flip(bwd_out, dims=[1])
        bwd_out = self.dropout_lstm_b(bwd_out)

        l_sumz = torch.cat([fwd_out, bwd_out], dim=2)  # (B, seq_size, 2*n_lstm)

        l_sumz2x = l_sumz.sum(dim=2, keepdim=True)  # Sum_last_ax -> (B, seq_size, 1)
        l_sumz2x = l_sumz2x.expand(
            batch, self.seq_size, self.vocab_size
        )  # broadcast to vocab axis via repeat+reshape
        l_sumz2x = l_sumz2x * x  # ElemwiseMerge(l_in, l_sumz2x, mul)

        l_profile = l_sumz2x.sum(dim=2)  # Sum_last_ax -> (B, seq_size)
        l_profile = self.dropout_out(l_profile)

        out = torch.sigmoid(self.output(l_profile))  # l_out
        return out


def build_deepclip() -> DeepCLIP:
    return DeepCLIP(seq_size=50, vocab_size=VOCAB_SIZE)


def example_input_deepclip() -> torch.Tensor:
    torch.manual_seed(0)
    batch = 2
    seq_size = 50
    idx = torch.randint(0, VOCAB_SIZE, (batch, seq_size))
    onehot = F.one_hot(idx, num_classes=VOCAB_SIZE).float()
    return onehot


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepCLIP", "build_deepclip", "example_input_deepclip", 2020, "ported-pytorch"),
]

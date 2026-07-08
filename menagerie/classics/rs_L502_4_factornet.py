# FAITHFUL PORT of uci-cbcl/FactorNet @ master (utils.py `make_meta_model`)
# (original framework: keras 1.2.2 / Theano)
#
# FactorNet (Quang & Xie, Methods 2019) predicts cell-type-specific
# transcription-factor (TF) binding from DNA sequence + strand-specific
# DNase-seq/bigWig accessibility tracks + per-cell-type metadata (e.g. gene
# expression PCA features), and was a top performer in the ENCODE-DREAM TF
# binding challenge. The real network (`utils.make_meta_model`, confirmed
# against a real trained model.json export: shared Convolution1D -> Dropout
# -> TimeDistributed(Dense) -> MaxPooling1D -> Bidirectional(LSTM) ->
# Dropout -> Flatten -> Dense stack) is applied identically ("weight-tied",
# via Keras's shared-layer-object `get_output(input, hidden_layers)`
# pattern) to the forward-strand and reverse-complement-strand sequence+
# accessibility windows; each branch's pooled features are concatenated
# with a shared metadata vector, passed through a second (also
# weight-tied-across-strands) Dropout -> Dense -> Dropout -> Dense(sigmoid)
# head, and the two strands' sigmoid outputs are averaged
# (`merge([forward_output, reverse_output], mode='ave')`) into the final
# per-TF binding-probability prediction. This is a faithful torch
# transcription of that exact layer graph (every layer/kernel-width/
# pool-width/hidden-width below matches the real Keras 1.2.2 functional
# model, including the weight-sharing between the two strand branches);
# only the Keras (channels-last, NLC) -> torch (channels-first, NCL) Conv1d
# axis convention and the CLI/data-loading/training glue are new.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"

# real train.py CLI defaults (utils.py `L`/`w`/`w2` globals set from these):
SEQ_LEN = 1000  # -L / --seqlen default
MOTIF_WIDTH = 26  # -w / --motifwidth default
POOL_WIDTH = MOTIF_WIDTH // 2  # utils.w2 = w / 2
NUM_MOTIFS = 32  # -k / --kernels default
NUM_RECURRENT = 32  # -r / --recurrent default
NUM_DENSE = 64  # -d / --dense default
DROPOUT_RATE = 0.5  # -p / --dropout default
NUM_TRACKS = 6  # real: 4 one-hot bases + num_bws strand-specific bigwig tracks (e.g. DNase)
NUM_TFS = 1  # real: number of TFs predicted (multi-task models predict several at once)
NUM_META = 14  # real: per-cell-type metadata feature count (e.g. RNA-seq PCA dims)


class FactorNetBranch(nn.Module):
    """The weight-tied per-strand feature extractor.

    Keras: Convolution1D(nb_filter=NUM_MOTIFS, filter_length=MOTIF_WIDTH,
    border_mode='valid', activation='relu') -> Dropout(0.1) ->
    TimeDistributed(Dense(NUM_MOTIFS, activation='relu')) ->
    MaxPooling1D(pool_length=POOL_WIDTH, stride=POOL_WIDTH) ->
    Bidirectional(LSTM(NUM_RECURRENT, return_sequences=True)) ->
    Dropout(dropout_rate) -> Flatten() -> Dense(NUM_DENSE, activation='relu').
    The SAME module instance is applied to both the forward-strand and
    reverse-complement-strand inputs (Keras shared-layer weight tying).
    """

    def __init__(
        self,
        num_tracks,
        motif_width,
        num_motifs,
        pool_width,
        num_recurrent,
        num_dense,
        dropout_rate,
        seq_len,
    ):
        super().__init__()
        self.conv = nn.Conv1d(num_tracks, num_motifs, kernel_size=motif_width)
        self.conv_act = nn.ReLU()
        self.conv_drop = nn.Dropout(0.1)
        # keras TimeDistributed(Dense(num_motifs)) applied per conv position == a
        # position-wise Linear in torch (equivalent to a 1x1 conv over channels).
        self.timedist_dense = nn.Linear(num_motifs, num_motifs)
        self.timedist_act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_width, stride=pool_width)
        self.lstm = nn.LSTM(num_motifs, num_recurrent, batch_first=True, bidirectional=True)
        self.lstm_drop = nn.Dropout(dropout_rate)

        conv_out_len = seq_len - motif_width + 1
        pooled_len = conv_out_len // pool_width
        self.flatten_dim = pooled_len * (2 * num_recurrent)
        self.dense = nn.Linear(self.flatten_dim, num_dense)
        self.dense_act = nn.ReLU()

    def forward(self, x):
        # x arrives channels-first (N, C, L); real Keras input is channels-last
        # (N, L, C) -- Convolution1D there == torch Conv1d here over dim=1.
        x = self.conv_act(self.conv(x))
        x = self.conv_drop(x)
        # position-wise dense: (N, C, L) -> (N, L, C) -> dense -> (N, L, C) -> (N, C, L)
        x = x.transpose(1, 2)
        x = self.timedist_act(self.timedist_dense(x))
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)  # LSTM expects (N, L, C) with batch_first=True
        x, _ = self.lstm(x)
        x = self.lstm_drop(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense_act(self.dense(x))
        return x


class FactorNet(nn.Module):
    """Faithful port of `utils.make_meta_model`: two weight-tied strand
    branches, each fused with a shared metadata vector and pushed through a
    second (also weight-tied across strands) dense head, then the two
    strands' sigmoid outputs are averaged.
    """

    def __init__(
        self,
        num_tracks=NUM_TRACKS,
        num_tfs=NUM_TFS,
        num_meta=NUM_META,
        motif_width=MOTIF_WIDTH,
        num_motifs=NUM_MOTIFS,
        pool_width=POOL_WIDTH,
        num_recurrent=NUM_RECURRENT,
        num_dense=NUM_DENSE,
        dropout_rate=DROPOUT_RATE,
        seq_len=SEQ_LEN,
    ):
        super().__init__()
        # keras `hidden_layers` list of shared layer objects, reused for both strands.
        self.branch = FactorNetBranch(
            num_tracks,
            motif_width,
            num_motifs,
            pool_width,
            num_recurrent,
            num_dense,
            dropout_rate,
            seq_len,
        )

        # keras: shared dropout2_layer / dense2_layer / dropout3_layer / sigmoid_layer,
        # each applied identically to the forward and reverse branch outputs.
        self.dropout2 = nn.Dropout(0.1)
        self.dense2 = nn.Linear(num_dense + num_meta, num_dense)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.sigmoid_layer = nn.Linear(num_dense, num_tfs)
        self.sigmoid_act = nn.Sigmoid()

    def _head(self, branch_features, meta):
        x = torch.cat([branch_features, meta], dim=-1)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.dropout3(x)
        x = self.sigmoid_layer(x)
        return self.sigmoid_act(x)

    def forward(self, forward_seq, reverse_seq, meta):
        forward_features = self.branch(forward_seq)
        reverse_features = self.branch(reverse_seq)
        forward_out = self._head(forward_features, meta)
        reverse_out = self._head(reverse_features, meta)
        # keras: merge([forward_output, reverse_output], mode='ave')
        return (forward_out + reverse_out) / 2


# ---------------------------------------------------------------------------
# menagerie staging entry point
# ---------------------------------------------------------------------------
# Use a tiny sequence length while keeping the real motif_width/pool_width
# ratio so the conv+pool stack yields a small positive pooled length.
_TINY_SEQ_LEN = 100  # -> conv_out_len = 100-26+1 = 75, pooled_len = 75 // 13 = 5


def build_factornet():
    return FactorNet(seq_len=_TINY_SEQ_LEN)


def example_input_factornet():
    forward_seq = torch.randn(2, NUM_TRACKS, _TINY_SEQ_LEN)
    reverse_seq = torch.randn(2, NUM_TRACKS, _TINY_SEQ_LEN)
    meta = torch.randn(2, NUM_META)
    return (forward_seq, reverse_seq, meta)


MENAGERIE_ENTRIES = [
    ("FactorNet", build_factornet, example_input_factornet, 2019, "SOURCE_AVAILABLE"),
]

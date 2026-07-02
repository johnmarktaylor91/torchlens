# FAITHFUL PORT of muhaochen/seq_ppi @ 9172fbdc9a9bd9c0c1234bde3314ae0b6bff8dd0
# (original framework: TensorFlow 1.x / Keras with CuDNNGRU)
#
# PIPR (Chen et al., Bioinformatics 2019, "Multifaceted protein-protein interaction
# prediction based on Siamese residual RCNN") -- from
# binary/model/lasagna/rcnn.py::build_model(). The real repo is a training script (not a
# packaged nn.Module class) written against TensorFlow 1.x Keras with `CuDNNGRU`, an
# API that no longer exists in any TF/Keras version compatible with this environment
# (CuDNNGRU was removed after TF1.x/legacy multi-backend Keras; the base env only has
# torch, not any TensorFlow build at all). The dependency genuinely cannot be reasonably
# installed here, so the architecture from `build_model()` is transcribed faithfully into
# a self-contained torch `nn.Module`, preserving every layer and the exact residual-RCNN
# topology of the original: a Siamese (shared-weight) branch applied independently to two
# protein sequence-embedding tensors, each branch = 5 stacked blocks of
# (Conv1D -> BiGRU -> concat[BiGRU_out, conv_out] -> MaxPool1D), followed by one more
# Conv1D and GlobalAveragePooling1D; the two branch pooled vectors are combined via
# elementwise multiply (`keras.layers.multiply`), then passed through a small MLP head
# (Dense(100)+LeakyReLU(0.3) -> Dense(hidden) LeakyReLU(0.3) -> Dense(2) softmax) matching
# `build_model()`'s `main_output` exactly. Padding is set to `padding="same"` for the
# Conv1D layers to match Keras `Conv1D`'s default (`padding='valid'` in Keras defaults to
# 'valid', matching torch's default no-padding Conv1d -- so PyTorch Conv1d default
# (padding=0) already matches Keras's build_model() call, which passes no explicit
# `padding=` kwarg and therefore uses Keras's own default of 'valid').
#
# MENAGERIE_ZOO = "ported-pytorch"

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class PIPRBranch(nn.Module):
    """One Siamese-shared RCNN branch: 5x (Conv1D -> BiGRU -> concat -> MaxPool1D)
    followed by a final Conv1D + GlobalAveragePooling1D, matching build_model()'s
    per-sequence stack (l1..l6, r1..r5) applied to seq_input1 / seq_input2."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # l1..l6: Conv1D(hidden_dim, kernel_size=3), Keras default padding='valid'.
        # After each `concatenate([r_i(s), s])`, channel width is
        # 2*hidden_dim (bidirectional GRU) + hidden_dim (conv-pool output) = 3*hidden_dim.
        self.l1 = nn.Conv1d(in_dim, hidden_dim, kernel_size=3)
        self.l2 = nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=3)
        self.l3 = nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=3)
        self.l4 = nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=3)
        self.l5 = nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=3)
        self.l6 = nn.Conv1d(3 * hidden_dim, hidden_dim, kernel_size=3)

        # r1..r5: Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
        self.r1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.r2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.r3 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.r4 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.r5 = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.pool = nn.MaxPool1d(3)

    def _conv(self, conv, x_bcn):
        # x_bcn: [B, C, N] (torch Conv1d layout)
        return conv(x_bcn)

    def forward(self, seq_input):
        # seq_input: [B, N, C] (Keras layout: batch, seq_len, dim)
        x_bcn = seq_input.transpose(1, 2)  # [B, C, N]

        s = self.pool(self._conv(self.l1, x_bcn))  # MaxPooling1D(3)(l1(seq_input))
        s_bnc = s.transpose(1, 2)
        r_out, _ = self.r1(s_bnc)
        s = torch.cat([r_out, s_bnc], dim=-1).transpose(1, 2)  # concat -> [B, C, N]

        s = self.pool(self._conv(self.l2, s))
        s_bnc = s.transpose(1, 2)
        r_out, _ = self.r2(s_bnc)
        s = torch.cat([r_out, s_bnc], dim=-1).transpose(1, 2)

        s = self.pool(self._conv(self.l3, s))
        s_bnc = s.transpose(1, 2)
        r_out, _ = self.r3(s_bnc)
        s = torch.cat([r_out, s_bnc], dim=-1).transpose(1, 2)

        s = self.pool(self._conv(self.l4, s))
        s_bnc = s.transpose(1, 2)
        r_out, _ = self.r4(s_bnc)
        s = torch.cat([r_out, s_bnc], dim=-1).transpose(1, 2)

        s = self.pool(self._conv(self.l5, s))
        s_bnc = s.transpose(1, 2)
        r_out, _ = self.r5(s_bnc)
        s = torch.cat([r_out, s_bnc], dim=-1).transpose(1, 2)

        s = self._conv(self.l6, s)  # s1 = l6(s1)
        s = s.mean(dim=-1)  # GlobalAveragePooling1D()
        return s


class PIPR(nn.Module):
    """Siamese residual RCNN for protein-protein interaction prediction from paired
    protein sequence embeddings. Faithful port of build_model() in
    binary/model/lasagna/rcnn.py."""

    def __init__(self, in_dim, hidden_dim=25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.branch = PIPRBranch(in_dim, hidden_dim)  # weight-shared Siamese branch

        self.dense1 = nn.Linear(hidden_dim, 100)
        self.leaky1 = nn.LeakyReLU(0.3)
        self.dense2 = nn.Linear(100, int((hidden_dim + 7) / 2))
        self.leaky2 = nn.LeakyReLU(0.3)
        self.dense3 = nn.Linear(int((hidden_dim + 7) / 2), 2)

    def forward(self, seq_input1, seq_input2):
        s1 = self.branch(seq_input1)
        s2 = self.branch(seq_input2)  # same weights -- Siamese
        merge_text = s1 * s2  # keras.layers.multiply([s1, s2])
        x = self.leaky1(self.dense1(merge_text))
        x = self.leaky2(self.dense2(x))
        main_output = F.softmax(self.dense3(x), dim=-1)
        return main_output


def build_pipr():
    # Real repo default: hidden_dim=25 (embeddings/*.txt use dim=5..7 per default_onehot /
    # string_vec5 / CTCoding_onehot / vec5_CTC; we use dim=7 to match CTCoding_onehot's
    # feature width, a real embedding-file option `use_emb=2` in the original script).
    return PIPR(in_dim=7, hidden_dim=25)


def example_input_pipr():
    # seq_size=2000 matches the real script's default protein-sequence padding length
    # (`seq_size = 2000` in rcnn.py); 5 rounds of Conv1D(k=3,valid)+MaxPool1D(3) need a
    # long enough sequence to avoid collapsing to zero length.
    torch.manual_seed(0)
    batch_size, seq_len, dim = 2, 2000, 7
    seq_input1 = torch.randn(batch_size, seq_len, dim)
    seq_input2 = torch.randn(batch_size, seq_len, dim)
    return (seq_input1, seq_input2)


MENAGERIE_ENTRIES = [
    ("PIPR", "build_pipr", "example_input_pipr", 2019, "ported-pytorch"),
]

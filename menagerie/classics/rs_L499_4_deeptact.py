# FAITHFUL PORT of liwenran/DeepTACT @ master (DeepTACT.py) (original framework: Keras 1.2.0 + Theano)
#
# DeepTACT (Wang et al., Nucleic Acids Research 2019) predicts promoter-enhancer /
# promoter-promoter (P-E / P-P) chromatin interactions from paired DNA-sequence
# windows and DNase-hypersensitivity tracks. The real repo's `model_def()` builds
# a Keras 1.x functional-ish `Sequential`+`Merge` graph:
#   - two "sequence" Conv2D branches (one per region, e.g. enhancer + promoter),
#     each Convolution2D(1024, kernel=(NUM_SEQ, 40), relu) -> MaxPooling2D((1,20))
#     -> Reshape to (1024, L) with the pooled window length L = (RESIZED_LEN-40+1)//20
#   - two analogous "DNase" Conv2D branches (kernel height NUM_REP, the number of
#     DNase replicate experiments, instead of NUM_SEQ)
#   - Merge(seq branches, mode='concat') along the window-length axis, and
#     likewise for the DNase branches
#   - Merge(merged_seq, merged_DNase, mode='concat', concat_axis=-2) concatenating
#     along the 1024-channel axis -> Permute((2,1)) so the sequence-position axis
#     becomes the RNN time axis
#   - BatchNorm -> Dropout -> Bidirectional(LSTM(100), merge_mode='concat')
#     (200-d per timestep) -> a custom additive-attention `AttLayer` (tanh score,
#     softmax pool over time) -> BatchNorm -> Dropout
#   - Dense(925) -> BatchNorm -> ReLU -> Dropout -> Dense(1, sigmoid)
# Ported layer-for-layer with the same branch structure, channel counts (1024),
# kernel geometry ((region-channels, 40) conv + (1,20) pool), concat axes,
# BiLSTM hidden size (100 each direction), and the same tanh/softmax additive
# attention pooling mechanism as the real `AttLayer.call()`.
import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class SeqConvBranch(nn.Module):
    """Faithful port of one conv_*_seq / conv_*_DNase Sequential branch.

    Real Keras: Convolution2D(n_filters, region_channels, kernel_len, relu,
    border_mode='valid', dim_ordering='th') on (1, region_channels, region_len)
    -> MaxPooling2D((1, pool_len), border_mode='valid') -> Reshape((n_filters, L)).
    """

    def __init__(self, region_channels, region_len, n_filters=1024, kernel_len=40, pool_len=20):
        super().__init__()
        self.conv = nn.Conv2d(1, n_filters, kernel_size=(region_channels, kernel_len))
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d((1, pool_len))

    def forward(self, x):
        # x: (N, 1, region_channels, region_len)
        x = self.act(self.conv(x))  # (N, n_filters, 1, region_len-kernel_len+1)
        x = self.pool(x)  # (N, n_filters, 1, L)
        x = x.squeeze(2)  # (N, n_filters, L)  == Reshape((n_filters, L))
        return x


class AttLayer(nn.Module):
    """Faithful port of the real repo's additive-attention AttLayer.

    Real Keras/Theano call():
        M = tanh(x)
        alpha = dot(M, W)                    # W: (features,), alpha: (N, T)
        ai = exp(alpha)
        weights = ai / sum(ai, axis=1)       # softmax over time axis T
        weighted_input = x * weights[:, :, None]
        return tanh(sum(weighted_input, axis=1))   # (N, features)
    """

    def __init__(self, features):
        super().__init__()
        self.W = nn.Parameter(torch.empty(features).normal_(0, 0.05))

    def forward(self, x):
        # x: (N, T, features)
        m = torch.tanh(x)
        alpha = torch.matmul(m, self.W)  # (N, T)
        weights = torch.softmax(alpha, dim=1)  # exp/sum == softmax
        weighted = x * weights.unsqueeze(-1)  # (N, T, features)
        return torch.tanh(weighted.sum(dim=1))  # (N, features)


class DeepTACT(nn.Module):
    """Faithful port of model_def() for the P-E interaction type."""

    def __init__(
        self,
        num_seq=4,
        num_rep=3,
        enhancer_len=2000,
        promoter_len=1000,
        n_filters=1024,
        kernel_len=40,
        pool_len=20,
        lstm_hidden=100,
        dense_hidden=925,
    ):
        super().__init__()
        self.conv_enhancer_seq = SeqConvBranch(
            num_seq, enhancer_len, n_filters, kernel_len, pool_len
        )
        self.conv_promoter_seq = SeqConvBranch(
            num_seq, promoter_len, n_filters, kernel_len, pool_len
        )
        self.conv_enhancer_dnase = SeqConvBranch(
            num_rep, enhancer_len, n_filters, kernel_len, pool_len
        )
        self.conv_promoter_dnase = SeqConvBranch(
            num_rep, promoter_len, n_filters, kernel_len, pool_len
        )

        self.bn1 = nn.BatchNorm1d(2 * n_filters)
        drop_rate = 0.5
        self.dropout1 = nn.Dropout(drop_rate)
        self.lstm = nn.LSTM(
            input_size=2 * n_filters, hidden_size=lstm_hidden, batch_first=True, bidirectional=True
        )
        self.attn = AttLayer(2 * lstm_hidden)
        self.bn2 = nn.BatchNorm1d(2 * lstm_hidden)
        self.dropout2 = nn.Dropout(drop_rate)

        self.fc1 = nn.Linear(2 * lstm_hidden, dense_hidden)
        self.bn3 = nn.BatchNorm1d(dense_hidden)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(dense_hidden, 1)

    def forward(self, enhancer_seq, promoter_seq, enhancer_dnase, promoter_dnase):
        # each branch: (N, 1, region_channels, region_len) -> (N, n_filters, L)
        e_seq = self.conv_enhancer_seq(enhancer_seq)
        p_seq = self.conv_promoter_seq(promoter_seq)
        merged_seq = torch.cat([e_seq, p_seq], dim=-1)  # concat along window-length axis

        e_dnase = self.conv_enhancer_dnase(enhancer_dnase)
        p_dnase = self.conv_promoter_dnase(promoter_dnase)
        merged_dnase = torch.cat([e_dnase, p_dnase], dim=-1)

        # Merge([merged_seq, merged_dnase], mode='concat', concat_axis=-2)
        # concatenates along the channel (n_filters) axis == dim=1 here.
        merged = torch.cat([merged_seq, merged_dnase], dim=1)  # (N, 2*n_filters, L_total)

        merged = self.bn1(merged)
        merged = self.dropout1(merged)

        # Permute((2, 1)): sequence-position axis becomes the RNN time axis.
        merged = merged.permute(0, 2, 1).contiguous()  # (N, L_total, 2*n_filters)
        merged, _ = self.lstm(merged)  # (N, L_total, 2*lstm_hidden)

        pooled = self.attn(merged)  # (N, 2*lstm_hidden)
        pooled = self.bn2(pooled)
        pooled = self.dropout2(pooled)

        out = self.fc1(pooled)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = torch.sigmoid(self.fc2(out))
        return out


def build_deeptact():
    # Real config (P-E, demo run): NUM_SEQ=4, NUM_REP=3, enhancer RESIZED_LEN=2000,
    # promoter RESIZED_LEN=1000, n_filters=1024. Shrink region lengths and filter
    # count for a tiny trace while keeping every branch/merge/LSTM/attention stage
    # from the real graph intact.
    return DeepTACT(
        num_seq=4,
        num_rep=3,
        enhancer_len=120,
        promoter_len=100,
        n_filters=8,
        kernel_len=10,
        pool_len=4,
        lstm_hidden=6,
        dense_hidden=16,
    )


def example_input_deeptact():
    n = 2
    enhancer_seq = torch.randn(n, 1, 4, 120)
    promoter_seq = torch.randn(n, 1, 4, 100)
    enhancer_dnase = torch.randn(n, 1, 3, 120)
    promoter_dnase = torch.randn(n, 1, 3, 100)
    return (enhancer_seq, promoter_seq, enhancer_dnase, promoter_dnase)


MENAGERIE_ENTRIES = [
    ("DeepTACT", build_deeptact, example_input_deeptact, 2019, "ported-pytorch"),
]

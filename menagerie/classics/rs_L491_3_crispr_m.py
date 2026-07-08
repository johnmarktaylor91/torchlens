# FAITHFUL PORT of lyotvincent/CRISPR-M @ master (original framework: TensorFlow 2.9 /
# Keras 2.9)
#
# CRISPR-M (PLOS Comp. Biol. 2024) is a three-branch multi-view network for sgRNA
# off-target-effect prediction with mismatches AND indels: three token sequences (sgRNA,
# on-target, off-target encodings) share one Embedding + PositionalEncoding stack, feed a
# self-attention block each, then split into four parallel Conv2D/pooling/BiLSTM branches
# (branch_2 reuses `attention_1`'s output a second time, and branch_3 skips pooling --
# exactly as in the upstream code) that concatenate into a dense classification head. Per
# the repo's own README, the final model is `m81212_n13` in
# `test/2encoding_test/mine/test_model.py` (backed by `codes/positional_encoding.py`); the
# repo ships only Keras/TF source (no torch), so the real functional-model graph was
# transcribed layer-for-layer, verified against the real TF/Keras shapes:
#
#   inputs_{1,2,3}: (batch, 24) int token ids
#   shared Embedding(30, 7) + shared PositionalEncoding(max_steps=24, max_dims=7)
#   attention_{1,2,3} = MultiHeadAttention(heads=8, key_dim=6)(pe_i, pe_i)  -> (b, 24, 7)
#
#   branch_1 (from attention_1): Reshape(24,7,1) -> Dropout(.2) -> Conv2D(32,(1,4),relu) ->
#     BN -> Dropout(.2) -> Conv2D(64,(1,4),relu) -> BN -> squeeze -> (b,24,64)
#     -> AveragePooling1D/MaxPooling1D(channels_first, pool=2) -> concat -> (b,24,64)
#     -> Bidirectional(LSTM(32), return_sequences=True) -> (b,24,64) -> Flatten -> 1536
#   branch_2 (from attention_1 AGAIN): Reshape -> Dropout(.2) -> Conv2D(64,(1,7),relu) ->
#     BN -> squeeze -> (b,24,64) -> avg/max pool -> concat -> BiLSTM(32) -> Flatten -> 1536
#   branch_3 (from attention_2): Reshape -> Dropout(.2) -> Conv2D(64,(1,7),relu) -> BN ->
#     squeeze -> (b,24,64) -> BiLSTM(32) [NO pooling step] -> Flatten -> 1536
#   branch_4 (from attention_3): identical shape to branch_3 -> Flatten -> 1536
#
#   concat([flatten_1..4], axis=-1) -> 6144 -> Dropout(.2) -> Dense(256, relu) -> BN ->
#     Dropout(.2) -> Dense(64, relu) -> BN -> Dropout(.8) -> Dense(1, sigmoid)
#
# Shapes above were verified against a live `tensorflow.keras` run of the real
# PositionalEncoding + MultiHeadAttention + Conv2D/pooling stack at VOCABULARY_SIZE=30,
# MAX_STEPS=24, EMBED_SIZE=7 (the function's own defaults) before writing this port, so the
# torch layer dimensions below are not guessed.
import math

import torch
from torch import nn

MENAGERIE_ZOO = "ported-pytorch"


class PositionalEncoding(nn.Module):
    """Faithful port of codes/positional_encoding.py's PositionalEncoding Keras layer."""

    def __init__(self, max_steps, max_dims):
        super().__init__()
        even_dims = max_dims + 1 if max_dims % 2 == 1 else max_dims
        pos_emb = torch.zeros(1, max_steps, even_dims)
        position = torch.arange(max_steps, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, even_dims, 2, dtype=torch.float32) * (-math.log(10000.0) / even_dims)
        )
        pos_emb[0, :, 0::2] = torch.sin(position * div_term)
        pos_emb[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_embedding", pos_emb[:, :, :max_dims])

    def forward(self, x):
        seq_len, dims = x.shape[-2], x.shape[-1]
        return x + self.positional_embedding[:, :seq_len, :dims]


class ConvPoolBranch(nn.Module):
    """branch_1 / branch_2 style: two (or one) Conv2D(relu)+BN stages, squeeze, then
    channel-wise AveragePooling1D/MaxPooling1D(pool=2) concatenated, then a BiLSTM."""

    def __init__(self, embed_size, two_stage, kernel_w_1, kernel_w_2=None, lstm_units=32):
        super().__init__()
        self.two_stage = two_stage
        self.conv_1 = nn.Conv2d(1, 32 if two_stage else 64, kernel_size=(1, kernel_w_1))
        self.bn_1 = nn.BatchNorm2d(32 if two_stage else 64, eps=1e-3)
        self.dropout_1 = nn.Dropout(p=0.2)
        if two_stage:
            self.conv_2 = nn.Conv2d(32, 64, kernel_size=(1, kernel_w_2))
            self.bn_2 = nn.BatchNorm2d(64, eps=1e-3)
        self.avg_pool = nn.AvgPool1d(kernel_size=2)
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.bilstm = nn.LSTM(
            input_size=64 if not two_stage else 64,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True,
        )
        self.relu = nn.ReLU()

    def forward(self, attn_out):
        # attn_out: (batch, 24, embed_size) -> Reshape(24, embed_size, 1) in Keras NHWC ==
        # torch NCHW (batch, 1, 24, embed_size).
        x = attn_out.unsqueeze(1)
        x = self.dropout_1(x)
        x = self.relu(self.conv_1(x))
        x = self.bn_1(x)
        if self.two_stage:
            x = self.dropout_1(x)
            x = self.relu(self.conv_2(x))
            x = self.bn_2(x)
        # x: (batch, 64, 24, 1) -> squeeze trailing width dim -> (batch, 64, 24)
        # -> Keras `Reshape` drops the length-1 axis leaving (batch, seq=24, channels=64);
        # channels_first pooling in Keras pools over the channel axis (64), so we pool the
        # (batch, channels=64, seq=24) layout directly (channels-first == torch native).
        x = x.squeeze(-1)  # (batch, 64, 24)
        avg = self.avg_pool(x)  # pools over seq axis? -- see note below
        # NOTE: Keras `AveragePooling1D(data_format='channels_first')` on a (batch, 24, 64)
        # tensor treats axis 1 (length 24) as the "channels" dim and pools over axis 2 (the
        # 64-wide feature axis) in windows of 2, producing (batch, 24, 32). To reproduce
        # that exactly in torch (which pools over the last axis natively), keep the tensor
        # as (batch, seq=24, channels=64) and pool the last axis directly.
        seq_x = x.permute(0, 2, 1)  # (batch, 24, 64)
        avg = self.avg_pool(seq_x)  # (batch, 24, 32)
        mx = self.max_pool(seq_x)  # (batch, 24, 32)
        pooled = torch.cat([avg, mx], dim=-1)  # (batch, 24, 64)
        lstm_out, _ = self.bilstm(pooled)  # (batch, 24, 64)
        return torch.flatten(lstm_out, start_dim=1)  # (batch, 1536)


class ConvOnlyBranch(nn.Module):
    """branch_3 / branch_4 style: single Conv2D(relu)+BN, squeeze, BiLSTM -- NO pooling."""

    def __init__(self, kernel_w=7, lstm_units=32):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=(1, kernel_w))
        self.bn = nn.BatchNorm2d(64, eps=1e-3)
        self.dropout = nn.Dropout(p=0.2)
        self.bilstm = nn.LSTM(
            input_size=64, hidden_size=lstm_units, batch_first=True, bidirectional=True
        )
        self.relu = nn.ReLU()

    def forward(self, attn_out):
        x = attn_out.unsqueeze(1)
        x = self.dropout(x)
        x = self.relu(self.conv(x))
        x = self.bn(x)
        x = x.squeeze(-1).permute(0, 2, 1)  # (batch, 24, 64)
        lstm_out, _ = self.bilstm(x)
        return torch.flatten(lstm_out, start_dim=1)


class CRISPRM(nn.Module):
    """Faithful port of CRISPR-M's final model, `m81212_n13`."""

    def __init__(self, vocabulary_size=30, max_steps=24, embed_size=7):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embed_size)
        self.positional_encoding = PositionalEncoding(max_steps, embed_size)
        # Keras `MultiHeadAttention(num_heads=8, key_dim=6)` projects Q/K/V to key_dim=6
        # per head internally and merges back to the query's last dim (embed_size=7) for
        # output -- torch's nn.MultiheadAttention instead requires embed_dim to be
        # divisible by num_heads and ties kdim/vdim directly to that embed_dim. embed_size
        # (7) is prime, so 8 heads is not representable; we use num_heads=1 (single-head
        # attention over the same embed_dim=7 Q/K/V), preserving the shape contract
        # ((batch, 24, 7) in -> (batch, 24, 7) out, verified above) and the residual-free
        # self-attention role the layer plays in the graph, while the per-head width no
        # longer matches upstream's 6-dim keys.
        self.attention_1 = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=1, batch_first=True
        )
        self.attention_2 = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=1, batch_first=True
        )
        self.attention_3 = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=1, batch_first=True
        )

        self.branch_1 = ConvPoolBranch(embed_size, two_stage=True, kernel_w_1=4, kernel_w_2=4)
        self.branch_2 = ConvPoolBranch(embed_size, two_stage=False, kernel_w_1=7)
        self.branch_3 = ConvOnlyBranch(kernel_w=7)
        self.branch_4 = ConvOnlyBranch(kernel_w=7)

        concat_dim = 4 * (max_steps * 64)  # 4 branches x (24 * 64) = 6144
        self.dropout_main_1 = nn.Dropout(p=0.2)
        self.dense_1 = nn.Linear(concat_dim, 256)
        self.bn_1 = nn.BatchNorm1d(256, eps=1e-3)
        self.dropout_main_2 = nn.Dropout(p=0.2)
        self.dense_2 = nn.Linear(256, 64)
        self.bn_2 = nn.BatchNorm1d(64, eps=1e-3)
        self.dropout_main_3 = nn.Dropout(p=0.8)
        self.output_layer = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2, input_3):
        embeddings_1 = self.positional_encoding(self.embedding(input_1))
        embeddings_2 = self.positional_encoding(self.embedding(input_2))
        embeddings_3 = self.positional_encoding(self.embedding(input_3))

        attention_1, _ = self.attention_1(embeddings_1, embeddings_1, embeddings_1)
        attention_2, _ = self.attention_2(embeddings_2, embeddings_2, embeddings_2)
        attention_3, _ = self.attention_3(embeddings_3, embeddings_3, embeddings_3)

        flatten_1 = self.branch_1(attention_1)
        flatten_2 = self.branch_2(attention_1)  # branch_2 reuses attention_1, per upstream
        flatten_3 = self.branch_3(attention_2)
        flatten_4 = self.branch_4(attention_3)

        con = torch.cat([flatten_1, flatten_2, flatten_3, flatten_4], dim=-1)
        main = self.dropout_main_1(con)
        main = self.relu(self.dense_1(main))
        main = self.bn_1(main)
        main = self.dropout_main_2(main)
        main = self.relu(self.dense_2(main))
        main = self.bn_2(main)
        main = self.dropout_main_3(main)
        return self.sigmoid(self.output_layer(main))


def build_crispr_m():
    return CRISPRM()


def example_input_crispr_m():
    input_1 = torch.randint(0, 30, (2, 24))
    input_2 = torch.randint(0, 30, (2, 24))
    input_3 = torch.randint(0, 30, (2, 24))
    return (input_1, input_2, input_3)


MENAGERIE_ENTRIES = [
    ("CRISPR-M", build_crispr_m, example_input_crispr_m, 2024, "ported-pytorch"),
]

# SOURCE: vendored from 13274086/DeepSC @ 6ede3fdb696424ddb87936941fcd78aed94205f4
# models/transceiver.py (real DeepSC architecture classes, unmodified) +
# a forward() driver composed from the REAL call sequence in utils.py::val_step
# (create_masks / model.encoder / model.channel_encoder / PowerNormalize /
# Channels.AWGN / model.channel_decoder / model.decoder / model.dense).
# The original repo never defines DeepSC.forward() -- callers (train_step/
# val_step/greedy_decode in utils.py) invoke the submodules directly in that
# exact order. This wrapper adds a mechanical forward() that reproduces that
# exact composition (AWGN channel branch) so the model is traceable end to end;
# no new layers/mechanisms are introduced.
"""DeepSC: Deep learning enabled Semantic Communication transformer (Xie et al. 2021)."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )  # math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        x = self.dropout(x)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)

        x = self.dense(x)
        x = self.dropout(x)

        return x

    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores += mask * -1e9
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)

        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)

        src_output = self.src_mha(x, memory, memory, trg_padding_mask)  # q, k, v
        x = self.layernorm2(x + src_output)

        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, num_layers, src_vocab_size, max_len, d_model, num_heads, dff, dropout=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, src_mask):
        "Pass the input (and mask) through each layer in turn."
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)

        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size, max_len, d_model, num_heads, dff, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)

        return x


class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()

        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)

        self.layernorm = nn.LayerNorm(size1, eps=1e-6)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)

        output = self.layernorm(x1 + x5)

        return output


class Channels:
    """utils.py::Channels -- vendored verbatim (AWGN branch used by forward())."""

    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.normal(0, n_var, size=Tx_sig.shape).to(Tx_sig.device)
        return Rx_sig


def _power_normalize(x):
    # utils.py::PowerNormalize, vendored verbatim
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    return x


def _subsequent_mask(size):
    # utils.py::subsequent_mask, vendored verbatim
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask)


def _create_masks(src, trg, padding_idx):
    # utils.py::create_masks, vendored verbatim (device-pin dropped; traced on src.device)
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
    look_ahead_mask = _subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)

    return src_mask.to(src.device), combined_mask.to(src.device)


class DeepSC(nn.Module):
    def __init__(
        self,
        num_layers,
        src_vocab_size,
        trg_vocab_size,
        src_max_len,
        trg_max_len,
        d_model,
        num_heads,
        dff,
        dropout=0.1,
    ):
        super(DeepSC, self).__init__()

        self.encoder = Encoder(
            num_layers, src_vocab_size, src_max_len, d_model, num_heads, dff, dropout
        )

        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16),
        )

        self.channel_decoder = ChannelDecoder(16, d_model, 512)

        self.decoder = Decoder(
            num_layers, trg_vocab_size, trg_max_len, d_model, num_heads, dff, dropout
        )

        self.dense = nn.Linear(d_model, trg_vocab_size)

        self._channels = Channels()
        self._pad_idx = 0
        self._n_var = 0.1  # fixed noise std, matches SNR_to_noise(~13dB) default sweep

    def forward(self, src, trg):
        # Mirrors utils.py::val_step's real call sequence (AWGN branch).
        trg_inp = trg[:, :-1]
        src_mask, look_ahead_mask = _create_masks(src, trg_inp, self._pad_idx)

        enc_output = self.encoder(src, src_mask)
        channel_enc_output = self.channel_encoder(enc_output)
        tx_sig = _power_normalize(channel_enc_output)

        rx_sig = self._channels.AWGN(tx_sig, self._n_var)

        channel_dec_output = self.channel_decoder(rx_sig)
        dec_output = self.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
        pred = self.dense(dec_output)
        return pred


def build_deepsc():
    return DeepSC(
        num_layers=2,
        src_vocab_size=32,
        trg_vocab_size=32,
        src_max_len=16,
        trg_max_len=16,
        d_model=32,
        num_heads=4,
        dff=64,
        dropout=0.1,
    )


def example_input_deepsc():
    src = torch.randint(1, 32, (1, 12))
    trg = torch.randint(1, 32, (1, 12))
    return (src, trg)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepSC", "build_deepsc", "example_input_deepsc", 2021, "vendored-pytorch"),
]

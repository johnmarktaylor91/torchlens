# SOURCE: vendored from https://github.com/Jhryu30/AnomalyBERT @ main
# (models/transformer.py + models/anomaly_transformer.py + utils/functions.py::clone_layer)
#
# AnomalyBERT (Jeong et al., ICLR 2023 workshop / arXiv:2305.04468): a BERT-style
# masked-pretraining transformer for time-series anomaly detection. A pre-norm
# transformer encoder with 1D relative-position-embedding multi-head attention
# reconstructs a patch-embedded multivariate window; large reconstruction error
# on masked/degraded segments flags anomalies.
#
# Vendored real repo code (AnomalyTransformer + TransformerEncoder/EncoderLayer/
# MultiHeadAttentionLayer/PositionWiseFeedForwardLayer/Sinusoidal+Absolute
# positional-encoding layers + the clone_layer helper). Only import paths were
# adjusted (relative `models.transformer` / `utils.functions` imports flattened
# into this single file); no layer, kernel size, dimension, or dataflow inside
# the architecture was changed. Non-architectural pieces (data preprocessing,
# training loop, estimate/metrics scripts) were dropped -- none of that is part
# of the model.

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_

MENAGERIE_ZOO = "vendored-pytorch"


def clone_layer(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_embed, n_head, max_seq_len=512, relative_position_embedding=True):
        super(MultiHeadAttentionLayer, self).__init__()
        assert d_embed % n_head == 0  # Check if d_model is divisible by n_head.

        self.d_embed = d_embed
        self.n_head = n_head
        self.d_k = d_embed // n_head
        self.scale = 1 / np.sqrt(self.d_k)

        self.word_fc_layers = clone_layer(nn.Linear(d_embed, d_embed), 3)
        self.output_fc_layer = nn.Linear(d_embed, d_embed)

        self.max_seq_len = max_seq_len
        self.relative_position_embedding = relative_position_embedding
        if relative_position_embedding:
            # Table of 1D relative position embedding
            self.relative_position_embedding_table = nn.Parameter(
                torch.zeros(2 * max_seq_len - 1, n_head)
            )
            trunc_normal_(self.relative_position_embedding_table, std=0.02)

            # Set 1D relative position embedding index.
            coords_h = np.arange(max_seq_len)
            coords_w = np.arange(max_seq_len - 1, -1, -1)
            coords = coords_h[:, None] + coords_w[None, :]
            self.relative_position_index = coords.flatten()

    def forward(self, x):
        """
        <input>
        x : (n_batch, n_token, d_embed)
        """
        n_batch = x.shape[0]

        # Apply linear layers.
        query = self.word_fc_layers[0](x)
        key = self.word_fc_layers[1](x)
        value = self.word_fc_layers[2](x)

        # Split heads.
        query_out = query.view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)
        key_out = key.view(n_batch, -1, self.n_head, self.d_k).contiguous().permute(0, 2, 3, 1)
        value_out = value.view(n_batch, -1, self.n_head, self.d_k).transpose(1, 2)

        # Compute attention and concatenate matrices.
        scores = torch.matmul(query_out * self.scale, key_out)

        # Add relative position embedding
        if self.relative_position_embedding:
            position_embedding = self.relative_position_embedding_table[
                self.relative_position_index
            ].view(self.max_seq_len, self.max_seq_len, -1)
            position_embedding = position_embedding.permute(2, 0, 1).contiguous().unsqueeze(0)
            scores = scores + position_embedding

        probs = F.softmax(scores, dim=-1)
        attention_out = torch.matmul(probs, value_out)

        # Convert 4d tensor to proper 3d output tensor.
        attention_out = attention_out.transpose(1, 2).contiguous().view(n_batch, -1, self.d_embed)

        return self.output_fc_layer(attention_out)


class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_embed, d_ff, dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.first_fc_layer = nn.Linear(d_embed, d_ff)
        self.second_fc_layer = nn.Linear(d_ff, d_embed)
        self.activation_layer = nn.GELU()
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.first_fc_layer(x)
        out = self.dropout_layer(self.activation_layer(out))
        return self.second_fc_layer(out)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_seq_len=512, dropout=0.1):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)

        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        denominators = torch.exp(
            torch.arange(0, d_embed, 2) * (np.log(0.0001) / d_embed)
        ).unsqueeze(0)
        encoding_matrix = torch.matmul(positions, denominators)

        encoding = torch.empty(1, max_seq_len, d_embed)
        encoding[0, :, 0::2] = torch.sin(encoding_matrix)
        encoding[0, :, 1::2] = torch.cos(encoding_matrix[:, : (d_embed // 2)])

        self.register_buffer("encoding", encoding)

    def forward(self, x):
        return self.dropout_layer(x + self.encoding)


class AbsolutePositionEmbedding(nn.Module):
    def __init__(self, d_embed, max_seq_len=512, dropout=0.1):
        super(AbsolutePositionEmbedding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_embed))
        trunc_normal_(self.embedding, std=0.02)

    def forward(self, x):
        return self.dropout_layer(x + self.embedding)


class EncoderLayer(nn.Module):
    def __init__(self, attention_layer, feed_forward_layer, norm_layer, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention_layer = attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.norm_layers = clone_layer(norm_layer, 2)
        self.dropout_layer = nn.Dropout(p=dropout)

        for p in self.attention_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        for p in self.feed_forward_layer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def forward(self, x):
        out1 = self.norm_layers[0](x)  # Layer norm first
        out1 = self.attention_layer(out1)
        out1 = self.dropout_layer(out1) + x

        out2 = self.norm_layers[1](out1)
        out2 = self.feed_forward_layer(out2)
        return self.dropout_layer(out2) + out1


class TransformerEncoder(nn.Module):
    def __init__(self, positional_encoding_layer, encoder_layer, n_layer):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = clone_layer(encoder_layer, n_layer)

        self.positional_encoding = True if positional_encoding_layer is not None else False
        if self.positional_encoding:
            self.positional_encoding_layer = positional_encoding_layer

    def forward(self, x):
        """
        <input>
        x : (n_batch, n_token, d_embed)
        """
        if self.positional_encoding:
            out = self.positional_encoding_layer(x)
        else:
            out = x

        for layer in self.encoder_layers:
            out = layer(out)

        return out


def get_transformer_encoder(
    d_embed=512,
    positional_encoding=None,
    relative_position_embedding=True,
    n_layer=6,
    n_head=8,
    d_ff=2048,
    max_seq_len=512,
    dropout=0.1,
):
    if positional_encoding in ("Sinusoidal", "sinusoidal", "sin"):
        positional_encoding_layer = SinusoidalPositionalEncoding(d_embed, max_seq_len, dropout)
    elif positional_encoding in ("Absolute", "absolute", "abs"):
        positional_encoding_layer = AbsolutePositionEmbedding(d_embed, max_seq_len, dropout)
    elif positional_encoding in (None, "None"):
        positional_encoding_layer = None
    else:
        raise ValueError(f"Unknown positional_encoding: {positional_encoding}")

    attention_layer = MultiHeadAttentionLayer(
        d_embed, n_head, max_seq_len, relative_position_embedding
    )
    feed_forward_layer = PositionWiseFeedForwardLayer(d_embed, d_ff, dropout)
    norm_layer = nn.LayerNorm(d_embed, eps=1e-6)
    encoder_layer = EncoderLayer(attention_layer, feed_forward_layer, norm_layer, dropout)

    return TransformerEncoder(positional_encoding_layer, encoder_layer, n_layer)


class AnomalyTransformer(nn.Module):
    """<class init args>
    linear_embedding : embedding layer to feed data into Transformer encoder
    transformer_encoder : Transformer encoder body
    mlp_layers : MLP layers to return output data
    d_embed : embedding dimension (in Transformer encoder)
    patch_size : number of data points for an embedded vector
    max_seq_len : maximum length of sequence (= window size)
    """

    def __init__(
        self, linear_embedding, transformer_encoder, mlp_layers, d_embed, patch_size, max_seq_len
    ):
        super(AnomalyTransformer, self).__init__()
        self.linear_embedding = linear_embedding
        self.transformer_encoder = transformer_encoder
        self.mlp_layers = mlp_layers

        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        self.data_seq_len = patch_size * max_seq_len

    def forward(self, x):
        """
        <input info>
        x : (n_batch, n_token, d_data) = (_, max_seq_len*patch_size, _)
        """
        n_batch = x.shape[0]

        embedded_out = x.view(n_batch, self.max_seq_len, self.patch_size, -1).view(
            n_batch, self.max_seq_len, -1
        )
        embedded_out = self.linear_embedding(embedded_out)  # linear embedding

        transformer_out = self.transformer_encoder(embedded_out)  # Encode data.
        output = self.mlp_layers(transformer_out)  # Reconstruct data.
        return output.view(n_batch, self.max_seq_len, self.patch_size, -1).view(
            n_batch, self.data_seq_len, -1
        )


def get_anomaly_transformer(
    input_d_data,
    output_d_data,
    patch_size,
    d_embed=512,
    hidden_dim_rate=4.0,
    max_seq_len=512,
    positional_encoding=None,
    relative_position_embedding=True,
    transformer_n_layer=12,
    transformer_n_head=8,
    dropout=0.1,
):
    """
    <input info>
    input_d_data : data input dimension
    output_d_data : data output dimension
    patch_size : number of data points per embedded feature
    d_embed : embedding dimension (in Transformer encoder)
    hidden_dim_rate : hidden layer dimension rate to d_embed
    max_seq_len : maximum length of sequence (= window size)
    positional_encoding : positional encoding for embedded input; None/Sinusoidal/Absolute
    relative_position_embedding : relative position embedding option
    transformer_n_layer : number of Transformer encoder layers
    transformer_n_head : number of heads in multi-head attention module
    dropout : dropout rate
    """
    hidden_dim = int(hidden_dim_rate * d_embed)

    linear_embedding = nn.Linear(input_d_data * patch_size, d_embed)
    transformer_encoder = get_transformer_encoder(
        d_embed=d_embed,
        positional_encoding=positional_encoding,
        relative_position_embedding=relative_position_embedding,
        n_layer=transformer_n_layer,
        n_head=transformer_n_head,
        d_ff=hidden_dim,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
    mlp_layers = nn.Sequential(
        nn.Linear(d_embed, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_d_data * patch_size)
    )

    nn.init.xavier_uniform_(linear_embedding.weight)
    nn.init.zeros_(linear_embedding.bias)
    nn.init.xavier_uniform_(mlp_layers[0].weight)
    nn.init.zeros_(mlp_layers[0].bias)
    nn.init.xavier_uniform_(mlp_layers[2].weight)
    nn.init.zeros_(mlp_layers[2].bias)

    return AnomalyTransformer(
        linear_embedding, transformer_encoder, mlp_layers, d_embed, patch_size, max_seq_len
    )


def build_anomalybert():
    # Tiny config: 3-channel input data, patch_size=4, max_seq_len=8 (window
    # size 32), small embedding/head/layer counts -- keeps the real
    # architecture (relative-position-embedding attention + pre-norm encoder)
    # intact while staying fast to trace.
    return get_anomaly_transformer(
        input_d_data=3,
        output_d_data=3,
        patch_size=4,
        d_embed=32,
        hidden_dim_rate=4.0,
        max_seq_len=8,
        positional_encoding=None,
        relative_position_embedding=True,
        transformer_n_layer=2,
        transformer_n_head=4,
        dropout=0.0,
    )


def example_input_anomalybert():
    # (n_batch, max_seq_len * patch_size, input_d_data) = (2, 8*4, 3)
    return torch.randn(2, 32, 3)


MENAGERIE_ENTRIES = [
    (
        "AnomalyBERT",
        build_anomalybert,
        example_input_anomalybert,
        2023,
        MENAGERIE_ZOO,
    ),
]

# SOURCE: vendored from zcao0420/MOFormer @ main
# File: model/transformer.py
#
# MOFormer (Cao et al., JACS 2023): a structure-agnostic Transformer for metal-organic
# framework (MOF) property prediction. A tokenized MOFid string (SMILES-like topology +
# metal-node identifier encoding) is embedded, scaled by sqrt(d_model), given sinusoidal
# positional encodings, and passed through a standard `nn.TransformerEncoder` stack;
# `TransformerRegressor` takes the graph-token ([CLS]-style position 0) embedding and feeds
# it through a 4-layer regression head (`regressoionHead`) down to a scalar property
# prediction. Verbatim from the real repo file (only import path unchanged since the module
# had no local imports beyond torch/numpy/pandas).
#
# MENAGERIE_ZOO = "vendored-pytorch"

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class regressoionHead(nn.Module):
    def __init__(self, d_embedding: int):
        super().__init__()
        self.layer1 = nn.Linear(d_embedding, d_embedding // 2)
        self.layer2 = nn.Linear(d_embedding // 2, d_embedding // 4)
        self.layer3 = nn.Linear(d_embedding // 4, d_embedding // 8)
        self.layer4 = nn.Linear(d_embedding // 8, 1)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))

        return self.layer4(x)


class Transformer(nn.Module):
    def __init__(
        self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.1
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.xavier_normal_(self.token_encoder.weight)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.token_encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class TransformerRegressor(nn.Module):
    def __init__(self, transformer, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.transformer = transformer
        self.regressionHead = regressoionHead(d_model)

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        output = self.transformer(src)
        output = self.regressionHead(output[:, 0:1, :])
        return output


def build_moformer():
    d_model = 32
    transformer = Transformer(
        ntoken=256, d_model=d_model, nhead=4, d_hid=64, nlayers=2, dropout=0.0
    )
    return TransformerRegressor(transformer, d_model=d_model)


def example_input_moformer():
    # tokenized MOFid string, batch_first: [batch_size, seq_len]
    return (torch.randint(0, 256, (2, 16)),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("MOFormer", "build_moformer", "example_input_moformer", 2023, "vendored"),
]

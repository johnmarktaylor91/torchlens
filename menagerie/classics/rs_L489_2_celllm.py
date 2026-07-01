# SOURCE: vendored from MeiHou0204/CellLM @ main (model/decoder_only_pretrained_model.ipynb)
#
# CellLM (Zhao et al. 2023, "Cell-LM: A Cell Language Model for Single-Cell RNA-Seq
# Analysis"): the repo's `model/__init__.py` module file is empty upstream; the actual
# pretrained-model architecture ships only inside the evaluation notebook
# `model/decoder_only_pretrained_model.ipynb` ("step 2. Training a decoder-only
# pretrained model" cell), as the `SingleCellDecoder` class. It is a decoder-only
# Transformer over gene-expression vectors: a linear embedding into hidden space, an
# `nn.TransformerDecoder` stack (memory fed as an all-zero tensor, i.e. no encoder --
# a self-attention-only "decoder" used autoregressively over the expression profile),
# and a linear output-projection head that reconstructs the gene-expression vector.
# Copied verbatim from the notebook cell (only the surrounding hyperparameter-search /
# Ray Tune driver code, which is not part of the architecture, is omitted).
import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class SingleCellDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.1,
        output_dim: int = None,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        # embedding layer, embed the input gene expression to a higher-dimensional space
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # one decoder-only layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # multiple decoder-only layer
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # output layer, hidden layer to gene expression reconstruction.
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: x: (batch_size, seq_len, input_dim)  # single cell gene expression matrix
        output: (batch_size, seq_len, input_dim)     # reconstruct gene expression matrix
        """
        # 1. embedding
        x_embed = self.embedding(x)  # (batch_size, seq_len, hidden_dim)

        # 2. self-attention decoder-only layer
        memory = torch.zeros_like(x_embed)  # without encoder, only autoregression
        output = self.transformer_decoder(
            tgt=x_embed,
            memory=memory,
        )  # output shape (batch_size, seq_len, hidden_dim)

        # 3. output reconstructed gene expression matrix
        return self.output_layer(output)


def build_celllm_singlecelldecoder():
    return SingleCellDecoder(input_dim=64, hidden_dim=32, num_layers=2, nhead=4, dropout=0.1)


def example_input_celllm_singlecelldecoder():
    # (batch_size, seq_len, input_dim); a handful of "cells" (seq positions) each with a
    # small gene-expression vector, matching the real forward signature.
    return torch.randn(2, 6, 64)


MENAGERIE_ENTRIES = [
    (
        "CellLM-SingleCellDecoder",
        build_celllm_singlecelldecoder,
        example_input_celllm_singlecelldecoder,
        2023,
        "SOURCE_AVAILABLE",
    ),
]

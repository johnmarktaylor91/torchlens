# SOURCE: vendored from facebookresearch/PAQ @ main
#
# https://github.com/facebookresearch/PAQ
# https://raw.githubusercontent.com/facebookresearch/PAQ/main/paq/retrievers/retriever_utils.py
#
# Lewis et al. 2021 "PAQ: 65 Million Probably-Asked Questions and What You Can
# Do With Them" -- the official PAQ repo. RePAQ's dense retriever is a
# DPR-style bi-encoder: `RetrieverEncoder` (`paq/retrievers/retriever_utils.py`)
# wraps a HuggingFace transformer body (`AutoModel`, typically ALBERT/BERT in
# the shipped checkpoints -- the class strips an `'albert.'` state-dict prefix
# in `from_pretrained`) with an optional linear projection head on the
# `[CLS]`/pooled first-token representation, used identically to embed both
# questions and passages (two `RetrieverEncoder` instances form the bi-encoder
# used to build the PAQ HNSW index and to retrieve/rerank at inference).
#
# The `RetrieverEncoder` class below is copied verbatim from
# `paq/retrievers/retriever_utils.py`. Only the `from_pretrained` classmethod
# (which loads a checkpoint directory's `pytorch_model.bin` from disk) is
# dropped, since this vendor constructs the encoder directly via `__init__`
# with a tiny random-init HF config instead of loading pretrained PAQ
# checkpoint weights -- the `forward` computation (the actual architectural
# contribution: HF backbone -> take `[CLS]`/pooled first token -> optional
# `encode_proj` linear) is unchanged.

import torch
from torch import nn
from transformers import AutoModel, BertConfig


class RetrieverEncoder(nn.Module):
    """A wrapper for HF models, with an optional projection"""

    def __init__(self, config, proj_dim):
        super().__init__()
        self.model = AutoModel.from_config(config)
        self.encode_proj = nn.Linear(config.hidden_size, proj_dim) if proj_dim is not None else None
        self.model.init_weights()

    def forward(self, *args, **kwargs):
        seq_outputs = self.model(*args, **kwargs)["last_hidden_state"]
        return (
            self.encode_proj(seq_outputs[:, 0])
            if self.encode_proj is not None
            else seq_outputs[:, 0]
        )


def build_repaq():
    config = BertConfig(
        vocab_size=200,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    # RePAQ shipped checkpoints project the backbone hidden state down to a
    # smaller retrieval embedding dim (the `encode_proj` head) -- exercised
    # here with proj_dim=16.
    return RetrieverEncoder(config, proj_dim=16)


def example_input_repaq():
    input_ids = torch.randint(0, 200, (2, 12))
    return (input_ids,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RePAQ (PAQ Retriever)", "build_repaq", "example_input_repaq", 2021, "vendored"),
]

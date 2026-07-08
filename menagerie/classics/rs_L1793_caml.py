# SOURCE: vendored from jamesmullenbach/caml-mimic @ 44a47455070d3d5c6ee69fb5305e32caec104960
# https://raw.githubusercontent.com/jamesmullenbach/caml-mimic/master/learn/models.py
#
# Mullenbach, Wiegreffe, Duke, Sun, Eisenstein 2018 (NAACL) "Explainable Prediction of
# Medical Codes from Clinical Text" -- CAML (Convolutional Attention network for Multi-Label
# classification). `ConvAttnPool` is the paper's flagship architecture: a word-embedding
# layer feeds a single wide `Conv1d` over the whole document, then a PER-LABEL attention
# mechanism (`self.U`, one attention vector per ICD code) computes label-specific
# softmax-weighted sums of the convolutional features (`alpha = softmax(U.weight @
# conv_features)`, `m = alpha @ conv_features`), and a final per-label linear layer
# (implemented as an elementwise `weight.mul(m).sum(dim=2)` rather than `nn.Linear`, so
# each of the Y labels gets its own classifier row applied to its own attended
# representation) produces the multi-label logits -- this label-wise attention pooling is
# the "explainable" part of the architecture (the attention weights double as a
# code-to-text alignment). `ConvAttnPool`/`BaseModel` are copied verbatim from the real
# `learn/models.py` (module-level `from constants import *` / `from dataproc import
# extract_wvs` / gensim-based `_code_emb_init` code-embedding path and description
# regularization path are dropped -- they only fire when `embed_file`/`code_emb`/`lmbda`
# are supplied, which this build never does; `xavier_uniform` from `torch.nn.init` and
# `F.tanh` are the real (now-deprecated-but-still-functional) torch APIs the original
# code calls, left unchanged. No architectural changes.

from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform


class BaseModel(nn.Module):
    def __init__(self, Y, dicts, dropout=0.5, embed_size=100):
        super(BaseModel, self).__init__()
        torch.manual_seed(1337)
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)

        # add 2 to include UNK and PAD
        vocab_size = len(dicts["ind2w"])
        self.embed = nn.Embedding(vocab_size + 2, embed_size, padding_idx=0)

    def _get_loss(self, yhat, target):
        return F.binary_cross_entropy_with_logits(yhat, target)


class ConvAttnPool(BaseModel):
    def __init__(self, Y, kernel_size, num_filter_maps, dicts, embed_size=100, dropout=0.5):
        super(ConvAttnPool, self).__init__(Y, dicts, dropout=dropout, embed_size=embed_size)

        # initialize conv layer as in 2.1
        self.conv = nn.Conv1d(
            self.embed_size,
            num_filter_maps,
            kernel_size=kernel_size,
            padding=int(floor(kernel_size / 2)),
        )
        xavier_uniform(self.conv.weight)

        # context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.U.weight)

        # final layer: create a matrix to use for the L binary classifiers as in 2.3
        self.final = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.final.weight)

    def forward(self, x, target):
        # get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        # apply convolution and nonlinearity (tanh)
        x = F.tanh(self.conv(x).transpose(1, 2))
        # apply attention
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # document representations are weighted sums using the attention. Can compute all at once as a matmul
        m = alpha.matmul(x)
        # final layer classification
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)

        yhat = y
        loss = self._get_loss(yhat, target)
        return yhat, loss, alpha


def build_caml():
    dicts = {"ind2w": {i: str(i) for i in range(200)}}
    return ConvAttnPool(
        Y=10, kernel_size=5, num_filter_maps=16, dicts=dicts, embed_size=20, dropout=0.0
    )


def example_input_caml():
    x = torch.randint(1, 200, (2, 32))
    target = torch.zeros(2, 10)
    return (x, target)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("CAML", "build_caml", "example_input_caml", 2018, "vendored"),
]

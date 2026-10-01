# SOURCE: vendored from zshihang/MemN2N @ ef5cf330c764daa6069121c46c8defd6130486f3
# https://raw.githubusercontent.com/zshihang/MemN2N/ef5cf330c764daa6069121c46c8defd6130486f3/model.py
#
# Sukhbaatar, Szlam, Weston & Fergus 2015 "End-To-End Memory Networks"
# (https://arxiv.org/abs/1503.08895). Real architecture: input/output memory
# embeddings (A/C) per hop with layer-wise or adjacent (RNN-like) weight tying,
# position-encoding or bag-of-words sentence representations, learned temporal
# encoding matrices (TA/TC), a multi-hop soft-attention read over the memory bank
# that iteratively updates the query "state", and (in linear-start / non-tied-hop
# variants) a hop-updating linear layer `H`. This copy is the code exactly as
# written (class body byte-for-byte unchanged aside from the two shims noted
# below); it is a widely used community-maintained reference implementation of the
# paper's PyTorch port.
#
# Minimal, non-architectural changes made (only import-time / hardware-portability
# shims; no computation changed):
#   - `self.vocab = vocab`/`vocab.stoi['<pad>']`/`len(vocab)` in the real code read
#     a torchtext `Field.vocab` object built from the bAbI dataset loader
#     (helpers.py -> torchtext.datasets.BABI20, an optional/legacy dependency this
#     repo doesn't ship a stable API for). A minimal stand-in `_Vocab` (only
#     `.stoi['<pad>']` and `len(vocab)` are read by the traced path) replaces it.
#   - `params` in the real code is a `namedtuple` built from CLI/argparse config in
#     `helpers.get_params`; replaced with an equivalent local `_Params` namedtuple
#     with the same field names, populated directly (no argparse/CLI parsing).
#   - `compute_weights`'s real code calls `.cuda()` unconditionally when
#     `torch.cuda.is_available()`, which would silently move weights returned to a
#     CUDA model onto a mismatched device when torchlens runs it on a CPU-only host
#     with a GPU present but the model kept on CPU; changed to follow the model's
#     own embedding-weight device (`self.A[0].weight.device`) instead of a bare
#     `cuda.is_available()` check. This is a device-placement correctness fix, not
#     an architecture change (same weighting tensor, same math).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from collections import namedtuple


class _Vocab:
    """Stand-in for the torchtext Field.vocab object (only stoi/len read here)."""

    def __init__(self, n_words, pad_index=1):
        self.stoi = {"<pad>": pad_index}
        self._n_words = n_words

    def __len__(self):
        return self._n_words


_Params = namedtuple(
    "Params",
    ["embed_size", "memory_size", "num_hops", "use_bow", "use_lw", "use_ls"],
)


class MemN2N(nn.Module):
    def __init__(self, params, vocab):
        super(MemN2N, self).__init__()
        self.input_size = len(vocab)
        self.embed_size = params.embed_size
        self.memory_size = params.memory_size
        self.num_hops = params.num_hops
        self.use_bow = params.use_bow
        self.use_lw = params.use_lw
        self.use_ls = params.use_ls
        self.vocab = vocab

        # create parameters according to different type of weight tying
        pad = self.vocab.stoi["<pad>"]
        self.A = nn.ModuleList([nn.Embedding(self.input_size, self.embed_size, padding_idx=pad)])
        self.A[-1].weight.data.normal_(0, 0.1)
        self.C = nn.ModuleList([nn.Embedding(self.input_size, self.embed_size, padding_idx=pad)])
        self.C[-1].weight.data.normal_(0, 0.1)
        if self.use_lw:
            for _ in range(1, self.num_hops):
                self.A.append(self.A[-1])
                self.C.append(self.C[-1])
            self.B = nn.Embedding(self.input_size, self.embed_size, padding_idx=pad)
            self.B.weight.data.normal_(0, 0.1)
            self.out = nn.Parameter(
                I.normal_(torch.empty(self.input_size, self.embed_size), 0, 0.1)
            )
            self.H = nn.Linear(self.embed_size, self.embed_size)
            self.H.weight.data.normal_(0, 0.1)
        else:
            for _ in range(1, self.num_hops):
                self.A.append(self.C[-1])
                self.C.append(nn.Embedding(self.input_size, self.embed_size, padding_idx=pad))
                self.C[-1].weight.data.normal_(0, 0.1)
            self.B = self.A[0]
            self.out = self.C[-1].weight

        # temporal matrix
        self.TA = nn.Parameter(I.normal_(torch.empty(self.memory_size, self.embed_size), 0, 0.1))
        self.TC = nn.Parameter(I.normal_(torch.empty(self.memory_size, self.embed_size), 0, 0.1))

    def forward(self, story, query):
        sen_size = query.shape[-1]
        weights = self.compute_weights(sen_size)
        state = (self.B(query) * weights).sum(1)

        sen_size = story.shape[-1]
        weights = self.compute_weights(sen_size)
        for i in range(self.num_hops):
            memory = (
                (self.A[i](story.view(-1, sen_size)) * weights).sum(1).view(*story.shape[:-1], -1)
            )
            memory += self.TA
            output = (
                (self.C[i](story.view(-1, sen_size)) * weights).sum(1).view(*story.shape[:-1], -1)
            )
            output += self.TC

            probs = (memory @ state.unsqueeze(-1)).squeeze()
            if not self.use_ls:
                probs = F.softmax(probs, dim=-1)
            response = (probs.unsqueeze(1) @ output).squeeze()
            if self.use_lw:
                state = self.H(response) + state
            else:
                state = response + state

        return F.log_softmax(F.linear(state, self.out), dim=-1)

    def compute_weights(self, J):
        d = self.embed_size
        if self.use_bow:
            weights = torch.ones(J, d)
        else:
            func = lambda j, k: 1 - (j + 1) / J - (k + 1) / d * (1 - 2 * (j + 1) / J)  # noqa: E731  (0-based indexing)
            weights = torch.from_numpy(np.fromfunction(func, (J, d), dtype=np.float32))
        return weights.to(self.A[0].weight.device)


def build_memn2n():
    vocab = _Vocab(n_words=40, pad_index=1)
    params = _Params(
        embed_size=20, memory_size=10, num_hops=3, use_bow=False, use_lw=False, use_ls=False
    )
    model = MemN2N(params, vocab)
    model.eval()
    return model


def example_input_memn2n():
    batch_size, memory_size, sen_size, query_len = 2, 10, 6, 5
    vocab_n = 40
    story = torch.randint(2, vocab_n, (batch_size, memory_size, sen_size))
    query = torch.randint(2, vocab_n, (batch_size, query_len))
    return (story, query)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "MemN2N (End-To-End Memory Networks)",
        "build_memn2n",
        "example_input_memn2n",
        2015,
        "vendored",
    ),
]

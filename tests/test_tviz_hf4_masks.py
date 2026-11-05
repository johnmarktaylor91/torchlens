"""Attention-view masks and score decomposition on transformers 4.x eager layouts.

transformers 4.x eager BERT adds an extended key-padding mask shaped
``[batch, 1, 1, src]``; 4.x eager GPT-2 masks twice (a ``torch.where`` with
``finfo.min`` from its causal buffer, then the additive 4D mask), so its masked
scores overflow to ``-inf``. The modules below reproduce both layouts under the
BERT core's class name, so the built-in recipe routes them on any installed
transformers version.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.transformer as tviz

_HEADS = 2
_D_HEAD = 4
_HIDDEN = _HEADS * _D_HEAD
_MIN = torch.finfo(torch.float32).min


class BertSelfAttention(nn.Module):
    """Eager core with HF attribute names; ``layout`` picks the 4.x mask shape."""

    def __init__(self, layout: str) -> None:
        """Initialize q/k/v projections and the mask layout."""

        super().__init__()
        self.layout = layout
        self.num_attention_heads = _HEADS
        self.attention_head_size = _D_HEAD
        self.query = nn.Linear(_HIDDEN, _HIDDEN)
        self.key = nn.Linear(_HIDDEN, _HIDDEN)
        self.value = nn.Linear(_HIDDEN, _HIDDEN)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``[B, H, S, d_head]`` from ``[B, S, hidden]``."""

        return x.view(*x.shape[:-1], _HEADS, _D_HEAD).transpose(1, 2)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Eager attention with the layout's mask terms."""

        q = self._split(self.query(hidden))
        k = self._split(self.key(hidden))
        v = self._split(self.value(hidden))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(_D_HEAD)
        if self.layout == "gpt2":
            n = hidden.shape[1]
            causal = torch.tril(torch.ones(n, n, dtype=torch.bool))[None, None]
            scores = torch.where(causal, scores, torch.full([], _MIN))
        scores = scores + attention_mask
        context = torch.matmul(torch.softmax(scores, dim=-1), v)
        return context.transpose(1, 2).reshape(*hidden.shape[:-1], _HIDDEN)


class _Model(nn.Module):
    """Builds the layout's additive mask inside the traced forward."""

    def __init__(self, layout: str) -> None:
        """Initialize the core under the address ``attn``."""

        super().__init__()
        self.layout = layout
        self.attn = BertSelfAttention(layout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run the core with a 4.x-shaped additive mask."""

        if self.layout == "gpt2":
            n = x.shape[1]
            allowed = torch.tril(torch.ones(n, n, dtype=torch.bool))[None, None]
            additive = torch.where(allowed, torch.tensor(0.0), torch.tensor(_MIN))
        elif self.layout == "scalar":
            additive = torch.full([], _MIN) * attention_mask.amax()
        else:
            additive = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * _MIN
        return self.attn(x, additive)


def _trace(layout: str, padding: torch.Tensor) -> tl.Trace:
    torch.manual_seed(0)
    model = _Model(layout).eval()
    x = torch.randn(padding.shape[0], padding.shape[1], _HIDDEN)
    return tl.trace(model, x, input_kwargs={"attention_mask": padding})


_PADDED = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])


def test_bert_key_padding_mask_reads_each_batch_element() -> None:
    trace = _trace("bert", _PADDED)
    padded = tviz.attention_view(trace, "attn", batch_index=1)
    assert padded.mask is not None
    assert padded.mask.source == "eager_additive_mask"
    expected = (_PADDED[1] == 0).expand(5, 5)
    assert torch.equal(padded.mask.mask, expected)
    full = tviz.attention_view(trace, "attn", batch_index=0)
    assert full.mask is not None
    assert not bool(full.mask.mask.any())


def test_gpt2_double_mask_is_causal_and_its_scores_decompose() -> None:
    trace = _trace("gpt2", torch.ones(1, 5, dtype=torch.long))
    scores = trace.modules["attn"].facets["scores"].value
    assert bool(torch.isneginf(scores).any())
    view = tviz.attention_view(trace, "attn")
    assert view.mask is not None
    causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.equal(view.mask.mask, causal)
    masked = tviz.score_decomposition(trace, "attn", head=1, destination=1, source=3)
    assert masked.additive_terms == {"mask": float("-inf")}
    assert masked.total == masked.reference_score == float("-inf")
    open_pair = tviz.score_decomposition(trace, "attn", head=1, destination=3, source=1)
    assert open_pair.additive_terms == {}
    assert open_pair.total == pytest.approx(open_pair.reference_score, abs=1e-5)


def test_scalar_fill_is_not_read_as_a_mask() -> None:
    trace = _trace("scalar", torch.ones(1, 5, dtype=torch.long))
    assert tviz.attention_view(trace, "attn").mask is None

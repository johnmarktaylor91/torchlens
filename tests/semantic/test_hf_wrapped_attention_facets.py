"""Attention facets on Hugging Face's wrapped self-attention layout.

transformers 4.x ``ViTAttention`` and ``BertAttention`` (every transformers
line) keep q/k/v one level down, on a core child (``attention`` / ``self``),
and the output projection at ``output.dense`` beside it. The classes below
mirror those layouts by name so the built-in recipes route them, independent
of the installed transformers version.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.transformer as tviz
from torchlens.errors import TorchLensWarning
from torchlens.mechinterp._heads import _attention_modules

pytestmark = pytest.mark.usefixtures("root_edits_in_place")

_HEADS = 2
_D_HEAD = 4
_HIDDEN = _HEADS * _D_HEAD


class _EagerCore(nn.Module):
    """Eager q/k/v self-attention core with HF attribute names."""

    def __init__(self) -> None:
        """Initialize q/k/v projections and head metadata."""

        super().__init__()
        self.num_attention_heads = _HEADS
        self.attention_head_size = _D_HEAD
        self.query = nn.Linear(_HIDDEN, _HIDDEN)
        self.key = nn.Linear(_HIDDEN, _HIDDEN)
        self.value = nn.Linear(_HIDDEN, _HIDDEN)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``[B, H, S, d_head]`` from ``[B, S, hidden]``."""

        return x.view(*x.shape[:-1], _HEADS, _D_HEAD).transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run eager matmul/softmax/matmul attention and merge heads."""

        q = self._split(self.query(hidden))
        k = self._split(self.key(hidden))
        v = self._split(self.value(hidden))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(_D_HEAD)
        context = torch.matmul(torch.softmax(scores, dim=-1), v)
        return context.transpose(1, 2).reshape(*hidden.shape[:-1], _HIDDEN)


class ViTSelfAttention(_EagerCore):
    """transformers 4.x ViT q/k/v core."""


class BertSelfAttention(_EagerCore):
    """BERT q/k/v core."""


class ViTSelfOutput(nn.Module):
    """transformers 4.x ViT output projection (no residual inside)."""

    def __init__(self) -> None:
        """Initialize the dense projection."""

        super().__init__()
        self.dense = nn.Linear(_HIDDEN, _HIDDEN)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Project the merged heads."""

        return self.dense(context)


class BertSelfOutput(nn.Module):
    """BERT output projection, residual add and LayerNorm."""

    def __init__(self) -> None:
        """Initialize the dense projection and LayerNorm."""

        super().__init__()
        self.dense = nn.Linear(_HIDDEN, _HIDDEN)
        self.LayerNorm = nn.LayerNorm(_HIDDEN)

    def forward(self, context: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Project, add the residual and normalize."""

        return self.LayerNorm(self.dense(context) + hidden)


class ViTAttention(nn.Module):
    """transformers 4.x ViT wrapper: ``attention`` core plus ``output.dense``."""

    def __init__(self) -> None:
        """Initialize the core and output children."""

        super().__init__()
        self.attention = ViTSelfAttention()
        self.output = ViTSelfOutput()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the core, then the output projection."""

        return self.output(self.attention(hidden))


class BertAttention(nn.Module):
    """BERT wrapper: ``self`` core plus ``output.dense``."""

    def __init__(self) -> None:
        """Initialize the core and output children."""

        super().__init__()
        self.self = BertSelfAttention()
        self.output = BertSelfOutput()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the core, then the projection with residual and LayerNorm."""

        return self.output(self.self(hidden), hidden)


class _Block(nn.Module):
    """One attention wrapper under the stable address ``attn``, plus a head."""

    def __init__(self, wrapper: nn.Module) -> None:
        """Initialize with a wrapper block."""

        super().__init__()
        self.attn = wrapper
        self.head = nn.Linear(_HIDDEN, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the wrapper and a linear readout."""

        return self.head(self.attn(x))


def _trace(wrapper: nn.Module, seed: int) -> tuple[nn.Module, torch.Tensor, Any]:
    """Return ``(model, x, trace)`` for a seeded one-block model."""

    torch.manual_seed(seed)
    model = _Block(wrapper).eval()
    x = torch.randn(2, 5, _HIDDEN)
    return model, x, tl.trace(model, x, save="all")


def _output(trace: Any) -> torch.Tensor:
    """Return the trace's first output tensor."""

    return trace[trace.output_layers[0]].out


def _zero_head_reference(model: nn.Module, core: str, x: torch.Tensor, head: int) -> torch.Tensor:
    """Return the plain-PyTorch output with ``head`` zeroed in query, key and value."""

    def zero_head(_module: nn.Module, _inputs: Any, out: torch.Tensor) -> torch.Tensor:
        out = out.clone()
        out.view(*out.shape[:-1], _HEADS, _D_HEAD)[..., head, :] = 0
        return out

    core_module = getattr(model.attn, core)
    handles = [
        getattr(core_module, name).register_forward_hook(zero_head)
        for name in ("query", "key", "value")
    ]
    try:
        with torch.no_grad():
            return model(x)
    finally:
        for handle in handles:
            handle.remove()


def _metric(trace: Any) -> torch.Tensor:
    """Return one scalar of the trace output."""

    return _output(trace)[0, 0, 0]


@pytest.mark.parametrize(
    ("wrapper_cls", "core"), [(ViTAttention, "attention"), (BertAttention, "self")]
)
def test_wrapper_serves_core_head_facets_and_module_head_edit(
    wrapper_cls: type[nn.Module], core: str
) -> None:
    """The wrapper serves q/k/v per head, and ``module & head`` edits them.

    Before the fix the 4.x ViT wrapper (routed to the q_proj recipe) and the
    BERT wrapper (no recipe) served no q/k/v and the selector matched 0
    write-capable facets.
    """

    model, x, clean = _trace(wrapper_cls(), seed=3)
    wrapper = clean.modules["attn"].facets
    inner = clean.modules[f"attn.{core}"].facets
    for name in ("q", "k", "v"):
        assert tuple(wrapper[name].shape) == (2, 5, _HEADS, _D_HEAD)
        assert torch.equal(wrapper[name], inner[name])
    assert wrapper.n_heads == _HEADS
    assert wrapper.d_head == _D_HEAD
    assert torch.equal(wrapper.pattern, inner.pattern)

    site = tl.module("attn") & tl.head(1)
    with pytest.warns(TorchLensWarning, match=r"writes 3 sites"):
        ablated = clean.fork().do(site, tl.zero_ablate())
    edited_q = ablated.modules["attn"].facets.q
    assert torch.count_nonzero(edited_q[..., 1, :]) == 0
    assert torch.equal(edited_q[..., 0, :], wrapper.q[..., 0, :])
    assert not torch.allclose(_output(ablated), _output(clean))
    reference = _zero_head_reference(model, core, x, head=1)
    assert torch.allclose(_output(ablated), reference, atol=1e-6)


def test_bert_wrapper_attn_out_is_the_pre_residual_projection() -> None:
    """``attn_out`` on the wrapper is ``output.dense``, not the LayerNorm output."""

    _model, _x, clean = _trace(BertAttention(), seed=5)
    dense_out = clean.modules["attn.output.dense"].out
    assert torch.equal(clean.modules["attn"].facets.attn_out, dense_out)
    assert not torch.allclose(clean.modules["attn"].out, dense_out)


@pytest.mark.parametrize(
    ("wrapper_cls", "core"), [(ViTAttention, "attention"), (BertAttention, "self")]
)
def test_wrapper_and_core_result_read_the_shared_output_projection(
    wrapper_cls: type[nn.Module], core: str
) -> None:
    """Per-head ``result`` uses ``output.dense`` on the wrapper and on the core.

    The rows sum (plus bias) to the captured projection output; the core
    reads the projection its wrapper holds, so the two rows agree.
    """

    model, _x, clean = _trace(wrapper_cls(), seed=7)
    wrapper_result = clean.modules["attn"].facets.result
    core_result = clean.modules[f"attn.{core}"].facets.result
    assert tuple(wrapper_result.shape) == (2, 5, _HEADS, _HIDDEN)
    assert torch.allclose(wrapper_result, core_result)
    summed = wrapper_result.sum(dim=-2) + model.attn.output.dense.bias.detach()
    assert torch.allclose(summed, clean.modules["attn.output.dense"].out, atol=1e-5)


def test_whole_model_head_counts_each_bert_head_once() -> None:
    """Wrapper and core both match ``tl.head``; each q/k/v home is written once."""

    model, x, clean = _trace(BertAttention(), seed=11)
    with pytest.warns(TorchLensWarning, match=r"tl\.head\(1\) writes 3 sites") as fanout:
        ablated = clean.fork().do(tl.head(1), tl.zero_ablate())
    assert [w.message.fields["site_count"] for w in fanout] == [3]
    for address in ("attn", "attn.self"):
        view = ablated.modules[address].facets
        for name in ("q", "k", "v"):
            assert torch.count_nonzero(view.head(1)[name]) == 0
    reference = _zero_head_reference(model, "self", x, head=1)
    assert torch.allclose(_output(ablated), reference, atol=1e-6)


@pytest.mark.parametrize(
    ("wrapper_cls", "core"), [(ViTAttention, "attention"), (BertAttention, "self")]
)
def test_layer_listings_count_wrapper_and_core_as_one_layer(
    wrapper_cls: type[nn.Module], core: str
) -> None:
    """Per-layer listings return the core once, not the wrapper as a second layer.

    Attention views, the head-contribution scan and the per-layer patching
    sweeps each list one entry per attention layer.
    """

    model, x, clean = _trace(wrapper_cls(), seed=17)
    views = tviz.attention_views(clean)
    assert [view.layer for view in views] == [f"attn.{core}"]
    assert [module.address for module in _attention_modules(clean)] == [f"attn.{core}"]
    corrupted = torch.randn(2, 5, _HIDDEN)
    patching = tl.facets.patching
    by_layer = patching.activation_patch_attention_output(model, x, corrupted, _metric)
    assert tuple(by_layer.shape) == (1,)
    by_head = patching.activation_patch_attention_heads(model, x, corrupted, _metric)
    assert tuple(by_head.shape) == (1, _HEADS)


def test_bert_core_alone_keeps_its_head_edit() -> None:
    """``tl.module("attn.self") & tl.head(i)`` behaves as before the wrapper recipe."""

    _model, _x, clean = _trace(BertAttention(), seed=13)
    keys = set(clean.modules["attn.self"].facets.keys())
    assert {"q", "k", "v", "pattern", "scores", "z", "attn_out", "input"} <= keys
    with pytest.warns(TorchLensWarning, match=r"writes 3 sites"):
        ablated = clean.fork().do(tl.module("attn.self") & tl.head(0), tl.zero_ablate())
    assert torch.count_nonzero(ablated.modules["attn.self"].facets.head(0).q) == 0
    assert not torch.allclose(_output(ablated), _output(clean))

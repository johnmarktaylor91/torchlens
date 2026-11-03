"""``tl.module(address) & tl.head(i)`` is the scoped facet selector at every door.

Walk ruling RC-252: composing a module selector with a head (facet) selector,
in either order, must mean what ``tl.head(i).in_module(address)`` means wherever
a selector is accepted: ``trace[sel]``, ``tl.when``, ``do`` and ``attach_hooks``.
The attention block is the two-head ``MultiHeadSelfAttention`` of the SAE
multi-call tests, called once here so module scoping is unambiguous.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.test_sae4_multi_call_modules import MultiHeadSelfAttention, _head_trace, _output
from torchlens._selection.errors import SelectorCompositionError
from torchlens._selection.selectors import FacetSelector
from torchlens.options import InterventionOptions


class _AttnOnce(nn.Module):
    """One attention block between two activations."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.attn(x))


def _case() -> tuple[nn.Module, torch.Tensor, tl.Trace]:
    torch.manual_seed(4)
    model, x = _AttnOnce().eval(), torch.randn(2, 3, 8)
    return model, x, _head_trace(model, x)


def _outcome(fn: Any) -> Any:
    """The value ``fn`` returns, or the typed code and class of what it raises."""

    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - the comparison wants the refusal itself
        return (type(exc), getattr(exc, "fields", {}).get("code"))


def test_module_and_head_fold_into_the_scoped_facet_selector() -> None:
    folded = tl.module("attn") & tl.head(1)
    reversed_fold = tl.head(1) & tl.module("attn")
    explicit = tl.head(1).in_module("attn")

    assert isinstance(folded, FacetSelector)
    assert folded == explicit
    assert reversed_fold == explicit
    assert (folded.name, folded.head_index, folded.module_address) == (None, 1, "attn")
    named = tl.module("attn") & tl.head(0, "q")
    assert named == tl.head(0, "q").in_module("attn")
    assert tl.module("attn") & tl.head(1).in_module("attn") == explicit
    with pytest.raises(SelectorCompositionError) as caught:
        tl.module("attn") & tl.head(1).in_module("act")
    assert caught.value.fields["code"] == "selector_module_facet_scope_conflict"


def test_folded_selector_resolves_like_the_explicit_spelling_at_the_trace_door() -> None:
    _model, _x, trace = _case()
    folded = tl.module("attn") & tl.head(1)
    explicit = tl.head(1).in_module("attn")

    assert _outcome(lambda: trace[folded]) == _outcome(lambda: trace[explicit])
    assert _outcome(lambda: trace.find_sites(folded)) == _outcome(
        lambda: trace.find_sites(explicit)
    )


def test_folded_selector_writes_at_attach_hooks_and_do() -> None:
    model, x, clean = _case()
    clean_out = _output(clean).clone()

    hooked = clean.fork("hooked")
    hooked.attach_hooks(tl.module("attn") & tl.head(1), tl.zero_ablate())
    hooked.run(model, x)
    assert torch.count_nonzero(hooked.modules["attn"].facets.head(1).q) == 0
    assert not torch.allclose(_output(hooked), clean_out)

    explicit = clean.fork("explicit")
    explicit.attach_hooks(tl.head(1).in_module("attn"), tl.zero_ablate())
    explicit.run(model, x)
    assert torch.equal(_output(hooked), _output(explicit))

    done = clean.fork("do").do(
        tl.head(1) & tl.module("attn"),
        tl.zero_ablate(),
        model=model,
        inputs=x,
        intervention=InterventionOptions(engine="rerun"),
    )
    assert torch.equal(_output(done), _output(explicit))


def test_folded_selector_binds_through_when() -> None:
    model, x, _trace = _case()

    with torch.no_grad():
        folded = tl.when(tl.module("attn") & tl.head(1), tl.zero_ablate()).bind(model)(x)
        explicit = tl.when(tl.head(1).in_module("attn"), tl.zero_ablate()).bind(model)(x)
        plain = model(x)
    assert torch.equal(folded, explicit)
    assert not torch.equal(folded, plain)

"""Facet writes on a model that calls one module twice.

A module record that ran more than once raises ``module_call_ambiguous`` when
its ``facets`` view is read, so the facet write door skips such records unless
a selector is scoped to one, which refuses with
``facet_selector_module_multi_call``. The ops of each call still serve, so an
unscoped head selector and an SAE latent both write on a model whose one
``nn.ReLU`` runs twice. Latent expectations come from the hand-set SAE of
``test_sae4_latent_facets`` (``fc1`` is the identity, so the site equals X).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.errors import TorchLensWarning

pytestmark = pytest.mark.usefixtures("root_edits_in_place")  # D.12: no per-call mutation keyword

#: Stage a sticky hook on the trace without propagating it.
_STAGE = tl.options.InterventionOptions(engine="set_only")


class MultiHeadSelfAttention(nn.Module):
    """Tiny attention block matching the built-in DistilBERT head recipe's class name."""

    def __init__(self) -> None:
        super().__init__()
        self.n_heads = 2
        self.dim = 8
        self.q_lin = nn.Linear(8, 8)
        self.k_lin = nn.Linear(8, 8)
        self.v_lin = nn.Linear(8, 8)
        self.out_lin = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_lin(self.q_lin(x) + self.k_lin(x) + self.v_lin(x))


class _AttnReluTwice(nn.Module):
    """``act`` (one ``nn.ReLU``) runs before and after the attention block."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.attn(self.act(x)))


def _head_trace(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    return tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(save_args=True),
        save="all",
    )


def _output(trace: tl.Trace) -> torch.Tensor:
    return trace[trace.output_layers[0]].out


def test_head_selector_writes_on_a_model_that_calls_a_relu_twice() -> None:
    torch.manual_seed(4)
    model, x = _AttnReluTwice().eval(), torch.randn(2, 3, 8)
    clean = _head_trace(model, x)
    clean_out = _output(clean).clone()
    edited = clean.fork("head")
    with pytest.warns(TorchLensWarning, match=r"tl\.head\(1\) writes 3 sites") as fanout:
        edited.do({tl.head(1): tl.zero_ablate()}, intervention=_STAGE)
    assert [w.message.fields["code"] for w in fanout] == ["facet_selector_fanout"]
    edited.run(model, x)
    assert not torch.allclose(_output(edited), clean_out)
    assert torch.count_nonzero(edited.modules["attn"].facets.head(1).q) == 0


def test_a_selector_scoped_to_a_module_that_ran_twice_refuses_naming_its_calls() -> None:
    torch.manual_seed(4)
    model, x = _AttnReluTwice().eval(), torch.randn(2, 3, 8)
    clean = _head_trace(model, x)
    with pytest.raises(InvalidArgumentError) as caught:
        clean.fork("scoped").do(
            tl.head(1).in_module("act"),
            tl.zero_ablate(),
            intervention=tl.options.InterventionOptions(engine="set_only"),
        )
    assert caught.value.fields["code"] == "facet_selector_module_multi_call"
    calls = caught.value.fields["call_addresses"]
    assert tuple(calls) == ("act:1", "act:2")
    assert f"trace[{calls[0]!r}].facets" in str(caught.value)
    # The remedy runs: each call's facets are read at its call-qualified address.
    for address in calls:
        assert isinstance(clean[address].facets.keys(), list)

"""Zero-source ``scatter_add``: the index cannot reach the output.

A recurrent graph cell starts from a zero hidden state and aggregates it along
edges with ``new_zeros(...).scatter_add_(0, index.expand_as(msg), msg)``. With
``msg`` exactly zero, every valid index adds zero, so perturbing the index
leaves the output unchanged and validation failed the edge as
``perturbation_insensitive``. The ``zero_source_index_irrelevant`` value proof
reads the saved source (all zero, no tolerance) and the saved output (equal to
the saved destination). A non-zero source, a perturbed destination or source,
and a frozen replay over a non-zero source all still fail.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import torch
import torch.nn as nn
from _validation_capture import _capture

from torchlens.validation import validate_forward_pass
from torchlens.validation._saved_value_proofs import _zero_source_add_decision
from torchlens.validation.core import _check_whether_func_on_saved_parents_yields_saved_tensor


class _EdgeAggregate(nn.Module):
    """Aggregate a hidden state along edges; ``zero_state`` mimics a first step."""

    def __init__(self, zero_state: bool) -> None:
        super().__init__()
        self.zero_state = zero_state
        self.register_buffer("src", torch.tensor([0, 1, 2, 3, 1]))
        self.register_buffer("dst", torch.tensor([1, 2, 3, 0, 0]))
        self.register_buffer("norm", torch.tensor([[0.5], [1.0], [2.0], [1.5], [0.25]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.zeros_like(x) if self.zero_state else x * 2.0
        msg = hidden.index_select(0, self.src) * self.norm
        index = self.dst.view(-1, 1).expand_as(msg)
        return msg.new_zeros(x.shape).scatter_add_(0, index, msg) + x


def _scatter_and_index(model: nn.Module):
    trace, _ = _capture(model, torch.arange(8.0).reshape(4, 2) + 1.0)
    scatter = [op for op in trace._layer_list if op.func_name == "scatter_add_"][0]
    return trace, scatter, scatter.parent_arg_positions["args"][2]


def _perturb_index(trace, scatter, index_label: str):
    return _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, scatter.label, perturb=True, layers_to_perturb=[index_label]
    )


def test_zero_source_scatter_add_index_is_exempt() -> None:
    """The zero-state aggregation's index edge is proved irrelevant, and the model validates."""

    trace, scatter, index_label = _scatter_and_index(_EdgeAggregate(zero_state=True))
    result = _perturb_index(trace, scatter, index_label)
    assert result.decision == "exempted"
    assert result.reason == "zero_source_index_irrelevant"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(3):
            model = _EdgeAggregate(zero_state=True)
            assert validate_forward_pass(model, torch.ones(4, 2), seed=seed) is True


def test_zero_source_proof_refuses_live_and_frozen_edges() -> None:
    """A non-zero source, other slots, a forged output and a frozen live edge are not proved."""

    trace, scatter, index_label = _scatter_and_index(_EdgeAggregate(zero_state=False))
    assert _perturb_index(trace, scatter, index_label).reason == "perturbation_changed"
    saved = scatter.out.detach().clone()
    object.__setattr__(scatter, "func", lambda *args, **kwargs: saved.clone())
    frozen = _perturb_index(trace, scatter, index_label)
    assert (frozen.decision, frozen.reason) == ("failed", "perturbation_insensitive")

    destination = torch.zeros(4, 2)
    index = torch.tensor([[1, 1], [2, 2], [3, 3], [0, 0], [0, 0]])
    zero_source = torch.zeros(5, 2)
    args = (destination, 0, index, zero_source)
    layer = SimpleNamespace(out=destination.clone(), saved_kwargs={})
    assert _zero_source_add_decision(layer, {2}, args).exempt
    assert not _zero_source_add_decision(layer, {3}, args).exempt
    assert not _zero_source_add_decision(layer, {0}, args).exempt
    assert not _zero_source_add_decision(layer, {2, 3}, args).exempt
    live_source = zero_source.clone()
    live_source[4, 1] = 1e-30
    assert not _zero_source_add_decision(layer, {2}, (destination, 0, index, live_source)).exempt
    forged = SimpleNamespace(out=torch.ones(4, 2), saved_kwargs={})
    assert not _zero_source_add_decision(forged, {2}, args).exempt
    scaled = SimpleNamespace(out=destination.clone(), saved_kwargs={"alpha": 2})
    assert not _zero_source_add_decision(scaled, {2}, args).exempt

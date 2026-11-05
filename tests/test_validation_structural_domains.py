"""Perturbed replays that kept raising on a structural input now reach a verdict.

``embedding_bag`` offsets must start at 0 and stay ordered, and a traced
``view`` size next to a ``-1`` wildcard must divide what the wildcard absorbs.
Every generic draw and +1/-1 step broke that contract, so the edge ended
``perturbation_execution_exception`` (unverified). The in-domain moves keep the
call valid; a live edge then validates and a frozen (dead) replay still fails.
The fairseq shape read (``shape_as_tensor`` feeding ``view(bsz, seq_len, -1)``)
also crashed the receptive-field walk, which applied the data operand's axis
map to the 0-d size parents.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.receptive_field._validation import check_geometric_metadata_invariants
from torchlens.validation import validate_forward_pass
from torchlens.validation._core import _check_whether_func_on_saved_parents_yields_saved_tensor
from torchlens.validation._structural_domains import (
    embedding_bag_offsets_values,
    reshape_size_values,
)


class _BagWithOffsets(nn.Module):
    """FastText-style bag: token ids and bag offsets are model inputs."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.bag = nn.EmbeddingBag(50, 8, mode="mean")
        self.fc = nn.Linear(8, 3)

    def forward(self, text: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bag(text, offsets))


class _ShapeReadView(nn.Module):
    """fairseq ``SinusoidalPositionalEmbedding``: sizes read by ``shape_as_tensor``."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.register_buffer("weights", torch.randn(64, 16))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bspair = torch.onnx.operators.shape_as_tensor(tokens)
        bsz, seq_len = bspair[0], bspair[1]
        positions = torch.cumsum(torch.ones_like(tokens), 1) + 1
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1)


def _bag_inputs() -> list[torch.Tensor]:
    return [torch.tensor([3, 9, 4, 11, 7, 2, 30, 41, 5, 8, 1, 6]), torch.tensor([0, 3, 7])]


def _capture(model: nn.Module, inputs: list[torch.Tensor]) -> Any:
    torch.manual_seed(0)
    return tl.trace(model, inputs, capture=CaptureOptions(save_args=True, seed=0), save="all")


def _op(trace: Any, func_name: str, index: int = 0) -> Any:
    return [op for op in trace._layer_list if op.func_name == func_name][index]


def _perturb(trace: Any, op: Any, label: str) -> Any:
    return _check_whether_func_on_saved_parents_yields_saved_tensor(
        trace, op.label, perturb=True, layers_to_perturb=[label]
    )


def _freeze(op: Any) -> None:
    """Make ``op``'s replay return its saved call's result whatever its inputs (a dead edge)."""

    args = [a.clone() if isinstance(a, torch.Tensor) else a for a in op.call.saved_args]
    saved = op.func(*args, **dict(op.call.saved_kwargs or {}))

    def frozen(*_args: Any, **_kwargs: Any) -> Any:
        if isinstance(saved, tuple):
            return tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in saved)
        return saved.clone()

    object.__setattr__(op, "func", frozen)


def _quiet_validate(model: nn.Module, inputs: list[torch.Tensor]) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return validate_forward_pass(model, inputs, seed=0)


def test_embedding_bag_with_offsets_validates() -> None:
    assert _quiet_validate(_BagWithOffsets().eval(), _bag_inputs()) is True


@pytest.mark.parametrize("output_index", [0, 1, 2])
def test_offsets_edge_is_verified_and_a_frozen_replay_still_fails(output_index: int) -> None:
    trace = _capture(_BagWithOffsets().eval(), _bag_inputs())
    bags = [op for op in trace._layer_list if op.func_name == "embedding_bag"]
    op = next(bag for bag in bags if (bag.multi_output_index or 0) == output_index)
    offsets_label = op.call.parent_arg_positions["args"][2]
    verified = _perturb(trace, op, offsets_label)
    assert verified.decision == "validated", (verified.decision, verified.reason)
    _freeze(op)
    frozen = _perturb(trace, op, offsets_label)
    assert frozen.decision == "failed", (frozen.decision, frozen.reason)


def test_offsets_move_keeps_the_contract() -> None:
    trace = _capture(_BagWithOffsets().eval(), _bag_inputs())
    op = _op(trace, "embedding_bag")
    moved = embedding_bag_offsets_values(op, torch.tensor([0, 3, 7]))
    assert moved is not None and moved.tolist() == [0, 2, 7]
    assert embedding_bag_offsets_values(op, torch.tensor([0])).tolist() == [0, 6]
    assert embedding_bag_offsets_values(op, torch.tensor([0, 0, 0])).tolist() == [0, 0, 1]


def test_single_bag_offsets_edge_is_verified_and_a_frozen_replay_still_fails() -> None:
    """One bag admits only ``offsets == [0]``; splitting it is the in-domain move."""

    text, _offsets = _bag_inputs()
    model = _BagWithOffsets().eval()
    assert _quiet_validate(model, [text, torch.tensor([0])]) is True
    trace = _capture(model, [text, torch.tensor([0])])
    op = _op(trace, "embedding_bag")
    offsets_label = op.call.parent_arg_positions["args"][2]
    verified = _perturb(trace, op, offsets_label)
    assert verified.decision == "validated", (verified.decision, verified.reason)
    _freeze(op)
    frozen = _perturb(trace, op, offsets_label)
    assert frozen.decision == "failed", (frozen.decision, frozen.reason)


class _PaddedBag(nn.Module):
    """torch_frame-style bag: mean mode, ``padding_idx=0`` and singleton bags led by padding."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.bag = nn.EmbeddingBag(3, 4, mode="mean", padding_idx=0)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.bag(ids, offsets)


def test_offsets_move_skips_a_padding_entry() -> None:
    """Handing the padding entry between the first two bags changes nothing; the move skips it."""

    inputs = [torch.tensor([0, 1, 2, 0, 1, 2]), torch.tensor([0, 1, 1, 3, 4, 4])]
    model = _PaddedBag().eval()
    assert _quiet_validate(model, inputs) is True
    trace = _capture(model, inputs)
    op = _op(trace, "embedding_bag")
    assert embedding_bag_offsets_values(op, inputs[1]).tolist() == [0, 1, 2, 3, 4, 4]
    offsets_label = op.call.parent_arg_positions["args"][2]
    verified = _perturb(trace, op, offsets_label)
    assert verified.decision == "validated", (verified.decision, verified.reason)
    _freeze(op)
    frozen = _perturb(trace, op, offsets_label)
    assert frozen.decision == "failed", (frozen.decision, frozen.reason)


class _RepeatToSize(nn.Module):
    """torch_frame offsets to bag ids: ``repeat_interleave`` with a traced ``output_size``."""

    def forward(self, lengths: torch.Tensor) -> torch.Tensor:
        ids = torch.arange(lengths.numel())
        return torch.repeat_interleave(ids, lengths, output_size=lengths.sum()) * 1.0


def test_repeats_edge_with_pinned_total_is_verified_and_a_frozen_replay_still_fails() -> None:
    inputs = [torch.tensor([1, 0, 2, 1, 0, 2])]
    trace = _capture(_RepeatToSize().eval(), inputs)
    op = _op(trace, "repeat_interleave")
    repeats_label = op.call.parent_arg_positions["args"][1]
    verified = _perturb(trace, op, repeats_label)
    assert verified.decision == "validated", (verified.decision, verified.reason)
    _freeze(op)
    frozen = _perturb(trace, op, repeats_label)
    assert frozen.decision == "failed", (frozen.decision, frozen.reason)


def test_shape_read_view_validates_and_walks_the_receptive_field() -> None:
    model, inputs = _ShapeReadView().eval(), [torch.ones(1, 12, dtype=torch.long)]
    trace = tl.trace(model, inputs)
    assert check_geometric_metadata_invariants(trace)
    assert _quiet_validate(model, inputs) is True


def test_traced_view_size_edge_is_verified_and_a_frozen_replay_still_fails() -> None:
    trace = _capture(_ShapeReadView().eval(), [torch.ones(1, 12, dtype=torch.long)])
    op = next(
        op
        for op in trace._layer_list
        if op.func_name == "view" and len(op.call.parent_arg_positions["args"]) == 3
    )
    seq_label = op.call.parent_arg_positions["args"][2]
    assert reshape_size_values(op, seq_label, torch.tensor(12)).item() in {8, 16}
    verified = _perturb(trace, op, seq_label)
    assert verified.decision == "validated", (verified.decision, verified.reason)
    _freeze(op)
    frozen = _perturb(trace, op, seq_label)
    assert frozen.decision == "failed", (frozen.decision, frozen.reason)

"""End to end, order-reading and counting edges validate when the default draw cannot move them.

The default random draw usually moves an ``argmax`` over token ids or the true
count of a mask, so a seeded end-to-end run passes whether or not the retry
ladder can move the child. Here the draw is pinned to the minimal ``step_up``,
which keeps every integer order, every equality and the true count of a
balanced mask: the run validates only if a retry rung moves the child, as when
an unlucky draw left it unchanged. Without the non-uniform rungs these edges
fail with ``perturbation_insensitive``.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from torchlens.validation import core, validate_forward_pass


class _EndTokenPosition(nn.Module):
    """``ids.argmax(-1)``: the end-of-sequence id is the largest token id."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.int32).argmax(dim=-1)


class _MaskCount(nn.Module):
    """``(x > 0).sum()`` over a balanced two-entry mask."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x > 0).sum()


class _RunCounts(nn.Module):
    """``unique_consecutive(ids, return_counts=True)`` over distinct sorted ids."""

    def forward(self, x: torch.Tensor):
        return torch.unique_consecutive(x.long(), return_counts=True)


class _ClassCounts(nn.Module):
    """``unique(ids, return_inverse=True, return_counts=True)`` over distinct ids."""

    def forward(self, x: torch.Tensor):
        return torch.unique(x.long(), return_inverse=True, return_counts=True)


class _ShapeOfIds(nn.Module):
    """``torch._shape_as_tensor(ids)``, as ``torch.onnx.operators.shape_as_tensor`` calls it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch._shape_as_tensor(x.long())


def _token_ids() -> torch.Tensor:
    return torch.tensor([[49406.0, 74.0, 117.0, 33.0, 49407.0]])


_CASES = {
    "argmax_over_token_ids": (_EndTokenPosition, _token_ids),
    "count_of_balanced_mask": (_MaskCount, lambda: torch.tensor([-1.0, 2.0])),
    "run_counts_of_distinct_ids": (_RunCounts, lambda: torch.tensor([0.0, 1.0, 2.0])),
    "class_counts_of_distinct_ids": (_ClassCounts, lambda: torch.tensor([3.0, 1.0, 2.0])),
    "shape_of_ids": (_ShapeOfIds, _token_ids),
}


def _order_preserving_draw(parent_values: torch.Tensor, _child_out: torch.Tensor) -> torch.Tensor:
    """Stand-in for the random draw that keeps orders, equalities and balanced counts."""

    return core._directional_step_perturb(parent_values, "step_up")


@pytest.mark.parametrize("case", sorted(_CASES))
def test_edges_validate_when_the_draw_cannot_move_the_child(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the draw pinned, only the retry ladder can supply the evidence."""

    model_type, make_input = _CASES[case]
    monkeypatch.setattr(core, "_perturb_layer_outs", _order_preserving_draw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert validate_forward_pass(model_type(), make_input(), seed=0) is True
